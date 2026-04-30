"""
KYC document OCR extraction.

Supports Aadhaar/PAN uploads as PDFs or images. Text-based PDFs are read with
PyMuPDF first; scanned PDFs and image files are decoded to PIL/numpy images
before being sent to EasyOCR.
"""
import io
import logging
import re
from functools import lru_cache
from typing import Optional

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_reader():
    import easyocr

    return easyocr.Reader(["en"], gpu=False, verbose=False)


def _clean_text(text: str) -> str:
    text = text.replace("|", "I").replace("Â°", "0")
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _pdf_text(pdf_bytes: bytes) -> str:
    try:
        import fitz

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        parts = [page.get_text("text") for page in doc]
        doc.close()
        text = _clean_text("\n".join(parts))
        if len(re.sub(r"\s", "", text)) >= 30:
            return text
    except Exception as exc:
        logger.warning("[KYC] PDF text extraction failed: %s", exc)
    return ""


def _images_from_upload(file_bytes: bytes) -> list[Image.Image]:
    if file_bytes.lstrip().startswith(b"%PDF"):
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError("PyMuPDF is required to read PDF KYC documents.") from exc

        doc = fitz.open(stream=file_bytes, filetype="pdf")
        images = []
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False)
            images.append(Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB"))
        doc.close()
        return images

    try:
        return [Image.open(io.BytesIO(file_bytes)).convert("RGB")]
    except Exception as exc:
        raise ValueError("Upload a readable Aadhaar/PAN PDF or image file.") from exc


def _preprocess_for_ocr(image: Image.Image) -> list[np.ndarray]:
    max_side = 2200
    img = ImageOps.exif_transpose(image.convert("RGB"))
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

    gray = ImageOps.grayscale(img)
    gray = ImageEnhance.Contrast(gray).enhance(1.8)
    sharp = ImageEnhance.Sharpness(gray).enhance(1.5)

    return [
        np.array(img),
        np.array(sharp.convert("RGB")),
    ]


def _ocr_images(images: list[Image.Image]) -> str:
    reader = _get_reader()
    lines = []
    seen = set()

    for image in images:
        for variant in _preprocess_for_ocr(image):
            for item in reader.readtext(variant, detail=0, paragraph=False):
                line = str(item).strip()
                key = re.sub(r"\s+", " ", line.lower())
                if line and key not in seen:
                    seen.add(key)
                    lines.append(line)

    return _clean_text("\n".join(lines))


def _extract_text(file_bytes: bytes) -> str:
    text = _pdf_text(file_bytes) if file_bytes.lstrip().startswith(b"%PDF") else ""
    images = _images_from_upload(file_bytes)
    ocr_text = _ocr_images(images)
    return _clean_text("\n".join(part for part in [text, ocr_text] if part))


def _lines(text: str) -> list[str]:
    return [line.strip(" :-\t") for line in text.splitlines() if line.strip(" :-\t")]


def _normalize_date(value: str) -> str:
    return value.replace("-", "/").replace(".", "/")


def _extract_dob(text: str) -> Optional[str]:
    patterns = [
        r"(?:DOB|D\.O\.B|Date of Birth|Birth|YOB|Year of Birth)\s*[:\-]?\s*(\d{2}[\/\-.]\d{2}[\/\-.]\d{4}|\d{4})",
        r"\b(\d{2}[\/\-.]\d{2}[\/\-.]\d{4})\b",
        r"\b(?:Year of Birth|YOB)\s*[:\-]?\s*(\d{4})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return _normalize_date(match.group(1))
    return None


def _extract_gender(text: str) -> Optional[str]:
    labeled = re.search(r"\b(?:gender|sex)\s*[:\-]?\s*([mMfF])\b", text, re.IGNORECASE)
    if labeled:
        return "Female" if labeled.group(1).lower() == "f" else "Male"
    if re.search(r"\b(female|femal)\b", text, re.IGNORECASE):
        return "Female"
    if re.search(r"\b(male|maie)\b", text, re.IGNORECASE):
        return "Male"
    if re.search(r"\btransgender\b", text, re.IGNORECASE):
        return "Transgender"
    return None


def _is_probable_name(value: str) -> bool:
    value = re.sub(r"[^A-Za-z .]", " ", value).strip()
    value = re.sub(r"\s+", " ", value)
    if len(value) < 3 or len(value) > 70:
        return False
    words = value.split()
    if not 1 <= len(words) <= 5:
        return False

    blocked = {
        "government", "india", "income", "tax", "department", "aadhaar",
        "unique", "identification", "authority", "male", "female", "dob",
        "birth", "father", "signature", "permanent", "account", "number",
        "card", "address", "vid", "help", "www", "gov",
    }
    lowered = {word.lower().strip(".") for word in words}
    return not lowered & blocked


def _clean_name(value: str) -> Optional[str]:
    value = re.sub(r"[^A-Za-z .]", " ", value)
    value = re.sub(r"\b(name|father'?s name|mother'?s name)\b", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"\s+", " ", value).strip(" .")
    if _is_probable_name(value):
        return value.title()
    return None


def _extract_labeled_name(lines: list[str]) -> Optional[str]:
    for index, line in enumerate(lines):
        match = re.match(r"^(?:name|full name|applicant name)\s*[:\-]?\s*(.*)$", line, re.IGNORECASE)
        if not match:
            continue

        inline = _clean_name(match.group(1))
        if inline:
            return inline

        for candidate in lines[index + 1:index + 4]:
            cleaned = _clean_name(candidate)
            if cleaned:
                return cleaned
    return None


def _extract_name_from_aadhaar(text: str) -> Optional[str]:
    lines = _lines(text)
    labeled = _extract_labeled_name(lines)
    if labeled:
        return labeled

    context_markers = re.compile(
        r"government of india|unique identification|aadhaar|dob|date of birth|year of birth|male|female",
        re.IGNORECASE,
    )
    for index, line in enumerate(lines):
        if not context_markers.search(line):
            continue
        window = lines[max(0, index - 4): index + 5]
        for candidate in window:
            cleaned = _clean_name(candidate)
            if cleaned:
                return cleaned

    for line in lines[:12]:
        cleaned = _clean_name(line)
        if cleaned:
            return cleaned
    return None


def _extract_name_from_pan(text: str) -> Optional[str]:
    lines = _lines(text)
    labeled = _extract_labeled_name(lines)
    if labeled:
        return labeled

    pan_line_index = next((i for i, line in enumerate(lines) if _extract_pan_number(line)), None)
    if pan_line_index is not None:
        for candidate in lines[max(0, pan_line_index - 5):pan_line_index]:
            cleaned = _clean_name(candidate)
            if cleaned:
                return cleaned

    for line in lines:
        cleaned = _clean_name(line)
        if cleaned:
            return cleaned
    return None


def _extract_aadhaar_number(text: str) -> Optional[str]:
    normalized = text.upper()
    patterns = [
        r"\b([2-9]\d{3}\s*[ -]?\d{4}\s*[ -]?\d{4})\b",
        r"\b([Xx*]{4}\s*[ -]?[Xx*]{4}\s*[ -]?\d{4})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            return re.sub(r"[^0-9X*]", "", match.group(1))
    return None


def _pan_candidate_to_value(candidate: str) -> Optional[str]:
    compact = re.sub(r"[^A-Z0-9]", "", candidate.upper())
    if len(compact) < 10:
        return None

    for start in range(0, len(compact) - 9):
        chunk = compact[start:start + 10]
        corrected = (
            chunk[:5].replace("0", "O").replace("1", "I").replace("5", "S").replace("8", "B")
            + chunk[5:9].replace("O", "0").replace("I", "1").replace("S", "5").replace("B", "8")
            + chunk[9]
        )
        if re.fullmatch(r"[A-Z]{5}\d{4}[A-Z]", corrected):
            return corrected
    return None


def _extract_pan_number(text: str) -> Optional[str]:
    for line in _lines(text):
        value = _pan_candidate_to_value(line)
        if value:
            return value

    lines = _lines(text)
    for index, line in enumerate(lines):
        if re.search(r"\b(permanent account number|account number|pan)\b", line, re.IGNORECASE):
            window = " ".join(lines[index:index + 4])
            value = _pan_candidate_to_value(window)
            if value:
                return value
    return None


def _extract_mobile(text: str) -> Optional[str]:
    patterns = [
        r"(?:mobile|mob|phone|contact)\s*[:\-]?\s*(?:\+91[- ]?)?([6-9]\d{9})",
        r"\b(?:\+91[- ]?)?([6-9]\d{9})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def _extract_address(text: str) -> Optional[str]:
    lines = _lines(text)
    start_keywords = re.compile(r"\b(address|s/o|w/o|d/o|c/o|house|flat|plot|village|ward|road|street|nagar|colony)\b", re.IGNORECASE)
    stop_keywords = re.compile(r"\b(aadhaar|vid|dob|male|female|government|mobile)\b", re.IGNORECASE)

    for index, line in enumerate(lines):
        if not start_keywords.search(line):
            continue

        collected = []
        for item in lines[index:index + 7]:
            if _extract_aadhaar_number(item):
                break
            if collected and stop_keywords.search(item):
                break
            collected.append(item)

        address = ", ".join(collected)
        if len(address) > 12:
            return address
    return None


def extract_aadhaar_data(file_bytes: bytes) -> dict:
    raw = _extract_text(file_bytes)
    logger.info("[Aadhaar OCR] Extracted %s chars. Preview:\n%s", len(raw), raw[:600])
    return {
        "name": _extract_name_from_aadhaar(raw),
        "aadhaar_number": _extract_aadhaar_number(raw),
        "dob": _extract_dob(raw),
        "gender": _extract_gender(raw),
        "address": _extract_address(raw),
        "mobile": _extract_mobile(raw),
        "raw_text": raw[:1500],
    }


def extract_pan_data(file_bytes: bytes) -> dict:
    raw = _extract_text(file_bytes)
    logger.info("[PAN OCR] Extracted %s chars. Preview:\n%s", len(raw), raw[:600])
    return {
        "name": _extract_name_from_pan(raw),
        "pan_number": _extract_pan_number(raw),
        "dob": _extract_dob(raw),
        "raw_text": raw[:1500],
    }


def cross_verify(aadhaar_data: dict, pan_data: dict) -> dict:
    adob = (aadhaar_data.get("dob") or "").replace("-", "/")
    pdob = (pan_data.get("dob") or "").replace("-", "/")

    aname = (aadhaar_data.get("name") or "").lower().strip()
    pname = (pan_data.get("name") or "").lower().strip()
    if aname and pname:
        a_tokens = set(aname.split())
        p_tokens = set(pname.split())
        overlap = len(a_tokens & p_tokens)
        total = len(a_tokens | p_tokens)
        name_score = round(overlap / total, 2) if total else 0.0
    else:
        name_score = 0.0

    return {
        "dob_match": bool(adob and pdob and adob == pdob),
        "name_match_score": name_score,
        "name_match": name_score >= 0.5,
        "kyc_verified": bool(adob and pdob and adob == pdob and name_score >= 0.5),
    }
