import io
import re
from functools import lru_cache

import easyocr
import numpy as np
from PIL import Image

try:
    import fitz
except Exception:
    fitz = None


@lru_cache(maxsize=1)
def _get_reader():
    return easyocr.Reader(["en"], gpu=False)


def _images_from_upload(file_bytes):
    if file_bytes.lstrip().startswith(b"%PDF"):
        if fitz is None:
            raise ValueError("PDF payslips require PyMuPDF to be installed.")

        doc = fitz.open(stream=file_bytes, filetype="pdf")
        if doc.page_count == 0:
            raise ValueError("The uploaded PDF does not contain any pages.")

        images = []
        for page_index in range(min(doc.page_count, 2)):
            page = doc.load_page(page_index)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            images.append(np.array(image))
        return images

    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError("Upload a readable payslip image or PDF.") from exc

    return [np.array(image)]


def extract_payslip_data(file_bytes):
    """
    Extract salary and company name from a payslip image or PDF using EasyOCR.
    """
    reader = _get_reader()
    text_lines = []

    for image in _images_from_upload(file_bytes):
        result = reader.readtext(image, detail=0)
        text_lines.extend(t.strip() for t in result if t.strip())

    salary = None
    company_name = "Unknown Company"

    for line in text_lines[:8]:
        lower = line.lower()
        if (
            len(line) > 3
            and not re.search(r"\d", line)
            and "payslip" not in lower
            and "salary" not in lower
            and "statement" not in lower
        ):
            company_name = line
            break

    salary_labels = (
        "net pay",
        "net salary",
        "total salary",
        "take home",
        "amount paid",
        "gross salary",
    )

    for i, line in enumerate(text_lines):
        lower = line.lower()
        if any(label in lower for label in salary_labels):
            for j in range(i, min(i + 4, len(text_lines))):
                num_match = re.search(
                    r"(?:rs|inr|\u20b9)?\s*([\d,]+(?:\.\d{1,2})?)",
                    text_lines[j].lower(),
                )
                if num_match:
                    value = float(num_match.group(1).replace(",", ""))
                    if value > 1000:
                        salary = value
                        break
            if salary:
                break

    if not salary:
        values = []
        for line in text_lines:
            for num in re.findall(r"\d[\d,]*(?:\.\d{1,2})?", line):
                try:
                    value = float(num.replace(",", ""))
                except ValueError:
                    continue
                if value > 1000:
                    values.append(value)
        if values:
            salary = max(values)

    return {
        "company_name": company_name,
        "salary": salary,
        "text_preview": text_lines[:12],
    }
