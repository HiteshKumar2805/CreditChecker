from PyPDF2 import PdfReader
from PIL import Image
import pytesseract

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def extract_text_from_image(file):
    image = Image.open(file)
    return pytesseract.image_to_string(image)