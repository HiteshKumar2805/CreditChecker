import streamlit as st
import pytesseract

from src.document_reader import extract_text_from_pdf, extract_text_from_image
from src.transaction_parser import parse_transactions
from src.feature_engineering import add_features

# =========================
# ⚙️ CONFIG
# =========================
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

st.set_page_config(page_title="Financial Confidence System", layout="centered")
st.title("🏦 Financial Confidence Scoring System")

# =========================
# 📤 UPLOAD FILE
# =========================
uploaded_file = st.file_uploader(
    "Upload Bank Statement (PDF or Image)",
    type=["pdf", "png", "jpg", "jpeg"]
)

# =========================
# 🧾 USER INPUTS
# =========================
cibil = st.number_input("Enter CIBIL Score", 300, 900, value=650)
income = st.number_input("Enter Monthly Income", value=30000)

# =========================
# 🚀 MAIN PIPELINE
# =========================
if uploaded_file is not None:

    file_type = uploaded_file.name.split(".")[-1].lower()

    st.info("Processing bank statement...")

    # =========================
    # 📄 TEXT EXTRACTION
    # =========================
    if file_type == "pdf":
        text = extract_text_from_pdf(uploaded_file)
    else:
        text = extract_text_from_image(uploaded_file)

    # =========================
    # 📊 TRANSACTION PARSING
    # =========================
    parsed = parse_transactions(text)

    st.subheader("📊 Extracted Transactions")
    st.write(parsed)

    # =========================
    # ⚙️ FEATURE ENGINEERING
    # =========================
    features = add_features(parsed)

    salary_flag = features["salary_flag"]
    expense_score = features["expense_score"]
    income_score = features["income_score"]
    balance_score = features["balance_score"]
    fraud_risk = features["fraud_risk"]
    transaction_volume = features["transaction_volume"]

    st.subheader("📈 Financial Signals")
    st.write(features)

    # =========================
    # 🧠 RISK SCORING SYSTEM
    # =========================
    risk_score = 0

    if salary_flag == 0:
        risk_score += 25

    if fraud_risk > 0:
        risk_score += 40

    if balance_score < 0:
        risk_score += 20

    if transaction_volume < 5:
        risk_score += 15

    final_score = 100 - risk_score

    st.subheader("🧠 Loan Decision Score")
    st.metric("Financial Confidence Score", final_score)

    # =========================
    # 🎯 DECISION ENGINE
    # =========================
    if final_score >= 70:
        st.success("✅ Loan Approved")
    elif final_score >= 40:
        st.warning("⚠️ Manual Review Required")
    else:
        st.error("❌ Loan Rejected")

else:
    st.info("Please upload a bank statement to begin analysis.")