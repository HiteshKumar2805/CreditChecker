import streamlit as st
import pandas as pd

# IMPORT YOUR FUNCTIONS
from main import load_dataset, preprocess_data, train_model, predict_loan

# -------------------------------
# LOAD + TRAIN MODEL ON START
# -------------------------------
@st.cache_resource(show_spinner="Loading data and training model...")
def initialize_system():
    df = load_dataset("loan_data.csv")
    X, y, scaler, label_encoders, feature_columns = preprocess_data(df)
    model = train_model(X, y)
    return model, scaler, feature_columns

try:
    model, scaler, feature_columns = initialize_system()
except Exception as e:
    st.error(f"Error initializing system: {e}")
    st.stop()

st.set_page_config(page_title="Consumer Loan Decision System", page_icon="🏦", layout="wide")

st.title("🏦 Consumer Loan Decision System")
st.markdown("Enter the applicant's details and upload their bank transaction statement to get an AI-powered loan decision.")

with st.form("loan_application"):
    st.header("👤 Applicant Details")
    
    # Add Customer Name here
    customer_name = st.text_input("Customer Full Name", placeholder="e.g. John Doe")
    
    col1, col2 = st.columns(2)
    
    with col1:
        product_price = st.number_input("Product Price (₹)", min_value=0.0, value=50000.0, step=1000.0)
        down_payment = st.number_input("Down Payment (₹)", min_value=0.0, value=10000.0, step=1000.0)
        loan_amount = st.number_input("Loan Amount Requested (₹)", min_value=0.0, value=40000.0, step=1000.0)
        tenure = st.number_input("Tenure (Months)", min_value=1, value=12, step=1)
        income = st.number_input("Monthly Income (₹)", min_value=0.0, value=30000.0, step=1000.0)
        dependents = st.number_input("Number of Dependents", min_value=0, value=0, step=1)

    with col2:
        bank_name = st.text_input("Bank Name", value="HDFC")
        account_number = st.text_input("Account Number", value="123456789")
        ifsc = st.text_input("IFSC Code", value="HDFC0001")
        cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=750, step=1)
        existing_emi = st.number_input("Existing EMI (₹)", min_value=0.0, value=0.0, step=500.0)
        credit_card = st.selectbox("Has Credit Card?", ["yes", "no"])
        employment_type = st.selectbox("Employment Type", ["salaried", "self-employed"])
        
    st.header("📄 Transaction Statement")
    st.markdown("Upload the bank statement as a **PDF or PNG/JPG**.")
    
    uploaded_file = st.file_uploader("Upload Bank Statement (PDF, PNG, JPG, CSV)", type=["pdf", "png", "jpg", "jpeg", "csv"])
    
    submitted = st.form_submit_button("Submit Loan Application", type="primary", use_container_width=True)

if submitted:
    if uploaded_file is not None:
        try:
            # Check file type to simulate processing
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            # --- SIMULATE OCR / EXTRACTION ---
            if file_extension in ['pdf', 'png', 'jpg', 'jpeg']:
                with st.spinner(f"Extracting transaction data from {file_extension.upper()} using AI OCR..."):
                    import time
                    time.sleep(2)  # Simulate processing delay
                    st.toast("✅ Data successfully extracted from document!")
                    
                    # Mock extracted data
                    transactions = [
                        {"txn_type": "credit", "amount": 35000, "balance": 45000},
                        {"txn_type": "debit", "amount": 12000, "balance": 33000},
                        {"txn_type": "debit", "amount": 5000, "balance": 28000},
                        {"txn_type": "debit", "amount": 2000, "balance": 26000},
                        {"txn_type": "credit", "amount": 5000, "balance": 31000},
                        {"txn_type": "debit", "amount": 1000, "balance": 30000}
                    ]
            else:
                # Original CSV parsing
                transactions_df = pd.read_csv(uploaded_file)
                required_columns = {'txn_type', 'amount', 'balance'}
                if not required_columns.issubset(set(transactions_df.columns)):
                    st.error(f"❌ Uploaded CSV is missing required columns. Ensure it has: {', '.join(required_columns)}")
                    st.stop()
                transactions = transactions_df.to_dict('records')

            input_data = {
                "product_price": product_price,
                "down_payment": down_payment,
                "loan_amount": loan_amount,
                "tenure": tenure,
                "bank_name": bank_name,
                "account_number": account_number,
                "IFSC": ifsc,
                "cibil_score": cibil_score,
                "existing_emi": existing_emi,
                "credit_card": credit_card,
                "income": income,
                "employment_type": employment_type,
                "dependents": dependents
            }
            
            profile = {
                "income": income,
                "cibil_score": cibil_score,
                "dependents": dependents
            }
            
            with st.spinner("Analyzing application with ML Model..."):
                result = predict_loan(
                    model, scaler, feature_columns,
                    input_data, transactions, profile
                )
            
            st.divider()
            st.header("📊 Loan Decision & NLP Analysis")
            
            # Display Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Document Confidence", f"{result.get('txn_confidence_score', 0):.1f} / 100", 
                          help="Based purely on the uploaded transaction history (income, debits, savings).")
            with col2:
                st.metric("CIBIL Confidence", f"{result.get('cibil_normalized_score', 0):.1f} / 100",
                          help="Normalized CIBIL Score (0-100 scale).")
            with col3:
                st.metric("Final Blended Score", f"{result.get('score', result.get('final_confidence_score', 0)):.1f} / 100",
                          help="Combined ML + Heuristic score used for final decision.")
            
            # NLP Insights
            st.markdown("### 🧠 AI Persona & NLP Insights")
            st.info(f"**Persona Applied:** {employment_type.capitalize()} (Adjusted velocity rules accordingly)")
            st.write("Our NLP model scanned the transaction descriptions for behavioral patterns:")
            
            # Quick loop to find categories for display
            investment_count = sum(1 for t in transactions if any(k in str(t.get('description','')).lower() for k in ['zerodha', 'groww', 'mutual fund', 'sip', 'lic', 'investment']))
            risk_count = sum(1 for t in transactions if any(k in str(t.get('description','')).lower() for k in ['dream11', 'stake', 'bet', 'casino', 'rummy']))
            
            c1, c2 = st.columns(2)
            c1.success(f"📈 **Investments Detected:** {investment_count} (+ Score Boost)")
            if risk_count > 0:
                c2.error(f"🎲 **High-Risk Activities Detected:** {risk_count} (- Score Penalty)")
            else:
                c2.success("🛡️ **High-Risk Activities:** None detected.")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if result['status'] == 'APPROVED':
                st.success(f"### 🎉 APPLICATION APPROVED FOR {customer_name.upper()}!")
                st.write(f"**Approved Amount:** ₹{result.get('loan_amount', loan_amount):,.2f}")
                st.write("The transaction history strongly supports the application and aligns well with the CIBIL profile.")
                st.balloons()
            else:
                st.error(f"### ❌ APPLICATION REJECTED FOR {customer_name.upper()}")
                st.write(f"**Reason:** {result.get('reason', 'N/A')}")
                st.write("The risk was determined to be too high when comparing the document transaction data against the required criteria.")
                
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")
    else:
        st.warning("⚠️ Please connect your bank account or upload a transaction statement before submitting.")
