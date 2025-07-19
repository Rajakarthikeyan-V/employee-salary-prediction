import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("employee_salary_prediction.pkl")

# Load dataset for dropdown values
df = pd.read_csv("employee_salary.csv")
df = df.dropna()
df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce')
df = df.dropna(subset=['Salary'])

# Streamlit config
st.set_page_config(page_title="Salary Predictor", page_icon="💼", layout="centered")

# 🌙 Dark Theme Custom Styling
st.markdown("""
    <style>
    html, body {
        background-color: #121212;
        color: #ffffff;
        font-family: 'Segoe UI', sans-serif;
    }
    .stApp {
        background-color: #121212;
        color: #ffffff;
    }
    h1 {
        color: #90caf9;
        text-align: center;
        font-weight: 800;
    }
    .stForm label, .stSlider label, .stSelectbox label {
        color: #ffffff !important;
        font-weight: 600;
        font-size: 16px;
    }
    .stButton > button {
        background-color: #0d47a1;
        color: white;
        font-weight: 600;
        border-radius: 10px;
        font-size: 16px;
    }
    .stButton > button:hover {
        background-color: #1976d2;
    }
    .stSidebar, .css-1d391kg {
        background-color: #1e1e1e !important;
        color: #ffffff !important;
    }
    .stSlider .css-14el2xx .st-c1 {
        background-color: #90caf9 !important;
    }
    </style>
""", unsafe_allow_html=True)

# App Title
st.markdown("<h1>💼 Employee Salary Predictor</h1>", unsafe_allow_html=True)
st.markdown("### 🎯 Estimate employee **Monthly salary** using machine learning.")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About the App")
    st.markdown("""
    This app predicts salaries using:

    - Age  
    - Gender  
    - Education Level  
    - Job Title  
    - Years of Experience  
    - Experience-to-Age Ratio  
    """)
    st.success("Model used: Random Forest Regressor")

# Form UI
with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("🎂 Age", 18, 65, 30)
        gender = st.selectbox("🧑 Gender", sorted(df["Gender"].unique()))
        education = st.selectbox("🎓 Education Level", sorted(df["Education Level"].unique()))

    with col2:
        job = st.selectbox("💼 Job Title", sorted(df["Job Title"].unique()))
        experience = st.slider("📈 Years of Experience", 0, 40, 5)

    submit = st.form_submit_button("🔍 Predict Salary")

# Feature engineering
exp_to_age = experience / age if age > 0 else 0

input_df = pd.DataFrame([{
    "Age": age,
    "Gender": gender,
    "Education Level": education,
    "Job Title": job,
    "Years of Experience": experience,
    "Exp_to_Age": exp_to_age
}])

# Predict
if submit:
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"💰 **Estimated Monthly Salary**: ₹{prediction:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
