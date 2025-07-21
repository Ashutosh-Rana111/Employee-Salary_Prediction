import streamlit as st
import pandas as pd
import joblib

# Load trained pipeline
model = joblib.load("Employee_Salary_Predict.pkl")

st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")
st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

st.sidebar.header("Input Employee Details")

# Inputs
age = st.sidebar.slider("Age", 18, 65, 30)
workclass = st.sidebar.selectbox("Workclass", [
    "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
    "Local-gov", "State-gov", "Without-pay", "Never-worked"
])
education = st.sidebar.selectbox("Education", [
    "Bachelors", "HS-grad", "11th", "Masters", "9th", "Some-college", "Assoc-acdm",
    "Assoc-voc", "Doctorate", "Prof-school", "7th-8th", "12th", "10th", "1st-4th", "Preschool"
])
marital_status = st.sidebar.selectbox("Marital Status", [
    "Never-married", "Married-civ-spouse", "Divorced", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"
])
occupation = st.sidebar.selectbox("Occupation", [
    "Tech-support", "Craft-repair", "Other-service", "Sales",
    "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct",
    "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv",
    "Protective-serv", "Armed-Forces"
])
race = st.sidebar.selectbox("Race", ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"])
gender = st.sidebar.radio("Gender", ["Male", "Female"])
hours_per_week = st.sidebar.slider("Hours per week", 1, 100, 40)
native_country = st.sidebar.selectbox("Native Country", [
    "United-States", "India", "Mexico", "Philippines", "Germany", "Canada", "England", "China", "Other"
])

# Build input DataFrame
input_df = pd.DataFrame({
    "age": [age],
    "workclass": [workclass],
    "education": [education],
    "marital-status": [marital_status],
    "occupation": [occupation],
    "race": [race],
    "gender": [gender],
    "hours-per-week": [hours_per_week],
    "native-country": [native_country]
})

st.write("### 🔎 Input Data")
st.write(input_df)

# Prediction
if st.button("Predict Salary Class"):
    prediction = model.predict(input_df)
    st.success(f"✅ Predicted Income: {prediction[0]}")

# Batch prediction
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:")
    st.write(batch_data.head())

    try:
        batch_preds = model.predict(batch_data)
        batch_data["PredictedIncome"] = batch_preds

        st.write("✅ Predictions:")
        st.write(batch_data.head())

        csv = batch_data.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions CSV", csv, file_name='predicted_salaries.csv', mime='text/csv')
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
