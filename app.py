import streamlit as st
import pandas as pd
import joblib  # To load trained model

st.set_page_config(page_title="Readmission Predictor", page_icon="🏥")
st.title("🏥 Hospital Readmission Risk Predictor")

# User inputs
age = st.slider("Age", 0, 100, 50)
gender = st.selectbox("Gender", ['Male', 'Female'])
blood_type = st.selectbox("Blood Type", ['A+','A-','B+', 'B-','AB+','AB-','O+','O-'])
billing_amount = st.number_input("Billing Amount (₹)", min_value=0, value=5000)
medication = st.selectbox("Medication Given", ['Penicilin','Aspirin','Ibuprofen','Paracetamol','Lipitor','None'])
admission_type = st.selectbox("Admission Type", ['Emergency', 'Urgent','Elective'])
test_result = st.selectbox("Test Results", ['Normal', 'Abnormal','Inconclusive'])
condition = st.selectbox("Medical Condition", ['Diabetes','Hypertension','Obesity','Asthama','Cancer','Arthritis'])

# Predict using trained model
if st.button("Predict"):
    try:
        # Load model and input columns
        model = joblib.load("model.pkl")  # Your saved model
        model_columns = joblib.load("model_columns.pkl")  # Column names used in training

        # Build input as DataFrame
        user_input = pd.DataFrame([{
            'Age': age,
            'Gender': gender,
            'Blood Type': blood_type,
            'Billing Amount': billing_amount,
            'Medication': medication,
            'Admission Type': admission_type,
            'Test Results': test_result,
            'Medical Condition': condition
        }])

        # Convert to same format as training data
        user_input = pd.get_dummies(user_input)
        user_input = user_input.reindex(columns=model_columns, fill_value=0)

        prediction = model.predict(user_input)[0]

        if prediction == 1:
            st.error("🔴 Prediction: High Risk of Readmission")
        else:
            st.success("🟢 Prediction: Low Risk of Readmission")

    except Exception as e:
        st.error(f"Error during prediction: {e}")


