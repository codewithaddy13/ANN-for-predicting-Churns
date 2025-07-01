import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf

# Load model and encoders
model = tf.keras.models.load_model('model.h5')

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('le.pkl', 'rb') as f:
    le = pickle.load(f)

with open('ohe.pkl', 'rb') as f:
    ohe = pickle.load(f)

# App title and description
st.title("💼 Customer Churn Prediction")
st.markdown(
    """
    This app predicts whether a customer is likely to churn based on their demographics and account information.
    Fill in the details below and click **Predict** to see the result.
    """
)

# Create a form so prediction happens only on submit
with st.form("churn_form"):
    # Layout using columns
    col1, col2 = st.columns(2)
    
    with col1:
        geography = st.selectbox("🌍 Geography", ohe.categories_[0])
        gender = st.selectbox("👤 Gender", le.classes_)
        age = st.slider("🎂 Age", 18, 92, 30)
        credit_score = st.number_input("💳 Credit Score", min_value=300, max_value=900, value=650)
        balance = st.number_input("💰 Balance", min_value=0.0, value=0.0)
    with col2:
        estimated_salary = st.number_input("💵 Estimated Salary", min_value=0.0, value=50000.0)
        tenure = st.slider("📅 Tenure (years)", 0, 10, 3)
        num_of_products = st.slider("📦 Number of Products", 1, 4, 1)
        has_cr_card = st.selectbox("💳 Has Credit Card?", [0, 1])
        is_active_member = st.selectbox("✅ Is Active Member?", [0, 1])
    
    # Submit button
    submit = st.form_submit_button("Predict")

if submit:
    # Prepare data
    data = {
        'CreditScore': [credit_score],
        'Gender': [le.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    }
    input_df = pd.DataFrame(data)
    
    geo_encoded = ohe.transform([[geography]])
    geo_df = pd.DataFrame(geo_encoded, columns=ohe.get_feature_names_out())
    
    final_df = pd.concat([input_df, geo_df], axis=1)
    
    scaled_df = scaler.transform(final_df)
    
    prediction = model.predict(scaled_df)
    pred_prob = prediction[0][0]
    
    # Display result
    if pred_prob > 0.5:
        st.error(f"⚠️ The customer is **likely to churn**! (Probability: {pred_prob:.2f})")
    else:
        st.success(f"✅ The customer is **unlikely to churn**. (Probability: {pred_prob:.2f})")
