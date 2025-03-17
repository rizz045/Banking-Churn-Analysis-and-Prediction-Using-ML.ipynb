import streamlit as st
import joblib
import numpy as np
from MLEncoder import MultiLabelEncoder

# Load Model, Encoder & Scaler
model = joblib.load('BCA_model.pkl', 'r')
encoder = joblib.load('multi_label_encoder.pkl', 'r')
scaler = joblib.load('scaler.pkl', 'r')

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
st.image("https://d1muf25xaso8hp.cloudfront.net/https%3A%2F%2Ff2fa1cdd9340fae53fcb49f577292458.cdn.bubble.io%2Ff1711306378848x345911071866751300%2Fchurn.png?w=384&h=&auto=true&dpr=2.5&fit=crop", use_container_width=True)

# Custom Styling
st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>Customer Churn Prediction</h1>
    <p style='text-align: center;'>Enter customer details to predict churn.</p>
    <hr style='border: 1px solid #4CAF50;'>
""", unsafe_allow_html=True)

# App Title
st.markdown("<h1>📊 Banking Customer Churn Prediction</h1>", unsafe_allow_html=True)
st.write("Enter customer details below to predict churn.")

# Input Section
st.markdown("---")
st.subheader("🔹 Customer Details")

# Input Fields
with st.container():
    
    c_score = st.number_input("💳 Credit Score ", min_value=350, max_value=850, step=1)

    geo = st.selectbox("🌍 State Code", ['France','Germany','Spain'])
    gender = st.selectbox("🚻 Select Gender", ['Female', 'Male'])

    age = st.number_input("📆 Age ", min_value=18, max_value=92, step=1)
    tenure = st.slider("⏳ Tenure ", min_value=1, max_value=10, step=1)

    balance = st.number_input("💵 Balance ", min_value=0.0, max_value=250898.09, step=1.0, format="%.2f")

    products = st.slider("📞 Number of Products ", min_value=1, max_value=4, step=1)

    credit_card = st.selectbox("💳 Has Credit Card? ", ['Yes', 'No'])
    active = st.selectbox("🎯 Active member? ", ['Yes', 'No'])

    estimated_sal = st.number_input("💵 Estimated Salary ", min_value=11.58, max_value=199992.48, step=1.0, format="%.2f")

    complain = st.selectbox("❗ Whether the cutomer raised a complaint? ", ['Yes', 'No'])

    ratings = st.slider("⭐ Raitings given by the customer ", min_value=1, max_value=5, step=1)

    card_type = st.selectbox("💳 Card type of the customer", ['DIAMOND', 'GOLD', 'PLATINUM', 'SILVER'])
    
    points_earned = st.number_input("✨ Points Earned ", min_value=119, max_value=1000, step=1)


# Convert Inputs
credit_card = 1 if credit_card == "Yes" else 0
active = 1 if active == "Yes" else 0
complain = 1 if complain == "Yes" else 0

# Encode categorical columns
geo, gender, card_type = encoder.transform(['Geography','Gender',"Card Type"],[geo, gender, card_type])

# Prepare Data for Prediction
# input_data = np.array([[float(c_score), float(geo), float(gender),
#                         float(voice_plan), float(voice_messages), float(intl_plan), float(intl_mins), float(intl_calls),
#                         float(day_mins), float(day_calls), float(eve_mins), float(eve_calls),
#                         float(night_mins), float(night_calls), float(customer_calls), float(total_charge)]])

input_data = np.array([c_score, geo, gender, age, tenure, balance, products, credit_card, active, estimated_sal, complain, ratings, card_type, points_earned])

input_data_scaled = scaler.transform(input_data)

# Predict Button with Loading Animation
if st.button("🔍 Predict Churn"):
    with st.spinner("Analyzing customer data..."):
        time.sleep(2)  # Short delay for loading animation
        prediction = model.predict(input_data_scaled)

    # Display Result with Icon
    if prediction[0] == 1:
        st.error("❌ This customer is **LIKELY** to churn.")
    else:
        st.success("✅ This customer is **NOT LIKELY** to churn.")

# Footer
st.markdown("---")
st.markdown("<h5 style='text-align:center;'>Do Visit Us Again ❤️</h5>", unsafe_allow_html=True)
