import streamlit as st
import pandas as pd
import joblib

# Page Configuration
st.set_page_config(
    page_title="Online Retail Analytics & Prediction",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ Customer Insights & Revenue Predictor")
st.markdown("Interactive Machine Learning dashboard for the Online Retail dataset.")

# Load Data & Artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load("retail_model.joblib")
    data = pd.read_csv("customer_features_sample.csv")
    return model, data

model, df = load_artifacts()

# Sidebar - User Inputs
st.sidebar.header("Customer Parameter Inputs")
total_orders = st.sidebar.number_input("Total Orders", min_value=1, max_value=500, value=5)
total_items = st.sidebar.number_input("Total Items Purchased", min_value=1, max_value=10000, value=250)
lifetime_days = st.sidebar.slider("Customer Lifetime (Days)", min_value=0, max_value=730, value=180)

# Main Section
col1, col2 = st.columns(2)

with col1:
    st.subheader("Data Overview")
    st.dataframe(df.head(10))

with col2:
    st.subheader("Model Prediction")
    if st.button("Predict Customer Value"):
        input_data = pd.DataFrame([[total_orders, total_items, lifetime_days]], 
                                  columns=["total_orders", "total_items", "lifetime_days"])
        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Lifetime Revenue: **${prediction:,.2f}**")