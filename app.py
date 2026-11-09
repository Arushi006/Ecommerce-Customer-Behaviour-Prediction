
import streamlit as st

st.set_page_config(
    page_title="E-commerce Behaviour Prediction",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ E-commerce Customer Behaviour Prediction")

st.markdown("""
### Welcome to the Prediction System!

This application uses Machine Learning to analyse
e-commerce customer behaviour and predict:

- **Product Returns:** Identify purchases that may be returned.
- **Customer Churn:** Identify customers who may stop purchasing.

### Explore the application

Use the navigation menu on the left to access:

1. Dashboard
2. Return Prediction
3. Churn Prediction
4. Customer Analytics
5. Model Comparison
""")

st.info(
    "Select Dashboard from the sidebar to explore "
    "the business overview and customer statistics."
)
