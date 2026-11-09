
import streamlit as st

from utils import load_model, create_prediction_form, show_prediction_result

st.set_page_config(
    page_title="Product Return Prediction",
    page_icon="📦",
    layout="wide"
)

st.title("📦 Product Return Prediction")

st.write(
    "Predict whether a customer is likely to return a purchased product."
)

st.markdown("""
This prediction is based on customer and transaction characteristics
used while training the return classification model.
""")

try:
    model = load_model("return_model.pkl")

    st.success("Return prediction model loaded successfully.")

    input_df = create_prediction_form(
        model,
        "return_prediction_form"
    )

    if input_df is not None:
        show_prediction_result(
            model,
            input_df,
            "Return Prediction"
        )

except Exception as e:
    st.error(f"Error loading or using the return model: {e}")
