
import streamlit as st

from utils import load_model, create_prediction_form, show_prediction_result

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="👥",
    layout="wide"
)

st.title("👥 Customer Churn Prediction")

st.write(
    "Identify customers who may stop purchasing from the business."
)

st.markdown("""
The churn model uses customer-level features created during
the data aggregation process in your third notebook.

For example, features may represent average purchase amount,
total quantity, transaction frequency and customer preferences.
""")

try:
    model = load_model("churn_model.pkl")

    st.success("Churn prediction model loaded successfully.")

    input_df = create_prediction_form(
        model,
        "churn_prediction_form"
    )

    if input_df is not None:
        show_prediction_result(
            model,
            input_df,
            "Churn Prediction"
        )

        st.markdown("### Suggested Business Action")

        prediction = model.predict(input_df)[0]

        if prediction == 1:
            st.warning(
                "This customer is predicted to be at risk of churn. "
                "Consider personalised offers, feedback collection "
                "or customer engagement campaigns."
            )
        else:
            st.success(
                "This customer is predicted to remain active. "
                "Continue providing a good customer experience."
            )

except Exception as e:
    st.error(f"Error loading or using the churn model: {e}")
