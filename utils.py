
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "ecommerce_customer_data_custom_ratios.csv.zip"
)


# Load trained ML model
@st.cache_resource
def load_model(model_name):

    model_path = os.path.join(MODEL_DIR, model_name)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    return joblib.load(model_path)


# Load dataset
@st.cache_data
def load_dataset():

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    return pd.read_csv(DATASET_PATH, compression="zip")


# Identify model input features
def get_model_features(model):

    preprocessor = model.named_steps["preprocessor"]

    numeric_defaults = {}
    categorical_options = {}

    for name, transformer, columns in preprocessor.transformers_:

        if name == "remainder" or transformer == "drop":
            continue

        columns = list(columns)

        steps = (
            transformer.named_steps
            if hasattr(transformer, "named_steps")
            else {}
        )

        if name == "num":

            imputer = steps.get("imputer")

            if imputer is not None:
                defaults = imputer.statistics_

            elif hasattr(transformer, "mean_"):
                defaults = transformer.mean_

            else:
                defaults = [0.0] * len(columns)

            for col, value in zip(columns, defaults):

                if not np.isfinite(value):
                    value = 0.0

                numeric_defaults[col] = float(value)

        elif name == "cat":

            encoder = steps.get("encoder")

            if encoder is None and hasattr(
                transformer, "categories_"
            ):
                encoder = transformer

            if encoder is not None:

                for col, categories in zip(
                    columns, encoder.categories_
                ):

                    valid = [
                        x for x in categories
                        if not pd.isna(x)
                    ]

                    categorical_options[col] = (
                        valid if valid else ["Unknown"]
                    )

    return numeric_defaults, categorical_options


# Create dynamic prediction form
def create_prediction_form(model, form_key):

    numeric_defaults, categorical_options = (
        get_model_features(model)
    )

    input_data = {}

    with st.form(form_key):

        st.subheader("Enter Customer Information")

        st.caption(
            "Enter the customer details below to generate a prediction."
        )

        st.markdown("### Numerical Features")

        for col, default in numeric_defaults.items():

            label = col.replace("_", " ").title()

            input_data[col] = st.number_input(
                label,
                value=float(default),
                step=1.0,
                key=f"{form_key}_{col}"
            )

        st.markdown("### Categorical Features")

        for col, options in categorical_options.items():

            label = col.replace("_", " ").title()

            input_data[col] = st.selectbox(
                label,
                options=options,
                key=f"{form_key}_{col}"
            )

        submitted = st.form_submit_button(
            "Generate Prediction",
            use_container_width=True
        )

    if submitted:

        expected_columns = list(model.feature_names_in_)

        input_df = pd.DataFrame(
            [input_data],
            columns=expected_columns
        )

        return input_df

    return None


# Display prediction result
def show_prediction_result(
    model, input_df, prediction_name
):

    prediction = model.predict(input_df)[0]

    st.divider()
    st.subheader("Prediction Result")

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(input_df)[0]

        classes = list(model.classes_)

        positive_index = (
            classes.index(1)
            if 1 in classes
            else len(classes) - 1
        )

        probability = float(
            probabilities[positive_index]
        )

        col1, col2 = st.columns(2)

        with col1:

            if prediction == 1:
                st.error(f"{prediction_name}: Positive")

            else:
                st.success(f"{prediction_name}: Negative")

        with col2:

            st.metric(
                "Probability of Positive Class",
                f"{probability * 100:.2f}%"
            )

        st.progress(probability)

        st.caption(
            "This is the model's estimated probability "
            "for class 1, not a guarantee of the outcome."
        )

    else:

        st.write(
            f"Predicted class: **{prediction}**"
        )
