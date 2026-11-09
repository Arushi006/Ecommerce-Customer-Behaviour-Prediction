
import os
import streamlit as st
import pandas as pd

from utils import BASE_DIR, load_model

st.set_page_config(
    page_title="Model Comparison",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Machine Learning Model Comparison")

st.write(
    "Compare the three algorithms trained for each prediction task."
)

model_groups = {
    "Product Return Prediction": [
        "return_logistic_regression.pkl",
        "return_decision_tree.pkl",
        "return_random_forest.pkl"
    ],
    "Customer Churn Prediction": [
        "churn_logistic_regression.pkl",
        "churn_decision_tree.pkl",
        "churn_random_forest.pkl"
    ]
}

for group_name, model_files in model_groups.items():

    st.subheader(group_name)

    comparison_rows = []

    for filename in model_files:

        try:
            model = load_model(filename)
            classifier = model.named_steps["classifier"]

            comparison_rows.append({
                "Model": filename.replace(".pkl", "").replace("_", " ").title(),
                "Algorithm": type(classifier).__name__,
                "Input Features": len(model.feature_names_in_)
            })

        except Exception as e:
            st.warning(f"Could not load {filename}: {e}")

    if comparison_rows:
        st.dataframe(
            pd.DataFrame(comparison_rows),
            use_container_width=True,
            hide_index=True
        )

    st.divider()


st.subheader("Evaluation Metrics")

metric_files = {
    "Return Prediction": "return_model_comparison.csv",
    "Churn Prediction": "churn_model_comparison.csv"
}

for task, filename in metric_files.items():

    path = os.path.join(BASE_DIR, "models", filename)

    st.markdown(f"**{task}**")

    if os.path.exists(path):
        metrics_df = pd.read_csv(path)
        st.dataframe(
            metrics_df,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info(
            f"{filename} is not available yet. "
            "Export results_df from the corresponding notebook "
            "to display Accuracy and ROC-AUC here."
        )
