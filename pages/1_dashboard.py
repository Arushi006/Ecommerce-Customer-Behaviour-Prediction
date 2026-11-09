
import streamlit as st
import plotly.express as px

from utils import load_dataset

st.set_page_config(
    page_title="E-commerce Behaviour Prediction",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ E-commerce Customer Behaviour Prediction")
st.write(
    "An intelligent machine learning system for predicting "
    "product returns and customer churn."
)

try:
    df = load_dataset()

    required_columns = ["Returns", "Churn"]

    if not all(col in df.columns for col in required_columns):
        st.error("The dataset is missing Returns or Churn columns.")
        st.stop()

    total_customers = (
        df["Customer ID"].nunique()
        if "Customer ID" in df.columns else len(df)
    )

    total_transactions = len(df)

    return_rate = df["Returns"].mean() * 100
    churn_rate = df["Churn"].mean() * 100

    st.markdown("## Business Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Customers", f"{total_customers:,}")
    c2.metric("Total Transactions", f"{total_transactions:,}")
    c3.metric("Return Rate", f"{return_rate:.2f}%")
    c4.metric("Churn Rate", f"{churn_rate:.2f}%")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Product Return Distribution")

        return_counts = (
            df["Returns"]
            .value_counts()
            .rename(index={0: "Not Returned", 1: "Returned"})
            .reset_index()
        )

        return_counts.columns = ["Return Status", "Count"]

        fig = px.pie(
            return_counts,
            names="Return Status",
            values="Count",
            hole=0.45,
            title="Returned vs Not Returned"
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Customer Churn Distribution")

        churn_counts = (
            df["Churn"]
            .value_counts()
            .rename(index={0: "Not Churned", 1: "Churned"})
            .reset_index()
        )

        churn_counts.columns = ["Customer Status", "Count"]

        fig = px.pie(
            churn_counts,
            names="Customer Status",
            values="Count",
            hole=0.45,
            title="Churned vs Active"
        )

        st.plotly_chart(fig, use_container_width=True)

    if "Product Category" in df.columns:
        st.subheader("Transactions by Product Category")

        category_counts = (
            df["Product Category"]
            .value_counts()
            .reset_index()
        )

        category_counts.columns = ["Category", "Transactions"]
        category_counts = category_counts.sort_values(
            "Transactions", ascending=False
        )
        fig = px.bar(
            category_counts,
            x="Category",
            y="Transactions",
            color="Category",
            title="Product Category Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Use the navigation menu on the left to explore "
        "return prediction, churn prediction, model comparison "
        "and customer analytics."
    )

except Exception as e:
    st.error(f"Unable to load dashboard: {e}")
