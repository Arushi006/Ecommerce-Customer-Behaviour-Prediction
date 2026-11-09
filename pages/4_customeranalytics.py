
import streamlit as st
import plotly.express as px

from utils import load_dataset

st.set_page_config(
    page_title="Customer Analytics",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Customer Behaviour Analytics")

st.write(
    "Explore purchasing patterns, customer demographics "
    "and product return behaviour."
)

try:
    df = load_dataset()

    # Product category analysis
    if "Product Category" in df.columns:

        st.subheader("Purchases by Product Category")

        category_data = (
            df.groupby("Product Category")
            .size()
            .reset_index(name="Transactions")
            .sort_values("Transactions", ascending=False)
        )

        fig = px.bar(
            category_data,
            x="Product Category",
            y="Transactions",
            color="Product Category",
            title="Transactions by Category"
        )

        st.plotly_chart(fig, use_container_width=True)

    # Payment method analysis
    if "Payment Method" in df.columns:

        st.subheader("Payment Method Distribution")

        payment_data = (
            df["Payment Method"]
            .value_counts()
            .reset_index()
        )

        payment_data.columns = ["Payment Method", "Count"]

        fig = px.pie(
            payment_data,
            names="Payment Method",
            values="Count",
            hole=0.4
        )

        st.plotly_chart(fig, use_container_width=True)

    # Age distribution
    age_column = (
        "Customer Age"
        if "Customer Age" in df.columns
        else "Age" if "Age" in df.columns else None
    )

    if age_column:

        st.subheader("Customer Age Distribution")

        fig = px.histogram(
            df,
            x=age_column,
            nbins=20,
            title="Customer Age Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    # Purchase amount analysis
    if "Total Purchase Amount" in df.columns:

        st.subheader("Purchase Amount Distribution")

        fig = px.histogram(
            df,
            x="Total Purchase Amount",
            nbins=30,
            title="Total Purchase Amount Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    # Return rate by category
    if (
        "Product Category" in df.columns
        and "Returns" in df.columns
    ):

        st.subheader("Return Rate by Product Category")

        return_data = (
            df.groupby("Product Category")["Returns"]
            .mean()
            .reset_index()
        )

        return_data["Return Rate (%)"] = (
            return_data["Returns"] * 100
        )

        fig = px.bar(
            return_data,
            x="Product Category",
            y="Return Rate (%)",
            title="Product Return Rate by Category"
        )

        st.plotly_chart(fig, use_container_width=True)

    # Churn by gender
    if "Gender" in df.columns and "Churn" in df.columns:

        st.subheader("Customer Churn by Gender")

        churn_data = (
            df.groupby("Gender")["Churn"]
            .mean()
            .reset_index()
        )

        churn_data["Churn Rate (%)"] = (
            churn_data["Churn"] * 100
        )

        fig = px.bar(
            churn_data,
            x="Gender",
            y="Churn Rate (%)",
            title="Churn Rate by Gender"
        )

        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Unable to generate analytics: {e}")
