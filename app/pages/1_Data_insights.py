import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📊 Data Insights Dashboard")

df = pd.read_csv("data/processed/clean_loan_data.csv")

st.subheader("Dataset Overview")

st.dataframe(df.head())

st.write("Shape :", df.shape)

# =====================
# LOAN STATUS
# =====================

fig = px.pie(
    df,
    names="Loan_Status",
    title="Loan Approval Distribution"
)

st.plotly_chart(fig, use_container_width=True)

# =====================
# INCOME DISTRIBUTION
# =====================

fig2 = px.histogram(
    df,
    x="ApplicantIncome",
    title="Applicant Income Distribution"
)

st.plotly_chart(fig2, use_container_width=True)

# =====================
# CREDIT HISTORY
# =====================

fig3 = px.histogram(
    df,
    x="Credit_History",
    color="Loan_Status",
    barmode="group",
    title="Credit History vs Loan Approval"
)

st.plotly_chart(fig3, use_container_width=True)

# =====================
# CORRELATION
# =====================

corr = df.corr(numeric_only=True)

st.subheader("Correlation Matrix")

st.dataframe(corr)