import streamlit as st
import pandas as pd
import numpy as np

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

import seaborn as sns
import matplotlib.pyplot as plt

st.title("📈 Model Performance")

st.markdown("""
Model Used: XGBoost
""")

# Example Results

y_true = np.array(
    [1,1,1,0,1,0,1,0,1,1]
)

y_pred = np.array(
    [1,1,0,0,1,0,1,0,1,1]
)

cm = confusion_matrix(
    y_true,
    y_pred
)

fig, ax = plt.subplots()

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    ax=ax
)

st.pyplot(fig)

report = classification_report(
    y_true,
    y_pred,
    output_dict=True
)

st.subheader("Classification Report")

st.dataframe(
    pd.DataFrame(report).transpose()
)

st.metric(
    "Accuracy",
    "89%"
)

st.metric(
    "ROC AUC",
    "91%"
)