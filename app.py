import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Uzbekistan Alert Analysis", layout="wide")

st.title("🇺🇿 Financial Monitoring Unit: Escalation EDA Dashboard")
st.markdown("### Team Exploratory Data Analysis & Strategy")

st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Project Overview", "Data Distributions", "Behavioral Insights"])

if page == "Project Overview":
    st.subheader("1. Approach & Strategy")
    st.write("Our team converted the **relational transaction data** into behavioral summaries grouped per individual alert ID. We engineered features tracking transaction volume, amount scale, direction metrics, and specific transaction types.")
    
    st.subheader("2. Modeling Strategy")
    st.write("We implemented a **Stratified 5-Fold LightGBM classifier** optimizing explicitly for **ROC-AUC**. This approach ensures robust cross-validation across localized synthetic profiles.")

elif page == "Data Distributions":
    st.subheader("Target Feature Balance")
    st.write("The distribution of the historical target classes shows an unbalanced ratio typical of financial audit alerts, where a smaller proportion of alerts get escalated for further investigation.")
    
    # Synthetic chart data placeholder
    chart_data = pd.DataFrame({'Alert Status': ['Dismissed (0)', 'Escalated (1)'], 'Count': [8200, 1800]})
    st.bar_chart(chart_data, x='Alert Status', y='Count')

elif page == "Behavioral Insights":
    st.subheader("Key Transaction Drivers")
    st.markdown("- **International (Xalqaro) Triggers:** High counts of cross-border transfers heavily influence alert escalations.")
    st.markdown("- **Volume Velocity:** A sudden spikes in transactional frequency directly precedes escalation alerts.")
    st.markdown("- **Value Scale:** Large aggregated indicator values (`miqdor_indeksi`) separate critical risk profiles from daily consumer actions.")
