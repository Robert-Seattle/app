import streamlit as st
import pandas as pd

st.set_page_config(page_title="FMU Uzbekistan Alert Audit Unit", layout="wide")

st.title("🇺🇿 Central Bank Monitoring Unit — Advanced Escalation Analysis")
st.markdown("---")

st.sidebar.title("App Navigation")
page = st.sidebar.radio("Go to Section:", ["Methodological Approach", "Structural Data Profiles", "Tabular Model Configuration"])

if page == "Methodological Approach":
    st.header("💡 Analytical Preprocessing Summary")
    st.write("Our team evaluated a high-volume relational framework connecting 6.9 million ledger rows to 14,000 corporate financial alert profiles.")
    
    st.subheader("The Chronological Deconstruction Discovery")
    st.error("A core exploration breakthrough occurred when our diagnostics revealed that traditional calendar timestamp windowing degraded validation metrics. Because the dataset's date attributes are synthesized uniformly out-of-order, treating time linearly added massive structural noise. We resolved this wall by transitioning exclusively to **Time-Agnostic Channel Cross-Aggregations**—evaluating how numeric values (`miqdor_indeksi`) intersect natively inside processing streams.")

elif page == "Structural Data Profiles":
    st.header("📊 Multi-Table Target & Volume Balances")
    st.write("The structural balance parameters derived across our final engineered features:")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Training Alert Records", "14,000 Rows")
    col2.metric("Relational Transaction Logs", "6,987,663 Events")
    col3.metric("Imbalanced Positive Target Skew", "17.2% Escalations")
    
    chart_df = pd.DataFrame({
        'Alert State': ['Dismissed (0)', 'Escalated (1)'],
        'Distribution %': [82.8, 17.2]
    })
    st.bar_chart(chart_df, x='Alert State', y='Distribution %')

elif page == "Tabular Model Configuration":
    st.header("🤖 Dual Gradient Boosting Architecture")
    st.write("To completely mitigate feature redundancy and multi-collinearity overfit, we locked down a multi-model hybrid blend:")
    
    st.markdown("- **Growth Pattern Ensembling:** We blended a leaf-wise **LightGBM** classifier with a depth-wise **XGBoost** classifier.")
    st.markdown("- **Outlier Preservation:** Our data tests proved that log-transforming or scaling continuous metrics dropped model resolution. The synthetic generation rules rely heavily on raw, hard numeric value thresholds.")
    st.markdown("- **leaderboard Strategy:** Averaging out-of-fold probability vectors across both structural frameworks stabilized public scoreboard variance and completely isolated the subtle category count patterns.")
