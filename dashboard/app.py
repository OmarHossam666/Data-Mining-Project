import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import shap
import sys
from pathlib import Path

# --- Path Setup ---
# Dynamically point to the root directory to access models and features
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR / "src"))
import config

# --- Page Config ---
st.set_page_config(page_title="ProGrade Dashboard", page_icon="🎓", layout="wide")
st.title("🎓 ProGrade: Academic Intervention Dashboard")
st.markdown("Real-time predictive analytics and risk assessment for student success.")

# --- Data Loading (Cached for speed) ---
@st.cache_data
def load_data():
    # Load the perfectly formatted test data for inference
    test_df = pd.read_csv(ROOT_DIR / "features" / "checkpoint_12_test.csv")
    X = test_df.drop(columns=['target'])
    y = test_df['target']
    
    # Load the checkpoint comparison for the timeline chart
    results_df = pd.read_csv(ROOT_DIR / "results" / "checkpoint_comparison.csv")
    
    return X, y, results_df

@st.cache_resource
def load_model():
    return joblib.load(ROOT_DIR / "models" / "xgb_model.joblib")

X, y, results_df = load_data()
model = load_model()

# --- Predict & Map Tiers ---
# Calculate probabilities and map to your 3-tier system
probs = model.predict_proba(X)[:, 1]

def map_tier(p):
    if p < 0.3: return "Low Risk"
    elif p <= 0.6: return "Medium Risk"
    else: return "High Risk"

X_display = X.copy()
X_display['Risk Probability'] = probs
X_display['Intervention Tier'] = [map_tier(p) for p in probs]
X_display['True Label'] = y.values
# Generate dummy Student IDs for the dashboard display
X_display['Student ID'] = [f"STU-{1000 + i}" for i in range(len(X_display))]

# --- Sidebar Filters ---
st.sidebar.header("Dashboard Filters")
selected_tier = st.sidebar.multiselect(
    "Filter by Risk Tier:",
    options=["High Risk", "Medium Risk", "Low Risk"],
    default=["High Risk", "Medium Risk", "Low Risk"]
)

filtered_df = X_display[X_display['Intervention Tier'].isin(selected_tier)]

# --- Top Level Metrics ---
col1, col2, col3 = st.columns(3)
col1.metric("Total Students Analyzed", len(filtered_df))
col2.metric("High Risk Students", len(filtered_df[filtered_df['Intervention Tier'] == 'High Risk']), delta_color="inverse")
col3.metric("Avg Risk Probability", f"{filtered_df['Risk Probability'].mean():.1%}")

st.divider()

# --- Row 1: Risk Distribution & Checkpoint Accuracy ---
row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    st.subheader("Student Risk Distribution")
    # Custom colors mapping your tiers
    color_map = {"Low Risk": "#2ca02c", "Medium Risk": "#ff7f0e", "High Risk": "#d62728"}
    fig_pie = px.pie(
        filtered_df, 
        names='Intervention Tier', 
        hole=0.4,
        color='Intervention Tier',
        color_discrete_map=color_map
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with row1_col2:
    st.subheader("System F1-Score Evolution")
    # Line chart showing performance from week 4 to 12
    if not results_df.empty:
        # Filter just to XGBoost to keep the chart clean
        xgb_results = results_df[results_df['Model'] == 'XGBoost']
        fig_line = px.line(
            xgb_results, 
            x='Checkpoint', 
            y='Val_F1_Macro', 
            markers=True,
            title="XGBoost Predictive Power Over Time",
            labels={'Val_F1_Macro': 'F1-Macro Score'}
        )
        fig_line.update_yaxes(range=[0.4, 1.0])
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("Run Phase 5 to generate checkpoint comparison data.")

st.divider()

# --- Row 2: Global Feature Importance (SHAP) ---
st.subheader("Global Risk Drivers (Top 10 Features)")
with st.spinner("Calculating SHAP values..."):
    # Calculate SHAP on a fast sample to keep the dashboard snappy
    explainer = shap.Explainer(model, X.sample(min(100, len(X)), random_state=42))
    shap_values = explainer(X)
    
    # Calculate mean absolute SHAP for plotting
    shap_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance (Mean |SHAP|)': np.abs(shap_values.values).mean(0)
    }).sort_values(by='Importance (Mean |SHAP|)', ascending=False).head(10)
    
    fig_shap = px.bar(
        shap_df, 
        x='Importance (Mean |SHAP|)', 
        y='Feature', 
        orientation='h',
        color='Importance (Mean |SHAP|)',
        color_continuous_scale='Reds'
    )
    fig_shap.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_shap, use_container_width=True)

st.divider()

# --- Row 3: Actionable Student Table ---
st.subheader("Actionable Student Roster")
st.markdown("Use this table to identify students requiring immediate intervention.")

# Clean up table for advisors
table_display = filtered_df[['Student ID', 'Intervention Tier', 'Risk Probability', 'True Label']].copy()
table_display['Risk Probability'] = table_display['Risk Probability'].apply(lambda x: f"{x:.1%}")

def recommend_action(tier):
    if tier == "High Risk": return "Schedule 1-on-1 counseling immediately."
    elif tier == "Medium Risk": return "Monitor attendance; suggest tutoring."
    else: return "No action required."

table_display['Recommended Action'] = table_display['Intervention Tier'].apply(recommend_action)

# Interactive Streamlit dataframe
st.dataframe(
    table_display,
    column_config={
        "Intervention Tier": st.column_config.TextColumn(
            "Intervention Tier",
            help="Based on ML prediction",
        )
    },
    hide_index=True,
    use_container_width=True
)