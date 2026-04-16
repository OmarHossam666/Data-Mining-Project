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
    # Load Gold Layer BI dataset
    gold_df = pd.read_csv(ROOT_DIR / "data" / "gold" / "student_risk_mart.csv")
    
    # Load the perfectly formatted test data for SHAP inference
    test_df = pd.read_csv(ROOT_DIR / "features" / "checkpoint_12_test.csv")
    X = test_df.drop(columns=['target'], errors='ignore')

    # Load the checkpoint comparison for the timeline chart
    results_df = pd.read_csv(ROOT_DIR / "results" / "checkpoint_comparison.csv")

    return gold_df, X, results_df

@st.cache_resource
def load_model():
    return joblib.load(ROOT_DIR / "models" / "xgb_model.joblib")

@st.cache_data
def compute_shap_values(_model, _X):
    """Compute SHAP values once and cache them for dashboard performance."""
    sample = _X.sample(min(100, len(_X)), random_state=42)
    explainer = shap.Explainer(_model, sample)
    shap_vals = explainer(_X)
    shap_df = pd.DataFrame({
        'Feature': _X.columns,
        'Importance (Mean |SHAP|)': np.abs(shap_vals.values).mean(0)
    }).sort_values(by='Importance (Mean |SHAP|)', ascending=False).head(10)
    return shap_df

gold_df, X, results_df = load_data()
model = load_model()

# --- Sidebar Filters ---
st.sidebar.header("Dashboard Filters")
selected_schools = st.sidebar.multiselect(
    "Filter by School:",
    options=gold_df['school'].unique(),
    default=gold_df['school'].unique()
)

selected_tiers = st.sidebar.multiselect(
    "Filter by Risk Tier:",
    options=["High Risk", "Low Risk"],
    default=["High Risk", "Low Risk"]
)

selected_gpa = st.sidebar.multiselect(
    "Filter by GPA Bracket:",
    options=gold_df['gpa_bracket'].unique(),
    default=gold_df['gpa_bracket'].unique()
)

# Apply filters to Gold Data
filtered_df = gold_df[
    (gold_df['school'].isin(selected_schools)) &
    (gold_df['risk_status'].isin(selected_tiers)) &
    (gold_df['gpa_bracket'].isin(selected_gpa))
]

# --- Top Level Metrics ---
col1, col2, col3 = st.columns(3)
high_risk_count = len(filtered_df[filtered_df['risk_status'] == 'High Risk'])
col1.metric("Total Students Analyzed", len(filtered_df))
col2.metric("High Risk Students", high_risk_count, delta_color="inverse")
col3.metric("Avg Absences", f"{filtered_df['absences'].mean():.1f}")

st.divider()

# --- Row 1: Risk Distribution & Checkpoint Accuracy ---
row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    st.subheader("Student Risk Distribution")
    color_map = {"Low Risk": "#2ca02c", "High Risk": "#d62728"}
    fig_pie = px.pie(
        filtered_df,
        names='risk_status',
        hole=0.4,
        color='risk_status',
        color_discrete_map=color_map
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with row1_col2:
    st.subheader("System F1-Score Evolution")
    if not results_df.empty:
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

# --- Row 2: BI Insights & SHAP Drivers ---
row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    st.subheader("Risk Distribution by School")
    fig_bar = px.histogram(
        filtered_df, 
        x="school", 
        color="risk_status",
        barmode="group",
        color_discrete_map=color_map,
        labels={"school": "School"}
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with row2_col2:
    st.subheader("Global Risk Drivers (Top Pre-Processed Features)")
    shap_df = compute_shap_values(model, X)
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
st.markdown("Use this table to identify students requiring immediate intervention. Contextual BI data provided by the Gold Layer.")

def recommend_action(tier):
    if tier == "High Risk": return "Schedule 1-on-1 counseling immediately."
    elif tier == "Medium Risk": return "Monitor attendance; suggest tutoring."
    else: return "No action required."

table_display = filtered_df[['student_id', 'age', 'sex', 'school', 'subject', 'gpa_bracket', 'attendance_tier', 'absences', 'risk_status']].copy()
table_display['Recommended Action'] = table_display['risk_status'].apply(recommend_action)
table_display = table_display.rename(columns={
    'student_id': 'Student ID',
    'age': 'Age',
    'sex': 'Gender',
    'school': 'School',
    'subject': 'Subject',
    'gpa_bracket': 'GPA Bracket',
    'attendance_tier': 'Attendance Tier',
    'absences': 'Absences',
    'risk_status': 'Intervention Tier'
})

st.dataframe(
    table_display,
    column_config={
        "Intervention Tier": st.column_config.TextColumn(
            "Intervention Tier",
            help="High Risk flags students needing immediate attention",
        ),
        "Student ID": st.column_config.TextColumn("Student ID") # Display as text without commas
    },
    hide_index=True,
    use_container_width=True
)