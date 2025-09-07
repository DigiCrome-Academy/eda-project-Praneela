# app.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# ===========================
# Load Dataset
# ===========================
@st.cache_data
def load_data():
    df = pd.read_csv("data/student_habits_performance.csv")
    return df

df = load_data()

# ===========================
# App Layout
# ===========================
st.set_page_config(page_title="Student Performance Dashboard", layout="wide")

st.title("📊 Student Performance Dashboard")
st.markdown("""
This interactive dashboard allows you to explore **student performance and lifestyle habits**.  
Use the sidebar to filter data and visualize results dynamically.
""")

# ===========================
# Sidebar Filters
# ===========================
st.sidebar.header("🔍 Filters")

# Gender filter
gender_filter = st.sidebar.multiselect(
    "Select Gender",
    options=df["gender"].unique(),
    default=df["gender"].unique()
)

# Race/Ethnicity filter (if present)
if "race/ethnicity" in df.columns:  # Your dataset may not have this
    race_filter = st.sidebar.multiselect(
        "Select Race/Ethnicity",
        options=df["race/ethnicity"].unique(),
        default=df["race/ethnicity"].unique()
    )
else:
    race_filter = None

# Diet Quality filter
diet_filter = st.sidebar.multiselect(
    "Select Diet Quality",
    options=df["diet_quality"].unique(),
    default=df["diet_quality"].unique()
)

# Apply filters
filtered_df = df[
    (df["gender"].isin(gender_filter)) &
    (df["diet_quality"].isin(diet_filter))
]

if race_filter:
    filtered_df = filtered_df[filtered_df["race/ethnicity"].isin(race_filter)]

# ===========================
# Key Metrics
# ===========================
st.subheader("📈 Key Metrics")

col1, col2, col3 = st.columns(3)
col1.metric("Total Students", len(filtered_df))
col2.metric("Average Exam Score", f"{filtered_df['exam_score'].mean():.2f}")
col3.metric("Average Study Hours", f"{filtered_df['study_hours_per_day'].mean():.2f}")

# ===========================
# Visualizations
# ===========================

st.subheader("📊 Exam Score Distribution")
fig1 = px.histogram(filtered_df, x="exam_score", nbins=20, title="Exam Score Distribution", color="gender")
st.plotly_chart(fig1, use_container_width=True)

st.subheader("📉 Study Hours vs Exam Score")
fig2 = px.scatter(
    filtered_df, 
    x="study_hours_per_day", 
    y="exam_score", 
    color="gender",
    size="attendance_percentage",
    hover_data=["diet_quality"]
)
fig2.update_layout(title="Study Hours vs Exam Score", xaxis_title="Study Hours", yaxis_title="Exam Score")
st.plotly_chart(fig2, use_container_width=True)

st.subheader("📦 Exam Score by Diet Quality")
fig3, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=filtered_df, x="diet_quality", y="exam_score", palette="pastel")
plt.title("Exam Score by Diet Quality")
st.pyplot(fig3)
