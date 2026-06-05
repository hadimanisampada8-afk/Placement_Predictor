import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(
    page_title="Placement Analytics Dashboard",
    page_icon="🎓",
    layout="wide"
)

# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("placementdata.csv")

df = load_data()

# ------------------------------------------------
# HEADER
# ------------------------------------------------

st.title("🎓 Student Placement Analytics Dashboard")
st.markdown(
"""
Analyze student profiles, placement trends,
and predict placement chances.
"""
)

# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------

st.sidebar.header("Dashboard Controls")

numeric_columns = [
    'CGPA',
    'Internships',
    'Projects',
    'Workshops/Certifications',
    'AptitudeTestScore',
    'SoftSkillsRating',
    'SSC_Marks',
    'HSC_Marks'
]

selected_feature = st.sidebar.selectbox(
    "Select Feature",
    numeric_columns
)

# ------------------------------------------------
# KPI SECTION
# ------------------------------------------------

st.subheader("📌 Key Metrics")

placed = (df["PlacementStatus"] == "Placed").sum()
not_placed = (df["PlacementStatus"] == "NotPlaced").sum()

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Students", len(df))
c2.metric("Placed", placed)
c3.metric("Not Placed", not_placed)
c4.metric(
    "Placement Rate",
    f"{(placed/len(df))*100:.1f}%"
)

# ------------------------------------------------
# DATA PREVIEW
# ------------------------------------------------

st.subheader("📋 Dataset Preview")

st.dataframe(df.head())

# ------------------------------------------------
# PLACEMENT DISTRIBUTION
# ------------------------------------------------

st.subheader("🎯 Placement Distribution")

fig = px.pie(
    df,
    names="PlacementStatus",
    hole=0.5,
    title="Placement Status"
)

st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------
# FEATURE DISTRIBUTION
# ------------------------------------------------

st.subheader("📈 Feature Distribution")

fig = px.histogram(
    df,
    x=selected_feature,
    nbins=30,
    title=f"{selected_feature} Distribution"
)

st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------
# CGPA VS PLACEMENT
# ------------------------------------------------

st.subheader("🎓 CGPA vs Placement")

fig = px.box(
    df,
    x="PlacementStatus",
    y="CGPA",
    color="PlacementStatus"
)

st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------
# APTITUDE ANALYSIS
# ------------------------------------------------

st.subheader("🧠 Aptitude Score Analysis")

fig = px.violin(
    df,
    x="PlacementStatus",
    y="AptitudeTestScore",
    box=True
)

st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------
# CORRELATION HEATMAP
# ------------------------------------------------

st.subheader("🔥 Correlation Heatmap")

numeric_df = df[numeric_columns]

corr = numeric_df.corr()

fig = px.imshow(
    corr,
    text_auto=".2f",
    aspect="auto"
)

st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------
# TRAIN MODEL
# ------------------------------------------------

st.subheader("🤖 Placement Prediction Model")

model_df = df.copy()

le = LabelEncoder()

model_df["PlacementStatus"] = le.fit_transform(
    model_df["PlacementStatus"]
)

model_df["ExtracurricularActivities"] = le.fit_transform(
    model_df["ExtracurricularActivities"]
)

model_df["PlacementTraining"] = le.fit_transform(
    model_df["PlacementTraining"]
)

X = model_df.drop(
    ["PlacementStatus", "StudentID"],
    axis=1
)

y = model_df["PlacementStatus"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

st.success(
    f"Model Accuracy : {accuracy:.2%}"
)

# ------------------------------------------------
# CONFUSION MATRIX
# ------------------------------------------------

st.subheader("📊 Confusion Matrix")

cm = confusion_matrix(y_test, pred)

fig = px.imshow(
    cm,
    text_auto=True,
    title="Confusion Matrix"
)

st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------
# FEATURE IMPORTANCE
# ------------------------------------------------

st.subheader("🏆 Feature Importance")

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)

fig = px.bar(
    importance_df.head(10),
    x="Importance",
    y="Feature",
    orientation="h",
    title="Top Factors Influencing Placement"
)

st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------
# STUDENT PREDICTOR
# ------------------------------------------------

st.subheader("🔮 Placement Chance Predictor")

col1, col2 = st.columns(2)

cgpa = col1.slider("CGPA", 0.0, 10.0, 8.0)
internships = col1.slider("Internships", 0, 10, 2)
projects = col1.slider("Projects", 0, 10, 2)
workshops = col1.slider("Certifications", 0, 10, 2)

aptitude = col2.slider("Aptitude Score", 0, 100, 75)
softskills = col2.slider("Soft Skills", 0.0, 5.0, 4.0)
ssc = col2.slider("SSC Marks", 0, 100, 75)
hsc = col2.slider("HSC Marks", 0, 100, 75)

if st.button("Predict Placement"):

    sample = pd.DataFrame([{
        "CGPA": cgpa,
        "Internships": internships,
        "Projects": projects,
        "Workshops/Certifications": workshops,
        "AptitudeTestScore": aptitude,
        "SoftSkillsRating": softskills,
        "ExtracurricularActivities": 1,
        "PlacementTraining": 1,
        "SSC_Marks": ssc,
        "HSC_Marks": hsc
    }])

    result = model.predict(sample)[0]

    if result == 1:
        st.success("🎉 Likely To Be Placed")
    else:
        st.error("⚠️ Placement Probability Low")

# ------------------------------------------------
# INSIGHTS
# ------------------------------------------------

st.subheader("💡 AI Insights")

top_feature = importance_df.iloc[0]["Feature"]

st.info(f"""
✔ Placement Rate : {(placed/len(df))*100:.2f}%

✔ Most Influential Factor : {top_feature}

✔ Students with higher aptitude scores
generally show better placement outcomes.

✔ Practical experience through internships
improves placement probability.

✔ Strong academic performance contributes
significantly to placement success.
""")

# ------------------------------------------------
# FOOTER
# ------------------------------------------------

st.markdown("---")
st.caption("Developed with Streamlit | Placement Analytics Dashboard")
