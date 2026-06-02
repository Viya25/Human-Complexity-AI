import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -------------------------
# Load Files
# -------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(BASE_DIR / "models" / "rf_model.pkl")
scaler = joblib.load(BASE_DIR / "models" / "scaler.pkl")
features = joblib.load(BASE_DIR / "models" / "features.pkl")

# -------------------------
# Page Config
# -------------------------

st.set_page_config(
    page_title="Human Complexity AI",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Human Complexity AI")
st.subheader("Academic Risk Prediction & Recommendation System")

st.markdown("---")

# -------------------------
# Inputs
# -------------------------

col1, col2 = st.columns(2)

with col1:

    age = st.slider("Age", 15, 22, 18)

    failures = st.slider(
        "Previous Academic Failures",
        0, 4, 0
    )

    absences = st.slider(
        "Absences",
        0, 100, 5
    )

    goout = st.slider(
        "Going Out Frequency",
        1, 5, 3
    )

with col2:

    health = st.slider(
        "Health Score",
        1, 5, 3
    )

    famrel = st.slider(
        "Family Relationship Score",
        1, 5, 4
    )

    studytime = st.slider(
        "Study Time",
        1, 4, 2
    )

    Walc = st.slider(
        "Weekend Alcohol Consumption",
        1, 5, 1
    )

# -------------------------
# Feature Engineering
# -------------------------

engagement_score = studytime / (absences + 1)

socioeconomic_score = famrel + health

lifestyle_risk = goout + Walc

# -------------------------
# Prediction Button
# -------------------------

if st.button("Predict Risk"):

    sample = pd.DataFrame({

        "age":[age],
        "failures":[failures],
        "famrel":[famrel],
        "goout":[goout],
        "health":[health],
        "Walc":[Walc],
        "absences":[absences],
        "engagement_score":[engagement_score],
        "socioeconomic_score":[socioeconomic_score],
        "lifestyle_risk":[lifestyle_risk]

    })

    # Add missing columns
    for col in features:
        if col not in sample.columns:
            sample[col] = 0

    sample = sample[features]

    sample_scaled = scaler.transform(sample)

    prediction = model.predict(sample_scaled)

    st.markdown("---")

    if prediction[0] == 1:

        st.error("🔴 HIGH RISK STUDENT")

    else:

        st.success("🟢 LOW RISK STUDENT")

    # -------------------------
    # Recommendations
    # -------------------------

    recommendations = []

    if failures > 1:
        recommendations.append(
            "📚 Academic mentoring recommended"
        )

    if absences > 10:
        recommendations.append(
            "🏫 Improve attendance"
        )

    if goout > 3:
        recommendations.append(
            "⏰ Reduce social distractions"
        )

    if studytime < 2:
        recommendations.append(
            "📖 Increase study hours"
        )

    if health < 3:
        recommendations.append(
            "🏥 Health support suggested"
        )

    if famrel < 3:
        recommendations.append(
            "👨‍👩‍👧 Improve family engagement"
        )

    if Walc > 3:
        recommendations.append(
            "🚫 Reduce alcohol consumption"
        )

    if engagement_score < 0.20:
        recommendations.append(
            "🎯 Student engagement is very low"
        )

    if socioeconomic_score < 6:
        recommendations.append(
            "💰 Additional support resources recommended"
        )

    if lifestyle_risk > 6:
        recommendations.append(
            "⚠ Lifestyle risk is high"
        )

    if len(recommendations) == 0:

        recommendations.append(
            "✅ Student is performing well. Maintain current habits."
        )

    st.subheader("Recommendations")

    for rec in recommendations:
        st.write(rec)

    # -------------------------
    # Student Profile Summary
    # -------------------------

    st.markdown("---")

    st.subheader("Student Profile")

    st.write(f"Age: {age}")
    st.write(f"Absences: {absences}")
    st.write(f"Failures: {failures}")
    st.write(f"Study Time: {studytime}")
    st.write(f"Health: {health}")

    st.markdown("---")

    st.info(
        "Human Complexity AI combines academic, social, lifestyle and engagement indicators to assess student risk."
    )