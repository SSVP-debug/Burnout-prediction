"""Streamlit dashboard for the burnout risk model.

This is a thin presentation layer -- all data generation, preprocessing,
and training logic lives in src/ and data/, not here. If no trained
model is found, the app trains one on startup rather than requiring a
pre-committed binary.
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.data_loader import FEATURE_COLUMNS

MODEL_PATH = Path(__file__).resolve().parent / "models" / "burnout_model.joblib"

st.set_page_config(page_title="Burnout Risk Predictor", page_icon="🧭", layout="centered")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.warning("No trained model found — training one now (first run only)...")
        from src.train import main as train_main

        train_main()
    return joblib.load(MODEL_PATH)


def main():
    st.title("Burnout Risk Predictor")
    st.caption(
        "Trained on synthetic data. This is a portfolio/demo project, "
        "not a validated clinical or HR tool. See README for methodology "
        "and limitations."
    )

    model = load_model()

    st.subheader("Workload")
    col1, col2 = st.columns(2)
    with col1:
        meetings_count = st.slider("Meetings per day", 0, 15, 4)
        total_meeting_minutes = st.slider("Total meeting minutes / day", 0, 480, 180)
        back_to_back_meetings = st.slider("Back-to-back meetings", 0, 10, 1)
        avg_meeting_gap_minutes = st.slider("Avg. gap between meetings (min)", 0, 120, 25)
    with col2:
        deep_work_minutes = st.slider("Deep work minutes / day", 0, 480, 150)
        context_switches = st.slider("Context switches / day", 0, 30, 8)
        tasks_completed = st.slider("Tasks completed / day", 0, 25, 6)
        overdue_tasks = st.slider("Overdue tasks", 0, 15, 1)

    st.subheader("After-hours & recovery")
    col3, col4 = st.columns(2)
    with col3:
        emails_after_hours = st.slider("Emails sent after hours", 0, 20, 3)
        after_hours_work_minutes = st.slider("After-hours work (min)", 0, 300, 35)
        sleep_hours_prev_night = st.slider("Sleep last night (hrs)", 2.0, 10.0, 6.8, step=0.1)
        exercise_minutes = st.slider("Exercise minutes yesterday", 0, 180, 20)
    with col4:
        mood_prev_day = st.slider("Self-rated mood yesterday (1=bad, 5=great)", 1, 5, 3)
        interruptions_count = st.slider("Interruptions / day", 0, 25, 5)
        caffeine_cups = st.slider("Caffeine (cups) / day", 0, 10, 2)
        commute_minutes = st.slider("Commute minutes / day", 0, 120, 25)

    st.subheader("Context")
    col5, col6 = st.columns(2)
    with col5:
        wfh = st.checkbox("Working from home today")
    with col6:
        sick_or_caregiving = st.checkbox("Sick or caregiving responsibilities")

    input_dict = {
        "meetings_count": meetings_count,
        "total_meeting_minutes": total_meeting_minutes,
        "back_to_back_meetings": back_to_back_meetings,
        "avg_meeting_gap_minutes": avg_meeting_gap_minutes,
        "deep_work_minutes": deep_work_minutes,
        "context_switches": context_switches,
        "emails_after_hours": emails_after_hours,
        "after_hours_work_minutes": after_hours_work_minutes,
        "sleep_hours_prev_night": sleep_hours_prev_night,
        "exercise_minutes": exercise_minutes,
        "mood_prev_day": mood_prev_day,
        "tasks_completed": tasks_completed,
        "overdue_tasks": overdue_tasks,
        "interruptions_count": interruptions_count,
        "caffeine_cups": caffeine_cups,
        "wfh": int(wfh),
        "commute_minutes": commute_minutes,
        "sick_or_caregiving": int(sick_or_caregiving),
    }

    input_df = pd.DataFrame([input_dict])[FEATURE_COLUMNS]  # enforce column order

    if st.button("Predict burnout risk", type="primary"):
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]
        classes = model.classes_

        color = {"Low": "green", "Moderate": "orange", "High": "red"}.get(prediction, "blue")
        st.markdown(f"### Predicted risk: :{color}[{prediction}]")

        proba_df = pd.DataFrame({"Risk level": classes, "Probability": proba}).set_index("Risk level")
        st.bar_chart(proba_df)

    with st.expander("Model details"):
        st.write(
            "Model: whichever of Logistic Regression / Random Forest scored higher "
            "on 5-fold cross-validated accuracy at training time (see "
            "`reports/metrics.json` for the actual selected model and full metrics)."
        )
        st.write("Trained and evaluated on synthetic data — see README for full methodology.")


if __name__ == "__main__":
    main()
