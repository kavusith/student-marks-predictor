# app.py
import streamlit as st
import pickle
import pandas as pd

st.set_page_config(page_title="Student Mark Predictor", layout="centered")
st.title("🎓 Student Mark Predictor")

# Load model safely
try:
    with open('student_marks.pkl', 'rb') as f:
        data = pickle.load(f)
    # If Pickle contains a dictionary
    if isinstance(data, dict):
        model = data['model']
        feature_cols = data.get('feature_cols', ['number_courses', 'time_study'])
        target_col = data.get('target_col', 'Marks')
    else:
        model = data
        feature_cols = ['number_courses', 'time_study']
        target_col = 'Marks'
except FileNotFoundError:
    st.error("❌ student_marks.pkl not found. Make sure it is uploaded to the repo.")
    st.stop()

st.markdown(f"**Model target:** `{target_col}`  \n**Features (order):** `{feature_cols}`")

# Input widgets
number_courses = st.number_input(
    label="Number of courses",
    min_value=0,
    step=1,
    value=3
)

time_study = st.number_input(
    label="Study time (hours)",
    min_value=0.0,
    step=0.25,
    format="%.3f",
    value=4.0
)

# Build input dataframe
input_df = pd.DataFrame([[number_courses, time_study]], columns=feature_cols)

if st.button("Predict"):
    try:
        X = input_df.astype(float)
        pred = model.predict(X)[0]
        st.success(f"Predicted {target_col}: {pred:.3f}")
        st.write("Input used:", X.to_dict(orient="records")[0])
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.caption("Make sure `student_marks.pkl` exists in this folder (created by train.py).")
