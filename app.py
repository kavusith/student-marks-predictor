# app.py
import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.title("Student Mark Predictor")

# Load model
with open('student_marks.pkl', 'rb') as f:
    data = pickle.load(f)

model = data['model']
feature_cols = data.get('feature_cols', ['Courses', 'StudyTime'])

st.write("Model expects features in this order:", feature_cols)

# Inputs - use appropriate input types
# If number of courses is integer, use number_input with step=1
courses = st.number_input("Enter the number of courses", min_value=0, step=1, value=1)
study_time = st.number_input("Enter the study time (hours)", min_value=0.0, step=0.5, value=1.0)

# If you have more features, add other inputs here.

if st.button("Predict"):
    # Build input in the SAME order as feature_cols
    inp_df = pd.DataFrame([[courses, study_time]], columns=feature_cols)
    # Optional: ensure numeric type
    inp_df = inp_df.astype(float)
    pred = model.predict(inp_df)[0]
    st.success(f"Predicted Marks: {pred:.2f}")

st.caption("Make sure the dataset columns and input fields match the training script.")
