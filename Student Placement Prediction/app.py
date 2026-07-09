import streamlit as st
import joblib
import numpy as np

# Load the saved model and scaler
model = joblib.load('placement_model.h5')
scaler = joblib.load('scaler.pkl')

st.title("Student Placement Predictor")
st.write("Enter the following details to predict placement status:")

# Input fields - ensure the order matches the columns in your training data
cgpa = st.number_input("CGPA", 0.0, 10.0)
internships = st.number_input("Number of Internships", 0, 5)
projects = st.number_input("Number of Projects", 0, 10)
workshops = st.number_input("Workshops/Certifications", 0, 10)
aptitude = st.number_input("Aptitude Test Score", 0, 100)
soft_skills = st.number_input("Soft Skills Rating", 0.0, 5.0)
extracurricular = st.selectbox("Extracurricular Activities", [0, 1]) # 0: No, 1: Yes
training = st.selectbox("Placement Training", [0, 1]) # 0: No, 1: Yes
ssc = st.number_input("SSC Marks", 0, 100)
hsc = st.number_input("HSC Marks", 0, 100)

if st.button("Predict Placement"):
    # Combine inputs into a feature array
    input_features = np.array([[cgpa, internships, projects, workshops, aptitude, 
                                soft_skills, extracurricular, training, ssc, hsc]])
    
    # Scale inputs using the loaded scaler
    scaled_features = scaler.transform(input_features)
    
    # Predict
    prediction = model.predict(scaled_features)
    
    # Display result
    result = "Placed" if prediction[0] == 1 else "Not Placed"
    st.subheader(f"Prediction: {result}")
