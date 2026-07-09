import streamlit as st
import joblib
import numpy as np

# Load the trained model and vectorizer
model = joblib.load('spam_ham_model.h5')
vectorizer = joblib.load('vectorizer.pkl')

st.title("Spam/Ham Message Predictor")
st.write("Enter the Message to predict whether it is spam/ham: ")

# Input Fields
msg = st.text_input("Enter the Message ")

if st.button("Predict Spam/Ham"):
  if msg:
    message = vectorizer.transform([msg])
    prediction = model.predict(message)

    # Decode the prediction
    if prediction[0] == 1:
        result = "Spam"
    else:
        result = "Ham"

    st.success(f"The message is: {result}")
  else:
    st.warning("Please enter a message to classify.")
