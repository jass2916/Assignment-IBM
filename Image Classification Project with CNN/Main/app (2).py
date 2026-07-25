import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐾", layout="centered")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('cat_dog_cnn_model.h5')
    return model

with st.spinner('Loading model... Please wait!'):
    model = load_model()

st.title("🐾 Cat vs. Dog Image Classification")
st.write("Upload an image of a cat or a dog, and the Convolutional Neural Network (CNN) will predict what it is!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_display = Image.open(uploaded_file)
    # Updated parameter to avoid the warning message completely
    st.image(image_display, caption='Uploaded Image', use_container_width=True)
    
    st.write("")
    if st.button('Predict!'):
        with st.spinner('Analyzing the image...'):
            img = image_display.resize((150, 150))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x /= 255.0  

            prediction = model.predict(x)
            score = prediction[0][0]

            st.markdown("---")
            st.subheader("Prediction Result:")
            if score > 0.5:
                confidence = score * 100
                st.success(f"### It's a DOG! 🐶")
                st.write(f"**Confidence:** {confidence:.2f}%")
            else:
                confidence = (1 - score) * 100
                st.error(f"### It's a CAT! 🐱")
                st.write(f"**Confidence:** {confidence:.2f}%")
