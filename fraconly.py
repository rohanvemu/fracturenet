import os
import streamlit as st
import torch
import pandas as pd
import numpy as np
import io
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
from tensorflow.keras.metrics import Precision, Recall
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import cv2


@tf.keras.utils.register_keras_serializable(package='Custom', name='F1Score')
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = Precision()
        self.recall = Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + K.epsilon()))

    def reset_states(self):
        self.precision.reset_state()
        self.recall.reset_state()

# Load the classification model for 3-view diagnosis
@st.cache_resource
def load_classification_model():
    return load_model("DN169_3View.keras", custom_objects={"F1Score": F1Score})

# Preprocess image for classification model
def preprocess_for_classification(image):
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = image / 255.0  # Normalize
    return image

# Save form input to a CSV file
def save_to_csv(name, email, message):
    data = {"Name": name, "Email": email, "Message": message}
    df = pd.DataFrame([data])
    df.to_csv("team_requests.csv", mode="a", index=False, header=False)

# Page setup
st.set_page_config(page_title="FractureNet", page_icon="🦴", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main-title { font-size: 8rem; color: #FFFFFF; background-color: #4A4E69; text-align: center; font-weight: bold; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1.5rem; }
    .sub-title { font-size: 1.5rem; color: #FFFFFF; text-align: center; margin-bottom: 2rem; }
    .sidebar-title { font-size: 1.2rem; color: #9A8C98; font-weight: bold; }
    .contact-header { font-size: 1.8rem; color: #FFFFFF; margin-top: 2rem; }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-title" style="font-size: 36px;">FractureNet: AI-Powered Fracture Management Assistant</p>', unsafe_allow_html=True)

st.markdown("""
    <p class="sub-title">
        Welcome to FractureNet, an advanced AI tool for assessing the need for intervention for a variety of fractures. 
        Upload your images and let AI assist you with your management plan.
    </p>
""", unsafe_allow_html=True)

# Upload individual images for AP, Oblique, Lateral
st.markdown('<p style="font-size: 20px;">🖼️ Upload X-ray views for classification:</p>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    ap_file = st.file_uploader("AP View", type=["jpg", "jpeg", "png"], key="ap")
with col3:
    oblique_file = st.file_uploader("Oblique View", type=["jpg", "jpeg", "png"], key="oblique")
with col2:
    lat_file = st.file_uploader("Lateral View", type=["jpg", "jpeg", "png"], key="lat")

classification_model = load_classification_model()

if ap_file and oblique_file and lat_file:
    images = [Image.open(file).convert("RGB") for file in [ap_file, lat_file, oblique_file]]
    views = ["AP", "Lateral", "Oblique"]
    for view, img in zip(views, images):
        st.image(img, caption=f"{view} View", width=300)

    if st.button("📊 Click to Run AI Model!"):
        with st.spinner("Assessing if intervention is necessary..."):
            preprocessed = [preprocess_for_classification(img) for img in images]
            input_list = [np.expand_dims(arr, axis=0) for arr in preprocessed]

            prob = classification_model.predict(input_list)[0][0]
            label = "No Intervention" if prob >= 0.5 else "Needs Intervention"
            st.success(f"Intervention Status: {label}")
            
else:
    st.warning("Please upload all three views: AP, Lateral, and Oblique.")
    


# Join the Team
st.markdown('<p style="font-size: 20px;">🚀 Join the Team</p>', unsafe_allow_html=True)
name = st.text_input("Your Name")
email = st.text_input("Your Email")
message = st.text_area("How can you contribute or what questions do you have?")

if st.button("📩 Submit"):
    if name and email and message:
        save_to_csv(name, email, message)
        st.success(f"Thank you for your interest, {name}! We will contact you soon.")
    else:
        st.error("Please fill out all the fields.")

# Contact Info
st.markdown('<p style="font-size: 20px;"> 👥 Contact Information</p>', unsafe_allow_html=True)
contact_df = pd.DataFrame({
    "Role": ["Principal Investigator", "Medical Student", "Medical Student"],
    "Name": ["Lorenzo Deveza, MD PhD", "Rohan Vemu, MSE", "Jacob Scioscia, BS"],
    "Email": ["lorenzo.deveza@bcm.edu", "rohan.vemu@bcm.edu", "jacob.scioscia@bcm.edu"]
})
st.table(contact_df)

# Footer
st.markdown("""
    <hr>
    <p style="text-align: center; color: #4A4E69;">
        Made with ❤️ by the FractureNet Team | <a href="https://devezaresearch.org/" target="_blank">Visit Our Lab</a>
    </p>
""", unsafe_allow_html=True)
