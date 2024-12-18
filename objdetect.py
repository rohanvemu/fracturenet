#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced FractureNet Streamlit App
Created on Mon Dec 16 17:54:31 2024
@author: rohanvemu
"""

import streamlit as st
import torch
import pandas as pd
import io
from PIL import Image

# Load the custom YOLOv5 model
@st.cache_resource
def load_model():
    model = torch.hub.load("ultralytics/yolov5", "custom", path="best.pt")  # Replace with your model path
    return model

# Save form input to a CSV file
def save_to_csv(name, email, message):
    data = {"Name": name, "Email": email, "Message": message}
    df = pd.DataFrame([data])
    df.to_csv("team_requests.csv", mode="a", index=False, header=False)

# Set page configuration
st.set_page_config(
    page_title="FractureNet",
    page_icon="🦴",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
    /* Title styling */
    .main-title { 
        font-size: 8rem; 
        color: #FFFFFF; 
        background-color: #4A4E69;
        text-align: center;
        font-weight: bold;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
    }
    /* Subtitle styling */
    .sub-title {
        font-size: 1.5rem;
        color: #FFFFFF;
        text-align: center;
        margin-bottom: 2rem;
    }
    /* Sidebar title styling */
    .sidebar-title {
        font-size: 1.2rem;
        color: #9A8C98;
        font-weight: bold;
    }
    /* Section headers */
    .contact-header {
        font-size: 1.8rem;
        color: #FFFFFF;
        margin-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Main Title
st.markdown('<p class="main-title" style="font-size: 36px;">FractureNet: AI-Powered Fracture Detection</p>', unsafe_allow_html=True)

# Subtitle
st.markdown("""
    <p class="sub-title">
        Welcome to FractureNet, an advanced AI tool for fracture detection and localization. 
        Upload your images, adjust detection settings, and let AI assist you with fracture management.
    </p>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("🔧 Settings")
st.sidebar.markdown('<p class="sidebar-title">Adjust Detection Parameters</p>', unsafe_allow_html=True)
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)

# Load YOLOv5 model
model = load_model()

# Image Upload Section
st.markdown('<p style="font-size: 20px;">🖼️ Upload an X-ray image:</p>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", width=400)
    
 # Run Detection
    if st.button("⚙️ Run Detection"):
        with st.spinner("Detecting fractures... 🦴"):
            # Run YOLOv5 inference
            results = model(img)

            # Extract detection data safely
            detected_data = results.pandas().xyxy[0]  # Use 'xyxy' for bounding box format
            detected_image = results.render()[0] 
            st.subheader("Detection Results")

            # Check if expected columns exist
            if {"xmin", "ymin", "xmax", "ymax", "confidence", "name"}.issubset(detected_data.columns):
                # Filter results based on confidence threshold
                detected_data = detected_data[detected_data["confidence"] >= conf_threshold]
                
                if detected_data.empty:
                    st.write("No fractures detected above the selected confidence threshold.")
                else:
                    # Display relevant columns
                    st.dataframe(
                        detected_data[["name", "confidence", "xmin", "ymin", "xmax", "ymax"]].style.set_caption(
                            "Detected Fractures"
                        )
                    )

                    # Display image with bounding boxes
                    st.image(results.render()[0], caption="Processed Image with Bounding Boxes", width=400)
            else:
                st.error("Unexpected detection data format. Columns are missing.")
                # Convert the image to bytes for downloading
                
            image_bytes = io.BytesIO()
            Image.fromarray(detected_image).save(image_bytes, format="JPEG")
            image_bytes.seek(0)  # Reset file pointer to the start
        
            # Add download button for the processed image
            st.download_button(
                label="📥 Download Processed Image",
                data=image_bytes,
                file_name="fracture_detection_result.jpg",
                mime="image/jpeg"
            )


#Join the Team
st.markdown('<p style="font-size: 20px;">🚀 Join the Team</p>', unsafe_allow_html=True)
st.write("Interested in contributing to FractureNet? Fill out the form below!")

name = st.text_input("Your Name")
email = st.text_input("Your Email")
message = st.text_area("How can you contribute or what questions do you have?")

if st.button("📩 Submit"):
    if name and email and message:
        save_to_csv(name, email, message)
        st.success(f"Thank you for your interest, {name}! We will contact you soon.")
    else:
        st.error("Please fill out all the fields.")

# Contact Information Section
st.markdown('<p style="font-size: 20px;"> 👥 Contact Information</p>', unsafe_allow_html=True)
st.write("For inquiries or collaborations, reach out to our team:")

contact_data = {
    "Role": ["Principal Investigator", "Medical Student"],
    "Name": ["Lorenzo Deveza, MD PhD", "Rohan Vemu, MSE"],
    "Email": ["lorenzo.deveza@bcm.edu", "rohan.vemu@bcm.edu"]
}

contact_df = pd.DataFrame(contact_data)
st.table(contact_df)

# Footer Section
st.markdown("""
    <hr>
    <p style="text-align: center; color: #4A4E69;">
        Made with ❤️ by the FractureNet Team | <a href="https://devezaresearch.org/" target="_blank">Visit Our Lab</a>
    </p>
""", unsafe_allow_html=True)
