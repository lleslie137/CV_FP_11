import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Function to detect and crop face
def detect_and_crop_face(image):
    # Convert PIL image to numpy array for OpenCV
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

    # Load OpenCV's Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # If no face is detected, return None
    if len(faces) == 0:
        return None

    # Crop the first detected face
    for (x, y, w, h) in faces:
        cropped_face = image_np[y:y+h, x:x+w]
        return Image.fromarray(cropped_face)

    return None

# Streamlit app
st.title("Face Detection App")
st.write("Upload an image, and the app will detect and crop the face (if any).")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Detect and crop face
    st.write("Detecting face...")
    cropped_face = detect_and_crop_face(image)

    # Display results
    if cropped_face is None:
        st.error("No face detected. Please upload an image with a clear face.")
    else:
        st.write("Face detected and cropped.")
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_column_width=True)
        with col2:
            st.image(cropped_face, caption="Cropped Face", use_column_width=True)
