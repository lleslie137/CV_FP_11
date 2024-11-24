import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import ViTForImageClassification, ViTImageProcessor

# Load the pre-trained model
# model_path = "./vit-deepfake-detector"  # Path to your trained model
# model = ViTForImageClassification.from_pretrained(model_path)
# processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

# Set model to evaluation mode
# model.eval()

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

# Function to make predictions
def predict_image(image):
    # Process the image with the ViT processor
    inputs = processor(images=image, return_tensors="pt")

    # Get model predictions
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    confidence, prediction = torch.max(probabilities, dim=-1)

    return prediction.item(), confidence.item(), probabilities.tolist()

# Streamlit app
st.title("Deepfake Detection App")
st.write("Upload an image, and the app will detect if it is real or fake.")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Detect and crop face
    st.write("Detecting face...")
    cropped_face = detect_and_crop_face(image)

    if cropped_face is None:
        st.error("No face detected. Please upload an image with a clear face.")
    else:
        st.image(cropped_face, caption="Cropped Face", use_column_width=True)

        # Make prediction
        st.write("Making prediction...")
        prediction, confidence, probabilities = predict_image(cropped_face)

        # Display the results
        label = "Real" if prediction == 0 else "Fake"
        st.write(f"Prediction: **{label}**")
        st.write(f"Confidence: **{confidence * 100:.2f}%**")
        st.write("Class Probabilities:")
        st.json({"Real": probabilities[0][0], "Fake": probabilities[0][1]})
