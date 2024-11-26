import streamlit as st
import mediapipe as mp
import cv2
import numpy as np
from PIL import Image
import torch
from safetensors.torch import load_file
from transformers import ViTForImageClassification, ViTImageProcessor
import gdown  # For downloading files from Google Drive
# Google Drive file ID of your safetensors file
FILE_ID = "1QsgZ_2wamTjt491U2iDSGXZQB-4jXxqm"
MODEL_FILE = "model088-2.safetensors"  # Local filename for the downloaded model
# Function to download the model file from Google Drive
@st.cache_data
def download_model_from_drive(file_id, output_file):
    """
    Downloads the model file from Google Drive.

    Args:
        file_id (str): File ID from the Google Drive shareable link.
        output_file (str): Local path where the file should be saved.

    Returns:
        str: Path to the downloaded file.
    """
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_file, quiet=False)
    return output_file

# Download the safetensors model file
st.write("Downloading model file from Google Drive...")
downloaded_model_path = download_model_from_drive(FILE_ID, MODEL_FILE)
st.write("Model downloaded successfully.")


# Download the safetensors model file
st.write("Downloading model file from Google Drive...")
downloaded_model_path = download_model_from_drive(FILE_ID, MODEL_FILE)
st.write("Model downloaded successfully.")

# Load the model
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=2
)

# Load the weights from the safetensors file
state_dict = load_file(downloaded_model_path)  # Use safetensors to load the file
model.load_state_dict(state_dict)

# Set the model to evaluation mode
model.eval()

# Load the processor
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

# Function to detect and crop face
def detect_and_crop_face(image, target_size=224, margin=20):
    """
    Detects a face in the image using Mediapipe and properly crops the face, scaling it to the desired size.

    Args:
        image (PIL.Image): Input image.
        target_size (int): Desired output size for the cropped face (e.g., 224x224).
        margin (int): Extra pixels to include around the detected face.

    Returns:
        PIL.Image or None: Cropped and resized face image, or None if no face is detected.
    """
    mp_face_detection = mp.solutions.face_detection

    # Convert PIL image to numpy array
    image_np = np.array(image)

    # Mediapipe requires RGB format
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))

        # If no face is detected, return None
        if not results.detections:
            return None

        # Get the first detected face
        for detection in results.detections:
            # Extract bounding box
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image_np.shape
            x = int(bboxC.xmin * iw)
            y = int(bboxC.ymin * ih)
            w = int(bboxC.width * iw)
            h = int(bboxC.height * ih)

            # Add margin around the bounding box
            x1 = max(x - margin, 0)
            y1 = max(y - margin, 0)
            x2 = min(x + w + margin, iw)
            y2 = min(y + h + margin, ih)

            # Crop the face properly (ensure aspect ratio is preserved)
            cropped_face = image_np[y1:y2, x1:x2]

            # Resize the cropped face to the target size
            resized_face = cv2.resize(cropped_face, (target_size, target_size), interpolation=cv2.INTER_AREA)

            return Image.fromarray(resized_face)

    return None

# Function to make predictions
def predict_image(image):
    """
    Predicts whether an image is real or fake.

    Args:
        image (PIL.Image): Input image.

    Returns:
        tuple: Prediction label ("Real" or "Fake"), confidence score, and probabilities.
    """
    # Preprocess the image
    inputs = processor(images=image, return_tensors="pt")

    # Get predictions from the model
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)

    # Extract prediction and confidence
    confidence, prediction = torch.max(probabilities, dim=-1)
    label = "Real" if prediction.item() == 0 else "Fake"

    return label, confidence.item(), probabilities.tolist()

# Streamlit app
st.title("Deepfake Detection App")
st.write("Upload an image, and the app will detect if it is real or fake.")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the uploaded image
    image = Image.open(uploaded_file)

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
            st.image(image, caption="Original Image", use_container_width=True)
        with col2:
            st.image(cropped_face, caption="Cropped Face", use_container_width=True)

        # Make prediction
        st.write("Making prediction...")
        label, confidence, probabilities = predict_image(cropped_face)

        # Display the results
        st.write(f"**Prediction:** {label}")
        st.write(f"**Confidence:** {confidence * 100:.2f}%")
        st.write("**Class Probabilities:**")
        st.json({"Real": probabilities[0][0], "Fake": probabilities[0][1]})
