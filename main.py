import streamlit as st
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from safetensors.torch import load_file
from transformers import ViTForImageClassification, ViTImageProcessor
import gdown


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
st.write("Downloading model file from remote...")
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
st.empty()


def visualize_attention(image):
    """
    Visualizes the attention weights for the ViT model.

    Args:
        image (PIL.Image): Input image.

    Returns:
        PIL.Image: Heatmap overlaid on the original image.
    """
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs, output_attentions=True)
    attentions = outputs.attentions  # List of attention matrices

    # Use the last layer's attention weights
    last_attention = attentions[-1]  # (batch_size, num_heads, num_tokens, num_tokens)
    mean_attention = last_attention.mean(dim=1)  # Average over all heads

    # Extract attention for the [CLS] token and map it to the image
    cls_attention = mean_attention[:, 0, 1:]  # Ignore the [CLS] and [SEP] tokens
    cls_attention = cls_attention.reshape(14, 14)  # Assuming 14x14 patches

    # Detach and convert to NumPy
    cls_attention_np = cls_attention.detach().cpu().numpy()

    # Resize attention to match the input image size
    cls_attention_resized = cv2.resize(cls_attention_np, image.size, interpolation=cv2.INTER_LINEAR)

    # Normalize the heatmap for better visualization
    cls_attention_resized = (cls_attention_resized - cls_attention_resized.min()) / (
        cls_attention_resized.max() - cls_attention_resized.min()
    )
    cls_attention_resized = (cls_attention_resized * 255).astype(np.uint8)

    # Create a heatmap and overlay it on the image
    heatmap = cv2.applyColorMap(cls_attention_resized, cv2.COLORMAP_JET)
    heatmap_image = cv2.addWeighted(np.array(image), 0.5, heatmap, 0.5, 0)

    return Image.fromarray(heatmap_image)

def predict_image(image):
    label, confidence, probabilities = original_prediction(image)  # Call existing prediction logic

    # Generate attention visualization
    attention_image = visualize_attention(image)

    return label, confidence, probabilities, attention_image


def original_prediction(image):
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


st.title("Deepfake Detection App")
st.write("Upload an image, and the app will detect if it is real or fake.")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.write("Making prediction...")
    label, confidence, probabilities, attention_image = predict_image(image)
    st.write("**Visualization:**")
    col1, col2 = st.columns(2)
    with col1:
        st.image(attention_image, caption="Attention Heatmap", use_container_width=True)
    with col2:
        st.image(image, caption="Original Image", use_container_width=True)

    st.write(f"**Prediction:** {label}")
    st.write(f"**Confidence:** {confidence * 100:.2f}%")
    st.write("**Class Probabilities:**")
    st.json({"Real": probabilities[0][0], "Fake": probabilities[0][1]})




