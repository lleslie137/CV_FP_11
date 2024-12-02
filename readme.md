# Computer Vision - FP11
IEEE Deepfake Detection Application.
<br>
## Deployment
Final model deployment uses the model from VisionTransformer.py. The model is deployed using a Streamlit application that allow users to upload images, which will be classified as a real or deepfake image. Attention weights are extracted from the last layer of the model that are then used to generate a heatmap. The heatmap is overlaid onto the original image to highlight parts of the image that contributed the most to the model's decision. The heatmap provides visual feedback to the user, showing which regions influenced the prediction, thus, increasing the user's trust in the model's decision
<br><br>
Link to Streamlit app: https://comvisfp11.streamlit.app/

## Dataset
Dataset is given by the IEEE Signal Processing Society as part of the DFWildCup: Deepfake Face Detection In The Wild Competition. The dataset is taken from publicly available datasets, such as Celeb-DF-v1, Celeb-DF-v2, FaceForensics++,
DeepfakeDetection, FaceShifter, UADFV, Deepfake Detection Challenge Preview, and Deepfake Detection Challenge, which are then identically pre-processed by the organizers for the competition. All images provided are labelled as deepfake or real images.
<br><br>
Further details about the dataset and competition can be found in the [2025 SP Cup Official Document](https://2025.ieeeicassp.org/wp-content/uploads/sites/489/2025-SP-Cup-Competition-Official-Document_-Version-1_-FINAL.pdf).
<br><br>
Link to dataset: https://drive.google.com/drive/folders/1_4OnSzv1Ag290Hr0abwjks5eKBuva4Rx
