# !pip install datasets

import os
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
from google.colab import drive
from transformers import ViTForImageClassification, AutoFeatureExtractor, Trainer, TrainingArguments
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset

from google.colab import drive
drive.mount('/content/drive')

input_dir = "/content/drive/My Drive/Newfolder/DFWildCup"

train10k = {
    "train_fake": "fake_3_1/fake_3_1",
    "train_real": "real_4_1/real_4_1",
}
train10k2 = {
    "train_fake": "fake_3_1/fake_3_2",
    "train_real": "real_4_1/real_4_2",
}
train40k= {"train_fake": "fake_1", "train_real": "real_1","train_real": "real_2"}
batch_size = 256
lr=5e-5


class DeepfakeDataset(Dataset):
    def __init__(self, input_dir, folder_mapping, transform=None):
        """
        Args:
            input_dir (str): Base path to the dataset directory.
            folder_mapping (dict): Maps dataset types (train_fake, train_real, etc.) to subfolder labels.
            transform (callable, optional): Transformations to apply to images.
        """
        self.image_paths = []
        self.labels = []
        self.transform = transform

        # Collect image paths and assign labels
        for folder_name, label_subfolder in folder_mapping.items():
            folder_path = os.path.join(input_dir, folder_name, label_subfolder)
            if os.path.exists(folder_path):
                label = 0 if "real" in folder_name else 1  # Assign label: real -> 0, fake -> 1
                for file_name in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file_name)
                    if os.path.isfile(file_path):  # Ensure it's a file
                        self.image_paths.append(file_path)
                        self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    from PIL import Image, UnidentifiedImageError

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            # Attempt to open the image
            image = Image.open(img_path).convert("RGB")
        except (OSError, UnidentifiedImageError) as e:
            print(f"Skipping corrupted file: {img_path}")
            # Return a dummy image and label if the file is corrupted
            image = Image.new("RGB", (224, 224), color=(0, 0, 0))
            label = -1  # Assign a dummy label (if needed for handling)

        if self.transform:
            image = self.transform(image)

        return {"pixel_values": image, "labels": label}




from sklearn.metrics import accuracy_score, precision_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    acc = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='binary')  # Adjust for multi-class if needed
    return {"accuracy": acc, "precision": precision}



data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to match model input size
        transforms.RandomHorizontalFlip(),  # Data augmentation: random horizontal flip
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color jitter
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
}

# Create train and validation datasets
train_dataset = DeepfakeDataset(
    input_dir=input_dir,
    folder_mapping=train10k,
    transform=data_transforms['train']
)

val_dataset = DeepfakeDataset(
    input_dir=input_dir,
    folder_mapping={"valid_fake": "fake", "valid_real": "real"},
    transform=data_transforms['val']
)

# Example of dataset output
print(train_dataset[0])  # Should output: {'pixel_values': tensor(...), 'labels': 0 or 1}

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

from transformers import ViTForImageClassification, TrainingArguments, Trainer

# Load the Vision Transformer model
model_name = "google/vit-base-patch16-224-in21k"
model = ViTForImageClassification.from_pretrained(model_name, num_labels=2)

# Define training arguments
TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",       # Log validation metrics at the end of every epoch
    save_strategy="epoch",             # Save model checkpoints after every epoch
    logging_strategy="epoch",          # Log training metrics after every epoch
    save_steps=None,                   # Disable step-based saving, rely on epoch-based saving
    logging_steps=None,                # Disable step-based logging
    num_train_epochs=10,               # Number of epochs
    per_device_train_batch_size=480,
    per_device_eval_batch_size=480,
    learning_rate=5e-5,
    weight_decay=1e-4,
    load_best_model_at_end=True,       # Load the best model after training
    save_total_limit=10,                # Limit the number of checkpoints
    fp16=True,
)


# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=None,
    compute_metrics=compute_metrics,
)

# from transformers import ViTForImageClassification, ViTConfig
# from torch import nn

# # Adjust dropout rates in the configuration
# config = ViTConfig.from_pretrained(
#     "google/vit-base-patch16-224-in21k",
#     num_labels=2,
#     hidden_dropout_prob=0.2,            # Dropout for hidden layers
#     attention_probs_dropout_prob=0.2   # Dropout for attention heads
# )

# # Initialize the model with the modified configuration
# model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", config=config)

# # Add additional dropout to the classification head (optional)
# model.classifier = nn.Sequential(
#     nn.Dropout(0.3),  # Add 30% dropout before the classifier
#     model.classifier  # Original classifier
# )

# # Define training arguments
# training_args = TrainingArguments(
#     output_dir="./results",
#     evaluation_strategy="epoch",
#     save_strategy="epoch",
#     learning_rate=1e-5,
#     per_device_train_batch_size=480,
#     per_device_eval_batch_size=480,
#     num_train_epochs=10,
#     weight_decay=0.01,
#     logging_dir="./logs",
#     logging_steps=10,
#     load_best_model_at_end=True,
#     save_total_limit=5,
#     fp16=True
# )

# # Create the Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
#     tokenizer=None,
#     compute_metrics=compute_metrics,
# )

# # Start training
# trainer.train()


trainer.train()




# 7. Evaluate the Model
metrics = trainer.evaluate()
print(metrics)


# 8. Save the Model
trainer.save_model("./vit-deepfake-detector")


# After training, extract logs
logs = trainer.state.log_history  # Contains all logged metrics

# Extract specific metrics
training_loss = [log["loss"] for log in logs if "loss" in log]
validation_loss = [log["eval_loss"] for log in logs if "eval_loss" in log]
accuracy = [log["eval_accuracy"] for log in logs if "eval_accuracy" in log]
precision = [log["eval_precision"] for log in logs if "eval_precision" in log]
epochs = range(1, len(validation_loss) + 1)  # Assuming one log per epoch

# Plot the metrics
import matplotlib.pyplot as plt

# Plot Loss
plt.figure()
plt.plot(training_loss, label="Training Loss")
plt.plot(epochs, validation_loss, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.show()

# Plot Accuracy
plt.figure()
plt.plot(epochs, accuracy, label="Validation Accuracy")
plt.plot(epochs, precision, label="Validation Precision")
plt.xlabel("Epoch")
plt.ylabel("Validation")
plt.title("Validation Accuracy and Precision")
plt.legend()
plt.show()
