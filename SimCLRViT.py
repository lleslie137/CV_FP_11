from transformers import ViTModel
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
import os

class DeepfakeDataset(Dataset):
    def __init__(self, input_dir, folder_mapping, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform

        for folder_name, label_subfolder in folder_mapping.items():
            folder_path = os.path.join(input_dir, label_subfolder)
            if os.path.exists(folder_path):
                label = 0 if "real" in folder_name else 1
                for file_name in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file_name)
                    if os.path.isfile(file_path):
                        self.image_paths.append(file_path)
                        self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except (OSError, UnidentifiedImageError):
            image = Image.new("RGB", (224, 224), color=(0, 0, 0))
            label = -1

        if self.transform:
            image = self.transform(image)

        return {"pixel_values": image, "labels": label}

# Pretraining Model Class: SimCLRViT
class SimCLRViT(nn.Module):
    def __init__(self, vit_model_name="google/vit-base-patch16-224-in21k", projection_dim=128):
        super(SimCLRViT, self).__init__()
        self.backbone = ViTModel.from_pretrained(vit_model_name)
        in_features = self.backbone.config.hidden_size
        self.projector = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )

    def forward(self, x):
        features = self.backbone(pixel_values=x).pooler_output
        projections = self.projector(features)
        return nn.functional.normalize(projections, dim=1)

# Fine-Tuning Model Class: DeepfakeViT
class DeepfakeViT(nn.Module):
    def __init__(self, pretrained_backbone, num_classes=2):
        super(DeepfakeViT, self).__init__()
        self.backbone = pretrained_backbone
        in_features = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.backbone(pixel_values=x).pooler_output
        output = self.classifier(features)
        return output

# Data Preparation
input_dir = "/content/drive/My Drive/Newfolder/DFWildCup"
train10k = {"train_fake": "fake_3_1/fake_3_1", "train_real": "real_4_1/real_4_1"}
val_mapping = {"valid_fake": "fake", "valid_real": "real"}

data_transforms = {
    'train': transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]),
    'val': transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
}

train_dataset = DeepfakeDataset(input_dir, train10k, transform=data_transforms['train'])
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_dataset = DeepfakeDataset(input_dir, val_mapping, transform=data_transforms['val'])
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)


"""  SimCLR Pretraining Logic """
def train_simclr(simclr_model, train_loader, epochs=10, lr=1e-4):
    optimizer = torch.optim.Adam(simclr_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    simclr_model.train()

    for epoch in range(epochs):
        epoch_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            images = batch['pixel_values'].to(device)
            z = simclr_model(images)
            loss = criterion(z, torch.eye(z.size(0)).to(device))  # Simplified NT-Xent loss for contrastive learning
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(train_loader)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
simclr_model = SimCLRViT().to(device)
train_simclr(simclr_model, train_loader)

# Save Pretrained Weights
torch.save(simclr_model.backbone.state_dict(), "simclr_vit_backbone.pth")


"""  Vision Transformer Fine-Tuning Logic """
# Fine-Tuning Step
pretrained_backbone = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
pretrained_backbone.load_state_dict(torch.load("simclr_vit_backbone.pth"))
fine_tune_model = DeepfakeViT(pretrained_backbone, num_classes=2).to(device)

# Fine-Tuning Logic (Implement Training Loop as Needed)
def compute_metrics(predictions, labels):
    predictions = predictions.argmax(axis=1)
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='binary')  # Adjust for binary classification
    return {"accuracy": accuracy, "precision": precision}

def fine_tune_model(model, train_loader, val_loader, epochs=10, lr=1e-4):
    """
    Fine-tune the DeepfakeViT model.
    Args:
        model: DeepfakeViT model.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
    """
    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_labels = []
        train_preds = []

        for batch in train_loader:
            images = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            train_loss += loss.item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track predictions and labels
            train_preds.extend(outputs.detach().cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        # Compute metrics for training
        train_metrics = compute_metrics(torch.tensor(train_preds), train_labels)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_loader):.4f}, "
              f"Train Accuracy: {train_metrics['accuracy']:.4f}, Train Precision: {train_metrics['precision']:.4f}")

        # Validation loop
        model.eval()
        val_loss = 0
        val_labels = []
        val_preds = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Track predictions and labels
                val_preds.extend(outputs.detach().cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        # Compute metrics for validation
        val_metrics = compute_metrics(torch.tensor(val_preds), val_labels)
        print(f"Validation Loss: {val_loss / len(val_loader):.4f}, "
              f"Validation Accuracy: {val_metrics['accuracy']:.4f}, Validation Precision: {val_metrics['precision']:.4f}")

# Load pretrained SimCLR backbone
pretrained_backbone = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
pretrained_backbone.load_state_dict(torch.load("simclr_vit_backbone.pth"))

# Initialize the fine-tuning model
fine_tune_model = DeepfakeViT(pretrained_backbone=pretrained_backbone, num_classes=2)

# Prepare DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Fine-tune the model
fine_tune_model(fine_tune_model, train_loader, val_loader, epochs=10, lr=5e-5)