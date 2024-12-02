import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import logging
import sys
from datetime import datetime
import json
import logging

def setup_logger():
    logger = logging.getLogger('DeepfakeDetector')
    logger.setLevel(logging.INFO)
    
    # Create console handler with formatting
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    
    # Add handler if it doesn't exist
    if not logger.handlers:
        logger.addHandler(ch)
    
    return logger

class MetricsVisualizer:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
    
    def update(self, train_loss, val_loss, train_acc, val_acc):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)
    
    def plot_metrics(self):
        # Set style for better visualizations
        plt.style.use('seaborn')
        
        # Create a figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Training Metrics Over Time', fontsize=16, y=1.05)
        
        # Plot losses
        epochs = range(1, len(self.train_losses) + 1)
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2, marker='o')
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2, marker='o')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend(loc='upper right')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Add loss values as text annotations
        for i, (train_loss, val_loss) in enumerate(zip(self.train_losses, self.val_losses)):
            ax1.annotate(f'{train_loss:.3f}', 
                        (i+1, train_loss), 
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center',
                        fontsize=8)
            ax1.annotate(f'{val_loss:.3f}', 
                        (i+1, val_loss), 
                        textcoords="offset points", 
                        xytext=(0,-15), 
                        ha='center',
                        fontsize=8)
        
        # Plot accuracies
        ax2.plot(epochs, self.train_accs, 'g-', label='Training Accuracy', linewidth=2, marker='o')
        ax2.plot(epochs, self.val_accs, 'm-', label='Validation Accuracy', linewidth=2, marker='o')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend(loc='lower right')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Add accuracy values as text annotations
        for i, (train_acc, val_acc) in enumerate(zip(self.train_accs, self.val_accs)):
            ax2.annotate(f'{train_acc:.1f}%', 
                        (i+1, train_acc), 
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center',
                        fontsize=8)
            ax2.annotate(f'{val_acc:.1f}%', 
                        (i+1, val_acc), 
                        textcoords="offset points", 
                        xytext=(0,-15), 
                        ha='center',
                        fontsize=8)
        
        # Adjust layout and display
        plt.tight_layout()
        plt.show()

class DeepfakeDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Dataset class for loading deepfake and real images
        Args:
            data_dir (str): Directory containing 'real' and 'fake' subdirectories
            transform: Optional transform to be applied to images
        """
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load real images
        real_dir = os.path.join(data_dir, 'real')
        for img_name in os.listdir(real_dir):
            self.images.append(os.path.join(real_dir, img_name))
            self.labels.append(0)  # 0 for real
            
        # Load fake images
        fake_dir = os.path.join(data_dir, 'fake')
        for img_name in os.listdir(fake_dir):
            self.images.append(os.path.join(fake_dir, img_name))
            self.labels.append(1)  # 1 for fake

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class DeepfakeRCNN(nn.Module):
    def __init__(self, num_classes=2):
        """
        Simplified R-CNN model for deepfake detection
        Args:
            num_classes (int): Number of output classes (2 for binary classification)
        """
        super(DeepfakeRCNN, self).__init__()
        
        # Use ResNet34 as backbone
        self.backbone = models.resnet34(pretrained=True)
        
        # Remove the last fully connected layer
        for param in list(self.backbone.parameters())[:-4]:
            param.requires_grad = False
            
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # Extract features using backbone
        features = self.backbone(x)
        
        # Reshape features
        features = features.view(features.size(0), -1)
        
        # Classification
        output = self.classifier(features)
        return output

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, logger=None):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.logger = logger

    def __call__(self, val_loss, model, path='checkpoint.pt'):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, path)
        elif val_loss > self.best_loss + self.min_delta:
            self.counter += 1
            if self.logger:
                self.logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.logger:
            self.logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss

def train_model(model, train_loader, val_loader, num_epochs=10, device='cuda', patience=7, save_dir='model_outputs'):
    """
    Training function with logger and separated visualization
    """
    logger = setup_logger()
    visualizer = MetricsVisualizer()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=patience, logger=logger)
    
    model = model.to(device)
    logger.info(f"Training started on device: {device}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss/len(train_loader)
        train_acc = 100. * correct / total
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = val_running_loss/len(val_loader)
        val_acc = 100. * correct / total
        
        # Update visualizer
        visualizer.update(train_loss, val_loss, train_acc, val_acc)
        
        # Log progress
        logger.info(f'Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Early stopping check
        early_stopping(val_loss, model, path=f'{save_dir}/best_model.pth')
        if early_stopping.early_stop:
            logger.info('Early stopping triggered')
            break
    
    # Create final plots
    visualizer.plot_metrics()
    logger.info(f'Training completed.')
    
    # Load best model
    return model, {'train_losses': visualizer.train_losses, 'val_losses': visualizer.val_losses,
                  'train_accs': visualizer.train_accs, 'val_accs': visualizer.val_accs}

def main():
    # Set up logging directory
    log_dir = 'logs'
    save_dir = 'model_outputs'
    
    # Create directories if they don't exist
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets and loaders
    train_dataset = DeepfakeDataset(
        data_dir='D:\\deep_learn\\IEEE\\train\\',
        transform=transform
    )
    val_dataset = DeepfakeDataset(
        data_dir='D:\\deep_learn\\IEEE\\val\\',
        transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    model = DeepfakeRCNN(num_classes=2)
    
    # Train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trained_model, history = train_model(
        model, 
        train_loader, 
        val_loader, 
        num_epochs=10, 
        device=device,
        patience=7
    )

if __name__ == '__main__':
    main()

