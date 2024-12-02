import os
import torch
import timm
import logging
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deepfake_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_weights = None
        
    def __call__(self, model, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    logger.info('Restoring best weights')
        else:
            self.best_loss = val_loss
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
                logger.info('Saving best weights')
            self.counter = 0
            
        return self.early_stop

class DeepfakeDataset(Dataset):
    """Custom Dataset for loading deepfake detection images"""
    def __init__(self, root_dir, transform=None, split='train'):
        """
        Args:
            root_dir (str): Root directory containing real/ and fake/ subdirectories
            transform (callable, optional): Optional transform to be applied on images
            split (str): Whether this is 'train' or 'val' dataset
        """
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.classes = ['real', 'fake']
        
        # Initialize lists to store images and labels
        self.image_paths = []
        self.labels = []
        
        # Load images from both real and fake directories
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                raise RuntimeError(f'Directory not found: {class_dir}')
                
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(class_idx)
        
        logger.info(f'Found {len(self.image_paths)} images in {split} split')
        logger.info(f'Real images: {self.labels.count(0)}, Fake images: {self.labels.count(1)}')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class DeepfakeDetector:
    def __init__(self, model_name='tf_efficientnet_b0_ns', batch_size=32, num_classes=2):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'Using device: {self.device}')
        
        self.batch_size = batch_size
        
        # Create model
        logger.info(f'Creating model: {model_name}')
        self.model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
        self.model = self.model.to(self.device)
        
        # Get model configuration and create transform
        self.config = resolve_data_config({}, model=self.model)
        self.transform = create_transform(**self.config)
        
    def train(self, train_loader, val_loader, epochs=100, learning_rate=0.001, patience=7):
        """Train the model"""
        logger.info('Starting training...')
        
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        early_stopping = EarlyStopping(patience=patience)
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_images, batch_labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}'):
                batch_images = batch_images.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_images)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += batch_labels.size(0)
                train_correct += predicted.eq(batch_labels).sum().item()
            
            train_loss = train_loss / len(train_loader)
            train_acc = 100. * train_correct / train_total
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_images, batch_labels in val_loader:
                    batch_images = batch_images.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    outputs = self.model(batch_images)
                    loss = criterion(outputs, batch_labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += batch_labels.size(0)
                    val_correct += predicted.eq(batch_labels).sum().item()
            
            val_loss = val_loss / len(val_loader)
            val_acc = 100. * val_correct / val_total
            
            # Log metrics
            logger.info(f'Epoch {epoch + 1}/{epochs}:')
            logger.info(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            logger.info(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
            
            # Early stopping check
            if early_stopping(self.model, val_loss):
                logger.info('Early stopping triggered')
                break
    
    def predict_image(self, image_path):
        """Predict on a single image"""
        logger.info(f'Processing image: {image_path}')
        
        image = Image.open(image_path).convert('RGB')
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(tensor)
            probability = torch.nn.functional.softmax(output, dim=1)
            fake_prob = probability[0, 1].item()
            
        result = {
            'probability': fake_prob,
            'prediction': 'FAKE' if fake_prob > 0.5 else 'REAL'
        }
        
        logger.info(f'Prediction: {result}')
        return result

def main():
    # Initialize detector
    detector = DeepfakeDetector(
        model_name='tf_efficientnet_b0_ns',
        batch_size=32,
        num_classes=2
    )
    
    # Create datasets
    train_dataset = DeepfakeDataset(
        root_dir='D:\\deep_learn\\IEEE\\train\\',
        transform=detector.transform,
        split='train'
    )
    
    val_dataset = DeepfakeDataset(
        root_dir='D:\\deep_learn\\IEEE\\val\\',
        transform=detector.transform,
        split='val'
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Train the model
    detector.train(train_loader, val_loader, epochs=10, learning_rate=0.001, patience=7)
    
    # Save the trained model
    torch.save(detector.model.state_dict(), 'deepfake_detector.pth')

if __name__ == '__main__':
    main()