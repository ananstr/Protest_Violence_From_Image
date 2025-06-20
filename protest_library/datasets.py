"""
Dataset classes for protest detection and violence prediction.
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class ProtestDataset(Dataset):
    """Dataset class for protest detection."""
    
    def __init__(self, images, labels=None, transform=None):
        """
        Args:
            images: Array of images
            labels: Array of labels (optional for inference)
            transform: Torchvision transforms to apply
        """
        self.images = images
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        
        # Convert to PIL Image if necessary (for transforms)
        if self.transform:
            image = Image.fromarray((image * 255).astype(np.uint8))
            image = self.transform(image)
        else:
            # PyTorch expects image data in format [C, H, W]
            image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)
        
        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return image, label
        else:
            return image


class UnifiedMultiTaskDataset(Dataset):
    """Unified dataset for multi-task learning (protest, attributes, violence)"""
    def __init__(self, images, protest_labels=None, attribute_labels=None, violence_labels=None, transform=None):
        """
        Args:
            images: List/array of images (numpy arrays or paths)
            protest_labels: Optional protest binary labels
            attribute_labels: Optional attribute vectors
            violence_labels: Optional violence scores
            transform: Torchvision transforms
        """
        self.images = images
        self.protest_labels = protest_labels
        self.attribute_labels = attribute_labels
        self.violence_labels = violence_labels
        self.transform = transform
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load and transform image
        image = self.images[idx]
        
        if self.transform:
            if isinstance(image, np.ndarray):
                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                image = Image.fromarray(image)
            image = self.transform(image)
        else:
            if isinstance(image, np.ndarray):
                image = torch.FloatTensor(image).permute(2, 0, 1)
        
        # Initialize all labels as None
        protest_label = attribute_label = violence_label = None
        
        # Process protest label if available
        if self.protest_labels is not None:
            protest_label = torch.tensor(
                self.protest_labels[idx], 
                dtype=torch.float32
            )
        
        # Process attribute labels if available
        if self.attribute_labels is not None:
            attribute_label = torch.tensor(
                self.attribute_labels[idx], 
                dtype=torch.float32
            )
        
        # Process violence label with NaN handling
        if self.violence_labels is not None:
            val = self.violence_labels[idx]
            if isinstance(val, float) and np.isnan(val):
                val = 0.0
            violence_label = torch.tensor(val, dtype=torch.float32)
        
        # Return only available labels
        labels = tuple(label for label in 
                      [protest_label, attribute_label, violence_label] 
                      if label is not None)
        
        return (image, *labels) if labels else image