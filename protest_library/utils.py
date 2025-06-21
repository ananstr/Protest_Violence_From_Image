"""
Utility functions for protest detection project.
"""

import torch
import torchvision.transforms as transforms
import numpy as np


def get_device():
    """Get the best available device (CUDA or CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def get_transforms():
    """Get standard preprocessing transforms for the models."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet expects 224x224 images
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    return transform

def augment_data():
    """Transforms with augmentation for training data"""
    return transforms.Compose([
        # Augmentation (random operations)
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3),
        
        # Preprocessing (deterministic operations)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def setup_data_loaders(train_dataset, val_dataset, test_dataset, batch_size=32):
    """Set up data loaders for training, validation, and testing."""
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader


def save_model(model, optimizer, metrics, save_path):
    """Save model checkpoint with training information."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        **metrics
    }, save_path)
    print(f"Model saved to {save_path}")


def load_model(model, checkpoint_path, device):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from {checkpoint_path}")
    return model


def print_class_distribution(labels, dataset_name="Dataset"):
    """Print class distribution for a dataset."""
    class_counts = np.bincount(labels.astype(int))
    
    if len(class_counts) > 1:
        print(f"{dataset_name} class distribution: Non-protest: {class_counts[0]}, Protest: {class_counts[1]}")
    else:
        print(f"{dataset_name} class distribution: Non-protest: {class_counts[0]}, Protest: 0")


def filter_violence_data(data, include_zero_violence=False):
    """
    Filter data to include only samples with valid violence scores.
    
    Args:
        data: Dictionary containing loaded data
        include_zero_violence: Whether to include samples with violence score 0
        
    Returns:
        Filtered data arrays
    """
    if include_zero_violence:
        # Use all non-NaN violence scores (including 0)
        mask = ~np.isnan(data['train_violence'])
    else:
        # Use only positive violence scores
        mask = (~np.isnan(data['train_violence'])) & (data['train_violence'] > 0)
    
    filtered_data = {
        'train_images': data['train_images'][mask],
        'train_labels': data['train_labels'][mask],
        'train_violence': data['train_violence'][mask],
        'train_paths': [data['train_paths'][i] for i in range(len(mask)) if mask[i]]
    }
    
    # Similar filtering for test data
    if include_zero_violence:
        test_mask = ~np.isnan(data['test_violence'])
    else:
        test_mask = (~np.isnan(data['test_violence'])) & (data['test_violence'] >= 0)
    
    filtered_data.update({
        'test_images': data['test_images'][test_mask],
        'test_labels': data['test_labels'][test_mask],
        'test_violence': data['test_violence'][test_mask],
        'test_paths': [data['test_paths'][i] for i in range(len(test_mask)) if test_mask[i]]
    })
    
    return filtered_data


def create_performance_summary(results):
    """Create a performance summary from evaluation results."""
    summary = {
        'Test Accuracy': f"{results['test_accuracy']:.4f}",
        'Test Loss': f"{results['test_loss']:.4f}"
    }
    
    if 'test_violence_rmse' in results:  # Multi-task results
        summary.update({
            'Protest Accuracy': f"{results['test_protest_accuracy']:.4f}",
            'Violence RMSE': f"{results['test_violence_rmse']:.4f}",
            'Violence R²': f"{results['test_violence_r2']:.4f}"
        })
    
    return summary
