# Protest Detection and Violence Prediction Library

A comprehensive Python library for training and evaluating deep learning models for protest detection and violence prediction in images.

## Overview

This library provides a complete toolkit for:
- **Protest Detection**: Binary classification to identify protests in images
- **Violence Prediction**: Regression to predict violence levels (0-1 scale)
- **Multi-task Learning**: Joint training for both tasks
- **Comprehensive Evaluation**: Detailed metrics and visualizations

## Library Structure

```
library/
├── __init__.py              # Main library imports
├── datasets.py              # PyTorch Dataset classes
├── models.py                # Neural network architectures
├── data_loader.py           # Data loading and preprocessing
├── training.py              # Training functions
├── evaluation.py            # Model evaluation utilities
├── visualization.py         # Plotting and analysis
└── utils.py                # General utility functions
```

## Quick Start

### 1. Import the Library

```python
from library import *

# Or import specific modules
from library import (
    load_protest_data, ProtestDataset, UnifiedMultiTaskModel,
    train_model, evaluate_model, plot_stylish_confusion_matrix
)
```

### 2. Load and Prepare Data

```python
# Load annotations
annot_train, annot_test = load_annotations('data/annot_train.txt', 'data/annot_test.txt')
annot_train, annot_test = clean_binary_columns(annot_train, annot_test)

# Load image data
data = load_protest_data('data/train', 'data/test', annot_train, annot_test)

# Set up transforms and device
device = get_device()
transform = get_transforms()
```

### 3. Create Datasets and Data Loaders

```python
from sklearn.model_selection import train_test_split

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    data['train_images'], data['train_labels'], 
    test_size=0.2, random_state=42, stratify=data['train_labels']
)

# Create datasets
train_dataset = ProtestDataset(X_train, y_train, transform=transform)
val_dataset = ProtestDataset(X_val, y_val, transform=transform)
test_dataset = ProtestDataset(data['test_images'], data['test_labels'], transform=transform)

# Set up data loaders
train_loader, val_loader, test_loader = setup_data_loaders(
    train_dataset, val_dataset, test_dataset, batch_size=32
)
```

### 4. Train a Model

```python
import torch.nn as nn
import torch.optim as optim

# Initialize model
model = UnifiedMultiTaskModel().to(device)
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)

# Train the model
history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
    num_epochs=10
)
```

### 5. Evaluate and Visualize

```python
# Evaluate the model
results = evaluate_model(model, test_loader, device)

# Create visualizations
plot_training_history(history)
plot_stylish_confusion_matrix(results['labels'], results['predictions'])
plot_stylish_roc_curve(results['labels'], results['scores'])
plot_stylish_misclassifications(data['test_paths'], results['labels'], results['predictions'], results['scores'])
```

## Multi-Task Learning

For joint protest detection and violence prediction:

```python
# Load data with violence scores
data = load_data_with_violence('data/train', 'data/test', annot_train, annot_test)
data = filter_violence_data(data, include_zero_violence=True)

# Create multi-task dataset
train_dataset = MultiTaskDataset(X_train, y_protest_train, y_violence_train, transform=transform)

# Initialize multi-task model
model = MultiTaskModel().to(device)

# Train multi-task model
history = train_multi_task_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    device=device,
    num_epochs=15
)

# Evaluate multi-task model
results = evaluate_multi_task_model(model, test_loader, device)
```

## Available Models

- **UnifiedMultiTaskModel**: Unified ResNet-50 based model for protest detection, visual attributes, and violence prediction
- **ProtestResNet**: More complex ResNet-50 with custom classifier layers
- **MultiTaskModel**: Joint model for protest detection and violence prediction

## Key Features

### Data Loading
- Automatic image preprocessing and normalization
- Support for various image formats (RGB, grayscale, RGBA)
- Efficient path-based loading for high-resolution visualization
- Binary feature cleaning and validation

### Training
- Early stopping with validation monitoring
- Learning rate scheduling
- Progress tracking with tqdm
- Automatic best model saving

### Evaluation
- Comprehensive metrics (accuracy, precision, recall, F1, AUC)
- Multi-task evaluation support
- Single image prediction capability

### Visualization
- Stylish confusion matrices
- ROC and Precision-Recall curves
- Training history plots
- High-quality misclassification analysis
- Violence level analysis by features
- Correlation heatmaps

## Requirements

- Python 3.7+
- PyTorch 1.9+
- torchvision
- scikit-learn
- matplotlib
- seaborn
- pandas
- numpy
- opencv-python
- imageio
- tqdm
- PIL

## File Organization

When using this library, organize your data as follows:

```
project/
├── data/
│   ├── train/           # Training images
│   ├── test/            # Test images
│   ├── annot_train.txt  # Training annotations
│   └── annot_test.txt   # Test annotations
├── results/             # Model outputs and visualizations
├── library/             # This library
└── your_notebook.ipynb  # Your analysis notebook
```

## Citation

If you use this library in your research, please cite:

```
@misc{protest_detection_library,
  title={Protest Detection and Violence Prediction Library},
  author={Your Name},
  year={2025},
  url={https://github.com/your-repo/protest-detection}
}
```
