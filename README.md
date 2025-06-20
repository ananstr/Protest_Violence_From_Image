# Protest Detection and Violence Prediction

This notebook uses custom `protest_library` package.

- Class definitions for models are located in `models.py`
- Data loading functions are located in `data_loader.py`
- Training functions are located in `training.py`
- Evaluation functions are located in `evaluation.py`
- Visualization functions are located in `visualization.py`
- Utility functions are located in `utils.py`

## Library Structure:
```
protest_library/
├── __init__.py          # Imports
├── datasets.py          # ProtestDataset, MultiTaskDataset
├── models.py            # UnifiedMultiTaskModel for all tasks
├── data_loader.py       # load_annotations, load_protest_data, etc.
├── training.py          # train_model, train_multi_task_model
├── evaluation.py        # evaluate_model, evaluate_multi_task_model
├── visualization.py     # plot_stylish_*, analyze_violence_by_features
├── utils.py             # get_device, get_transforms, setup_data_loaders
└── README.md            # Documentation
```

## Approach:

This project uses a multi-task learning (MTL) approach using a single Convolutional Neural Network (CNN) backbone.

Related tasks (like recognizing protest, violence, and visual attributes in an image) rely on extracting similar underlying visual features. Instead of training separate models for each task, a multi-task model trains one network that learns these shared features efficiently.

### 1. Model Architecture
**Input:** The model takes a full image as input.
**Shared Feature Extractor (Backbone):**
- It's based on a 50-layer ResNet. ResNet is a powerful CNN architecture known for its deep convolutional layers, which are excellent at extracting hierarchical visual features (from simple edges and textures to complex objects and patterns).
- It includes batch normalization (to stabilize training) and ReLU layers (for non-linearity), standard components of modern CNNs.
- Crucially, the features computed through convolutional layers are all shared by linear layers for multiple classification tasks. This means the main convolutional part of the ResNet acts as a single feature extractor, learning representations that are useful for all the downstream tasks.
- Multiple Output Heads (Linear Layers for Specific Tasks): After the shared ResNet backbone extracts these features, the network branches into several separate linear (fully connected) layers, each responsible for a specific prediction task:
- Binary Image Category (Protest or Non-Protest): 1 output neuron. This is a binary classification task.
- Visual Attributes: 10 output neurons. Each neuron likely corresponds to the presence/absence of a specific attribute (e.g., "crowd," "sign," "police," etc.), making this 10 separate binary classifications.
- Perceived Violence: 1 output neuron. This is a regression task, predicting a continuous score for violence.
Total Outputs: In total, this single model outputs 1+10+1=12 prediction scores.

### 2. Training Strategy: Joint Learning
- Joint Training: I jointly train the model such that all parameters for 3 different tasks – protest classification, violence score and visual attribute classification – are updated jointly.
During each training step, an image is passed through the model.
Predictions are generated for all 12 outputs simultaneously.
Losses are calculated for each of the tasks separately based on their respective ground truth labels.
- Loss Functions:
Binary Cross-Entropy (BCE) Loss: Used for the binary variables:
Protest classification (1 output).
Visual attributes classification (10 outputs, likely 10 independent BCE losses or one BCEWithLogitsLoss over all 10 outputs).
Mean Squared Error (MSE) Loss: Used for the regression dimensions:
Perceived violence (1 output).
Image sentiment (4 outputs).
- Loss Combination: These individual loss values (BCE for protest, BCE for attributes, MSE for violence) are summed up to form a single total loss.
- Backpropagation: This total loss is then used to backpropagate gradients through all the linear output layers and the shared 50-layer ResNet backbone. This means the features learned by the ResNet are optimized to be good for all tasks simultaneously.

### How it "Works" (Conceptual Flow):
An image enters the ResNet-50 backbone.
The ResNet extracts a rich, high-level feature representation of the image.
This single feature representation is then fed into multiple specialized linear layers.
Each linear layer makes a prediction for its specific task (protest, attributes, violence, sentiment).
Based on these predictions and the true labels, separate loss values are calculated for each task (using BCE or MSE as appropriate).
All these individual losses are combined.
The combined loss guides the gradient descent process, updating the weights throughout the entire network – from the shared convolutional layers all the way to the individual output layers.
This joint training allows the model to leverage common visual cues across tasks. For example, learning what constitutes a "protest" image (for binary classification) might help in recognizing "crowds" (a visual attribute) or detecting "anger" (a sentiment dimension), as the shared ResNet learns features relevant to all these aspects.