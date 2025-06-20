"""
Neural network models for protest detection and violence prediction.
"""

import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights


class UnifiedMultiTaskModel(nn.Module):
    """Unified multi-task model for comprehensive protest analysis.
    
    Single ResNet backbone with three task-specific heads:
    1. Binary protest classification (1 output)
    2. Visual attributes classification (10 outputs) 
    3. Violence regression (1 output)
    Total: 12 outputs
    """
    
    def __init__(self, dropout_rate=0.3):
        """
        Args:
            dropout_rate: Dropout rate for regularization
        """
        super(UnifiedMultiTaskModel, self).__init__()
        
        # Shared ResNet50 backbone
        self.backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        num_features = self.backbone.fc.in_features
        
        # Remove original fully connected layer
        self.backbone.fc = nn.Identity()
        
        # Shared feature extractor after backbone
        self.shared_features = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(1024)
        )
        
        # Task 1: Binary protest classification (1 output)
        self.protest_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
        # Task 2: Visual attributes classification (10 outputs)
        # Each output is a binary classification for: 
        # 'sign', 'photo', 'fire', 'police', 'children', 'group_20', 'group_100', 'flag', 'night', 'shouting'
        self.attributes_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 10),
            nn.Sigmoid()  # Each attribute is binary
        )
        
        # Task 3: Violence regression (1 output)
        self.violence_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 1),
            nn.Sigmoid()  # Violence score between 0-1
        )
        
        # Attribute names for reference
        self.attribute_names = ['sign', 'photo', 'fire', 'police', 'children', 
                               'group_20', 'group_100', 'flag', 'night', 'shouting']

    def forward(self, x):
        """
        Forward pass through all task heads.
        
        Returns:
            tuple: (protest_output, attributes_output, violence_output)
                - protest_output: (batch_size, 1) - binary classification
                - attributes_output: (batch_size, 10) - multi-label classification  
                - violence_output: (batch_size, 1) - regression
        """
        # Shared feature extraction
        x = self.backbone(x)
        x = self.shared_features(x)
        
        # Task-specific predictions
        protest_out = self.protest_head(x)           # (batch_size, 1)
        attributes_out = self.attributes_head(x)     # (batch_size, 10)
        violence_out = self.violence_head(x)         # (batch_size, 1)
        
        return protest_out, attributes_out, violence_out
    
    def get_attribute_predictions(self, x):
        """
        Get attribute predictions with named outputs.
        
        Returns:
            dict: Mapping of attribute names to predictions
        """
        _, attributes_out, _ = self.forward(x)
        
        # Convert to dictionary with attribute names
        predictions = {}
        for i, attr_name in enumerate(self.attribute_names):
            predictions[attr_name] = attributes_out[:, i:i+1]
        
        return predictions
