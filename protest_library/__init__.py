"""
Protest Detection and Violence Prediction Library

Modules:
- datasets: Dataset classes for PyTorch data loading
- models: Neural network architectures
- data_loader: Data loading and preprocessing utilities
- training: Training functions for single and multi-task models
- evaluation: Model evaluation utilities
- visualization: Plotting and analysis functions
- utils: General utility functions
"""

from .datasets import ProtestDataset, UnifiedMultiTaskDataset
from .models import UnifiedMultiTaskModel
from .data_loader import (
    load_annotations, 
    clean_binary_columns, 
    clean_violence_data,
    load_data_with_violence
)
from .training import train_unified_model
from .evaluation import evaluate_model, evaluate_multi_task_model, predict_single_image
from .visualization import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_pr_curve,
    plot_misclassifications,
    plot_training_history,
    analyze_violence_by_features,
    create_correlation_analysis,
    create_attention_analysis
)
from .utils import (
    get_device,
    get_transforms,
    augment_data,
    setup_data_loaders,
    save_model,
    load_model,
    print_class_distribution,
    filter_violence_data,
    create_performance_summary
)

__version__ = "1.0.0"
__author__ = "Anastasia Chernavskaia: Protest Detection"

__all__ = [
    # Dataset classes
    'ProtestDataset',
    'UnifiedMultiTaskDataset',
    # Model classes
    'UnifiedMultiTaskModel',
    
    # Data loading functions
    'load_annotations',
    'clean_binary_columns',
    'clean_violence_data',
    'load_data_with_violence',
    
    # Training functions
    'train_unified_model',
    
    # Evaluation functions
    'evaluate_model',
    'evaluate_multi_task_model',
    'predict_single_image',
    
    # Visualization functions
    'plot_confusion_matrix',
    'plot_roc_curve',
    'plot_pr_curve',
    'plot_misclassifications',
    'plot_training_history',
    'analyze_violence_by_features',
    'create_correlation_analysis',
    'create_attention_analysis',
    
    # Utility functions
    'get_device',
    'get_transforms',
    'augment_data',
    'setup_data_loaders',
    'save_model',
    'load_model',
    'print_class_distribution',
    'filter_violence_data',
    'create_performance_summary'
]