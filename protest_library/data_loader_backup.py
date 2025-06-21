"""
Data loading and preprocessing utilities for protest detection.
"""

import os
import cv2
import imageio
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
from time import ctime


def load_annotations(train_csv_path, test_csv_path):
    """
    Load and preprocess annotation files.
    
    Args:
        train_csv_path: Path to training annotation file
        test_csv_path: Path to test annotation file
        
    Returns:
        Tuple of (train_annotations, test_annotations)
    """
    annot_train = pd.read_csv(train_csv_path, sep='\t')
    annot_test = pd.read_csv(test_csv_path, sep='\t')    # Replace "-" with 0 due to the way the file is written (suppress future warning)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*Downcasting.*")
        annot_train = annot_train.replace("-", 0)
        annot_test = annot_test.replace("-", 0)
    
    return annot_train, annot_test


def clean_binary_columns(annot_train, annot_test):
    """
    Clean and standardize binary columns in annotation data.
    
    Args:
        annot_train: Training annotations DataFrame
        annot_test: Test annotations DataFrame
        
    Returns:
        Tuple of cleaned (annot_train, annot_test)
    """
    binary_cols = ['sign', 'fire', 'police', 'children', 'group_20', 'group_100']
      print("Cleaning binary columns...")
    for col in binary_cols:
        if col in annot_train.columns:
            # Convert string representations to numeric (suppress future warning)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning, message=".*Downcasting.*")
                annot_train[col] = annot_train[col].replace({'0': 0, '1': 1, '-': 0})
                annot_test[col] = annot_test[col].replace({'0': 0, '1': 1, '-': 0})
            
            # Convert to integer
            annot_train[col] = pd.to_numeric(annot_train[col], errors='coerce').fillna(0).astype(int)
            annot_test[col] = pd.to_numeric(annot_test[col], errors='coerce').fillna(0).astype(int)
    
    return annot_train, annot_test


def clean_violence_data(annot_train, annot_test):
    """
    Clean violence data columns.
    
    Args:
        annot_train: Training annotations DataFrame
        annot_test: Test annotations DataFrame
        
    Returns:
        Tuple of cleaned (annot_train, annot_test)
    """
    print("Cleaning violence data...")
      # Replace common string representations with NaN, then convert to numeric (suppress future warning)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*Downcasting.*")
        annot_train['violence'] = annot_train['violence'].replace(['-', '', 'nan', 'NaN', 'null'], np.nan)
        annot_test['violence'] = annot_test['violence'].replace(['-', '', 'nan', 'NaN', 'null'], np.nan)
    
    # Convert to numeric, forcing invalid values to NaN
    annot_train['violence'] = pd.to_numeric(annot_train['violence'], errors='coerce')
    annot_test['violence'] = pd.to_numeric(annot_test['violence'], errors='coerce')
    
    return annot_train, annot_test


def load_protest_data(train_folder, test_folder, annot_train, annot_test):
    """
    Load protest detection data with image paths stored for visualization.
    
    Args:
        train_folder: Path to training images folder
        test_folder: Path to test images folder
        annot_train: Training annotations DataFrame
        annot_test: Test annotations DataFrame
        
    Returns:
        Dictionary containing loaded data
    """
    print(f"Starting data loading: {ctime()}")
    
    train_images = []          # Low-resolution for model training
    train_labels = []
    train_image_paths = []     # Store file paths for visualization
    train_filenames = []       # Store filenames for reference
    test_images = []
    test_labels = []
    test_image_paths = []      # Store file paths for visualization
    test_filenames = []

    # Load training data
    print(f"Loading training data from {train_folder}")
    for file in tqdm(os.listdir(train_folder), desc="Loading training images"):
        if file.endswith(('.jpg', '.png', '.jpeg')):
            try:
                # Get the protest label for this image
                protest_value = 0
                if file in annot_train['fname'].values:
                    protest_value = annot_train[annot_train.fname == file]['protest'].iloc[0]
                
                # Store the full path for later visualization
                filepath = os.path.join(train_folder, file)
                train_image_paths.append(filepath)
                
                # Read image for model training
                image = imageio.imread(filepath)
                
                # Handle grayscale images (convert to RGB)
                if len(image.shape) == 2:
                    image = np.stack([image, image, image], axis=2)
                elif image.shape[2] == 4:  # Handle RGBA images
                    image = image[:, :, :3]
                
                # Resize image to 50x50 (for model training only)
                model_img = cv2.resize(image, (50, 50))
                model_img = model_img / 255.0  # Normalize to [0,1]
                
                # Add to datasets
                train_images.append(model_img)
                train_labels.append(protest_value)
                train_filenames.append(file)
                
            except Exception as e:
                print(f"Error processing {file}: {e}")

    # Load test data
    print(f"Loading test data from {test_folder}")
    for file in tqdm(os.listdir(test_folder), desc="Loading test images"):
        if file.endswith(('.jpg', '.png', '.jpeg')):
            try:
                # Get the protest label for this image if available
                protest_value = 0
                if file in annot_test['fname'].values:
                    protest_value = annot_test[annot_test.fname == file]['protest'].iloc[0]
                
                # Store the full path for later visualization
                filepath = os.path.join(test_folder, file)
                test_image_paths.append(filepath)
                
                # Read image for model training
                image = imageio.imread(filepath)
                
                # Handle grayscale images (convert to RGB)
                if len(image.shape) == 2:
                    image = np.stack([image, image, image], axis=2)
                elif image.shape[2] == 4:  # Handle RGBA images
                    image = image[:, :, :3]
                
                # Resize image to 50x50 (for model training)
                model_img = cv2.resize(image, (50, 50))
                model_img = model_img / 255.0  # Normalize to [0,1]
                
                # Add to datasets
                test_images.append(model_img)
                test_labels.append(protest_value)
                test_filenames.append(file)
                
            except Exception as e:
                print(f"Error processing {file}: {e}")

    # Convert lists to numpy arrays
    data = {
        'train_images': np.array(train_images),
        'train_labels': np.array(train_labels),
        'train_paths': train_image_paths,
        'train_filenames': train_filenames,
        'test_images': np.array(test_images),
        'test_labels': np.array(test_labels),
        'test_paths': test_image_paths,
        'test_filenames': test_filenames
    }

    print(f"Data loading completed: {ctime()}")
    print(f"Training data: {data['train_images'].shape[0]} images")
    print(f"Test data: {data['test_images'].shape[0]} images")
    
    return data


def load_data_with_violence(train_folder, test_folder, annot_train, annot_test):
    """
    Load data with violence scores for multi-task learning.
    
    Args:
        train_folder: Path to training images folder
        test_folder: Path to test images folder
        annot_train: Training annotations DataFrame
        annot_test: Test annotations DataFrame
        
    Returns:
        Dictionary containing loaded data with violence scores
    """
    train_images = []
    train_labels = []
    train_violence_scores = []
    train_image_paths = []
    test_images = []
    test_labels = []
    test_violence_scores = []
    test_image_paths = []
    
    # Load training data
    print(f"Loading training data from {train_folder}")
    for file in tqdm(os.listdir(train_folder), desc="Loading training images"):
        if file.endswith(('.jpg', '.png', '.jpeg')):
            try:
                # Get labels for this image
                protest_value = 0
                violence_value = np.nan  # Use NaN as default
                
                if file in annot_train['fname'].values:
                    row = annot_train[annot_train.fname == file]
                    protest_value = int(row['protest'].iloc[0])
                    
                    # Get violence value - it should now be numeric or NaN
                    violence_raw = row['violence'].iloc[0]
                    if pd.notna(violence_raw):
                        violence_value = float(violence_raw)
                    else:
                        violence_value = np.nan
                
                # Store the full path
                filepath = os.path.join(train_folder, file)
                train_image_paths.append(filepath)
                
                # Read image for model training
                image = imageio.imread(filepath)
                
                # Handle grayscale/RGBA images
                if len(image.shape) == 2:
                    image = np.stack([image, image, image], axis=2)
                elif image.shape[2] == 4:
                    image = image[:, :, :3]
                
                # Resize for model
                model_img = cv2.resize(image, (50, 50))
                model_img = model_img / 255.0
                
                train_images.append(model_img)
                train_labels.append(protest_value)
                train_violence_scores.append(violence_value)
                
            except Exception as e:
                print(f"Error processing {file}: {e}")
    
    # Load test data
    print(f"Loading test data from {test_folder}")
    for file in tqdm(os.listdir(test_folder), desc="Loading test images"):
        if file.endswith(('.jpg', '.png', '.jpeg')):
            try:
                # Get labels
                protest_value = 0
                violence_value = np.nan
                
                if file in annot_test['fname'].values:
                    row = annot_test[annot_test.fname == file]
                    protest_value = int(row['protest'].iloc[0])
                    
                    # Get violence value
                    violence_raw = row['violence'].iloc[0]
                    if pd.notna(violence_raw):
                        violence_value = float(violence_raw)
                    else:
                        violence_value = np.nan
                
                # Store the full path
                filepath = os.path.join(test_folder, file)
                test_image_paths.append(filepath)
                
                # Read and process image
                image = imageio.imread(filepath)
                
                if len(image.shape) == 2:
                    image = np.stack([image, image, image], axis=2)
                elif image.shape[2] == 4:
                    image = image[:, :, :3]
                
                model_img = cv2.resize(image, (50, 50))
                model_img = model_img / 255.0
                
                test_images.append(model_img)
                test_labels.append(protest_value)
                test_violence_scores.append(violence_value)
                
            except Exception as e:
                print(f"Error processing {file}: {e}")
    
    return {
        'train_images': np.array(train_images),
        'train_labels': np.array(train_labels, dtype=np.int32),
        'train_violence': np.array(train_violence_scores, dtype=np.float32),
        'train_paths': train_image_paths,
        'test_images': np.array(test_images),
        'test_labels': np.array(test_labels, dtype=np.int32),
        'test_violence': np.array(test_violence_scores, dtype=np.float32),
        'test_paths': test_image_paths
    }
