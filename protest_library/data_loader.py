"""
Data loading and preprocessing utilities for protest detection.
"""

import os
import imageio
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm
from PIL import Image
import imageio
from tqdm import tqdm
from joblib import Parallel, delayed




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
    annot_test = pd.read_csv(test_csv_path, sep='\t')
    
    # Replace "-" with 0 due to the way the file is written (suppress future warning)
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
    binary_cols = ['protest', 'sign', 'photo', 'fire', 'police', 'children', 'group_20', 'group_100', 'flag', 'night', 'shouting']
    
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
    
    # Replace common string representations with NaN, then convert to numeric (suppress future warning)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*Downcasting.*")
        annot_train['violence'] = annot_train['violence'].replace(['-', '', 'nan', 'NaN', 'null'], np.nan)
        annot_test['violence'] = annot_test['violence'].replace(['-', '', 'nan', 'NaN', 'null'], np.nan)
    
    # Convert to numeric, forcing invalid values to NaN
    annot_train['violence'] = pd.to_numeric(annot_train['violence'], errors='coerce')
    annot_test['violence'] = pd.to_numeric(annot_test['violence'], errors='coerce')
    
    return annot_train, annot_test


def process_image(file, folder, fnames_set, annot_dict):
    """Process single image with optimized operations"""
    try:
        filepath = os.path.join(folder, file)
        protest_value = 0
        violence_value = np.nan
        
        if file in fnames_set:
            row = annot_dict.get(file, {})
            protest_value = int(row.get('protest', 0))
            violence_raw = row.get('violence', np.nan)
            if pd.notna(violence_raw):
                violence_value = float(violence_raw)
        
        # Read and process image
        image = imageio.imread(filepath)
        
        # Handle image formats
        if len(image.shape) == 2:
            image = np.stack([image]*3, axis=-1)
        elif image.shape[2] == 4:
            image = image[..., :3]
        
        # Resize using PIL
        pil_img = Image.fromarray(image)
        pil_img = pil_img.resize((50, 50), Image.BILINEAR)
        model_img = np.array(pil_img, dtype=np.float32) / 255.0
        
        return {
            'image': model_img,
            'protest': protest_value,
            'violence': violence_value,
            'path': filepath,
            'filename': file
        }
    except Exception as e:
        print(f"Error processing {file}: {e}")
        return None

def load_data_with_violence(train_folder, test_folder, annot_train, annot_test, n_jobs=4):
    """Optimized data loader with parallel processing"""
    # Precompute annotation lookups
    train_dict = annot_train.set_index('fname').to_dict(orient='index')
    test_dict = annot_test.set_index('fname').to_dict(orient='index')
    train_fnames = set(annot_train['fname'])
    test_fnames = set(annot_test['fname'])
    
    # Find image files
    train_files = [f for f in os.listdir(train_folder) 
                  if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    test_files = [f for f in os.listdir(test_folder) 
                 if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    # Parallel processing
    print(f"Loading training data from {train_folder}")
    train_results = Parallel(n_jobs=n_jobs)(
        delayed(process_image)(f, train_folder, train_fnames, train_dict) 
        for f in tqdm(train_files, desc="Loading training images")
    )
    
    print(f"Loading test data from {test_folder}")
    test_results = Parallel(n_jobs=n_jobs)(
        delayed(process_image)(f, test_folder, test_fnames, test_dict) 
        for f in tqdm(test_files, desc="Loading test images")
    )
    
    # Filter failed results and unpack
    train_data = [r for r in train_results if r is not None]
    test_data = [r for r in test_results if r is not None]
    
    # Convert to arrays
    def unpack(data):
        return {
            'images': np.array([d['image'] for d in data]),
            'labels': np.array([d['protest'] for d in data], dtype=np.int32),
            'violence': np.array([d['violence'] for d in data], dtype=np.float32),
            'paths': [d['path'] for d in data],
            'filenames': [d['filename'] for d in data]
        }
    
    train_unpack = unpack(train_data)
    test_unpack = unpack(test_data)
    
    return {
        'train_images': train_unpack['images'],
        'train_labels': train_unpack['labels'],
        'train_violence': train_unpack['violence'],
        'train_paths': train_unpack['paths'],
        'test_images': test_unpack['images'],
        'test_labels': test_unpack['labels'],
        'test_violence': test_unpack['violence'],
        'test_paths': test_unpack['paths']
    }