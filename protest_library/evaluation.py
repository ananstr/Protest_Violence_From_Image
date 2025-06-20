"""
Evaluation utilities for protest detection models.
"""

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report


def evaluate_model(model, test_loader, device):
    """
    Evaluate a single-task protest detection model.
    
    Args:
        model: Trained PyTorch model
        test_loader: Test data loader
        device: Device to run evaluation on
        
    Returns:
        Dictionary containing evaluation results
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_scores = []
    
    test_loss = 0.0
    correct = 0
    total = 0
    
    criterion = torch.nn.BCELoss()
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating test data"):
            inputs = inputs.to(device)
            labels = labels.to(device).view(-1, 1)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Calculate statistics
            test_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store predictions and labels for detailed metrics
            all_preds.extend(predicted.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
            all_scores.extend(outputs.cpu().numpy().flatten())
    
    # Calculate overall metrics
    test_loss = test_loss / len(test_loader.dataset)
    test_accuracy = correct / total
    
    results = {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'predictions': all_preds,
        'labels': all_labels,
        'scores': all_scores,
        'classification_report': classification_report(all_labels, all_preds)
    }
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("\nClassification Report:")
    print(results['classification_report'])
    
    return results


def evaluate_multi_task_model(model, test_loader, device):
    """
    Evaluate a multi-task model for protest detection and violence prediction.
    
    Args:
        model: Trained multi-task PyTorch model
        test_loader: Test data loader
        device: Device to run evaluation on
        
    Returns:
        Dictionary containing evaluation results
    """
    model.eval()
    
    # Loss functions
    bce_loss = torch.nn.BCELoss()
    mse_loss = torch.nn.MSELoss()
    
    test_loss = 0.0
    test_protest_loss = 0.0
    test_violence_loss = 0.0
    protest_correct = 0
    protest_total = 0
    violence_mse_sum = 0.0
    
    all_protest_preds = []
    all_protest_labels = []
    all_violence_preds = []
    all_violence_labels = []
    
    protest_weight = 1.0
    violence_weight = 1.0
    
    with torch.no_grad():
        for inputs, (protest_labels, violence_labels) in tqdm(test_loader, desc="Evaluating test data"):
            inputs = inputs.to(device)
            protest_labels = protest_labels.to(device).view(-1, 1)
            violence_labels = violence_labels.to(device).view(-1, 1)
            
            # Forward pass
            protest_outputs, violence_outputs = model(inputs)
            
            # Calculate individual losses
            p_loss = bce_loss(protest_outputs, protest_labels)
            v_loss = mse_loss(violence_outputs, violence_labels)
            
            # Combined loss
            loss = protest_weight * p_loss + violence_weight * v_loss
            
            # Statistics
            batch_size = inputs.size(0)
            test_loss += loss.item() * batch_size
            test_protest_loss += p_loss.item() * batch_size
            test_violence_loss += v_loss.item() * batch_size
            
            protest_predicted = (protest_outputs > 0.5).float()
            protest_total += protest_labels.size(0)
            protest_correct += (protest_predicted == protest_labels).sum().item()
            
            violence_mse_sum += ((violence_outputs - violence_labels) ** 2).sum().item()
            
            # Store predictions and labels for detailed metrics
            all_protest_preds.extend(protest_predicted.cpu().numpy().flatten())
            all_protest_labels.extend(protest_labels.cpu().numpy().flatten())
            all_violence_preds.extend(violence_outputs.cpu().numpy().flatten())
            all_violence_labels.extend(violence_labels.cpu().numpy().flatten())
    
    # Calculate overall metrics
    test_loss = test_loss / len(test_loader.dataset)
    test_protest_loss = test_protest_loss / len(test_loader.dataset)
    test_violence_loss = test_violence_loss / len(test_loader.dataset)
    test_protest_acc = protest_correct / protest_total
    test_violence_mse = violence_mse_sum / len(test_loader.dataset)
    test_violence_rmse = np.sqrt(test_violence_mse)
    test_violence_r2 = 1 - (np.sum((np.array(all_violence_labels) - np.array(all_violence_preds))**2) / 
                            np.sum((np.array(all_violence_labels) - np.mean(all_violence_labels))**2))
    
    results = {
        'test_loss': test_loss,
        'test_protest_loss': test_protest_loss,
        'test_protest_accuracy': test_protest_acc,
        'test_violence_loss': test_violence_loss,
        'test_violence_rmse': test_violence_rmse,
        'test_violence_r2': test_violence_r2,
        'protest_predictions': all_protest_preds,
        'protest_labels': all_protest_labels,
        'violence_predictions': all_violence_preds,
        'violence_labels': all_violence_labels
    }
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Protest Loss: {test_protest_loss:.4f}")
    print(f"Test Protest Accuracy: {test_protest_acc:.4f}")
    print(f"Test Violence Loss (MSE): {test_violence_loss:.4f}")
    print(f"Test Violence RMSE: {test_violence_rmse:.4f}")
    print(f"Test Violence R²: {test_violence_r2:.4f}")
    
    return results


def predict_single_image(model, image_path, transform, device):
    """
    Make prediction on a single image.
    
    Args:
        model: Trained model
        image_path: Path to image file
        transform: Preprocessing transforms
        device: Device to run prediction on
        
    Returns:
        Dictionary containing prediction results
    """
    import imageio
    from PIL import Image
    import torch
    
    # Load and preprocess the image
    image = imageio.imread(image_path)
    
    # Handle grayscale/RGBA images
    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=2)
    elif image.shape[2] == 4:
        image = image[:, :, :3]
    
    # Apply transformations
    image_pil = Image.fromarray(image)
    image_tensor = transform(image_pil).unsqueeze(0).to(device)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        if hasattr(model, 'protest_classifier'):  # Multi-task model
            protest_output, violence_output = model(image_tensor)
            protest_prob = protest_output.item()
            violence_score = violence_output.item()
            protest_pred = 1 if protest_prob > 0.5 else 0
            
            return {
                'protest_prediction': protest_pred,
                'protest_probability': protest_prob,
                'violence_score': violence_score,
                'image': image
            }
        else:  # Single-task model
            output = model(image_tensor)
            protest_prob = output.item()
            protest_pred = 1 if protest_prob > 0.5 else 0
            
            return {
                'protest_prediction': protest_pred,
                'protest_probability': protest_prob,
                'image': image
            }
