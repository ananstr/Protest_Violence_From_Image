"""
Training utilities for protest detection models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                device, num_epochs=10, early_stopping_patience=4):
    """
    Train a single-task protest detection model.
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on (cuda/cpu)
        num_epochs: Number of epochs to train
        early_stopping_patience: Patience for early stopping
        
    Returns:
        Dictionary containing training history
    """
    best_val_acc = 0.0
    no_improve_epochs = 0
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            inputs = inputs.to(device)
            labels = labels.to(device).view(-1, 1)  # Reshape to match output
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                inputs = inputs.to(device)
                labels = labels.to(device).view(-1, 1)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'results/best_model.pt')
            print(f"Saved new best model with validation accuracy: {val_acc:.4f}")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    return {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_acc': train_accs,
        'val_acc': val_accs
    }


def train_unified_model(model, train_loader, val_loader, optimizer, scheduler, device, 
                       protest_criterion, attributes_criterion, violence_criterion,
                       protest_weight=1.0, attributes_weight=1.0, violence_weight=1.0,
                       num_epochs=10):
    """Train the unified multi-task model."""
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_protest_loss': [], 'val_protest_loss': [],
        'train_attributes_loss': [], 'val_attributes_loss': [],
        'train_violence_loss': [], 'val_violence_loss': [],
        'train_protest_acc': [], 'val_protest_acc': [],
        'train_attributes_acc': [], 'val_attributes_acc': [],
        'train_violence_mse': [], 'val_violence_mse': []
    }
    
    best_val_loss = float('inf')
    early_stopping_patience = 4
    no_improve_epochs = 0
    
    # gradient clipping threshold
    max_grad_norm = 1.0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_losses = {'total': 0, 'protest': 0, 'attributes': 0, 'violence': 0}
        train_metrics = {'protest_correct': 0, 'attributes_correct': 0, 'violence_mse_sum': 0}
        train_totals = {'protest': 0, 'attributes': 0, 'violence': 0}
        
        for batch_idx, (inputs, protest_labels, attribute_labels, violence_labels) in enumerate(tqdm(train_loader, desc="Training")):
            inputs = inputs.to(device)
            protest_labels = protest_labels.to(device).view(-1, 1)  # Reshape to (batch_size, 1)
            attribute_labels = attribute_labels.to(device)  # Already (batch_size, 10)
            violence_labels = violence_labels.to(device).view(-1, 1)  # Reshape to (batch_size, 1)
            
            optimizer.zero_grad()
            
            # Forward pass
            protest_out, attributes_out, violence_out = model(inputs)
            
            # Calculate individual losses
            loss_protest = protest_criterion(protest_out, protest_labels)
            loss_attributes = attributes_criterion(attributes_out, attribute_labels)
            loss_violence = violence_criterion(violence_out, violence_labels)
            
            # Combined loss
            total_loss = (protest_weight * loss_protest + 
                         attributes_weight * loss_attributes + 
                         violence_weight * loss_violence)
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            
            # Change: added gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                max_grad_norm
            )
            
            optimizer.step()
            
            # Accumulate losses
            batch_size = inputs.size(0)
            train_losses['total'] += total_loss.item() * batch_size
            train_losses['protest'] += loss_protest.item() * batch_size
            train_losses['attributes'] += loss_attributes.item() * batch_size
            train_losses['violence'] += loss_violence.item() * batch_size
            
            # Calculate metrics
            protest_pred = (protest_out > 0.5).float()
            attributes_pred = (attributes_out > 0.5).float()
            
            train_metrics['protest_correct'] += (protest_pred == protest_labels).sum().item()
            train_metrics['attributes_correct'] += (attributes_pred == attribute_labels).sum().item()
            train_metrics['violence_mse_sum'] += ((violence_out - violence_labels) ** 2).sum().item()
            
            train_totals['protest'] += protest_labels.numel()
            train_totals['attributes'] += attribute_labels.numel()
            train_totals['violence'] += violence_labels.numel()
        
        # Calculate training metrics
        train_loss = train_losses['total'] / len(train_loader.dataset)
        train_protest_loss = train_losses['protest'] / len(train_loader.dataset)
        train_attributes_loss = train_losses['attributes'] / len(train_loader.dataset)
        train_violence_loss = train_losses['violence'] / len(train_loader.dataset)
        
        train_protest_acc = train_metrics['protest_correct'] / train_totals['protest']
        train_attributes_acc = train_metrics['attributes_correct'] / train_totals['attributes']
        train_violence_mse = train_metrics['violence_mse_sum'] / train_totals['violence']
        
        # Validation phase
        model.eval()
        val_losses = {'total': 0, 'protest': 0, 'attributes': 0, 'violence': 0}
        val_metrics = {'protest_correct': 0, 'attributes_correct': 0, 'violence_mse_sum': 0}
        val_totals = {'protest': 0, 'attributes': 0, 'violence': 0}
        
        with torch.no_grad():
            for inputs, protest_labels, attribute_labels, violence_labels in tqdm(val_loader, desc="Validation"):
                inputs = inputs.to(device)
                protest_labels = protest_labels.to(device).view(-1, 1)  # Reshape to (batch_size, 1)
                attribute_labels = attribute_labels.to(device)  # Already (batch_size, 10)
                violence_labels = violence_labels.to(device).view(-1, 1)  # Reshape to (batch_size, 1)
                
                # Forward pass
                protest_out, attributes_out, violence_out = model(inputs)
                
                # Calculate losses
                loss_protest = protest_criterion(protest_out, protest_labels)
                loss_attributes = attributes_criterion(attributes_out, attribute_labels)
                loss_violence = violence_criterion(violence_out, violence_labels)
                
                total_loss = (protest_weight * loss_protest + 
                             attributes_weight * loss_attributes + 
                             violence_weight * loss_violence)
                
                # Accumulate losses
                batch_size = inputs.size(0)
                val_losses['total'] += total_loss.item() * batch_size
                val_losses['protest'] += loss_protest.item() * batch_size
                val_losses['attributes'] += loss_attributes.item() * batch_size
                val_losses['violence'] += loss_violence.item() * batch_size
                
                # Calculate metrics
                protest_pred = (protest_out > 0.5).float()
                attributes_pred = (attributes_out > 0.5).float()
                
                val_metrics['protest_correct'] += (protest_pred == protest_labels).sum().item()
                val_metrics['attributes_correct'] += (attributes_pred == attribute_labels).sum().item()
                val_metrics['violence_mse_sum'] += ((violence_out - violence_labels) ** 2).sum().item()
                
                val_totals['protest'] += protest_labels.numel()
                val_totals['attributes'] += attribute_labels.numel()
                val_totals['violence'] += violence_labels.numel()
        
        # Calculate validation metrics
        val_loss = val_losses['total'] / len(val_loader.dataset)
        val_protest_loss = val_losses['protest'] / len(val_loader.dataset)
        val_attributes_loss = val_losses['attributes'] / len(val_loader.dataset)
        val_violence_loss = val_losses['violence'] / len(val_loader.dataset)
        
        val_protest_acc = val_metrics['protest_correct'] / val_totals['protest']
        val_attributes_acc = val_metrics['attributes_correct'] / val_totals['attributes']
        val_violence_mse = val_metrics['violence_mse_sum'] / val_totals['violence']
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_protest_loss'].append(train_protest_loss)
        history['val_protest_loss'].append(val_protest_loss)
        history['train_attributes_loss'].append(train_attributes_loss)
        history['val_attributes_loss'].append(val_attributes_loss)
        history['train_violence_loss'].append(train_violence_loss)
        history['val_violence_loss'].append(val_violence_loss)
        history['train_protest_acc'].append(train_protest_acc)
        history['val_protest_acc'].append(val_protest_acc)
        history['train_attributes_acc'].append(train_attributes_acc)
        history['val_attributes_acc'].append(val_attributes_acc)
        history['train_violence_mse'].append(train_violence_mse)
        history['val_violence_mse'].append(val_violence_mse)
        
        # Print epoch results
        print(f"Loss - Train: {train_loss:.4f}, Val: {val_loss:.4f}")
        print(f"Protest Acc - Train: {train_protest_acc:.4f}, Val: {val_protest_acc:.4f}")
        print(f"Attributes Acc - Train: {train_attributes_acc:.4f}, Val: {val_attributes_acc:.4f}")
        print(f"Violence MSE - Train: {train_violence_mse:.4f}, Val: {val_violence_mse:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping and model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'results/best_unified_model.pt')
            print(f"Saved new best model (Val Loss: {val_loss:.4f})")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    return history
