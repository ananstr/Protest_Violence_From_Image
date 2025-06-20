"""
Visualization utilities for protest detection analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import imageio
import cv2
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from matplotlib.colors import LinearSegmentedColormap


# Custom color palette
colors = {
    'primary': '#3498db',    # Blue
    'secondary': "#fd57cb",  # Red
    'tertiary': "#fdad2c",   # Green
    'background': '#f9f9f9', # Light gray
    'text': "#7d34c2"        # Dark blue/gray
}


def plot_stylish_confusion_matrix(all_labels, all_preds, save_path='confusion_matrix.png'):
    """Plot a stylish confusion matrix."""
    # Set the visual style
    plt.style.use('default')
    sns.set_palette("deep")
    sns.set_context("notebook", font_scale=1.2)
    
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Create a custom colormap (blue to white)
    cmap = LinearSegmentedColormap.from_list('blue_cmap', ['white', colors['primary']], N=100)
    
    plt.figure(figsize=(10, 8))
    ax = plt.subplot()
    
    # Plot heatmap with custom style
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, linewidths=1, linecolor='white',
                cbar_kws={"shrink": 0.8}, square=True, annot_kws={"size": 16})
    
    # Configure appearance
    ax.set_xlabel('Predicted Label', fontsize=14, color=colors['text'])
    ax.set_ylabel('True Label', fontsize=14, color=colors['text'])
    ax.set_title('Confusion Matrix', fontsize=18, color=colors['text'], pad=20)
    
    # Set tick labels
    classes = ['Non-Protest', 'Protest']
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes, rotation=0)
    
    # Add overall accuracy
    acc = np.sum(np.diag(cm)) / np.sum(cm)
    plt.figtext(0.5, 0.01, f'Overall Accuracy: {acc:.2%}', 
                ha='center', fontsize=14, color=colors['text'])
    
    # Tight layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()


def plot_stylish_roc_curve(all_labels, all_scores, save_path='roc_curve.png'):
    """Plot a stylish ROC curve."""
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    roc_auc = auc(fpr, tpr)
    
    fig = plt.figure(figsize=(10, 8))
    
    # Create gradient background
    ax = plt.subplot()
    ax.set_facecolor(colors['background'])
    
    # Plot ROC curve with shadow for depth
    plt.plot(fpr, tpr, color=colors['primary'], lw=3, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color=colors['secondary'], lw=2, linestyle='--', alpha=0.7)
    
    # Add shaded area under curve
    plt.fill_between(fpr, tpr, alpha=0.2, color=colors['primary'])
    
    # Add grid with lower opacity
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Customize appearance
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14, color=colors['text'])
    plt.ylabel('True Positive Rate', fontsize=14, color=colors['text'])
    plt.title('Receiver Operating Characteristic (ROC) Curve', 
              fontsize=18, color=colors['text'], pad=20)
    
    # Add annotation for perfect classifier
    plt.annotate('Perfect Classifier', xy=(0.02, 0.98), xytext=(0.2, 0.8),
                arrowprops=dict(facecolor=colors['secondary'], shrink=0.05),
                fontsize=12, color=colors['text'])
    
    # Customize legend
    plt.legend(loc="lower right", frameon=True, facecolor='white', framealpha=0.9, fontsize=12)
    
    # Add watermark
    fig.text(0.95, 0.05, 'Protest Detection Model', 
             fontsize=12, color=colors['text'], ha='right', 
             style='italic', alpha=0.7)
    
    # Save and show
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()


def plot_stylish_pr_curve(all_labels, all_scores, save_path='precision_recall_curve.png'):
    """Plot a stylish precision-recall curve."""
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(all_labels, all_scores)
    
    # Calculate F1 score and optimal threshold
    f1_scores = 2 * recall[:-1] * precision[:-1] / (recall[:-1] + precision[:-1] + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    best_f1 = f1_scores[optimal_idx]
    
    plt.figure(figsize=(10, 8))
    ax = plt.subplot()
    ax.set_facecolor(colors['background'])
    
    # Plot precision-recall curve
    plt.plot(recall, precision, color=colors['primary'], lw=3)
    
    # Add shaded area
    plt.fill_between(recall, precision, alpha=0.2, color=colors['primary'])
    
    # Mark optimal threshold point
    plt.plot(recall[optimal_idx], precision[optimal_idx], 'o', 
             markersize=12, markerfacecolor=colors['tertiary'], 
             markeredgecolor='white', markeredgewidth=2)
    
    # Add annotation for optimal threshold
    plt.annotate(f'Optimal threshold: {optimal_threshold:.3f}\nF1: {best_f1:.3f}',
                xy=(recall[optimal_idx], precision[optimal_idx]),
                xytext=(recall[optimal_idx]-0.2, precision[optimal_idx]-0.2),
                arrowprops=dict(facecolor=colors['text'], shrink=0.05, width=2),
                fontsize=12, color=colors['text'], 
                bbox=dict(boxstyle="round,pad=0.5", fc='white', alpha=0.8))
    
    # Add grid with lower opacity
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Customize appearance
    plt.xlabel('Recall', fontsize=14, color=colors['text'])
    plt.ylabel('Precision', fontsize=14, color=colors['text'])
    plt.title('Precision-Recall Curve', fontsize=18, color=colors['text'], pad=20)
    
    # Add baseline
    no_skill = len(np.where(np.array(all_labels) == 1)[0]) / len(all_labels)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', color=colors['secondary'], 
             alpha=0.7, label=f'Baseline: {no_skill:.3f}')
    
    # Customize legend
    plt.legend(loc="lower left", frameon=True, facecolor='white', framealpha=0.9, fontsize=12)
    
    # Save and show
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    return optimal_threshold


def plot_stylish_misclassifications(image_paths, all_labels, all_preds, all_scores,  
                                   save_path='misclassifications.png'):
    """Plot misclassified examples by loading high-resolution images on demand."""
    # Find misclassified examples
    misclassified_indices = np.where(np.array(all_preds) != np.array(all_labels))[0]
    
    # Select up to 9 examples (or fewer if less available)
    num_examples = min(9, len(misclassified_indices))
    
    if num_examples == 0:
        print("No misclassifications found.")
        return
    
    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    fig.patch.set_facecolor(colors['background'])
    
    # Add title to the figure
    fig.suptitle('Misclassified Examples', fontsize=24, color=colors['text'], y=0.95)
    
    # Flatten axes for easy iteration
    axes = axes.flatten()
    
    for i in range(9):
        # Turn off all axes first
        axes[i].axis('off')
        
        # Only plot if we have a misclassified example
        if i < num_examples:
            idx = misclassified_indices[i]
            
            try:
                # Load high-quality image on demand
                path = image_paths[idx]
                img = imageio.imread(path)
                
                # Process image for display
                if len(img.shape) == 2:
                    img = np.stack([img, img, img], axis=2)
                elif img.shape[2] == 4:
                    img = img[:,:,:3]
                
                # Resize to a good display size with high-quality interpolation
                img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LANCZOS4)
                img = img / 255.0  # Normalize to [0,1]
                
                # Display the image
                axes[i].imshow(img)
                
                # Add styled border based on classification
                for spine in axes[i].spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(5)
                    spine.set_edgecolor(colors['secondary'])  # Red border for misclassification
                
                # Determine labels
                true_label = 'Protest' if all_labels[idx] == 1 else 'Non-protest'
                pred_label = 'Protest' if all_preds[idx] == 1 else 'Non-protest'
                confidence = all_scores[idx] if pred_label == 'Protest' else 1 - all_scores[idx]
                
                # Add styled text box with classification details
                text = f"True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f}"
                axes[i].text(0.5, -0.15, text, transform=axes[i].transAxes,
                            fontsize=12, color=colors['text'], ha='center',
                            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5',
                                     edgecolor=colors['primary'], linewidth=1.5))
                                     
            except Exception as e:
                print(f"Error loading image {path}: {e}")
                axes[i].text(0.5, 0.5, "Error loading image", 
                             ha='center', va='center', color='red')
    
    # Adjust spacing
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save and show
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=colors['background'])
    plt.show()


def plot_training_history(history, save_path='training_history.png'):
    """Plot training history with loss and accuracy curves."""
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Model Accuracy')
    
    # Mark the best model
    best_epoch = np.argmax(history['val_acc'])
    plt.plot(best_epoch, history['val_acc'][best_epoch], 'rx', markersize=10)
    plt.annotate(f"Best: {history['val_acc'][best_epoch]:.4f}", 
                 (float(best_epoch), float(history['val_acc'][best_epoch])),
                 xytext=(float(best_epoch+0.5), float(history['val_acc'][best_epoch])),
                 fontsize=9)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Model Loss')
    
    # Mark the best model
    best_epoch_loss = np.argmin(history['val_loss'])
    plt.plot(best_epoch_loss, history['val_loss'][best_epoch_loss], 'rx', markersize=10)
    plt.annotate(f"Best: {history['val_loss'][best_epoch_loss]:.4f}", 
                 (float(best_epoch_loss), float(history['val_loss'][best_epoch_loss])),
                 xytext=(float(best_epoch_loss+0.5), float(history['val_loss'][best_epoch_loss])),
                 fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def analyze_violence_by_features(annot_train):
    """Analyze violence levels by binary features."""
    binary_cols = ['sign', 'fire', 'police', 'children', 'group_20', 'group_100']
    
    # Check which binary columns actually exist and have data
    available_cols = []
    for col in binary_cols:
        if col in annot_train.columns:
            # Check if we have both 0 and 1 values
            unique_vals = annot_train[col].unique()
            if len(unique_vals) > 1 and (0 in unique_vals or 1 in unique_vals):
                available_cols.append(col)
    
    if not available_cols:
        print("No binary feature columns with sufficient variation found.")
        return
    
    print(f"Available binary columns with variation: {available_cols}")
    
    # Calculate number of subplots needed
    n_cols = len(available_cols)
    n_rows = (n_cols + 2) // 3  # Ceiling division for rows
    
    plt.figure(figsize=(15, n_rows * 4))
    
    plot_idx = 1
    
    for col in available_cols:
        # Get data for images with and without this feature
        # Only consider rows where violence is not NaN
        valid_violence = annot_train['violence'].notna()
        
        # Now properly filter by binary values (0 and 1)
        with_feature = annot_train[(annot_train[col] == 1) & valid_violence]['violence']
        without_feature = annot_train[(annot_train[col] == 0) & valid_violence]['violence']
        
        print(f"\n{col}:")
        print(f"  With {col}: {len(with_feature)} images (mean violence: {with_feature.mean():.3f})")
        print(f"  Without {col}: {len(without_feature)} images (mean violence: {without_feature.mean():.3f})")
        
        # Only plot if we have data for both conditions
        if len(with_feature) > 0 and len(without_feature) > 0:
            plt.subplot(n_rows, 3, plot_idx)
              # Create box plots
            data_to_plot = [without_feature.values, with_feature.values]
            
            bp = plt.boxplot(data_to_plot, patch_artist=True)
            plt.xticks([1, 2], [f'No {col.capitalize()}', f'Has {col.capitalize()}'])
            
            # Color the boxes
            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][1].set_facecolor('lightcoral')
            
            plt.ylabel('Violence Score')
            plt.title(f'Violence by {col.capitalize()} Presence')
            
            # Add mean markers as red diamonds
            means = [np.mean(without_feature), np.mean(with_feature)]
            plt.scatter([1, 2], means, marker='D', color='red', s=80, zorder=5)
            
            # Add mean values as text
            for j, mean in enumerate(means):
                plt.annotate(f'μ: {mean:.3f}', 
                             xy=(j+1, mean), 
                             xytext=(0, 10),
                             textcoords='offset points',
                             ha='center',
                             va='bottom',
                             fontsize=10,
                             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
            
            # Add statistical info
            plt.text(0.02, 0.98, f'n₀={len(without_feature)}\nn₁={len(with_feature)}', 
                     transform=plt.gca().transAxes, 
                     verticalalignment='top',
                     fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.5))
            
            # Add difference annotation
            diff = means[1] - means[0]
            plt.text(0.5, 0.02, f'Δμ: {diff:+.3f}', 
                     transform=plt.gca().transAxes, 
                     ha='center',
                     fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.3", 
                              facecolor='lightgreen' if diff > 0 else 'pink', 
                              alpha=0.7))
            
            plot_idx += 1
        else:
            print(f"  Skipping {col} - insufficient data")
    
    plt.tight_layout()
    plt.savefig('violence_by_feature_fixed.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_correlation_analysis(annot_train):
    """Create correlation analysis between violence and other features."""
    binary_cols = ['sign', 'fire', 'police', 'children', 'group_20', 'group_100']
    
    # Include additional columns if they exist
    additional_cols = ['flag', 'night', 'shouting', 'photo']
    all_cols = binary_cols + [col for col in additional_cols if col in annot_train.columns]
    
    # Create correlation matrix
    corr_cols = ['violence', 'protest'] + all_cols
    corr_data = annot_train[corr_cols].corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    
    mask = np.triu(np.ones_like(corr_data, dtype=bool))
    
    sns.heatmap(corr_data, 
                mask=mask,
                annot=True, 
                cmap='RdBu_r', 
                center=0,
                square=True,
                fmt='.3f',
                cbar_kws={'label': 'Correlation Coefficient'})
    
    plt.title('Correlation Matrix: Violence, Protest, and Visual Features', fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig('violence_correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print correlation values with violence
    print("\nCorrelation with violence (sorted by absolute value):")
    violence_corr = corr_data['violence'].drop('violence').sort_values(key=abs, ascending=False)
    for feature, corr in violence_corr.items():
        strength = "Strong" if abs(corr) > 0.3 else "Moderate" if abs(corr) > 0.1 else "Weak"
        direction = "positive" if corr > 0 else "negative"
        print(f"{feature:12}: {corr:+.3f} ({strength} {direction})")
