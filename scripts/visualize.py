import os
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np


def plot_training_curves(train_losses, normal_losses, anomaly_losses, val_aucs, save_path):
    """Plot training loss and validation AUC curves and save to file"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 8))
    
    # Plot total training loss
    ax1.plot(train_losses, 'b-', linewidth=2)
    ax1.set_title('Total Training Loss vs Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    
    # Plot individual losses
    ax2.plot(normal_losses, 'g-', linewidth=2, label='Normal Loss')
    ax2.plot(anomaly_losses, 'r-', linewidth=2, label='Anomaly Loss')
    ax2.set_title('Individual Losses vs Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot validation AUC
    if val_aucs:
        epochs = np.arange(1, len(val_aucs) + 1) 
        ax3.plot(epochs, val_aucs, 'r-o', linewidth=2, markersize=4)
        ax3.set_title('Validation ROC AUC vs Epoch')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('ROC AUC')
        ax3.set_ylim([0, 1])
        ax3.grid(True, alpha=0.3)
    
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to: {save_path}")


def plot_roc_curve(trainer, test_loader, save_path):
    """Plot ROC curve for final model and save to file"""
    trainer.feature_adaptor.eval()
    trainer.discriminator.eval()
    
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(trainer.device)
            labels = batch['label'].cpu().numpy()
            
            features = trainer.extract_features(images)
            scores = trainer.discriminator(features)
            
            # Take max score across spatial dimensions for image-level prediction
            scores_np = scores.view(scores.size(0), -1).max(dim=1)[0].cpu().numpy()
            
            all_scores.extend(scores_np)
            all_labels.extend(labels)
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Anomaly Detection (Image Level - Best Model)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved to: {save_path}")


def visualize_results(trainer, test_loader, train_losses, normal_losses, anomaly_losses, val_aucs, results_dir, category):
    """Generate all visualizations for the trained model and save to results directory"""
    os.makedirs(results_dir, exist_ok=True)
    
    print("\nGenerating and saving plots...")
    
    # Save training curves (now with individual losses)
    curves_path = os.path.join(results_dir, f"training_curves_{category}.png")
    plot_training_curves(train_losses, normal_losses, anomaly_losses, val_aucs, curves_path)
    
    # Save ROC curve
    roc_path = os.path.join(results_dir, f"roc_curve_{category}.png")
    plot_roc_curve(trainer, test_loader, roc_path)