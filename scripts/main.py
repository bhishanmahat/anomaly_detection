import os
import argparse

# --- GPU Configuration ---
GPU_ID = "5"
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

import torch

# Import modules 
from models import FeatureExtractor, FeatureAdaptor, Discriminator, ImprovedDiscriminator 
from dataloader import create_dataloaders
from gas import GlobalAnomalySynthesis
from trainer import GLASSTrainer
from visualize import visualize_results


def main():

    """Main training script"""

    # Configuration
    DATASET_PATH = "./data/anomaly_dataset"
    CATEGORY = "bracket_white"
    RESULTS_DIR = "./results"
    BATCH_SIZE = 8
    IMAGE_SIZE = 288
    EPOCHS = 100
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Ensure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print(f"Using device: {DEVICE}")
    print(f"Training on category: {CATEGORY}")
    
    # Create dataloaders
    train_loader, test_loader = create_dataloaders(
        DATASET_PATH, CATEGORY, BATCH_SIZE, IMAGE_SIZE
    )
    
    # Initialize model components
    feature_extractor = FeatureExtractor()
    
    # WideResNet50 layer2: 512 channels, layer3: 1024 channels
    # Total after concatenation: 512 + 1024 = 1536 channels
    feature_dim = 1536
    
    feature_adaptor = FeatureAdaptor(feature_dim)
    discriminator = ImprovedDiscriminator(feature_dim)
    
    # Initialize GAS
    gas = GlobalAnomalySynthesis(r1=1.0, r2=2.0, eta=0.1, noise_std=0.015)
    
    # Initialize trainer
    trainer = GLASSTrainer(
        feature_extractor=feature_extractor,
        feature_adaptor=feature_adaptor,
        discriminator=discriminator,
        gas=gas,
        device=DEVICE
    )
    
    # Train model
    train_losses, normal_losses, anomaly_losses, val_aucs = trainer.train(train_loader, test_loader, epochs=EPOCHS, save_dir=RESULTS_DIR, category=CATEGORY)
    
    # Load best checkpoint for visualization
    if not trainer.load_best_model(RESULTS_DIR, CATEGORY):
        print("⚠️  No saved best model found; using last epoch weights for visualization.")
    
    # Generate visualizations
    visualize_results(trainer, test_loader, train_losses, normal_losses, anomaly_losses, val_aucs, RESULTS_DIR, CATEGORY)
    
    print("Training completed!")


if __name__ == "__main__":
    main()