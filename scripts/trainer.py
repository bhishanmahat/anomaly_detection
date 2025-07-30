import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score


class GLASSTrainer:
    """Simple trainer for GLASS model with GAS"""
    
    def __init__(self, feature_extractor, feature_adaptor, discriminator, gas, device):
        self.device = device
        self.feature_extractor = feature_extractor.to(device).eval()
        self.feature_adaptor = feature_adaptor.to(device)
        self.discriminator = discriminator.to(device)
        self.gas = gas
        
        # Only train adaptor and discriminator (feature extractor is frozen)
        self.optimizer = optim.Adam(
            list(self.feature_adaptor.parameters()) + list(self.discriminator.parameters()),
            lr=0.0002
        )
        
        self.bce_loss = nn.BCELoss()
        self.train_losses = []
        self.normal_losses = []
        self.anomaly_losses = []
        self.val_aucs = []
        self.best_auc = 0.0
        self.best_epoch = 0
    
    def extract_features(self, images):
        """Extract and adapt features from images"""
        with torch.no_grad():
            raw_features = self.feature_extractor(images)
        return self.feature_adaptor(raw_features)
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.feature_adaptor.train()
        self.discriminator.train()
        
        epoch_total_loss = 0.0
        epoch_normal_loss = 0.0
        epoch_anomaly_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            images = batch['image'].to(self.device)
            
            # Extract normal features
            normal_features = self.extract_features(images)
            
            # Generate synthetic anomaly features using GAS
            synthetic_features = self.gas.synthesize(normal_features, self.discriminator)
            
            # Get discriminator scores
            normal_scores    = self.discriminator(normal_features)
            synthetic_scores = self.discriminator(synthetic_features)
            
            # Flatten scores for loss computation
            normal_scores_flat    = normal_scores.view(-1)
            synthetic_scores_flat = synthetic_scores.view(-1)
            
            # Create labels
            normal_labels  = torch.zeros_like(normal_scores_flat)
            anomaly_labels = torch.ones_like(synthetic_scores_flat)
            
            # Compute losses
            normal_loss = self.bce_loss(normal_scores_flat, normal_labels)
            anomaly_loss = self.bce_loss(synthetic_scores_flat, anomaly_labels)
            total_loss = normal_loss + anomaly_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            

            epoch_total_loss += total_loss.item()
            epoch_normal_loss += normal_loss.item()
            epoch_anomaly_loss += anomaly_loss.item()
            num_batches += 1

        # Guard against empty loader
        assert num_batches > 0, "No batches found in train_loader"
        avg_total_loss = epoch_total_loss / num_batches
        avg_normal_loss = epoch_normal_loss / num_batches
        avg_anomaly_loss = epoch_anomaly_loss / num_batches
        
        self.train_losses.append(avg_total_loss)
        self.normal_losses.append(avg_normal_loss)
        self.anomaly_losses.append(avg_anomaly_loss)
        
        return avg_total_loss, avg_normal_loss, avg_anomaly_loss
    
    def evaluate(self, test_loader):
        """Evaluate model and return ROC AUC"""
        self.feature_adaptor.eval()
        self.discriminator.eval()
        
        all_scores = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                images = batch['image'].to(self.device)
                labels = batch['label'].cpu().numpy()
                
                # Extract features and get anomaly scores
                features = self.extract_features(images)
                scores   = self.discriminator(features)
                
                # Image‑level score: max over spatial dims
                scores_np = scores.view(scores.size(0), -1).max(dim=1)[0].cpu().numpy()
                
                all_scores.extend(scores_np)
                all_labels.extend(labels)
        
        auc = roc_auc_score(all_labels, all_scores)
        self.val_aucs.append(auc)
        return auc
    

    def train(self, train_loader, test_loader, epochs=100, save_dir="./results", category=""):
        """Complete training loop with model saving"""
        os.makedirs(save_dir, exist_ok=True)
        print(f"Training for {epochs} epochs...")
        
        for epoch in range(epochs):
            total_loss, normal_loss, anomaly_loss = self.train_epoch(train_loader)
            
            # Always compute AUC:
            auc = self.evaluate(test_loader)

            print(f"Epoch {epoch+1}/{epochs} - Total Loss: {total_loss:.4f}, Normal: {normal_loss:.4f}, Anomaly: {anomaly_loss:.4f}, Image AUC: {auc:.4f}")
            
            # Compare AUC for every epoch
            if auc > self.best_auc:
                self.best_auc   = auc
                self.best_epoch = epoch + 1
                self.save_best_model(save_dir, category)
        
        print(f"Best AUC: {self.best_auc:.4f} at epoch {self.best_epoch}\n")
        return self.train_losses, self.normal_losses, self.anomaly_losses, self.val_aucs

    
    def save_best_model(self, save_dir, category):
        """Save the best performing model"""
        model_path = os.path.join(save_dir, f"best_model_{category}.pth")
        torch.save({
            'feature_adaptor_state_dict': self.feature_adaptor.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_auc': self.best_auc,
            'best_epoch': self.best_epoch,
        }, model_path)
        print(f"Best model saved: {model_path} (AUC: {self.best_auc:.4f})\n")
    
    def load_best_model(self, save_dir, category):
        """Load the best saved model"""
        model_path = os.path.join(save_dir, f"best_model_{category}.pth")
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.feature_adaptor.load_state_dict(checkpoint['feature_adaptor_state_dict'])
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            print(f"Loaded best model from epoch {checkpoint['best_epoch']} with AUC: {checkpoint['best_auc']:.4f}")
            return True
        return False
