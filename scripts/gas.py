import torch
import torch.nn.functional as F


class GlobalAnomalySynthesis:
    """Global Anomaly Synthesis (GAS) for manifold hypothesis"""
    
    def __init__(self, r1=1.0, r2=2.0, eta=0.1, noise_std=0.015, n_step=5, n_proj=1):
        self.r1 = r1  # minimum distance from normal features
        self.r2 = r2  # maximum distance from normal features  
        self.eta = eta  # learning rate for gradient ascent
        self.noise_std = noise_std  # standard deviation for Gaussian noise
        self.n_step = n_step  # number of gradient ascent iterations
        self.n_proj = n_proj  # interval for projection
    
    def add_gaussian_noise(self, features):
        """Step 1: Add Gaussian noise to normal features"""
        noise = torch.randn_like(features) * self.noise_std
        return features + noise
    
    def gradient_ascent_step(self, features, discriminator):
        """Single gradient ascent step"""
        # Clone and require grad for safety
        current_features = features.detach().clone().requires_grad_(True)
        
        # Forward pass through discriminator
        scores = discriminator(current_features)
        
        # Compute loss (we want to maximize anomaly score, so minimize negative score)
        loss = -scores.mean()
        
        # Compute gradients (no retain_graph needed for single use)
        grad = torch.autograd.grad(loss, current_features)[0]
        
        # Normalize gradient globally per sample (treat entire feature map as one vector)
        B, C, H, W = grad.shape
        grad_flat = grad.reshape(B, -1)  # (B, C*H*W) - use reshape instead of view
        grad_norm = torch.norm(grad_flat, dim=1, keepdim=True) + 1e-8  # (B, 1)
        normalized_grad_flat = grad_flat / grad_norm  # (B, C*H*W)
        normalized_grad = normalized_grad_flat.reshape(B, C, H, W)  # (B, C, H, W)
        
        # Update features in gradient direction
        updated_features = current_features + self.eta * normalized_grad
        
        return updated_features.detach()
    

    def truncated_projection(self, synthetic_features, normal_features):
        """Step 3: Apply truncated projection to constrain distance"""
        # Calculate distance from normal features (treat entire feature map as one vector)
        epsilon = synthetic_features - normal_features
        B, C, H, W = epsilon.shape
        epsilon_flat = epsilon.reshape(B, -1)  # (B, C*H*W)
        epsilon_norm = torch.norm(epsilon_flat, dim=1, keepdim=True)  # (B, 1)
        
        # Apply truncation based on manifold constraints (cleaner with clamp)
        alpha = epsilon_norm.clamp(min=self.r1, max=self.r2)
        
        # Project to constrained distance
        epsilon_hat_flat = (alpha / (epsilon_norm + 1e-8)) * epsilon_flat  # (B, C*H*W)
        epsilon_hat = epsilon_hat_flat.reshape(B, C, H, W)  # (B, C, H, W)
        
        return normal_features + epsilon_hat
    
    def synthesize(self, normal_features, discriminator):
        """Complete GAS pipeline following GAS Algorithm"""
        # Step 1: Add Gaussian noise
        current_features = self.add_gaussian_noise(normal_features)
        
        # Step 2: Gradient ascent loop with periodic projection
        for step in range(self.n_step):
            # Gradient ascent
            current_features = self.gradient_ascent_step(current_features, discriminator)
            
            # Apply truncated projection every n_proj steps
            if (step + 1) % self.n_proj == 0:
                with torch.no_grad():
                    current_features = self.truncated_projection(current_features, normal_features)
        
        # Final projection if needed (ensure constraints are met)
        if self.n_step % self.n_proj != 0:
            with torch.no_grad():
                current_features = self.truncated_projection(current_features, normal_features)
        
        return current_features