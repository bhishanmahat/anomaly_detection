import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights


class FeatureExtractor(nn.Module):
    """Extract and merge multi-scale features from WideResNet50."""
    
    def __init__(self, layers=['layer2', 'layer3'], neighborhood_size=3):
        super().__init__()
        self.layers = layers
        self.features = {}
        
        # Load pretrained backbone
        self.backbone = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        self.backbone.eval()
        
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Register hooks
        for name, module in self.backbone.named_modules():
            if name in self.layers:
                module.register_forward_hook(self._hook_fn(name))
        
        # Neighborhood aggregation
        self.pool = nn.AvgPool2d(neighborhood_size, stride=1, padding=neighborhood_size//2)
    
    def _hook_fn(self, layer_name):
        def hook(module, input, output):
            self.features[layer_name] = output
        return hook
    
    def forward(self, x):
        self.features = {}
        with torch.no_grad():
            _ = self.backbone(x)
        
        # Get features
        feat2 = self.pool(self.features['layer2'])  # Higher resolution
        feat3 = self.pool(self.features['layer3'])  # Lower resolution
        
        # Merge: upsample feat3 and concatenate
        feat3_up = F.interpolate(feat3, size=feat2.shape[-2:], mode='bilinear', align_corners=False)
        merged = torch.cat([feat2, feat3_up], dim=1)
        
        return merged


class FeatureAdaptor(nn.Module):
    """Linear transformation for domain adaptation."""
    
    def __init__(self, channels):
        super().__init__()
        self.linear = nn.Linear(channels, channels)
    
    def forward(self, x):
        # x: (B, C, H, W) -> (B, H, W, C) -> (B*H*W, C)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(-1, C)
        x = self.linear(x)
        return x.reshape(B, H, W, C).permute(0, 3, 1, 2)


class Discriminator(nn.Module):
    """MLP discriminator for anomaly detection."""
    
    def __init__(self, in_channels: int):
        super().__init__()
        hidden_dim = in_channels // 2
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: (B, C, H, W) -> (B, 1, H, W)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(-1, C)
        x = self.mlp(x)
        return x.reshape(B, H, W, 1).permute(0, 3, 1, 2)


class ImprovedDiscriminator(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.BatchNorm1d(in_channels // 2),
            nn.ReLU(),
            
            nn.Linear(in_channels // 2, in_channels // 4),
            nn.BatchNorm1d(in_channels // 4),
            nn.ReLU(),
            
            nn.Linear(in_channels // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: (B, C, H, W) -> (B, 1, H, W)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(-1, C)
        x = self.mlp(x)
        return x.reshape(B, H, W, 1).permute(0, 3, 1, 2)

