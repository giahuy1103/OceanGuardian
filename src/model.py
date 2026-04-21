import torch
import torch.nn as nn
from torchvision import models

class CoralFusionModel(nn.Module):
    def __init__(self, num_classes=2):
        super(CoralFusionModel, self).__init__()
        
        # -----------------------------------------------------
        # BRANCH 1: IMAGE FEATURE EXTRACTION (ResNet18)
        # -----------------------------------------------------
        # Initialize ResNet18 with pre-trained ImageNet weights
        self.cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Instead of keeping 512 dimensions, reduce to 128 dimensions to balance with the environmental branch
        # This prevents the model from focusing solely on the image and ignoring tabular data
        self.cnn.fc = nn.Linear(512, 128) 
        
        # -----------------------------------------------------
        # BRANCH 2: ENVIRONMENTAL FEATURE EXTRACTION (MLP)
        # -----------------------------------------------------
        # Process 6-dimensional vector (SST, pH, Lat, Lon, Month, Heatwave) into a 64-dimensional vector
        # Add BatchNorm1d to stabilize the distribution of tabular data
        self.mlp = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        
        # -----------------------------------------------------
        # EARLY FUSION & CLASSIFICATION LAYER
        # -----------------------------------------------------
        # Total dimensions after concatenation: 128 (image) + 64 (environment) = 192 dimensions
        self.fusion_norm = nn.BatchNorm1d(192)
        
        self.classifier = nn.Sequential(
            nn.Linear(192, 64),
            nn.ReLU(),
            nn.Dropout(0.4), # Increase Dropout to 0.4 for stronger Overfitting prevention
            nn.Linear(64, num_classes)
        )

    def forward(self, img, env_data):
        # 1. Extract both data streams simultaneously
        img_features = self.cnn(img)          # Shape: [batch_size, 128]
        env_features = self.mlp(env_data)     # Shape: [batch_size, 64]
        
        # 2. Early Fusion: Concatenate 2 vectors horizontally
        fused_features = torch.cat((img_features, env_features), dim=1) # Shape: [batch_size, 192]
        
        # Synchronize the scale of the fused vector before passing it to the classifier
        fused_features = self.fusion_norm(fused_features)
        
        # 3. Pass to the final classification layer
        out = self.classifier(fused_features)
        return out