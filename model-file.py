import torch
import torch.nn as nn
from typing import Dict

class DeepFusionNetwork(nn.Module):
    def __init__(self, input_dims: Dict[str, int], config: Dict):
        super().__init__()
        
        # Initialize network branches
        self.weather_branch = self._create_branch(
            input_dims['weather'],
            config['hidden_dims']['weather']
        )
        self.soil_branch = self._create_branch(
            input_dims['soil'],
            config['hidden_dims']['soil']
        )
        self.satellite_branch = self._create_branch(
            input_dims['satellite'],
            config['hidden_dims']['satellite']
        )
        
        # Fusion layers
        total_hidden = sum(config['hidden_dims'].values())
        self.fusion = self._create_fusion_layers(total_hidden)
        self.attention = nn.MultiheadAttention(total_hidden, 4)
        
        # Additional layers for uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(total_hidden, total_hidden // 2),
            nn.ReLU(),
            nn.Linear(total_hidden // 2, 1),
            nn.Softplus()  # Ensures positive uncertainty values
        )
    
    def _create_branch(self, input_dim: int, hidden_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2)
        )
    
    def _create_fusion_layers(self, input_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(input_dim // 2),
            nn.Dropout(0.3),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(),
            nn.BatchNorm1d(input_dim // 4),
            nn.Linear(input_dim // 4, 1)
        )
    
    def forward(self, weather_features, soil_features, satellite_features):
        # Process each branch
        weather_out = self.weather_branch(weather_features)
        soil_out = self.soil_branch(soil_features)
        satellite_out = self.satellite_branch(satellite_features)
        
        # Combine features with attention
        combined = torch.cat([weather_out, soil_out, satellite_out], dim=1)
        combined = combined.unsqueeze(0)
        attn_output, attention_weights = self.attention(combined, combined, combined)
        combined = attn_output.squeeze(0)
        
        # Generate prediction and uncertainty
        prediction = self.fusion(combined)
        uncertainty = self.uncertainty_head(combined)
        
        return prediction, uncertainty, attention_weights
