# deep_fusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepFusionModel(nn.Module):
    def __init__(self, num_districts=1000, num_crops=6):
        super().__init__()
        
        # Climate feature encoder
        self.climate_encoder = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # District embeddings
        self.district_embed = nn.Embedding(num_districts, 64)
        
        # Crop embeddings
        self.crop_embed = nn.Embedding(num_crops, 32)
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(128+64+32, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, inputs):
        """Input format:
        - climate: [batch_size, 3] (gdd, precip, solar_rad)
        - district: [batch_size] (district IDs)
        - crop: [batch_size] (crop IDs)
        """
        climate_feat = self.climate_encoder(inputs['climate'])
        district_emb = self.district_embed(inputs['district'])
        crop_emb = self.crop_embed(inputs['crop'])
        
        fused = torch.cat([climate_feat, district_emb, crop_emb], dim=1)
        return self.fusion(fused)
