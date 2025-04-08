# deep_fusion.py
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Embedding, Concatenate, Attention

class DeepFusionModel(tf.keras.Model):
    def __init__(self, num_districts=1000, num_crops=6):
        super().__init__()
        
        # Temporal features (3 features: GDD, precip, solar_rad)
        self.temporal_encoder = LSTM(128, return_sequences=True)
        self.temporal_attention = Attention()
        
        # Spatial features
        self.district_embedding = Embedding(num_districts, 32)
        self.soil_encoder = Dense(64, activation='relu')
        
        # Crop embedding
        self.crop_embedding = Embedding(num_crops, 16)
        
        # Fusion layers
        self.cross_attention = Attention()
        self.output_layer = Dense(1, activation='linear')
        
    def call(self, inputs):
        # Input shapes
        weather = inputs['weather']  # [batch, seq_len, 3]
        district_ids = inputs['district']  # [batch]
        soil = inputs['soil']  # [batch, 2]
        crop_ids = inputs['crop']  # [batch]
        
        # Temporal processing
        temporal_features = self.temporal_encoder(weather)
        temporal_context = self.temporal_attention(
            [temporal_features, temporal_features]
        )
        
        # Spatial processing
        district_emb = self.district_embedding(district_ids)
        soil_features = self.soil_encoder(soil)
        spatial_context = tf.concat([district_emb, soil_features], axis=-1)
        
        # Crop embedding
        crop_emb = self.crop_embedding(crop_ids)
        
        # Cross-modal fusion
        fused = self.cross_attention(
            [temporal_context, spatial_context]
        )
        fused = tf.concat([fused, crop_emb], axis=-1)
        
        return self.output_layer(fused)
