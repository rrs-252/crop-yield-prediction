# deep_fusion.py
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Embedding, Concatenate, Attention

class DeepFusionModel(tf.keras.Model):
    def __init__(self, num_districts=1000, num_crops=6):
        super().__init__()
        
        # Temporal stream (Climate + NDVI/VCI)
        self.temporal_encoder = tf.keras.Sequential([
            LSTM(128, return_sequences=True),
            Attention()
        ])
        
        # Spatial features
        self.district_embedding = Embedding(num_districts, 64)
        
        # Vegetation features
        self.vegetation_encoder = Dense(32, activation='relu')
        
        # Crop embedding
        self.crop_embedding = Embedding(num_crops, 32)
        
        # Fusion layers
        self.fusion = Concatenate(axis=-1)
        self.regressor = Dense(1, activation='linear')

    def call(self, inputs):
        # Process temporal features
        temporal = self.temporal_encoder(inputs['temporal'])
        
        # Process vegetation
        vegetation = self.vegetation_encoder(inputs['vegetation'])
        
        # District embedding
        district_emb = self.district_embedding(inputs['district'])
        
        # Crop embedding
        crop_emb = self.crop_embedding(inputs['crop'])
        
        # Feature fusion
        fused = self.fusion([
            tf.reduce_mean(temporal, axis=1),
            vegetation,
            district_emb,
            crop_emb
        ])
        
        return self.regressor(fused)
