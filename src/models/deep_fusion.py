# deep_fusion.py
import tensorflow as tf

class TemporalFusionModel(tf.keras.Model):
    def __init__(self, num_districts=1000):
        super().__init__()
        
        # Climate feature encoder
        self.climate_encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2)
        ])
        
        # District embedding
        self.district_embedding = tf.keras.layers.Embedding(num_districts, 32)
        
        # Temporal attention
        self.temporal_attention = tf.keras.layers.Attention()
        
        # Final regressor
        self.regressor = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')
        ])

    def call(self, inputs):
        # Input processing
        climate_features = self.climate_encoder(inputs['climate'])
        district_emb = self.district_embedding(inputs['district'])
        
        # Temporal fusion
        context = self.temporal_attention([
            climate_features, 
            tf.expand_dims(district_emb, 1)
        ])
        
        return self.regressor(context)
