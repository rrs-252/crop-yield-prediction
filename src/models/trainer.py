import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
import pyarrow.parquet as pq

def train_model(data_path: str = "data/processed"):
    """Train deep fusion model with climate data"""
    # Load and prepare dataset
    dataset = tf.data.Dataset.list_files(f"{data_path}/*/*.parquet") \
        .interleave(lambda x: tf.data.Dataset.from_parquet(x)) \
        .batch(64) \
        .prefetch(tf.data.AUTOTUNE)
    
    # Initialize model
    model = create_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='mse',
        metrics=['mae']
    )
    
    # Train model
    history = model.fit(
        dataset,
        epochs=200,
        validation_split=0.2,
        callbacks=[
            EarlyStopping(patience=15, restore_best_weights=True),
            ModelCheckpoint('model.keras', save_best_only=True)
        ]
    )
    
    return model, history

def create_model() -> tf.keras.Model:
    """Deep fusion model architecture"""
    inputs = {
        'weather': tf.keras.Input(shape=(None, 3), name='weather'),
        'district': tf.keras.Input(shape=(), dtype=tf.int32, name='district'),
        'crop': tf.keras.Input(shape=(), dtype=tf.int32, name='crop')
    }
    
    # Temporal stream
    temporal = tf.keras.layers.LSTM(128, return_sequences=True)(inputs['weather'])
    temporal = tf.keras.layers.GlobalAveragePooling1D()(temporal)
    
    # Spatial stream
    district_emb = tf.keras.layers.Embedding(1000, 32)(inputs['district'])
    
    # Crop stream
    crop_emb = tf.keras.layers.Embedding(6, 16)(inputs['crop'])
    
    # Fusion
    fused = tf.keras.layers.Concatenate()([temporal, district_emb, crop_emb])
    output = tf.keras.layers.Dense(1, activation='linear')(fused)
    
    return tf.keras.Model(inputs=inputs, outputs=output)
