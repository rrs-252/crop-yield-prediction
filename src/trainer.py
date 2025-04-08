import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from .data_processor import AgriDataProcessor
from .deep_fusion import DeepFusionModel

def train_model(data_path="processed_data"):
    processor = AgriDataProcessor("ICRISAT-District-Level-Data.csv")
    processor.process_dataset()
    
    model = DeepFusionModel()
    dataset = _create_dataset(data_path)
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    history = model.fit(
        dataset,
        epochs=100,
        validation_split=0.2,
        callbacks=[
            EarlyStopping(patience=10),
            ModelCheckpoint('model.keras', save_best_only=True)
        ]
    )
    
    _save_training_plots(history)
    return model

def _create_dataset(data_path):
    return tf.data.Dataset.list_files(f"{data_path}/*.parquet") \
        .interleave(lambda x: tf.data.Dataset.from_parquet(x)) \
        .batch(64) \
        .prefetch(2)

def _save_training_plots(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Training History')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.title('MAE Progress')
    plt.legend()
    plt.savefig('training_metrics.png')
