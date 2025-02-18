import argparse
from pathlib import Path
from datetime import datetime, timedelta
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple

from config import Config
from data_collector import DataCollector
from data_preprocessor import DataPreprocessor
from model import DeepFusionNetwork
from trainer import ModelTrainer, CropDataset

def parse_args():
    parser = argparse.ArgumentParser(description='Crop Yield Prediction System')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--api_key', type=str, required=True,
                       help='OpenWeather Agro API key')
    parser.add_argument('--mode', type=str, choices=['train', 'predict'],
                       required=True, help='Operation mode')
    parser.add_argument('--data_path', type=str,
                       help='Path to historical yield data (for training)')
    return parser.parse_args()

def load_historical_data(data_path: str) -> pd.DataFrame:
    """Load historical yield data"""
    return pd.read_csv(data_path)

def prepare_datasets(features_df: pd.DataFrame, 
                    yields: np.array,
                    batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """Prepare train and validation datasets"""
    X_train, X_valid, y_train, y_valid = train_test_split(
        features_df, yields, test_size=0.2, random_state=42
    )
    
    train_dataset = CropDataset(X_train, y_train)
    valid_dataset = CropDataset(X_valid, y_valid)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    
    return train_loader, valid_loader

def train_model(config: Config, args: argparse.Namespace):
    """Training pipeline"""
    # Load historical data
    historical_data = load_historical_data(args.data_path)
    
    # Initialize components
    collector = DataCollector(args.api_key, config)
    preprocessor = DataPreprocessor()
    
    # Process all available data
    print("Processing historical data...")
    all_processed_data = []
    all_yields = []
    
    for _, row in historical_data.iterrows():
        # Collect data for each historical record
        polygon_id = collector.create_field_polygon(row['latitude'], row['longitude'])
        
        weather_data = collector.collect_weather_data(
            polygon_id,
            datetime.strptime(row['planting_date'], '%Y-%m-%d'),
            datetime.strptime(row['harvest_date'], '%Y-%m-%d')
        )
        
        soil_data = collector.collect_soil_data(polygon_id)
        satellite_data = collector.collect_satellite_data(polygon_id)
        
        # Preprocess data
        processed_data, feature_dims = preprocessor.preprocess_all_data(
            weather_data,
            soil_data,
            satellite_data
        )
        
        all_processed_data.append(processed_data)
        all_yields.append(row['yield'])
    
    # Combine all processed data
    combined_features = pd.concat(all_processed_data, ignore_index=True)
    yields_array = np.array(all_yields)
    
    # Prepare datasets
    train_loader, valid_loader = prepare_datasets(
        combined_features,
        yields_array,
        config.MODEL_CONFIG['batch_size']
    )
    
    # Initialize and train model
    model = DeepFusionNetwork(feature_dims, config.MODEL_CONFIG)
    trainer = ModelTrainer(model, config.MODEL_CONFIG, config.MODELS_DIR)
    
    print("Starting training...")
    training_history = trainer.train(train_loader, valid_loader)
    
    print("Training completed!")
    return training_history

def predict_yield(config: Config, args: argparse.Namespace):
    """Prediction pipeline"""
    # Load the best model
    model_path = max(config.MODELS_DIR.glob('model_checkpoint_*.pt'))
    checkpoint = torch.load(model_path)
    
    model = DeepFusionNetwork(
        checkpoint['feature_dims'],
        checkpoint['config']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Initialize components
    collector = DataCollector(args.api_key, config)
    preprocessor = DataPreprocessor()
    
    # Get current field data
    polygon_id = collector.create_field_polygon(args.latitude, args.longitude)
    
    weather_data = collector.collect_weather_data(
        polygon_id,
        datetime.now() - timedelta(days=30),  # Last 30 days
        datetime.now()
    )
    
    soil_data = collector.collect_soil_data(polygon_id)
    satellite_data = collector.collect_satellite_data(polygon_id)
    
    # Preprocess data
    processed_data, _ = preprocessor.preprocess_all_data(
        weather_data,
        soil_data,
        satellite_data
    )
    
    # Make prediction
    with torch.no_grad():
        prediction, uncertainty, attention_weights = model(
            torch.FloatTensor(processed_data['weather'].values))
