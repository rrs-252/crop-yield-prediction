import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from models.model_creation import build_model

def train_model():
    """Train DFNN on preprocessed dataset."""
    # Load preprocessed dataset
    df = pd.read_parquet("data/processed_dataset.parquet")
    
    # Encode crop types (categorical -> numerical)
    le = LabelEncoder()
    df['crop'] = le.fit_transform(df['crop'])
    
    # Prepare features and targets
    soil_features = df[['soil_oc', 'soil_ph', 'soil_clay']].values
    climate_features = df[['temp', 'rain', 'humidity']].values
    crop_features = df['crop'].values.reshape(-1, 1)
    year_features = df['year'].values.reshape(-1, 1)  # Normalized years (2010-2020 -> 0-1)
    
    targets = df['yield'].values
    
    # Scale soil and climate features
    soil_scaler = StandardScaler()
    climate_scaler = StandardScaler()
    
    soil_scaled = soil_scaler.fit_transform(soil_features)
    climate_scaled = climate_scaler.fit_transform(climate_features)
    
    # Save scalers for future use in predictions
    import joblib
    joblib.dump(soil_scaler, "models/soil_scaler.joblib")
    joblib.dump(climate_scaler, "models/climate_scaler.joblib")
    joblib.dump(le, "models/crop_encoder.joblib")
    
    # Build DFNN model
    model = build_model()
    
    model.compile(
        optimizer='adamw',
        loss='huber',
        metrics=['mae']
    )
    
    # Train the model
    history = model.fit(
        [soil_scaled, climate_scaled, crop_features, year_features],
        targets,
        epochs=100,
        batch_size=2048,
        validation_split=0.2,
        verbose=1
    )
    
    # Save trained model weights
    model.save("models/model.keras")

if __name__ == "__main__":
    train_model()
