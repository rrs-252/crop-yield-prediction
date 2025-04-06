import numpy as np
import joblib
from models.model_creation import build_dfnn
from preprocessing.preprocess import DataPreprocessor

class YieldPredictor:
    def __init__(self):
        self.model = build_dfnn()
        self.model.load_weights("models/model.keras")
        
        # Load preprocessors
        self.soil_scaler = joblib.load("models/soil_scaler.joblib")
        self.climate_scaler = joblib.load("models/climate_scaler.joblib")
        self.crop_encoder = joblib.load("models/crop_encoder.joblib")
        
        # Initialize processors for fresh data extraction
        self.data_preprocessor = DataPreprocessor()

    def predict(self, lat: float, lon: float, crop: str):
        """Predict yield for given coordinates and crop."""
        try:
            # Extract soil properties
            soil_raw = list(self.data_preprocessor.soil_processor.get_soil_properties(lat, lon).values())
            soil_scaled = self.soil_scaler.transform([soil_raw])
            
            # Extract climate properties
            climate_raw = list(self.data_preprocessor.climate_processor.get_climate_data(lat, lon).values())
            climate_scaled = self.climate_scaler.transform([climate_raw])
            
            # Encode crop type
            crop_encoded = self.crop_encoder.transform([crop])[0]
            
            # Predict yield using pretrained model
            predicted_yield = self.model.predict([soil_scaled, climate_scaled, np.array([[crop_encoded]])])[0][0]
            
            return predicted_yield
        
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return None

if __name__ == "__main__":
    predictor = YieldPredictor()
    
    # Example prediction: Wheat yield in Delhi (28.6139°N, 77.2090°E)
    lat, lon, crop_type = 28.6139, 77.2090, "wheat"
    predicted_yield = predictor.predict(lat=lat, lon=lon, crop=crop_type)
    
    print(f"Predicted Yield for {crop_type} at ({lat}, {lon}): {predicted_yield:.2f} kg/ha")
