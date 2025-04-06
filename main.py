import numpy as np
import joblib
from models.model_creation import build_model
from preprocessing.preprocess import DataPreprocessor

class YieldPredictor:
    def __init__(self):
        self.model = build_model()
        self.model.load_weights("models/model.keras")
        
        self.soil_scaler = joblib.load("models/soil_scaler.joblib")
        self.climate_scaler = joblib.load("models/climate_scaler.joblib")
        self.crop_encoder = joblib.load("models/crop_encoder.joblib")
        
        self.preprocessor = DataPreprocessor()
    
    def predict(self, lat: float, lon: float, crop: str, year: int):
        """Predict yield for given coordinates."""
        try:
            # Extract spatial features (soil + climate)
            soil_raw = self.preprocessor._get_soil(lat, lon)[:3]
            climate_raw = list(self.preprocessor._get_climate(lat, lon).values())
            
            crop_encoded = self.crop_encoder.transform([crop])[0]
            year_normalized = (year - 2010) / 10
            
            # Scale features using saved preprocessors
            soil_scaled = self.soil_scaler.transform([soil_raw])
            climate_scaled = self.climate_scaler.transform([climate_raw])
            
            return self.model.predict([
                soil_scaled,
                climate_scaled,
                np.array([[crop_encoded]]),
                np.array([[year_normalized]])
            ])[0][0]
        
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return None

if __name__ == "__main__":
    predictor = YieldPredictor()
    
    lat, lon, crop_type, year_to_predict = 28.6139, 77.2090, 'wheat', 2025  # Example: Delhi wheat yield in 2025
    
    predicted_yield = predictor.predict(lat=lat, lon=lon,
                                        crop=crop_type,
                                        year=year_to_predict)
                                        
    print(f"Predicted Yield for {crop_type} in {year_to_predict}: {predicted_yield:.2f} kg/ha")
