import json
import numpy as np
from keras.models import load_model
from data_processing import DataProcessor
import requests

class YieldPredictor:
    def __init__(self):
        self.model = load_model('models/crop_yield_model.keras')  # Load saved model
        self.processor = DataProcessor()
        self.processor.load_preprocessors()  # Load preprocessors

    def fetch_climate_data(self, latitude, longitude):
        """Fetch climate data from NASA POWER API."""
        params = {"start": 2001, "end": 2020,
                  "latitude": latitude,
                  "longitude": longitude,
                  "parameters": ",".join(["T2M", "PRECTOTCORR", "RH2M", "CDD18_3"])}
        
        response = requests.get("https://power.larc.nasa.gov/api/temporal/climatology/point", params=params).json()
        return response

    def predict_yield(self, latitude, longitude, crop_type):
        """Predict crop yield using coordinates and crop type."""
        
        # Fetch climate data using coordinates
        climate_json = self.fetch_climate_data(latitude, longitude)
        
        # Process climate data
        climate_processed = self.processor.process_climate(climate_json)
        
        # Mock soil data (replace with actual soil lookup logic)
        soil_processed = np.array([1.2, 6.5, 28.0, 42.0, 15.0, 65.0])
        
        # Encode crop information (only crop type)
        crop_encoded = self.processor.encode_crop_info(crop_type)
        
        # Make prediction
        prediction = self.model.predict([np.array([climate_processed]), np.array([soil_processed]), crop_encoded])[0][0]
        
        print(f"Predicted Yield: {prediction:.2f} kg/ha")

# Example usage:
# predictor = YieldPredictor()
# predictor.predict_yield(latitude=14.6794, longitude=77.5983,
#                         crop_type='Groundnut')
