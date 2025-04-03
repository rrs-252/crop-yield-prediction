import json
import numpy as np
from keras.models import load_model
from data_processing import DataProcessor

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

    def predict_yield(self, district_name, crop_info):
        """Predict crop yield using climate, soil, and crop data."""
        
        # Get coordinates for the district
        latitude, longitude = self.processor.get_coordinates(district_name)
        
        # Fetch climate data using coordinates
        climate_json = self.fetch_climate_data(latitude, longitude)
        
        # Process climate data
        climate_processed = self.processor.process_climate(climate_json)
        
        # Replace with actual soil data lookup logic or mock data here:
        soil_processed = np.array([1.2, 6.5, 28.0, 42.0, 15.0, 65.0])  
        
        crop_encoded = self.processor.crop_encoder.transform([[crop_info["state"], crop_info["district"], crop_info["crop"], crop_info["season"]]])
        
        prediction = self.model.predict([np.array([climate_processed]), np.array([soil_processed]), crop_encoded])[0][0]
        
        print(f"Predicted Yield: {prediction:.2f} kg/ha")

# Example usage:
# predictor = YieldPredictor()
# predictor.predict_yield(district_name="ANANTAPUR",
#                         crop_info={'state': 'Andhra Pradesh', 
#                                    'district': 'ANANTAPUR',
#                                    'crop': 'Groundnut',
#                                    'season': 'Kharif'})
