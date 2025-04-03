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
        params = {
            "start": 2001,
            "end": 2020,
            "latitude": latitude,
            "longitude": longitude,
            "parameters": ",".join(["T2M", "PRECTOTCORR", "RH2M", "CDD18_3"])
        }
        
        response = requests.get("https://power.larc.nasa.gov/api/temporal/climatology/point", params=params).json()
        return response

    def predict_yield(self, latitude, longitude, crop_type):
        """Predict crop yield using coordinates and crop type."""
        
        # Fetch climate data using coordinates
        print("Fetching climate data...")
        climate_json = self.fetch_climate_data(latitude, longitude)
        
        # Process climate data
        climate_processed = self.processor.process_climate(climate_json)
        
        # Mock soil data (replace with actual soil lookup logic)
        soil_processed = np.array([1.2, 6.5, 28.0, 42.0, 15.0, 65.0])
        
        # Encode crop information (only crop type)
        crop_encoded = self.processor.encode_crop_info(crop_type)
        
        # Make prediction
        print("Predicting yield...")
        prediction = self.model.predict([np.array([climate_processed]), np.array([soil_processed]), crop_encoded])[0][0]
        
        print(f"Predicted Yield for {crop_type}: {prediction:.2f} kg/ha")


# Main function to handle user input
def main():
    print("Welcome to the Crop Yield Prediction System!")
    
    try:
        # Get user input for latitude and longitude
        latitude = float(input("Enter Latitude (e.g., 14.6794): "))
        longitude = float(input("Enter Longitude (e.g., 77.5983): "))
        
        # Get user input for crop type
        crop_type = input("Enter Crop Type (e.g., Groundnut): ").strip()
        
        # Initialize predictor and make predictions
        predictor = YieldPredictor()
        predictor.predict_yield(latitude, longitude, crop_type)
    
    except ValueError as e:
        print(f"Invalid input: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


# Run the main function when the script is executed
if __name__ == "__main__":
    main()
