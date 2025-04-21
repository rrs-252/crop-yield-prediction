from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import requests
import numpy as np
import joblib
import torch
import datetime
import logging
from models import DeepFusionNN

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Model and scaler loading with error handling
try:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    scaler = joblib.load("saved_models/feature_scaler.joblib")
    input_dim = 9
    output_dim = 6
    model = DeepFusionNN(input_dim, output_dim)
    model.load_state_dict(
        torch.load(
            "saved_models/best_DeepFusionNN.pth",
            map_location=device,
            weights_only=True
        )
    )
    model = model.to(device)
    model.eval()
    logger.info(f"Model loaded on device: {device}")
except Exception as e:
    logger.error(f"Model or scaler loading failed: {e}")
    raise RuntimeError(f"Model or scaler loading failed: {e}")

class LocationInput(BaseModel):
    latitude: float = Field(..., description="Latitude (-90 to 90)", example=19.076090)
    longitude: float = Field(..., description="Longitude (-180 to 180)", example=72.877426)

    @validator("latitude")
    def latitude_range(cls, v):
        if not -90 <= v <= 90:
            raise ValueError("Latitude must be between -90 and 90")
        return v

    @validator("longitude")
    def longitude_range(cls, v):
        if not -180 <= v <= 180:
            raise ValueError("Longitude must be between -180 and 180")
        return v

def get_nasa_power_data(lat: float, lon: float, days: int = 30):
    """Fetch 30-day weather data from NASA POWER API"""
    end_date = datetime.date.today() - datetime.timedelta(days=1)
    start_date = end_date - datetime.timedelta(days=days-1)
    url = f"https://power.larc.nasa.gov/api/projection/daily/point?start={start_date.strftime('%Y%m%d')}&end={end_date.strftime('%Y%m%d')}&latitude={lat}&longitude={lon}&community=ag&parameters=T2M%2CRH2M%2CT2M_MAX%2CT2M_MIN&format=json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()['properties']['parameter']
        dates = sorted(data['T2M'].keys())
        return {
            'current_temp': data['T2M'][dates[-1]],
            'current_humidity': data['RH2M'][dates[-1]],
            'tmax_history': [data['T2M_MAX'][d] for d in dates],
            'tmin_history': [data['T2M_MIN'][d] for d in dates]
        }
    except Exception as e:
        logger.error(f"NASA POWER API error: {e}")
        raise HTTPException(status_code=502, detail=f"NASA POWER API error: {str(e)}")

def calculate_gdd(tmax_list, tmin_list, base_temp=10):
    """Growing Degree Days (cumulative)"""
    return sum(
        max(((tmax + tmin) / 2 - base_temp), 0)
        for tmax, tmin in zip(tmax_list, tmin_list)
    )

def calculate_thermal_time(tmax, tmin, base_temp=10):
    """Daily Thermal Time (single day)"""
    return max(((tmax + tmin) / 2 - base_temp), 0)

def generate_moisture_stress(humidity):
    """Dummy function for moisture stress; replace with your logic."""
    return max(0, min(1, 1 - humidity / 100))

@app.post("/predict")
async def predict_yield(location: LocationInput):
    try:
        logger.info(f"Received coordinates: lat={location.latitude}, lon={location.longitude}")

        # Get NASA POWER data
        nasa_data = get_nasa_power_data(location.latitude, location.longitude)

        # Input validation for missing NASA data
        if nasa_data['current_temp'] == -999.0 or nasa_data['current_humidity'] == -999.0:
            raise HTTPException(status_code=400, detail="No weather data available for these coordinates.")

        # Calculate agricultural parameters
        gdd = calculate_gdd(nasa_data['tmax_history'], nasa_data['tmin_history'])
        daily_thermal_time = calculate_thermal_time(
            nasa_data['tmax_history'][-1],
            nasa_data['tmin_history'][-1]
        )
        moisture_stress = generate_moisture_stress(nasa_data['current_humidity'])

        # Use fixed NDVI/VCI for demo (replace with real data if available)
        ndvi = 0.5
        vci = 50.0

        # Prepare feature vector
        features = np.array([
            nasa_data['current_temp'],
            nasa_data['current_humidity'],
            ndvi,
            vci,
            gdd,
            moisture_stress,
            daily_thermal_time,
            location.latitude,
            location.longitude
        ]).reshape(1, -1)

        # Preprocess and predict
        scaled_features = scaler.transform(features)
        input_tensor = torch.FloatTensor(scaled_features).to(device)
        with torch.no_grad():
            prediction = model(input_tensor).cpu().numpy()

        # Define crop labels
        crop_labels = [
            "rice_yield",
            "wheat_yield",
            "pearl_millet_yield",
            "maize_yield",
            "finger_millet_yield",
            "barley_yield"
        ]

        # Create and sort predictions
        prediction_dict = dict(zip(crop_labels, prediction[0]))
        sorted_predictions = dict(
            sorted(prediction_dict.items(), key=lambda item: item[1], reverse=True)
        )

        # Convert all NumPy types to native Python types
        return {
            "predictions": {k: float(v) for k, v in sorted_predictions.items()},
            "features": {
                "temperature": float(nasa_data['current_temp']),
                "humidity": float(nasa_data['current_humidity']),
                "ndvi": float(ndvi),
                "vci": float(vci),
                "gdd": float(gdd),
                "moisture_stress": float(moisture_stress),
                "daily_thermal_time": float(daily_thermal_time),
                "latitude": float(location.latitude),
                "longitude": float(location.longitude)
            }
        }

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import nest_asyncio
    import uvicorn
    
    nest_asyncio.apply()  # Allow nested event loops
    uvicorn.run(app, host="0.0.0.0", port=8000)
