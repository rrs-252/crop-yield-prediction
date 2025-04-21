from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import numpy as np
import joblib
import torch
import datetime
from models import DeepFusionNN

app = FastAPI()

# Load models and scaler
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load scaler
scaler = joblib.load("saved_models/feature_scaler.joblib")

# Initialize model architecture
input_dim = 9  # Update with your actual input dimension
output_dim = 6  # Update with your actual output dimension
model = DeepFusionNN(input_dim, output_dim)

# Load trained weights
model.load_state_dict(
    torch.load(
        "saved_models/best_DeepFusionNN.pth",
        map_location=device,
        weights_only=True  # Security best practice
    )
)

# Set evaluation mode
model = model.to(device)
model.eval()

class LocationInput(BaseModel):
    latitude: float
    longitude: float

def get_nasa_power_data(lat: float, lon: float, days: int = 30):
    """Fetch 30-day weather data from NASA POWER API"""
    end_date = datetime.date.today() - datetime.timedelta(days=1)
    start_date = end_date - datetime.timedelta(days=days-1)

    url = (
        f"https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters=T2M,RH2M,T2M_MAX,T2M_MIN"
        f"&community=AG"
        f"&longitude={lon}"
        f"&latitude={lat}"
        f"&start={start_date.strftime('%Y%m%d')}"
        f"&end={end_date.strftime('%Y%m%d')}"
        f"&format=JSON"
    )

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

def calculate_gdd(tmax_list, tmin_list, base_temp=10):
    """Growing Degree Days (cumulative)"""
    return sum(
        max(((tmax + tmin) / 2 - base_temp), 0)
        for tmax, tmin in zip(tmax_list, tmin_list)
    )

def calculate_thermal_time(tmax, tmin, base_temp=10):
    """Daily Thermal Time (single day)"""
    return max(((tmax + tmin) / 2 - base_temp), 0)

@app.post("/predict")
async def predict_yield(location: LocationInput):
    try:
        # Get NASA POWER data
        nasa_data = get_nasa_power_data(location.latitude, location.longitude)
        
        # Calculate agricultural parameters
        gdd = calculate_gdd(nasa_data['tmax_history'], nasa_data['tmin_history'])
        daily_thermal_time = calculate_thermal_time(
            nasa_data['tmax_history'][-1],  # Latest day's Tmax
            nasa_data['tmin_history'][-1]   # Latest day's Tmin
        )
        moisture_stress = generate_moisture_stress(nasa_data['current_humidity'])

        # Prepare feature vector (matches original model training format)
        features = np.array([
            nasa_data['current_temp'],      # T2M
            nasa_data['current_humidity'],  # RH2M
            np.random.uniform(0, 1),        # NDVI (simulated)
            np.random.uniform(0, 100),      # VCI (simulated)
            gdd,                            # Cumulative GDD
            moisture_stress,
            daily_thermal_time,             # Daily Thermal Time
            location.latitude,
            location.longitude
        ]).reshape(1, -1)

        # Preprocess and predict
        scaled_features = scaler.transform(features)
        with torch.no_grad():
            prediction = model(torch.FloatTensor(scaled_features)).numpy()

        return {
            "prediction": prediction.tolist(),
            "features": {
                "temperature": nasa_data['current_temp'],
                "humidity": nasa_data['current_humidity'],
                "ndvi": features[0][2],
                "vci": features[0][3],
                "gdd": float(gdd),
                "moisture_stress": float(moisture_stress),
                "daily_thermal_time": float(daily_thermal_time),
                "latitude": location.latitude,
                "longitude": location.longitude
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import nest_asyncio
    import uvicorn
    
    nest_asyncio.apply()  # Allow nested event loops
    uvicorn.run(app, host="0.0.0.0", port=8000)
