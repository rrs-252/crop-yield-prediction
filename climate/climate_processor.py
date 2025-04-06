import requests
import numpy as np

class ClimateProcessor:
    def __init__(self):
        self.base_url = "https://power.larc.nasa.gov/api/temporal/climatology/point"
        self.params = "T2M,PRECTOTCORR,RH2M,CDD18_3"  # Temperature, Precipitation, Humidity, Growing Degree Days
        
    def get_climate_data(self, lat: float, lon: float) -> dict:
        """Fetch 20-year climate averages (2001-2020)"""
        params = {
            "latitude": lat,
            "longitude": lon,
            "parameters": self.params,
            "start": 2001,
            "end": 2020,
            "community": "AG"
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            return self._process_climate_json(response.json())
        except requests.exceptions.RequestException as e:
            print(f"Climate API Error: {str(e)}")
            return None

    def _process_climate_json(self, json_data: dict) -> dict:
        """Convert raw API response to seasonal features"""
        monthly_data = json_data['properties']['parameter']
        seasonal = {
            'Kharif': ['JUN', 'JUL', 'AUG', 'SEP'],  # June-September
            'Rabi': ['NOV', 'DEC', 'JAN', 'FEB']     # November-February
        }
        
        processed = {}
        for param in self.params.split(','):
            for season, months in seasonal.items():
                key = f"{param}_{season}"
                values = [monthly_data[param][month] for month in months]
                processed[key] = np.mean(values)
                
        return processed

# Example usage:
# climate = ClimateProcessor().get_climate_data(28.6139, 77.2090)  # Delhi coordinates
