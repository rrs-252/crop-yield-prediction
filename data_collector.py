import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional
from config import Config

class DataCollector:
    def __init__(self, api_key: str, config: Config):
        self.api_key = api_key
        self.config = config
        
    def create_field_polygon(self, lat: float, lon: float, field_size: float = 0.01) -> Optional[str]:
        """Create a polygon for field monitoring"""
        polygon_data = {
            "name": f"Field_{lat}_{lon}",
            "geo_json": {
                "type": "Feature",
                "properties":{},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [round(lon - field_size, 6), round(lat - field_size, 6)],
                        [round(lon + field_size, 6), round(lat - field_size, 6)],
                        [round(lon + field_size, 6), round(lat + field_size, 6)],
                        [round(lon - field_size, 6), round(lat + field_size, 6)],
                        [round(lon - field_size, 6), round(lat - field_size, 6)]
                    ]]
                }
            }
        }

        try:
            response = requests.post(
                f"{self.config.AGRO_API_BASE_URL}/polygons?appid={self.api_key}",
                json=polygon_data,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json().get('id')
        except requests.RequestException as e:
            print(f"Error creating polygon: {e}")
            print(f"Response: {response.content}")
            return None

    def collect_weather_data(self, polygon_id: str, start_date: datetime, 
                           end_date: datetime) -> pd.DataFrame:
        """Collect historical weather data"""
        params = {
            "polyid": polygon_id,
            "appid": self.api_key,
            "start": int(start_date.timestamp()),
            "end": int(end_date.timestamp())
        }
        
        weather_data = self._collect_multiple_weather_metrics(params)
        return pd.DataFrame(weather_data)
    
    def collect_soil_data(self, polygon_id: str) -> pd.DataFrame:
        """Collect soil data from multiple depths"""
        try:
            response = requests.get(
                f"{self.config.AGRO_API_BASE_URL}/soil?polyid={polygon_id}&appid={self.api_key}"
            )
            response.raise_for_status()
            return pd.DataFrame([response.json()])
        except requests.RequestException as e:
            print(f"Error fetching soil data: {e}")
            return pd.DataFrame()

    def collect_satellite_data(self, polygon_id: str) -> pd.DataFrame:
        """Collect vegetation indices from satellite data"""
        params = {
            "polyid": polygon_id,
            "appid": self.api_key
        }
        
        indices = ["ndvi", "evi", "lai"]
        satellite_data = {}
        
        for index in indices:
            try:
                response = requests.get(
                    f"{self.config.AGRO_API_BASE_URL}/satellite/indices/{index}",
                    params=params
                )
                response.raise_for_status()
                satellite_data[index] = response.json()
            except requests.RequestException as e:
                print(f"Error fetching {index} data: {e}")
                satellite_data[index] = None
                
        return pd.DataFrame(satellite_data)
    
    def _collect_multiple_weather_metrics(self, params: Dict) -> Dict:
        """Collect multiple weather metrics"""
        metrics = {
            "temperature": "accumulated_temperature",
            "precipitation": "accumulated_precipitation",
            "humidity": "accumulated_humidity"
        }
        
        weather_data = {}
        for metric, endpoint in metrics.items():
            try:
                weather_data[metric] = self._fetch_weather_metric(params, endpoint)
            except requests.RequestException as e:
                print(f"Error fetching {metric} data: {e}")
                weather_data[metric] = None
                
        return weather_data
    
    def _fetch_weather_metric(self, params: Dict, endpoint: str) -> Dict:
        """Fetch a specific weather metric"""
        url = f"{self.config.AGRO_API_BASE_URL}/weather/history/{endpoint}"
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
