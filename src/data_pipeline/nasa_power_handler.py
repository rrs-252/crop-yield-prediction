import requests
import pandas as pd
from retrying import retry

class NASAPowerClient:
    BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
    
    def __init__(self):
        self.session = requests.Session()
    
    @retry(stop_max_attempt_number=3, wait_fixed=2000)
    def fetch_climate_data(self, lat: float, lon: float, start: str, end: str) -> dict:
        """Fetch climate data from NASA POWER API"""
        params = {
            "parameters": "T2M,PRECTOTCORR,ALLSKY_SFC_SW_DWN",
            "latitude": lat,
            "longitude": lon,
            "start": start,
            "end": end,
            "format": "JSON"
        }
        
        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            return self._process_response(response.json())
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"API request failed: {str(e)}")

    def _process_response(self, data: dict) -> dict:
        """Extract and transform API response"""
        params = data['properties']['parameter']
        return {
            'gdd': self._calculate_gdd(params['T2M']),
            'precip': sum(params['PRECTOTCORR'].values()),
            'solar_rad': sum(params['ALLSKY_SFC_SW_DWN'].values()) / len(params['ALLSKY_SFC_SW_DWN'])
        }

    def _calculate_gdd(self, temp_data: dict, base_temp: float = 10.0) -> float:
        """Calculate Growing Degree Days"""
        return sum(
            (daily['max'] + daily['min']) / 2 - base_temp
            for daily in temp_data.values()
            if (daily['max'] + daily['min']) / 2 > base_temp
        )
