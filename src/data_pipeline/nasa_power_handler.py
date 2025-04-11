import requests
import pandas as pd
from retrying import retry
import logging

class NASAPowerClient:
    BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
    
    def __init__(self):
        self.session = requests.Session()
    
    @retry(stop_max_attempt_number=5, wait_exponential_multiplier=2000)
    def fetch_year_data(self, lat: float, lon: float, year: int) -> dict:
        """Fetch climate data for a specific year"""
        params = {
            "parameters": "T2M,PRECTOTCORR,ALLSKY_SFC_SW_DWN",
            "latitude": lat,
            "longitude": lon,
            "start": f"{year}0101",
            "end": f"{year}1231",
            "format": "JSON"
        }
        
        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            return self._process_year_data(response.json(), year)
        except requests.exceptions.RequestException as e:
            logging.error(f"API failed for {year}: {str(e)}")
            return {}

    def _process_year_data(self, data: dict, year: int) -> dict:
        """Calculate annual climate metrics"""
        params = data['properties']['parameter']
        return {
            'year': year,
            'gdd': self._calculate_gdd(params['T2M']),
            'precip': sum(params['PRECTOTCORR'].values()),
            'solar_rad': sum(params['ALLSKY_SFC_SW_DWN'].values())/365
        }

    def _calculate_gdd(self, temp_data: dict, base_temp: float = 10.0) -> float:
        """Growing Degree Days calculation"""
        return sum(
            max((daily['max'] + daily['min'])/2 - base_temp, 0)
            for daily in temp_data.values()
        )

    def fetch_historical_data(self, lat: float, lon: float) -> pd.DataFrame:
        """Get 2008-2017 climate data"""
        records = []
        for year in range(2008, 2018):
            yearly_data = self.fetch_year_data(lat, lon, year)
            if yearly_data:
                records.append(yearly_data)
        return pd.DataFrame(records)
