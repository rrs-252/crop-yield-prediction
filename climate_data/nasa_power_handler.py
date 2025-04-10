import requests
import pandas as pd
from retrying import retry

class NASAPowerClient:
    BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
    
    def __init__(self, geo_data_path):
        self.geo_data = pd.read_csv(geo_data_path)
    
    @retry(stop_max_attempt_number=3, wait_fixed=2000)
    def fetch_climate_data(self, lat, lon, start="20080101", end="20171231"):
        """Fetch climate data for given coordinates"""
        params = {
            "parameters": "T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M",
            "community": "AG",
            "latitude": lat,
            "longitude": lon,
            "start": start,
            "end": end,
            "format": "JSON"
        }
        response = requests.get(self.BASE_URL, params=params)
        response.raise_for_status()
        return self._process_response(response.json())
    
    def _process_response(self, data):
        """Convert API response to DataFrame"""
        df = pd.DataFrame(data["properties"]["parameter"])
        df = df.rename(columns={
            "T2M": "avg_temp",
            "T2M_MAX": "max_temp",
            "T2M_MIN": "min_temp",
            "PRECTOTCORR": "precipitation",
            "RH2M": "humidity"
        })
        return df.mean().to_dict()
    
    def fetch_all_districts(self):
        """Fetch climate data for all districts in geo_data"""
        results = []
        for _, row in self.geo_data.iterrows():
            try:
                climate_data = self.fetch_climate_data(row["Latitude"], row["Longitude"])
                results.append({
                    "District Name": row["District Name"],
                    **climate_data
                })
            except Exception as e:
                print(f"Failed to fetch data for {row['District Name']}: {str(e)}")
                results.append({
                    "District Name": row["District Name"],
                    "avg_temp": None,
                    "max_temp": None,
                    "min_temp": None,
                    "precipitation": None,
                    "humidity": None
                })
        return pd.DataFrame(results)
