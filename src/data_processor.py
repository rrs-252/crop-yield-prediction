# data_processor.py
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from geopy.geocoders import Nominatim
import requests
import logging
from concurrent.futures import ThreadPoolExecutor
import numpy as np

class AgriDataProcessor:
    def __init__(self, csv_path):
        self.df = self._load_data(csv_path)
        self.geolocator = Nominatim(user_agent="agri_analytics_v1")
        self.crops = ['RICE', 'WHEAT', 'MAIZE', 'PEARL MILLET', 
                     'FINGER MILLET', 'BARLEY']
        
    def _load_data(self, path):
        df = pd.read_csv(path)
        required_cols = ['Year', 'State Name', 'Dist Name']
        for crop in self.crops:
            required_cols += [f'{crop} YIELD (Kg per ha)']
        return df.dropna(subset=required_cols)

    def _get_coordinates(self, district, state):
        try:
            location = self.geolocator.geocode(f"{district}, {state}, India")
            return (location.latitude, location.longitude) if location else (None, None)
        except Exception as e:
            logging.error(f"Geocoding error: {e}")
            return (None, None)

    def _fetch_climate_data(self, lat, lon, year):
        if None in [lat, lon]: return {}
        try:
            url = "https://power.larc.nasa.gov/api/temporal/daily/point"
            params = {
                'parameters': 'T2M,PRECTOTCORR,ALLSKY_SFC_SW_DWN',
                'start': f"{year}0101",
                'end': f"{year}1231",
                'latitude': lat,
                'longitude': lon,
                'format': 'JSON'
            }
            response = requests.get(url, params=params, timeout=30)
            return self._process_climate(response.json())
        except Exception as e:
            logging.error(f"Climate API error: {e}")
            return {}

    def _process_climate(self, data):
        params = data['properties']['parameter']
        return {
            'gdd': self._calculate_gdd(params['T2M']),
            'precip': sum(params['PRECTOTCORR'].values()),
            'solar_rad': np.mean(list(params['ALLSKY_SFC_SW_DWN'].values()))
        }

    def _calculate_gdd(self, temp_data, base=10):
        return sum(max((v['max'] + v['min'])/2 - base, 0) 
                 for v in temp_data.values())

    def _fetch_soil_data(self, lat, lon):
        try:
            response = requests.get(f"https://rest.soilgrids.org/query?lat={lat}&lon={lon}")
            return {
                'ph': response.json()['properties']['phh2o']['0-5cm']['mean'],
                'organic_carbon': response.json()['properties']['oc']['0-5cm']['mean']
            }
        except Exception as e:
            logging.error(f"Soil API error: {e}")
            return {}

    def process_row(self, row):
        features = {}
        lat, lon = self._get_coordinates(row['Dist Name'], row['State Name'])
        climate = self._fetch_climate_data(lat, lon, row['Year'])
        soil = self._fetch_soil_data(lat, lon)
        
        features.update({
            'year': row['Year'],
            'lat': lat,
            'lon': lon,
            **climate,
            **soil
        })
        
        # Create crop-specific entries
        for crop in self.crops:
            yield {
                **features,
                'crop': crop.lower(),
                'yield': row[f'{crop} YIELD (Kg per ha)']
            }
            
    def process_dataset(self, output_path="processed_data"):
        records = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = executor.map(lambda row: list(self.process_row(row)), 
                                 self.df.iterrows())
            for crop_records in results:
                records.extend(crop_records)
                
        df = pd.DataFrame(records).dropna()
        table = pa.Table.from_pandas(df)
        pq.write_to_dataset(table, root_path=output_path, 
                          partition_cols=['year', 'crop'])

    
    # Recommended data_processor.py enhancement
    def integrate_soil_data(geo_df, soil_df):
    """Map state-level soil data to districts using weighted averages"""
    # Calculate weights based on district count per state
    state_weights = geo_df['State Name'].value_counts(normalize=True)
    
    merged = geo_df.merge(
        soil_df,
        left_on='State Name',
        right_on='State',
        how='left'
    )
    
    # Distribute state-level values proportionally
    soil_params = ['Nitrogen', 'Phosphorous', 'OC']
    for param in soil_params:
        merged[f'{param}_district'] = merged[f'{param} - Medium'] * state_weights[merged['State Name']]
    
    return merged[['District Name', 'Latitude', 'Longitude', 'Nitrogen_district', 
                  'Phosphorous_district', 'OC_district']]

