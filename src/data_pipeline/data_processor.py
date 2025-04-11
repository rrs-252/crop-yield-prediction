# data_processor.py
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import logging
from concurrent.futures import ThreadPoolExecutor
from .nasa_power_handler import NASAPowerClient
from .geospatial import GeoLocator

class AgriDataProcessor:
    def __init__(self, crop_csv, ndvi_csv, geo_csv):
        self.crop_df = pd.read_csv(crop_csv)
        self.ndvi_vci = pd.read_csv(ndvi_csv)
        self.geo_data = pd.read_csv(geo_csv)
        self.power_client = NASAPowerClient(geo_csv)
        self.geolocator = GeoLocator(geo_csv)
        
    def _merge_ndvi_vci(self, state, year):
        filtered = self.ndvi_vci[
            (self.ndvi_vci['State'] == state) &
            (self.ndvi_vci['Year'] == year)
        ]
        return filtered[['NDVI', 'VCI (%)']].mean().to_dict()
    
    def process_row(self, row):
        district = self.geolocator.district_from_coords(row['Latitude'], row['Longitude'])
        state = self.geo_data[self.geo_data['District Name'] == district]['State Name'].iloc[0]
        
        climate = self.power_client.fetch_climate_data(
            row['Latitude'], 
            row['Longitude'],
            start=f"{row['Year']}0101",
            end=f"{row['Year']}1231"
        )
        
        vegetation = self._merge_ndvi_vci(state, row['Year'])
        
        return {
            'year': row['Year'],
            'lat': row['Latitude'],
            'lon': row['Longitude'],
            'district': district,
            'state': state,
            **climate,
            **vegetation,
            'crop': row['Crop'].lower(),
            'yield': row['Yield']
        }
    
    def process_dataset(self, output_path="data/processed"):
        records = []
        merged = self.crop_df.merge(
            self.geo_data,
            on=['State Name', 'District Name']
        )
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(self.process_row, row) 
                      for _, row in merged.iterrows()]
            for future in futures:
                records.append(future.result())
                
        df = pd.DataFrame(records).dropna()
        table = pa.Table.from_pandas(df)
        pq.write_to_dataset(
            table,
            root_path=output_path,
            partition_cols=['year', 'crop']
        )
