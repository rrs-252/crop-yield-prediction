# data_processor.py
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from .nasa_power_handler import NASAPowerClient
from .geospatial import GeoLocator

class AgriDataProcessor:
    def __init__(self, crop_csv, ndvi_csv, geo_csv):
        self.crop_df = pd.read_csv(crop_csv)
        self.ndvi_vci = pd.read_csv(ndvi_csv)
        self.geo_data = pd.read_csv(geo_csv)
        self.power_client = NASAPowerClient()
        self.geolocator = GeoLocator(self.geo_data)
        
    def _get_climate_features(self, lat: float, lon: float) -> pd.DataFrame:
        """Get 10-year climate normals"""
        climate_df = self.power_client.fetch_historical_data(lat, lon)
        return climate_df.mean().to_dict()
    
    def _get_vegetation_features(self, state: str) -> dict:
        """Get NDVI/VCI averages"""
        return self.ndvi_vci[self.ndvi_vci['State'] == state][['NDVI', 'VCI (%)']].mean()
    
    def process_district(self, district_row):
        try:
            lat = district_row['Latitude']
            lon = district_row['Longitude']
            
            return {
                'district': district_row['District Name'],
                'state': district_row['State Name'],
                'lat': lat,
                'lon': lon,
                **self._get_climate_features(lat, lon),
                **self._get_vegetation_features(district_row['State Name'])
            }
        except Exception as e:
            logging.error(f"Failed {district_row['District Name']}: {str(e)}")
            return None

    def process_dataset(self, output_path="data/processed"):
        """Main processing workflow"""
        features = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(self.process_district, row) 
                      for _, row in tqdm(self.geo_data.iterrows(), total=len(self.geo_data))]
            for future in futures:
                if (result := future.result()) is not None:
                    features.append(result)
        
        # Merge with crop data
        full_df = pd.merge(
            pd.DataFrame(features),
            self.crop_df,
            left_on='district',
            right_on='Dist Name'
        ).dropna()
        
        # Save partitioned dataset
        pq.write_to_dataset(
            pa.Table.from_pandas(full_df),
            root_path=output_path,
            partition_cols=['year', 'crop']
        )

class ClimateDataset(torch.utils.data.Dataset):
    def __init__(self, parquet_path):
        self.df = pd.read_parquet(parquet_path)
        self.district_map = {d: i for i, d in enumerate(self.df['district'].unique())}
        self.crop_map = {
            'rice': 0, 'wheat': 1, 'maize': 2,
            'pearl_millet': 3, 'finger_millet': 4, 'barley': 5
        }
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            'climate': torch.tensor([row['gdd'], row['precip'], row['solar_rad']], dtype=torch.float32),
            'district': torch.tensor(self.district_map[row['district']], dtype=torch.long),
            'crop': torch.tensor(self.crop_map[row['crop']], dtype=torch.long),
            'yield': torch.tensor(row['yield'], dtype=torch.float32)
        }
