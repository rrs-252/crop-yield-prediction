import numpy as np
import pandas as pd
from .data_pipeline.geospatial import GeoLocator
from sklearn.neighbors import BallTree

class YieldPredictor:
    def __init__(self, model_path='model.keras'):
        self.model = tf.keras.models.load_model(model_path)
        self.geolocator = GeoLocator('data/raw/geo_data.csv')
        self.ndvi_vci = pd.read_csv('data/external/ndvi_vci.csv')
        
        # Load district metadata
        self.districts = pd.read_parquet('data/processed/').drop_duplicates(
            ['district', 'state']
        )
        
        # Build spatial index
        self.tree = BallTree(
            np.deg2rad(self.districts[['lat', 'lon']].values),
            metric='haversine'
        )
    
    def predict(self, lat: float, lon: float, crop: str, year: int):
        # Find nearest district
        _, idx = self.tree.query(np.deg2rad([[lat, lon]]), k=1)
        district = self.districts.iloc[idx[0][0]]
        
        # Get NDVI/VCI
        vegetation = self.ndvi_vci[
            (self.ndvi_vci['State'] == district['state']) &
            (self.ndvi_vci['Year'] == year)
        ][['NDVI', 'VCI (%)']].mean().values
        
        # Prepare inputs
        inputs = {
            'temporal': np.array([[
                district['gdd'],
                district['precip'],
                district['solar_rad'],
                vegetation[0],
                vegetation[1]
            ]]),
            'vegetation': vegetation.reshape(1, -1),
            'district': [district['district_id']],
            'crop': [self._crop_to_index(crop)]
        }
        
        return self.model.predict(inputs)[0][0]
    
    def _crop_to_index(self, crop_name):
        crop_mapping = {
            'rice': 0, 'wheat': 1, 'maize': 2,
            'pearl_millet': 3, 'finger_millet': 4, 'barley': 5
        }
        return crop_mapping[crop_name.lower()]
