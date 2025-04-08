import numpy as np
import pandas as pd
from geopy.distance import geodesic
from sklearn.neighbors import BallTree

class YieldPredictor:
    def __init__(self, model_path='model.keras', data_path='processed_data'):
        self.model = tf.keras.models.load_model(model_path)
        self.district_data = self._load_district_coordinates(data_path)
        self.tree = self._build_spatial_index()
        
    def _load_district_coordinates(self, data_path):
        """Load processed data with district coordinates and features"""
        df = pd.read_parquet(data_path)
        return df.groupby(['district', 'state']).agg({
            'lat': 'mean',
            'lon': 'mean',
            'gdd': 'mean',
            'precip': 'mean',
            'ph': 'mean',
            'organic_carbon': 'mean'
        }).reset_index()

    def _build_spatial_index(self):
        """Create BallTree for fast spatial queries"""
        coords = np.deg2rad(self.district_data[['lat', 'lon']].values)
        return BallTree(coords, metric='haversine')

    def _find_nearest_district(self, lat, lon):
        """Find closest district within 50km radius"""
        query_point = np.deg2rad(np.array([[lat, lon]]))
        dist, idx = self.tree.query(query_point, k=1)
        distance_km = dist[0][0] * 6371  # Convert to kilometers
        
        if distance_km > 50:
            raise ValueError("No district found within 50km radius")
            
        return self.district_data.iloc[idx[0][0]]

    def predict(self, lat, lon, crop_type):
        """Predict yield for given coordinates and crop"""
        district = self._find_nearest_district(lat, lon)
        
        inputs = {
            'weather': np.array([[
                district['gdd'],
                district['precip'],
                district.get('solar_rad', 0)  # Handle missing feature
            ]]),
            'district': [hash(f"{district['district']}{district['state']}") % 1000],
            'soil': [[district['ph'], district['organic_carbon']]],
            'crop': [self._crop_to_index(crop_type)]
        }
        
        return self.model.predict(inputs)[0][0]

    def _crop_to_index(self, crop_name):
        crop_mapping = {
            'rice': 0, 'wheat': 1, 'maize': 2,
            'pearl_millet': 3, 'finger_millet': 4, 'barley': 5
        }
        return crop_mapping[crop_name.lower()]
