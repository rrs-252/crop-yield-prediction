import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from functools import lru_cache

class GeoLocator:
    def __init__(self, data_path: str):
        self.data = self._load_geo_data(data_path)
        self.tree = self._build_spatial_index()
        
    @staticmethod
    @lru_cache(maxsize=1)
    def _load_geo_data(data_path: str) -> pd.DataFrame:
        """Cached dataset loader with validation"""
        df = pd.read_csv(data_path)
        required_cols = {'District Name', 'Latitude', 'Longitude'}
        if not required_cols.issubset(df.columns):
            raise ValueError("Dataset missing required columns")
        return df.dropna(subset=required_cols)

    def _build_spatial_index(self) -> BallTree:
        """Create optimized spatial index with radians conversion"""
        coords = np.deg2rad(self.data[['Latitude', 'Longitude']].values)
        return BallTree(coords, metric='haversine', leaf_size=40)

    def district_from_coords(self, lat: float, lon: float) -> str:
        """Get district with distance validation"""
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            raise ValueError("Invalid coordinates")
        
        dist, idx = self.tree.query(np.deg2rad([[lat, lon]]), k=1)
        if dist[0][0] > 0.5:  # ~55km threshold
            raise ValueError("No district within 50km radius")
            
        return self.data.iloc[idx[0][0]]['District Name']
