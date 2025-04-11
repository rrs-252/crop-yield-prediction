import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree

class GeoLocator:
    def __init__(self, geo_data: pd.DataFrame):
        self.data = geo_data
        self.tree = self._build_spatial_index()
    
    def _build_spatial_index(self) -> BallTree:
        """Create optimized spatial index"""
        coords = np.deg2rad(self.data[['lat', 'lon']].values)
        return BallTree(coords, metric='haversine', leaf_size=40)
    
    def find_nearest_district(self, lat: float, lon: float, max_km: float = 50) -> str:
        """Find closest district within specified radius"""
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            raise ValueError("Invalid coordinates")
            
        dist_rad, idx = self.tree.query(
            np.deg2rad([[lat, lon]]), 
            k=1,
            return_distance=True
        )
        
        distance_km = dist_rad[0][0] * 6371  # Convert to kilometers
        if distance_km > max_km:
            raise ValueError(f"No district within {max_km}km")
            
        return self.data.iloc[idx[0][0]]['district']
