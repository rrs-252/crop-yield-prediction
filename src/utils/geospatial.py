import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree

class GeoLocator:
    def __init__(self, data_path: str):
        self.data = pd.read_csv(data_path)
        self._build_spatial_index()
        
    def _build_spatial_index(self):
        """Create BallTree index for fast spatial queries"""
        coords = np.deg2rad(self.data[['Latitude', 'Longitude']].values)
        self.tree = BallTree(coords, metric='haversine')
        
    def district_from_coords(self, lat: float, lon: float) -> str:
        """Get district name from coordinates"""
        dist, idx = self.tree.query(np.deg2rad([[lat, lon]]), k=1)
        return self.data.iloc[idx[0][0]]['District Name']
    
    def coords_from_district(self, district: str) -> tuple:
        """Get coordinates from district name (case-insensitive partial match)"""
        match = self.data[
            self.data['District Name'].str.contains(district, case=False, na=False)
        ]
        if not match.empty:
            row = match.iloc[0]
            return (row['Latitude'], row['Longitude'])
        raise ValueError(f"District '{district}' not found")
