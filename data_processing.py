import pandas as pd
from sklearn.neighbors import BallTree
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import joblib

class DataProcessor:
    def __init__(self):
        self.climate_params = ["T2M", "PRECTOTCORR", "RH2M", "CDD18_3"]
        self.soil_features = ['T_OC', 'PH_H2O', 'T_CLAY', 'T_SAND', 'T_CEC', 'T_BS']
        self.crop_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        self.scaler = StandardScaler()
        
        # Load district coordinates
        self.district_coordinates = pd.read_csv('districts_coordinates.csv')
        self.tree = BallTree(
            np.radians(self.district_coordinates[['Latitude', 'Longitude']].values),
            metric='haversine'
        )

    def get_nearest_district(self, latitude, longitude):
        """Find the nearest district based on latitude and longitude."""
        coords = np.radians([[latitude, longitude]])
        dist, idx = self.tree.query(coords, k=1)
        nearest_district = self.district_coordinates.iloc[idx[0][0]]
        return nearest_district['District Name']

    def process_climate(self, json_data):
        """Process NASA POWER API response into seasonal features."""
        monthly_data = json_data['properties']['parameter']
        seasonal = {'Kharif': ['JUN', 'JUL', 'AUG', 'SEP'], 'Rabi': ['NOV', 'DEC', 'JAN', 'FEB']}
        return np.array([
            [np.mean([monthly_data[param][month] for month in seasonal['Kharif']]) 
             for param in self.climate_params] +
            [np.mean([monthly_data[param][month] for month in seasonal['Rabi']]) 
             for param in self.climate_params]
        ]).flatten()

    def process_soil(self, hwsd_row):
        """Process HWSD soil data with NPK estimation."""
        return np.array([
            hwsd_row['T_OC'] * 0.05,  # Nitrogen
            10 ** (-0.87 * hwsd_row['PH_H2O'] + 4.13),  # Phosphorus
            (hwsd_row['T_BS']/100 * hwsd_row['T_CEC']) * 0.1,  # Potassium
            hwsd_row['T_CLAY'],
            hwsd_row['T_SAND'],
            hwsd_row['T_OC']
        ])

    def encode_crop_info(self, crop_type):
        """Encode crop type."""
        encoded = self.crop_encoder.fit_transform([[crop_type]])
        return encoded

    def save_preprocessors(self):
        """Save preprocessors for future use."""
        joblib.dump(self.crop_encoder, 'preprocessing/crop_encoder.joblib')
        joblib.dump(self.scaler, 'preprocessing/scaler.joblib')

    def load_preprocessors(self):
        """Load preprocessors for inference."""
        self.crop_encoder = joblib.load('preprocessing/crop_encoder.joblib')
