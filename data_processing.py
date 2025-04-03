import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import joblib

class DataProcessor:
    def __init__(self):
        self.climate_params = ["T2M", "PRECTOTCORR", "RH2M", "CDD18_3"]
        self.soil_features = ['T_OC', 'PH_H2O', 'T_CLAY', 'T_SAND', 'T_CEC', 'T_BS']
        self.crop_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        self.scaler = StandardScaler()
        self.district_coordinates = pd.read_csv('districts_coordinates.csv')  # Load district coordinates

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

    def process_crop(self, df):
        """Clean and transform crop data."""
        df = df.dropna(subset=['Production', 'Area'])
        df['Yield'] = df['Production'] / df['Area']
        df = df[df['Yield'] < df.groupby('Crop')['Yield'].transform('mean') * 3]
        
        encoded = self.crop_encoder.fit_transform(df[['State_Name', 'District_Name', 'Crop', 'Season']])
        return encoded, df['Yield']

    def get_coordinates(self, district_name):
        """Fetch latitude and longitude for a given district."""
        district_info = self.district_coordinates[self.district_coordinates['District Name'] == district_name]
        if not district_info.empty:
            return district_info.iloc[0]['Latitude'], district_info.iloc[0]['Longitude']
        else:
            raise ValueError(f"Coordinates for district '{district_name}' not found.")

    def save_preprocessors(self):
        """Save preprocessors for future use."""
        joblib.dump(self.crop_encoder, 'preprocessing/crop_encoder.joblib')
        joblib.dump(self.scaler, 'preprocessing/scaler.joblib')

    def load_preprocessors(self):
        """Load preprocessors for inference."""
        self.crop_encoder = joblib.load('preprocessing/crop_encoder.joblib')
        self.scaler = joblib.load('preprocessing/scaler.joblib')
