from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from typing import Dict, Tuple

class DataPreprocessor:
    def __init__(self):
        self.scalers = {}
        
    def preprocess_all_data(self, 
                           weather_df: pd.DataFrame,
                           soil_df: pd.DataFrame,
                           satellite_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Preprocess all collected data"""
        # Preprocess each data type
        weather_processed = self.preprocess_weather_data(weather_df)
        soil_processed = self.preprocess_soil_data(soil_df)
        satellite_processed = self.preprocess_satellite_data(satellite_df)
        
        # Combine all features
        combined_df = self.combine_features(
            weather_processed,
            soil_processed,
            satellite_processed
        )
        
        # Get feature dimensions for model configuration
        feature_dims = {
            'weather': len(weather_processed.columns),
            'soil': len(soil_processed.columns),
            'satellite': len(satellite_processed.columns)
        }
        
        return combined_df, feature_dims
    
    def preprocess_weather_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess weather data"""
        df = df.copy()
        
        # Add derived features
        df['temp_variation'] = df['temperature'].rolling(window=7).std()
        df['rain_frequency'] = (df['precipitation'] > 0).rolling(window=30).mean()
        df['growing_degree_days'] = self._calculate_growing_degree_days(df['temperature'])
        
        # Handle missing values
        df = df.interpolate(method='time')
        
        return df
    
    def preprocess_soil_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess soil data"""
        df = df.copy()
        
        # Calculate soil health index
        df['soil_health_index'] = self._calculate_soil_health_index(df)
        
        # Add moisture stress indicator
        df['moisture_stress'] = (df['moisture'] < 0.2).astype(int)
        
        return df
    
    def preprocess_satellite_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess satellite data"""
        df = df.copy()
        
        # Calculate vegetation health index
        df['vhi'] = (
            df['ndvi'] * 0.5 +
            df['evi'] * 0.3 +
            df['lai'] * 0.2
        )
        
        # Add crop stress indicators
        df['vegetation_stress'] = (df['ndvi'] < 0.3).astype(int)
        
        return df
    
    def combine_features(self, 
                        weather_df: pd.DataFrame,
                        soil_df: pd.DataFrame,
                        satellite_df: pd.DataFrame) -> pd.DataFrame:
        """Combine all features into a single dataset"""
        combined_df = pd.concat([
            weather_df,
            soil_df,
            satellite_df
        ], axis=1)
        
        # Add temporal features
        combined_df['day_of_year'] = combined_df.index.dayofyear
        combined_df['month'] = combined_df.index.month
        combined_df['season'] = combined_df['month'].map(self._get_season_mapper())
        
        return combined_df
    
    def _calculate_growing_degree_days(self, temperatures: pd.Series) -> pd.Series:
        """Calculate growing degree days"""
        base_temp = 10  # Base temperature for most crops
        return np.maximum(temperatures - base_temp, 0)
    
    def _calculate_soil_health_index(self, df: pd.DataFrame) -> pd.Series:
        """Calculate soil health index"""
        return (
            df['moisture'] * 0.4 +
            df.get('ph', 7.0) * 0.3 +
            df.get('organic_matter', 2.0) * 0.3
        )
    
    def _get_season_mapper(self) -> Dict:
        """Map months to Indian agricultural seasons"""
        return {
            1: 'rabi', 2: 'rabi', 3: 'rabi',
            4: 'zaid', 5: 'zaid',
            6: 'kharif', 7: 'kharif', 8: 'kharif', 9: 'kharif', 10: 'kharif',
            11: 'rabi', 12: 'rabi'
        }
