import tensorflow as tf
import numpy as np
from src.data_pipeline.geospatial import GeoLocator

class YieldPredictor:
    def __init__(self, model_path: str = 'model.keras'):
        self.model = tf.keras.models.load_model(model_path)
        self.geo_data = pd.read_parquet('data/processed')
        self.geo_locator = GeoLocator(self.geo_data)
        
    def predict(self, lat: float, lon: float, crop: str) -> float:
        """Predict crop yield for given location"""
        try:
            # Find nearest district
            district = self.geo_locator.find_nearest_district(lat, lon)
            
            # Get district features
            features = self.geo_data[self.geo_data['district'] == district].iloc[0]
            
            # Prepare model inputs
            inputs = {
                'weather': np.array([[features['gdd'], features['precip'], features['solar_rad']]]),
                'district': np.array([features['district_id']]),
                'crop': np.array([self._crop_to_index(crop)])
            }
            
            return self.model.predict(inputs)[0][0]
            
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def _crop_to_index(self, crop_name: str) -> int:
        """Map crop name to model index"""
        return {
            'rice': 0, 'wheat': 1, 'maize': 2,
            'pearl_millet': 3, 'finger_millet': 4, 'barley': 5
        }[crop_name.lower()]

