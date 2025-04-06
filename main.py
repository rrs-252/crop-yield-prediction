import joblib
import numpy as np
from preprocessing.preprocess import DataPreprocessor

class YieldPredictor:
    def __init__(self):
        self.model = models.load_model("models/model.keras")
        self.scaler = joblib.load("models/scaler.joblib")
        self.le = joblib.load("models/label_encoder.joblib")
        self.preprocessor = DataPreprocessor()
    
    def predict(self, lat, lon, crop):
        # Get fresh data
        soil = self.preprocessor._get_soil(lat, lon)[:3]
        climate = self.preprocessor._get_climate(lat, lon)
        crop_encoded = self.le.transform([crop])[0]
        
        # Prepare inputs
        soil_scaled = self.scaler.transform([soil])
        climate_features = [climate['T2M'], climate['PRECTOTCORR'], climate['RH2M']]
        
        return self.model.predict([
            soil_scaled,
            np.array([climate_features]),
            np.array([[crop_encoded]])
        ])[0][0]

if __name__ == "__main__":
    # First-time setup
    # DataPreprocessor().create_dataset()
    # train_model()
    
    # Prediction
    predictor = YieldPredictor()
    print(predictor.predict(28.6139, 77.2090, 'wheat'))
