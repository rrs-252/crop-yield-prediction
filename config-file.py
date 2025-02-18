import json
from pathlib import Path

class Config:
    def __init__(self):
        self.BASE_DIR = Path(__file__).parent
        self.DATA_DIR = self.BASE_DIR / "data"
        self.MODELS_DIR = self.BASE_DIR / "models"
        
        # Create directories if they don't exist
        self.DATA_DIR.mkdir(exist_ok=True)
        self.MODELS_DIR.mkdir(exist_ok=True)
        
        # API Configuration
        self.AGRO_API_BASE_URL = "http://api.agromonitoring.com/agro/1.0"
        
        # Model Configuration
        self.MODEL_CONFIG = {
            "batch_size": 32,
            "epochs": 50,
            "learning_rate": 0.001,
            "hidden_dims": {
                "weather": 64,
                "soil": 32,
                "satellite": 32
            }
        }
        
        # Indian crop seasons
        self.CROP_SEASONS = {
            "rice": {
                "kharif": {"start": "06-15", "end": "11-15"},
                "rabi": {"start": "11-15", "end": "04-15"}
            },
            "wheat": {
                "rabi": {"start": "11-01", "end": "04-15"}
            }
        }
