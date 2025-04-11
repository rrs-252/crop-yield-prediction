import torch
import numpy as np
from sklearn.neighbors import BallTree

class YieldPredictor:
    def __init__(self, model_path='best_model.pth', data_path='data/processed'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = ClimateDataset(data_path)
        
        # Load model
        self.model = DeepFusionModel(num_districts=len(self.dataset.district_map))
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()
        
        # Spatial index
        self.tree = BallTree(np.deg2rad(self.dataset.df[['lat','lon']].values), metric='haversine')

    def predict(self, lat: float, lon: float, crop: str):
        # Find nearest district
        dist, idx = self.tree.query(np.deg2rad([[lat, lon]]), k=1)
        if dist[0][0] * 6371 > 50:
            raise ValueError("No district within 50km radius")
        
        district_data = self.dataset.df.iloc[idx[0][0]]
        
        # Prepare inputs
        inputs = {
            'climate': torch.tensor([
                district_data['gdd'],
                district_data['precip'],
                district_data['solar_rad']
            ], dtype=torch.float32).to(self.device),
            'district': torch.tensor(
                self.dataset.district_map[district_data['district']], 
                dtype=torch.long
            ).to(self.device),
            'crop': torch.tensor(
                self.dataset.crop_map[crop.lower()],
                dtype=torch.long
            ).to(self.device)
        }
        
        with torch.no_grad():
            return self.model(inputs).item()
