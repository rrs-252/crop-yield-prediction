import pandas as pd
import numpy as np
from tqdm import tqdm
from Climate.climate_processor import ClimateProcessor
from Soil.soil_processor import SoilProcessor
from Crop.crop_processor import CropProcessor

class DataPreprocessor:
    def __init__(self):
        self.climate_processor = ClimateProcessor()
        self.soil_processor = SoilProcessor()
        self.crop_processor = CropProcessor()
        self.grid = self._generate_india_grid()
        
    def _generate_india_grid(self, res=0.5):
        """Generate a 0.5° spatial grid covering India."""
        return [(lat, lon) 
                for lat in np.arange(8.0, 37.5, res)
                for lon in np.arange(68.0, 97.5, res)]
    
    def create_dataset(self, crops=['wheat', 'rice'], years=range(2010, 2021)):
        """Build spatial dataset combining soil, climate, and crop data."""
        records = []
        
        for lat, lon in tqdm(self.grid, desc="Processing grid points"):
            try:
                # Extract soil properties
                soil = self.soil_processor.get_soil_properties(lat, lon)
                if not soil:
                    continue
                
                # Extract climate properties
                climate = self.climate_processor.get_climate_data(lat, lon)
                if not climate:
                    continue
                
                # Process crop yield data
                for crop in crops:
                    yields = [self.crop_processor.get_crop_yield(lat, lon, crop, year) 
                              for year in years]
                    valid_yields = [y for y in yields if not np.isnan(y)]
                    
                    if valid_yields:
                        records.append({
                            'lat': lat,
                            'lon': lon,
                            **soil,
                            **climate,
                            'crop': crop,
                            'yield': np.mean(valid_yields)
                        })
            except Exception as e:
                print(f"Error at ({lat}, {lon}): {str(e)}")
        
        # Save dataset as Parquet file
        df = pd.DataFrame(records)
        df.to_parquet("data/processed_dataset.parquet", index=False)
        print("Dataset saved to data/processed_dataset.parquet")

# Example usage:
# DataPreprocessor().create_dataset()
