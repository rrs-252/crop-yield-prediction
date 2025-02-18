import pandas as pd
from config_file import Config
from data_collector import DataCollector
from data_preprocessor import DataPreprocessor
from datetime import datetime

# Initialize configurations and classes
config = Config()
data_collector = DataCollector(api_key="YOUR_API_KEY", config=config)
data_preprocessor = DataPreprocessor()

# Step 1: Create Field Polygon
polygon_id = data_collector.create_field_polygon(lat=28.6139, lon=77.2090)
if not polygon_id:
    print("Failed to create field polygon. Exiting.")
    exit()

# Step 2: Collect Weather, Soil, and Satellite Data
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 2, 1)
weather_df = data_collector.collect_weather_data(polygon_id, start_date, end_date)
soil_df = data_collector.collect_soil_data(polygon_id)
satellite_df = data_collector.collect_satellite_data(polygon_id)

# Step 3: Preprocess Data
combined_df, feature_dims = data_preprocessor.preprocess_all_data(
    weather_df, soil_df, satellite_df
)

# Step 4: Save Preprocessed Data and Feature Dimensions
combined_df.to_csv(config.DATA_DIR / 'preprocessed_data.csv', index=False)
with open(config.DATA_DIR / 'feature_dims.json', 'w') as f:
    import json
    json.dump(feature_dims, f, indent=4)

print("Data preprocessing complete!")
print("Feature dimensions:", feature_dims)
