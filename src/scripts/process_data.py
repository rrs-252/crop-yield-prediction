import pandas as pd
import pyarrow as pq
from src.data_pipeline import NASAPowerClient, GeoLocator

def main():
    """Process raw data into training format"""
    # Load raw data
    crop_data = pd.read_csv('data/raw/crop_yield.csv')
    geo_data = pd.read_csv('data/raw/district_coordinates.csv')
    
    # Initialize clients
    power_client = NASAPowerClient()
    geo_locator = GeoLocator(geo_data)
    
    # Process data
    records = []
    for _, row in crop_data.iterrows():
        try:
            # Get coordinates
            district = geo_locator.find_nearest_district(row['lat'], row['lon'])
            
            # Get climate data
            climate = power_client.fetch_climate_data(
                row['lat'], row['lon'],
                start=f"{row['year']}0101",
                end=f"{row['year']}1231"
            )
            
            records.append({
                'year': row['year'],
                'lat': row['lat'],
                'lon': row['lon'],
                'district': district,
                **climate,
                'crop': row['crop'],
                'yield': row['yield']
            })
            
        except Exception as e:
            print(f"Skipping {row}: {str(e)}")
    
    # Save processed data
    df = pd.DataFrame(records)
    pq.write_to_dataset(
        df,
        root_path='data/processed',
        partition_cols=['year', 'crop']
    )

if __name__ == "__main__":
    main()
