import pandas as pd
import requests
from retrying import retry
import numpy as np
from tqdm import tqdm
import time

# Load district coordinates data
df = pd.read_csv("UnApportionedIdentifiers.csv")
print(f"Loaded {len(df)} district records")

# NASA POWER API configuration
NASA_API = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMS = {
    "parameters": "T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,ALLSKY_SFC_SW_DWN",
    "community": "AG",
    "start": "20080101",
    "end": "20171231",
    "format": "JSON"
}

@retry(stop_max_attempt_number=3, wait_fixed=2000)
def fetch_nasa_data(lat, lon):
    """Fetch climate data from NASA POWER API with retry logic"""
    try:
        response = requests.get(
            NASA_API,
            params={**PARAMS, "latitude": lat, "longitude": lon},
            timeout=30
        )
        response.raise_for_status()
        return response.json()['properties']['parameter']
    except Exception as e:
        print(f"Error fetching data for {lat},{lon}: {str(e)}")
        return None

def preprocess_climate_data(raw_data):
    """Convert raw API response to processed DataFrame"""
    if not raw_data:
        return pd.DataFrame()
    
    # Convert to DataFrame and calculate annual aggregates
    df = pd.DataFrame(raw_data)
    return df.resample('Y').agg({
        'T2M': ['mean', 'std'],
        'T2M_MAX': 'max',
        'T2M_MIN': 'min',
        'PRECTOTCORR': 'sum',
        'ALLSKY_SFC_SW_DWN': 'mean'
    }).reset_index()

# Main processing loop
results = []
failed_districts = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    try:
        # Fetch raw climate data
        climate_data = fetch_nasa_data(row['Latitude'], row['Longitude'])
        
        if climate_data:
            # Preprocess and enhance with district info
            processed = preprocess_climate_data(climate_data)
            processed['District Code'] = row['District Code']
            processed['State Code'] = row['State Code']
            results.append(processed)
            
        # Respect API rate limits
        time.sleep(0.5)
        
    except Exception as e:
        failed_districts.append((row['District Name'], str(e)))

# Combine all results
final_df = pd.concat(results, ignore_index=True)

# Add metadata columns
meta_cols = ['State Code', 'District Code', 'State Name', 'District Name',
             'Latitude', 'Longitude', 'Agro Ecological Zones ICRISAT']
final_df = final_df.merge(df[meta_cols], on=['State Code', 'District Code'])

# Handle missing data
print(f"Missing data percentage: {final_df.isna().mean().round(4)*100}%")
final_df = final_df.dropna(subset=['T2M'])

# Save processed data
final_df.to_csv('district_climate_data.csv', index=False)
print(f"Successfully processed {len(final_df)} district-years")
print(f"Failed districts: {len(failed_districts)}")

# Export failure log
if failed_districts:
    pd.DataFrame(failed_districts, columns=['District', 'Error']
                ).to_csv('failed_requests.csv', index=False)
