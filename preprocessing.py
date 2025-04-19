import pandas as pd
import numpy as np

def load_and_preprocess():
    # Load all datasets
    climate = pd.read_csv("./data/raw/Climate_Data_Wide.csv")
    ndvi = pd.read_csv("./data/raw/NDVI_VCI_India_States.csv")
    icrisat = pd.read_csv("./data/raw/ICRISAT-District-Level-Data.csv")

    # Enhanced column standardization
    def standardize_columns(df):
        df.columns = (
            df.columns.str.strip()
            .str.upper()
            .str.replace(' ', '_')
            .str.replace('-', '_')
        )
        return df

    # Apply standardization
    climate = standardize_columns(climate)
    ndvi = standardize_columns(ndvi)
    icrisat = standardize_columns(icrisat)


    # Verify critical columns exist
    required_columns = {
        'icrisat': ['STATE_NAME', 'DIST_NAME', 'YEAR'],
        'ndvi': ['STATE', 'NDVI', 'VCI_(%)'],
        'climate': ['LATITUDE','LONGITUDE','DISTRICT_NAME', 'YEAR', 'RH2M', 'T2M', 'GDD', 'T2M_MAX']
    }

    # Validate all datasets
    for df_name, cols in required_columns.items():
        missing = [col for col in cols if col not in locals()[df_name].columns]
        if missing:
            print(f"{df_name.upper()} columns:", locals()[df_name].columns.tolist())
            raise ValueError(f"Missing in {df_name}: {missing}")

    # Perform merges with verified columns
    merged = icrisat.merge(
        ndvi[required_columns['ndvi']],
        left_on='STATE_NAME',
        right_on='STATE',
        how='left'
    ).merge(
        climate[required_columns['climate']],
        left_on=['DIST_NAME', 'YEAR'],
        right_on=['DISTRICT_NAME', 'YEAR'],
        how='inner'
    )
    # Feature engineering
    merged['RH2M'] = merged['RH2M'].replace(0, np.nan)
    merged['Moisture_Stress'] = merged['T2M'] / merged['RH2M']
    merged['Thermal_Time'] = merged['GDD'] * merged['T2M_MAX']
    
    # Final cleanup
    merged.to_csv("./data/processed/preprocessed_crop_data.csv", index=False)
    return merged.dropna(thresh=merged.shape[1]-3)

load_and_preprocess()
