# tests/test_data_validation.py
import pytest
import pandas as pd
import pyarrow.parquet as pq
import os

def test_required_columns():
    """Validate presence of required columns"""
    required = {'year', 'lat', 'lon', 'gdd', 'precip', 'solar_rad', 'crop', 'yield'}
    
    for root, _, files in os.walk('data/processed'):
        for file in files:
            if file.endswith('.parquet'):
                df = pq.read_table(os.path.join(root, file)).to_pandas()
                assert required.issubset(df.columns), f"Missing columns in {file}"

def test_climate_ranges():
    """Validate climate data ranges"""
    df = pd.read_parquet('data/processed/year=2010/crop=rice/part-0.parquet')
    
    assert df['gdd'].between(0, 5000).all(), "Invalid GDD values"
    assert df['precip'].between(0, 5000).all(), "Invalid precipitation"
    assert df['solar_rad'].between(0, 30).all(), "Invalid solar radiation"

def test_coordinate_ranges():
    """Validate geographic coordinates"""
    df = pd.read_parquet('data/processed/year=2015/crop=wheat/part-0.parquet')
    assert df['lat'].between(8.0, 37.6).all(), "Latitude out of India's range"
    assert df['lon'].between(68.7, 97.25).all(), "Longitude out of India's range"
