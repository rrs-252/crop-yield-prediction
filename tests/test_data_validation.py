# tests/test_data_validation.py
import pytest
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import os

def test_required_columns():
    """Validate presence of required columns in processed data"""
    required_cols = {
        'district', 'state', 'year', 'lat', 'lon',
        'gdd', 'precip', 'ph', 'organic_carbon',
        'rice_yield', 'wheat_yield', 'maize_yield',
        'pearl_millet_yield', 'finger_millet_yield', 'barley_yield'
    }
    
    # Check all parquet files in processed data
    for root, _, files in os.walk('data/processed'):
        for file in files:
            if file.endswith('.parquet'):
                df = pq.read_table(os.path.join(root, file)).to_pandas()
                assert required_cols.issubset(df.columns), \
                    f"Missing columns in {file}"

def test_yield_values():
    """Validate yield values are within realistic ranges"""
    valid_ranges = {
        'rice_yield': (500, 10000),
        'wheat_yield': (300, 8000),
        'maize_yield': (400, 10000),
        'pearl_millet_yield': (200, 5000),
        'finger_millet_yield': (150, 4000),
        'barley_yield': (100, 6000)
    }
    
    df = pq.read_table('data/processed/year=2010/state=Chhattisgarh/part-0.parquet').to_pandas()
    
    for crop, (min_yield, max_yield) in valid_ranges.items():
        assert df[crop].between(min_yield, max_yield).all(), \
            f"Invalid {crop} values detected"

def test_coordinate_ranges():
    """Validate geographic coordinates are within India's bounds"""
    df = pq.read_table('data/processed/year=2010/state=Maharashtra/part-0.parquet').to_pandas()
    
    # India's approximate geographic boundaries
    assert df['lat'].between(8.0, 37.6).all(), "Latitude out of India's range"
    assert df['lon'].between(68.7, 97.25).all(), "Longitude out of India's range"

def test_missing_values():
    """Validate no missing values in critical fields"""
    critical_fields = ['lat', 'lon', 'gdd', 'precip', 'ph']
    
    for root, _, files in os.walk('data/processed'):
        for file in files:
            if file.endswith('.parquet'):
                df = pq.read_table(os.path.join(root, file)).to_pandas()
                for field in critical_fields:
                    assert not df[field].isnull().any(), \
                        f"Missing values in {field} in {file}"

def test_temporal_coverage():
    """Validate data covers expected years"""
    processed_years = set()
    for entry in os.listdir('data/processed'):
        if entry.startswith('year='):
            processed_years.add(int(entry.split('=')[1]))
    
    assert processed_years == set(range(2008, 2018)), \
        "Missing years in processed data"

def test_soil_properties():
    """Validate soil properties within agronomic ranges"""
    df = pq.read_table('data/processed/year=2015/state=Karnataka/part-0.parquet').to_pandas()
    
    assert df['ph'].between(4.5, 9.0).all(), "Invalid pH values"
    assert df['organic_carbon'].between(0.1, 3.5).all(), "Invalid OC values"
