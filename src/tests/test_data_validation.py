# tests/test_data_validation.py
def test_temporal_coverage():
    """Validate 2008-2017 coverage"""
    years = set()
    for entry in os.listdir('data/processed'):
        if entry.startswith('year='):
            years.add(int(entry.split('=')[1]))
    assert years == set(range(2008, 2018)), "Missing years in processed data"

def test_climate_features():
    """Validate NASA-derived features"""
    df = pq.read_table('data/processed/year=2010/crop=rice/part-0.parquet').to_pandas()
    assert df['gdd'].between(0, 5000).all(), "Invalid GDD values"
    assert df['precip'].between(0, 5000).all(), "Invalid precipitation"
    assert df['solar_rad'].between(0, 30).all(), "Invalid solar radiation"

def test_ndvi_vci_ranges():
    """Validate vegetation indices"""
    df = pq.read_table('data/processed/year=2015/crop=wheat/part-0.parquet').to_pandas()
    assert df['NDVI'].between(-1, 1).all(), "Invalid NDVI values"
    assert df['VCI (%)'].between(0, 100).all(), "Invalid VCI percentages"

