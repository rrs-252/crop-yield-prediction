import pytest
from src.predictor import YieldPredictor

def test_prediction():
    predictor = YieldPredictor()
    yield = predictor.predict(21.1925, 81.2842, 'rice')
    assert 1000 < yield < 5000, "Yield prediction out of expected range"
