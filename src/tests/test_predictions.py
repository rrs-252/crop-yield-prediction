import pytest
from src.scripts.predict_yield import YieldPredictor

@pytest.mark.parametrize("lat, lon, crop, expected", [
    (28.6139, 77.2090, 'wheat', (3000, 6000)),  # Delhi
    (12.9716, 77.5946, 'rice', (2500, 5000)),   # Bengaluru
    (22.5726, 88.3639, 'maize', (2000, 4000))   # Kolkata
])
def test_yield_predictions(lat, lon, crop, expected):
    predictor = YieldPredictor()
    prediction = predictor.predict(lat, lon, crop)
    assert expected[0] <= prediction <= expected[1]

