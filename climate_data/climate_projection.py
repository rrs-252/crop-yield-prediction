import requests

# Correct API endpoint
url = "https://power.larc.nasa.gov/api/projection/daily/point"

# Define query parameters
params = {
    "start": 20250101, #YYYYMMDD
    "end": 20251231, #YYYYMMDD
    "latitude": 0,
    "longitude": 0,
    "community": "ag",
    "parameters": "T2M,T2M_MAX",
    "format":"json",
    "header":"true"
}

# Headers (optional but useful)
headers = {
    "User-Agent": "Mozilla/5.0"
}

try:
    # Make the GET request
    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)

    # Parse JSON response
    data = response.json()

    # Print fetched data
    print(data)
except requests.exceptions.RequestException as e:
    print(f"Error fetching data: {e}")
