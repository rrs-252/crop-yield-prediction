import requests

# Correct API endpoint
url = "https://power.larc.nasa.gov/api/application/indicators/point"

# Define query parameters
params = {
    "start": 2001,
    "end": 2020,
    "latitude": 0,
    "longitude": 0,
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
