import requests

# Correct API endpoint
url = "https://power.larc.nasa.gov/api/temporal/climatology/regional"

# Define query parameters
params = {
    "latitude-min": 0,
    "latitude-max": 10,
    "longitude-min": 0,
    "longitude-max": 10,
    "community": "ag",  # Agriculture community
    "parameters": "T2M", #One Parameter at a time
    "header": "true"
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
