import requests
import json

# NASA POWER API endpoint
url = "https://power.larc.nasa.gov/api/temporal/climatology/point"

# Define query parameters
params = {
    "start": 2001,
    "end": 2020,
    "latitude": 0,
    "longitude": 0,
    "community": "ag",  # Agriculture community
    "parameters": "T2M,PRECTOT,RH2M,QV2M,PS,T2MDEW,CDD18_3,T2M_MAX",
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


f = open("Climate_Data.json","w+")
json.dump(data,f)
f.close()