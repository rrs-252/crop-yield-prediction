import requests

'''
Temperature (T2M)
Precipitation (PRECTOT)
Wind Speed (WSC)
Relative Humidity (RH2M)
Specific Humidity (QV2M)
Surface Pressure (PS)
Corrected Atmospheric Pressure (PSC)
Dew/Frost Point (T2MDEW)
Cooling Degree Days (CDD18_3)
T2M,PRECTOT,WSC,RH2M,QV2M,PS,T2MDEW,CDD18_3,T2M_MAX,PRECTOT_MAX,WSC_MAX,RH2M_MAX,QV2M_MAX
'''

# Correct API endpoint
url = "https://power.larc.nasa.gov/api/application/windrose/point"

# Define query parameters
params = {
    "start": 20230101, #YYYYMMDD
    "end": 20250101, #YYYYMMDD
    "latitude": 0,
    "longitude": 0,
    "format":"json",
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
