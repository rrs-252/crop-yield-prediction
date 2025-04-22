import requests
import os


f = open('UnApportionedIdentifiers.csv','r')

data = f.readlines()

f.close()

L = []

for i in data:
    L.append(i.split(','))

latitudes = []
longitudes = []
for i in range (1,len(L)):
    longitudes.append(L[i][-1])
    latitudes.append(L[i][-2])

longitudes = [x.replace('\n','') for x in longitudes]


# API endpoint URL
url = "https://power.larc.nasa.gov/api/temporal/monthly/point"

for i in range (0,len(latitudes)):

    # Parameters extracted from your cURL
    params = {
        "start": "2008",
        "end": "2017",
        "latitude": latitudes[i],
        "longitude": longitudes[i],
        "community": "ag",
        "parameters": "T2M,PRECTOT,RH2M,QV2M,PS,T2MDEW,CDD18_3,T2M_MAX",
        "format": "csv",
        "header": "true"
    }

    # Headers (you used 'accept: application/json' but you're asking for CSV output)
    headers = {
        "accept": "application/json"
    }

    # Make the GET request
    response = requests.get(url, headers=headers, params=params)

    # Save the response content to a CSV file
    if response.status_code == 200:
        file_name = "nasa_power_data"+latitudes[i]+"_"+longitudes[i]+".csv"
        with open(file_name, "wb") as f:
            f.write(response.content)
        print(f"CSV saved successfully to {os.path.abspath(file_name)}")
    else:
        print(f"Error: {response.status_code}")
        print("Response body:", response.text)

