import requests
import json

# Test data
data = {
    "Location": "New York",
    "Temperature": 25.0,
    "Humidity": 70.0,
    "Wind_Speed": 15.0,
    "Precipitation": 0.0,
    "Cloud_Cover": 30.0,
    "Pressure": 1013.0
}

# Send request
response = requests.post('http://localhost:5000/predict', json=data)
print(json