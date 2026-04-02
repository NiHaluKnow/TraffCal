import openmeteo_requests

import pandas as pd
import requests_cache
from retry_requests import retry
from pathlib import Path

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
	"latitude": 34.0522,
	"longitude": -118.2437,
	"start_date": "2012-03-01",
	"end_date": "2012-06-30",
	"hourly": ["temperature_2m", "precipitation", "rain", "relative_humidity_2m", "wind_speed_100m", "wind_direction_100m", "dew_point_2m", "apparent_temperature", "wind_speed_10m", "wind_direction_10m"],
}
responses = openmeteo.weather_api(url, params=params)

# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]
print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
print(f"Elevation: {response.Elevation()} m asl")
print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")

# Process hourly data. The order of variables needs to be the same as requested.
hourly = response.Hourly()
hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
hourly_precipitation = hourly.Variables(1).ValuesAsNumpy()
hourly_rain = hourly.Variables(2).ValuesAsNumpy()
hourly_relative_humidity_2m = hourly.Variables(3).ValuesAsNumpy()
hourly_wind_speed_100m = hourly.Variables(4).ValuesAsNumpy()
hourly_wind_direction_100m = hourly.Variables(5).ValuesAsNumpy()
hourly_dew_point_2m = hourly.Variables(6).ValuesAsNumpy()
hourly_apparent_temperature = hourly.Variables(7).ValuesAsNumpy()
hourly_wind_speed_10m = hourly.Variables(8).ValuesAsNumpy()
hourly_wind_direction_10m = hourly.Variables(9).ValuesAsNumpy()

hourly_data = {"date": pd.date_range(
	start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
	end =  pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
	freq = pd.Timedelta(seconds = hourly.Interval()),
	inclusive = "left"
)}

hourly_data["temperature_2m"] = hourly_temperature_2m
hourly_data["precipitation"] = hourly_precipitation
hourly_data["rain"] = hourly_rain
hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
hourly_data["wind_speed_100m"] = hourly_wind_speed_100m
hourly_data["wind_direction_100m"] = hourly_wind_direction_100m
hourly_data["dew_point_2m"] = hourly_dew_point_2m
hourly_data["apparent_temperature"] = hourly_apparent_temperature
hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
hourly_data["wind_direction_10m"] = hourly_wind_direction_10m

hourly_dataframe = pd.DataFrame(data = hourly_data)
print("\nHourly data\n", hourly_dataframe)

# Save fetched archive weather for data pipeline integration.
output_path = Path("/home/nihal/Desktop/JOYbaba/traffic_project/data/weather/weather_model_style_sample.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
hourly_dataframe.to_csv(output_path, index=False)
print(f"\nSaved weather CSV to: {output_path}")
