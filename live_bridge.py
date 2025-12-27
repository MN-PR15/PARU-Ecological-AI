import requests
import pandas as pd
from datetime import datetime, timedelta

# Coordinates for Uttarakhand Districts (Approx Center)
DISTRICT_COORDS = {
    'Almora': {'lat': 29.61, 'lon': 79.66},
    'Bageshwar': {'lat': 29.84, 'lon': 79.77},
    'Chamoli': {'lat': 30.29, 'lon': 79.56},
    'Champawat': {'lat': 29.33, 'lon': 80.09},
    'Dehra Dun': {'lat': 30.31, 'lon': 78.03},
    'Haridwar': {'lat': 29.94, 'lon': 78.16},
    'Naini Tal': {'lat': 29.38, 'lon': 79.46},
    'Pauri Garhwal': {'lat': 30.15, 'lon': 78.77},
    'Pithoragarh': {'lat': 29.58, 'lon': 80.21},
    'Rudra Prayag': {'lat': 30.28, 'lon': 78.98},
    'Tehri Garhwal': {'lat': 30.38, 'lon': 78.48},
    'Udham Singh Nagar': {'lat': 28.98, 'lon': 79.40},
    'Uttarkashi': {'lat': 30.73, 'lon': 78.43}
}

def fetch_live_weather():
    print("📡 Connecting to Open-Meteo Satellites...")
    today = datetime.now().date()
    live_data = []

    for district, coords in DISTRICT_COORDS.items():
        try:
            # API Call 
            url = f"https://api.open-meteo.com/v1/forecast?latitude={coords['lat']}&longitude={coords['lon']}&current_weather=true&daily=temperature_2m_max,rain_sum&timezone=auto"
            r = requests.get(url).json()
            
            # Extract Data
            curr_temp = r['current_weather']['temperature']
            # Get latest rain sum
            daily_rain = r['daily']['rain_sum'][0] 
            
            # Append
            live_data.append({
                'district': district,
                'date': today,
                'Air_Temp': curr_temp,
                'Rain_Sum': daily_rain,
                # We assume these stay relatively stable for the forecast
                'NDVI': 0.0, # Placeholder (Model predicts this)
                'LST': curr_temp + 2.0, # Rough estimation for surface temp
                'Soil_Moisture': 0.2 + (daily_rain * 0.01) # Basic physics estimation
            })
            print(f"   ✅ Fetched: {district} ({curr_temp}°C, {daily_rain}mm)")
        except Exception as e:
            print(f"   ❌ Error {district}: {e}")

    return pd.DataFrame(live_data)

if __name__ == "__main__":
    df = fetch_live_weather()
    print("\nSAMPLE DATA:")
    print(df.head())