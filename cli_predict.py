# # cli_predict.py

# import pandas as pd
# import numpy as np
# import joblib
# from datetime import datetime, timedelta
# from geopy.geocoders import Nominatim
# from geopy.distance import geodesic
# import warnings
# import os
# import glob
# warnings.filterwarnings("ignore")

# MODEL_PATH = "bike_model.pkl"
# DATA_FOLDER = "data"
# YEARS_TO_USE = [2024]
# RADIUS_METERS = 1000
# LAG_COUNT = 3
# PREDICT_SHIFT = {15: 1, 30: 2, 45: 3, 60: 4}

# # --- 1. USER LOCATION ---
# loc_input = input("Enter your location (e.g., address, city name): ")
# geolocator = Nominatim(user_agent="bike-predict")
# location = geolocator.geocode(loc_input)
# if not location:
#     print("❌ Location not found.")
#     exit()
# user_coord = (location.latitude, location.longitude)
# print(f"📍 Coordinates: {user_coord[0]:.4f}, {user_coord[1]:.4f}")

# # --- 2. TIME CHOICE ---
# minutes = int(input("⏱️ Predict for how many minutes in the future? (15, 30, 45, 60): "))
# if minutes not in PREDICT_SHIFT:
#     print("❌ Invalid choice.")
#     exit()

# # --- 3. LOAD LATEST STATION DATA ---
# print("📂 Loading recent bike data...")
# stations_files = sorted(glob.glob(os.path.join(DATA_FOLDER, "*_STATIONS.csv")), reverse=True)
# info_files = sorted(glob.glob(os.path.join(DATA_FOLDER, "*_INFO.csv")), reverse=True)

# df_station, df_info = None, None
# for sf, inf in zip(stations_files, info_files):
#     try:
#         df_station = pd.read_csv(sf)
#         df_info = pd.read_csv(inf)
#         break
#     except Exception as e:
#         print(f"❌ Failed reading {sf} or {inf}: {e}")

# if df_station is None or df_info is None:
#     print("❌ Could not load recent data.")
#     exit()

# # --- 4. JOIN LAT/LON FROM INFO FILE ---
# df_info = df_info.drop_duplicates(subset="station_id")[['station_id', 'lat', 'lon']]
# df = pd.merge(df_station, df_info, on='station_id', how='left')
# df.dropna(subset=['lat', 'lon'], inplace=True)

# # --- 5. FILTER NEARBY STATIONS ---
# def is_close(row):
#     return geodesic((row['lat'], row['lon']), user_coord).meters <= RADIUS_METERS

# df = df[df.apply(is_close, axis=1)]
# if df.empty:
#     print("⚠️ No stations nearby.")
#     exit()

# # --- 6. FORMAT TIMESTAMP ---
# df['last_updated'] = pd.to_datetime(df['last_updated'], unit='s', errors='coerce')
# df = df.sort_values(['station_id', 'last_updated'])
# df = df.dropna(subset=['last_updated'])

# # --- 7. FEATURE ENGINEERING ---
# df['hour'] = df['last_updated'].dt.hour
# df['dayofweek'] = df['last_updated'].dt.dayofweek
# df['month'] = df['last_updated'].dt.month
# df['weekend'] = df['dayofweek'] >= 5
# df['station_id'] = df['station_id'].astype('category')
# df['station_encoded'] = df['station_id'].cat.codes

# # --- 8. ADD LAG FEATURES ---
# for i in range(1, LAG_COUNT + 1):
#     df[f'lag_{i}'] = df.groupby('station_id')['num_bikes_available'].shift(i)

# df.dropna(subset=[f'lag_{i}' for i in range(1, LAG_COUNT + 1)], inplace=True)

# # --- 9. SELECT MOST RECENT PER STATION ---
# latest_df = df.groupby('station_id').tail(1)

# features = ['hour', 'dayofweek', 'month', 'weekend',
#             'num_docks_available', 'station_encoded',
#             'lag_1', 'lag_2', 'lag_3']

# model = joblib.load(MODEL_PATH)
# pred = model.predict(latest_df[features])
# latest_df['predicted_bikes'] = pred

# # --- 10. PRINT RESULTS ---
# print(f"\n🚲 Prediction for +{minutes} minutes:")
# for _, row in latest_df.iterrows():
#     print(f" - 🅿️ Station {int(row['station_id'])} @ ({row['lat']:.4f}, {row['lon']:.4f})")
#     print(f"   📊 Now: {int(row['num_bikes_available'])} bikes | 🔮 Predicted: {int(row['predicted_bikes'])}\n")

    
# import pandas as pd
# import folium
# import webbrowser
# import os
# from datetime import datetime
# from geopy.geocoders import Nominatim
# from geopy.distance import geodesic

# # Replace this with actual recent station data
# def load_latest_bike_data():
#     return pd.DataFrame({
#         "station_id": [1, 2, 3, 4],
#         "lat": [41.3979, 41.3911, 41.4004, 41.4030],
#         "lon": [2.1801, 2.1966, 2.1924, 2.2054],
#         "num_bikes_available": [4, 5, 22, 8],
#         "hour": [datetime.now().hour] * 4,
#         "minute": [datetime.now().minute] * 4,
#         "weekday": [datetime.now().weekday()] * 4,
#     })

# # Dummy prediction logic (replace with model.predict())
# def predict_availability(df):
#     df["predicted"] = df["num_bikes_available"] + [5, 0, -1, 1]
#     return df

# def geocode_location(address):
#     geolocator = Nominatim(user_agent="bike_predictor")
#     location = geolocator.geocode(address)
#     if not location:
#         raise ValueError("Location not found.")
#     return (location.latitude, location.longitude)

# def filter_nearby_stations(df, user_coords, max_distance_km=1.0):
#     def is_nearby(row):
#         return geodesic(user_coords, (row.lat, row.lon)).km <= max_distance_km
#     return df[df.apply(is_nearby, axis=1)]

# def generate_map(df, user_lat, user_lon):
#     bike_map = folium.Map(location=[user_lat, user_lon], zoom_start=15)
#     for row in df.itertuples():
#         color = "green" if row.predicted >= 10 else "orange" if row.predicted >= 5 else "red"
#         popup = f"Station {row.station_id}<br>Now: {row.num_bikes_available} bikes<br>Predicted: {row.predicted}"
#         folium.CircleMarker(
#             location=[row.lat, row.lon],
#             radius=8,
#             color=color,
#             fill=True,
#             fill_color=color,
#             fill_opacity=0.8,
#             popup=folium.Popup(popup, max_width=250),
#         ).add_to(bike_map)
#     output_path = "bike_predictions_map.html"
#     bike_map.save(output_path)
#     webbrowser.open('file://' + os.path.realpath(output_path))

# def main():
#     address = input("Enter your location (e.g., address, city name): ")
#     minutes_ahead = int(input("⏱️ Predict for how many minutes in the future? (15, 30, 45, 60): "))

#     user_lat, user_lon = geocode_location(address)
#     print(f"📍 Coordinates: {round(user_lat, 4)}, {round(user_lon, 4)}")

#     print("📂 Loading recent bike data...")
#     df = load_latest_bike_data()
#     df = predict_availability(df)

#     nearby = filter_nearby_stations(df, (user_lat, user_lon))
#     if nearby.empty:
#         print("❌ No nearby stations found.")
#         return

#     print(f"\n🚲 Prediction for +{minutes_ahead} minutes:")
#     for row in nearby.itertuples():
#         print(f" - 🅿️ Station {row.station_id} @ ({row.lat}, {row.lon})")
#         print(f"   📊 Now: {row.num_bikes_available} bikes | 🔮 Predicted: {row.predicted} bikes\n")

#     generate_map(nearby, user_lat, user_lon)

# if __name__ == "__main__":
#     main()


import argparse
import joblib
import pickle
import pandas as pd

# Load model and encoder
model = joblib.load("bike_model.pkl")
with open("station_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

def parse_args():
    parser = argparse.ArgumentParser(description="Predict bike availability for a station.")
    parser.add_argument("--station_id", required=True, help="Station ID (e.g., '1')")
    parser.add_argument("--hour", type=int, required=True, help="Hour of day (0–23)")
    parser.add_argument("--dayofweek", type=int, required=True, help="Day of week (0=Mon ... 6=Sun)")
    parser.add_argument("--month", type=int, required=True, help="Month (1–12)")
    parser.add_argument("--weekend", type=int, choices=[0, 1], required=True, help="1 if weekend, 0 otherwise")
    parser.add_argument("--num_docks_available", type=int, required=True, help="Number of docks available")
    parser.add_argument("--lag_1", type=float, required=True, help="Lag 1: last bike count")
    parser.add_argument("--lag_2", type=float, required=True, help="Lag 2: second last bike count")
    parser.add_argument("--lag_3", type=float, required=True, help="Lag 3: third last bike count")
    return parser.parse_args()

def main():
    args = parse_args()

    try:
        station_encoded = encoder.transform([args.station_id])[0]
    except ValueError:
        print(f"❌ Station ID {args.station_id} not found in encoder.")
        return

    input_df = pd.DataFrame([{
        "hour": args.hour,
        "dayofweek": args.dayofweek,
        "month": args.month,
        "weekend": args.weekend,
        "num_docks_available": args.num_docks_available,
        "station_encoded": station_encoded,
        "lag_1": args.lag_1,
        "lag_2": args.lag_2,
        "lag_3": args.lag_3
    }])

    prediction = model.predict(input_df)[0]
    print(f"🚲 Predicted bikes available in 30 minutes: {prediction:.2f}")

if __name__ == "__main__":
    main()