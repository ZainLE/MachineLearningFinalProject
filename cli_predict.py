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
        print(f" Station ID {args.station_id} not found in encoder.")
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