import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import joblib
import pickle
import time
from sklearn.model_selection import train_test_split
import seaborn as sns

FILTER_BY_POSTAL_CODE = False 
POSTAL_CODE = "08005"         # Used if FILTER_BY_POSTAL_CODE is True
TOP_N_STATIONS = 100          # Used if FILTER_BY_POSTAL_CODE is False

DATA_DIR = "data"
INFO_FILE_YEAR = 2022  
INFO_FILE = f"{DATA_DIR}/{INFO_FILE_YEAR}_INFO.csv"
TRAIN_YEARS = [2021, 2022]
TEST_YEARS = [2023, 2024]
TEST_START = "2023-01-01"
TEST_END = "2024-04-01"

MODEL_OUT = "bike_model.pkl"
ENCODER_OUT = "station_encoder.pkl"
PREDICTIONS_OUT = "test_with_predictions.csv"

# Selecting Stations
def get_station_ids_by_postal_code(info_file, postal_code):
    info = pd.read_csv(info_file, usecols=["station_id", "post_code"])
    return info[info["post_code"] == int(postal_code)]["station_id"].astype(str).unique().tolist()

def get_top_n_active_stations(data_dir, years, n):
    files = sorted(glob.glob(os.path.join(data_dir, "*_STATIONS.csv")))
    files = [f for f in files if any(str(y) in f for y in years)]
    activity = {}
    for file in files:
        for chunk in pd.read_csv(file, usecols=["station_id", "num_bikes_available"], chunksize=500_000):
            chunk["station_id"] = chunk["station_id"].astype(str)
            chunk.sort_values(["station_id"], inplace=True)
            chunk["diff"] = chunk.groupby("station_id", observed=True)["num_bikes_available"].diff().abs()
            act = chunk.groupby("station_id", observed=True)["diff"].sum()
            for sid, val in act.items():
                activity[sid] = activity.get(sid, 0) + val
    top_stations = sorted(activity.items(), key=lambda x: -x[1])[:n]
    return [sid for sid, _ in top_stations]

if FILTER_BY_POSTAL_CODE:
    print(f"📂 Filtering stations by postal code {POSTAL_CODE}...")
    station_ids = get_station_ids_by_postal_code(INFO_FILE, POSTAL_CODE)
else:
    print(f"📂 Selecting top {TOP_N_STATIONS} most active stations...")
    station_ids = get_top_n_active_stations(DATA_DIR, TRAIN_YEARS, TOP_N_STATIONS)

if not station_ids:
    raise ValueError("No stations found for the selected filter.")
print(f"✅ Using {len(station_ids)} stations.")

# Loading Data 

def load_selected_files(folder, include_years, station_ids):
    all_files = sorted(glob.glob(os.path.join(folder, "*_STATIONS.csv")))
    dfs = []
    for file in all_files:
        if not any(str(year) in file for year in include_years):
            continue
        try:
            for chunk in pd.read_csv(file, usecols=[
                'station_id', 'num_bikes_available', 'num_docks_available', 'last_updated'
            ], low_memory=False, chunksize=500_000):
                chunk = chunk[chunk['station_id'].astype(str).isin(station_ids)]
                if chunk.empty:
                    continue
                chunk['last_updated'] = pd.to_datetime(chunk['last_updated'], unit='s')
                dfs.append(chunk)
        except Exception as e:
            print(f"❌ Failed to read {file}: {e}")
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

print("📂 Loading training and testing data...")
train_df = load_selected_files(DATA_DIR, TRAIN_YEARS, station_ids)
test_df = load_selected_files(DATA_DIR, TEST_YEARS, station_ids)
test_df = test_df[(test_df['last_updated'] >= TEST_START) & (test_df['last_updated'] < TEST_END)]

if train_df.empty or test_df.empty:
    raise ValueError("❌ No usable data for selected stations. Check file paths, years, or station filter.")

# Feature Engineering 

def add_features(group):
    group = group.sort_values('last_updated')
    group['hour'] = group['last_updated'].dt.hour
    group['dayofweek'] = group['last_updated'].dt.dayofweek
    group['month'] = group['last_updated'].dt.month
    group['weekend'] = group['dayofweek'] >= 5
    group['lag_1'] = group['num_bikes_available'].shift(1)
    group['lag_2'] = group['num_bikes_available'].shift(2)
    group['lag_3'] = group['num_bikes_available'].shift(3)
    group['target'] = group['num_bikes_available'].shift(-6)
    return group.dropna(subset=['target', 'lag_1', 'lag_2', 'lag_3'])

def prepare_df(df, encoder=None):
    df['station_id'] = df['station_id'].astype(str)
    if encoder is None:
        encoder = LabelEncoder()
        encoder.fit(df['station_id'])
    df['station_encoded'] = encoder.transform(df['station_id'])
    df = df.groupby('station_id', group_keys=False, observed=True).apply(add_features)
    return df.reset_index(drop=True), encoder

print("🧪 Engineering features...")
train_df, station_encoder = prepare_df(train_df)
test_df, _ = prepare_df(test_df, encoder=station_encoder)

# Train Test Split 
features = [
    'hour', 'dayofweek', 'month', 'weekend', 'num_docks_available',
    'station_encoded', 'lag_1', 'lag_2', 'lag_3'
]
X_train = train_df[features]
y_train = train_df['target']
X_test = test_df[features]
y_test = test_df['target']

# Model Training
print("🎯 Training LightGBM model...")
model = LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=10, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print(f"\n✅ MAE: {mae:.2f}")
print(f"✅ RMSE: {rmse:.2f}")
print(f"📊 Train rows: {len(X_train):,} | Test rows: {len(X_test):,}")

# Plotting

sample_station = station_ids[1]  # Picking the Second station in the list
sample_code = station_encoder.transform([sample_station])[0]
df_plot = test_df[test_df['station_encoded'] == sample_code].copy()
df_plot['prediction'] = model.predict(df_plot[features])

# Plotting only one week cuz of memory overload
if not df_plot.empty:
    week_start = pd.Timestamp('2023-02-01')
    week_end = pd.Timestamp('2023-02-08')
    df_plot = df_plot[(df_plot['last_updated'] >= week_start) & (df_plot['last_updated'] < week_end)]
    plt.figure(figsize=(14, 5))
    plt.plot(df_plot['last_updated'], df_plot['num_bikes_available'], label='Actual', alpha=0.8)
    plt.plot(df_plot['last_updated'], df_plot['prediction'], label='Predicted', alpha=0.8)
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.xticks(rotation=45)
    plt.title(f'Bike Availability Prediction (Station {sample_station}) - 1 Week View')
    plt.xlabel("Time")
    plt.ylabel("Bikes Available")
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print(f"No data to plot for station {sample_station} in the selected week.")

if not train_df.empty:
    train_df['hour'] = train_df['last_updated'].dt.hour
    hourly_avg = train_df.groupby('hour')['num_bikes_available'].mean()

    plt.figure(figsize=(10, 5))
    sns.lineplot(x=hourly_avg.index, y=hourly_avg.values)
    plt.title('Average Bike Availability by Hour')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Bikes Available')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("EDA skipped: train_df is empty.")
    # Saving Output
print("💾 Saving model and outputs...")
test_df['prediction'] = preds
test_df.to_csv(PREDICTIONS_OUT, index=False)
joblib.dump(model, MODEL_OUT)
with open(ENCODER_OUT, "wb") as f:
    pickle.dump(station_encoder, f)


# Baseline Model (Persistence)

print("Baseline (Persistence) ")
baseline_preds = test_df['lag_1']
baseline_mae = mean_absolute_error(y_test, baseline_preds)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_preds))
print(f"Baseline MAE: {baseline_mae:.2f}")
print(f"Baseline RMSE: {baseline_rmse:.2f}")


# Hyperparameter Tuning 

print(" Hyperparameter Tuning (Grid Search)")
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)
param_grid = [
    {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 10},
    {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 10},
    {'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 5},
]
best_mae = float('inf')
best_params = None
for params in param_grid:
    model_gs = LGBMRegressor(**params, random_state=42, n_jobs=-1)
    model_gs.fit(X_tr, y_tr)
    preds_gs = model_gs.predict(X_val)
    mae_gs = mean_absolute_error(y_val, preds_gs)
    print(f"Params: {params} -> MAE: {mae_gs:.2f}")
    if mae_gs < best_mae:
        best_mae = mae_gs
        best_params = params
print(f"Best params: {best_params} (MAE: {best_mae:.2f})")


# Time-based Cross-Validation

print("\Time-based Cross-Validation ")
# Fold 1: Train 2021, Validate 2022
train_cv = train_df[train_df['last_updated'].dt.year == 2021]
val_cv = train_df[train_df['last_updated'].dt.year == 2022]
X_train_cv = train_cv[features]
y_train_cv = train_cv['target']
X_val_cv = val_cv[features]
y_val_cv = val_cv['target']
model_cv = LGBMRegressor(**best_params, random_state=42, n_jobs=-1)
model_cv.fit(X_train_cv, y_train_cv)
preds_cv = model_cv.predict(X_val_cv)
mae_cv = mean_absolute_error(y_val_cv, preds_cv)
rmse_cv = np.sqrt(mean_squared_error(y_val_cv, preds_cv))
print(f"CV Fold: Train 2021, Validate 2022 -> MAE: {mae_cv:.2f}, RMSE: {rmse_cv:.2f}")
# Report bias/variance: compare train vs. val error
train_preds_cv = model_cv.predict(X_train_cv)
train_mae_cv = mean_absolute_error(y_train_cv, train_preds_cv)
print(f"Train MAE: {train_mae_cv:.2f}, Val MAE: {mae_cv:.2f} (Bias: {mae_cv-train_mae_cv:.2f})")

print("Completed")