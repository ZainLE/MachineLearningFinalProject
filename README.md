# Bicing Bike Availability Prediction

## 🚲 Project Motivation
Predicting bike availability at Barcelona Bicing stations helps users find bikes and helps the city optimize station management. This project uses real Bicing data to forecast the number of bikes available 30 minutes into the future at the busiest stations.

## 📊 Data Description
- **Source:** Barcelona Bicing open data (2019–2024)
- **Files:**
  - `*_STATIONS.csv`: Time series of bike/dock availability for each station
  - `*_INFO.csv`: Station metadata (location, capacity, postal code, etc.)
- **Features used:**
  - Time: hour, dayofweek, month, weekend
  - Station: encoded station_id
  - Lags: bike count 1, 2, 3 steps ago
  - Docks: num_docks_available

## 🛠️ Setup Instructions
1. Clone the repo and install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Place data files in the `/data` directory as described above.

## 🚀 How to Run
Run the main pipeline:
```bash
python main.py
```

## 🔮 How to Load the Model and Make Predictions
```python
import joblib
import pickle
import pandas as pd
# Load model and encoder
model = joblib.load('bike_model.pkl')
with open('station_encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)
# Prepare a sample row (with all features)
sample = pd.DataFrame([{  # fill with real values
    'hour': 8, 'dayofweek': 2, 'month': 5, 'weekend': False,
    'num_docks_available': 10, 'station_encoded': encoder.transform(['1'])[0],
    'lag_1': 5, 'lag_2': 6, 'lag_3': 7
}])
pred = model.predict(sample)
print('Predicted bikes available in 30 min:', pred[0])
```

## 📈 Exploratory Data Analysis (EDA)
- Checked for missing values, outliers, and data distributions
- Visualized bike usage patterns by hour, day, and station
- Motivated feature engineering (time, lags, station encoding)

## 🏁 Baseline vs. LightGBM Results
- **Baseline (Persistence):** Predicts last value (lag_1)
- **LightGBM:** Uses all features for improved accuracy
- **Metrics:**
  - MAE and RMSE reported for both baseline and LightGBM

## 🔬 Model Evaluation & Cross-Validation
- Hyperparameter tuning via grid search
- Time-based cross-validation (train on 2021, validate on 2022)
- Reported train/val MAE and bias

## 📊 Results
- LightGBM outperforms baseline on all metrics
- Model is robust to large, real-world data

## 📚 References
- [Barcelona Bicing Open Data](https://opendata-ajuntament.barcelona.cat/en/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)

---

For questions, contact: [Your Name] 