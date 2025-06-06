# 🚲 Bike Availability Prediction (Barcelona Bicing)

Predicting future bike availability at public stations using historical data from Barcelona's Bicing system.

## 🧠 Project Motivation

In urban environments, real-time knowledge of bike availability helps:
- Commuters plan better routes
- City services redistribute bikes more effectively
- Apps give better recommendations

This project uses machine learning (LightGBM) to predict the number of bikes at a station 30 minutes into the future.

---

## 📦 Dataset

- **Source**: Barcelona Bicing System (2019–2024)
- **Size**: ~250M rows
- **Granularity**: Data every 4 minutes per station
- **Features**: 
  - Station ID, timestamp, number of bikes/docks
  - GPS, station metadata

---

## 🧪 Features Engineered

- Time-based: `hour`, `dayofweek`, `month`, `weekend`
- Lag values: `lag_1`, `lag_2`, `lag_3`
- Target: bike availability 30 minutes later
- Station ID encoded numerically

---

## 📊 Model

- **Type**: Regression
- **Model**: LightGBM
- **Baseline**: Previous value (`lag_1`)
- **Train size**: ~20M rows
- **Test size**: ~11M rows
- **Cross-validation**: Time-aware (2021 → 2022)

---

## 🔧 Hyperparameter Tuning

Used Grid Search over 3 combinations:
```python
n_estimators: [100, 200]
learning_rate: [0.1, 0.05]
max_depth: [5, 10]
