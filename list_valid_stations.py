import pickle

with open("station_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

print(" Valid station IDs:")
print(list(encoder.classes_))