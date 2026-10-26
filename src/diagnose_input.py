
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

BASE_DIR = Path(__file__).resolve().parent.parent

# Load dataset and trained model
data = pd.read_csv(BASE_DIR / "data" / "Crop_recommendation.csv")
model = joblib.load(BASE_DIR / "models" / "crop_model.pkl")
features = joblib.load(BASE_DIR / "models" / "feature_names.pkl")

X = data[features]
y = data["label"]

# Use the values from your earlier website screenshot
input_values = {
    "N": 40.0,
    "P": 50.0,
    "K": 55.0,
    "temperature": 27.0,
    "humidity": 60.0,
    "ph": 6.5,
    "rainfall": 105.0
}

sample = pd.DataFrame(
    [[input_values[f] for f in features]],
    columns=features
)

# 1. Check input ranges
print("INPUT RANGE CHECK")
print("-----------------")

for feature in features:
    minimum = X[feature].min()
    maximum = X[feature].max()
    value = input_values[feature]

    status = (
        "Within dataset range"
        if minimum <= value <= maximum
        else "OUTSIDE dataset range"
    )

    print(
        f"{feature}: {value} | "
        f"Dataset range: {minimum:.2f} to {maximum:.2f} "
        f"| {status}"
    )

# 2. Display the model's predictions
probabilities = model.predict_proba(sample)[0]
order = np.argsort(probabilities)[::-1]

print("\nTOP FIVE PREDICTIONS")
print("--------------------")

for i in order[:5]:
    print(f"{model.classes_[i]}: {probabilities[i] * 100:.2f}%")

# 3. Find the five closest dataset examples
# Standardization prevents large-scale features from
# dominating the distance calculation.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
sample_scaled = scaler.transform(sample)

nearest = NearestNeighbors(n_neighbors=5)
nearest.fit(X_scaled)

distances, indices = nearest.kneighbors(sample_scaled)

neighbors = data.iloc[indices[0]].copy()
neighbors["Distance"] = distances[0]

print("\nFIVE CLOSEST DATASET EXAMPLES")
print("-----------------------------")

print(
    neighbors[
        features + ["label", "Distance"]
    ].to_string(index=False)
)

print("\nDiagnosis completed.")
