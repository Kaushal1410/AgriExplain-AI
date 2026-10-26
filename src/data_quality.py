
import pandas as pd

data = pd.read_csv("data/Crop_recommendation.csv")

features = [
    "N", "P", "K", "temperature",
    "humidity", "ph", "rainfall"
]

target = "label"

print("Dataset shape:", data.shape)

print("\nMissing values:")
print(data.isnull().sum())

print("\nDuplicate rows:", data.duplicated().sum())

print("\nUnique crop labels:", data[target].nunique())

print("\nCrop names:")
print(sorted(data[target].unique()))

print("\nInvalid numeric values:")
numeric = data[features].apply(pd.to_numeric, errors="coerce")
print(numeric.isnull().sum())

print("\nBasic statistics:")
print(numeric.describe())

print("\nData quality checks completed.")
