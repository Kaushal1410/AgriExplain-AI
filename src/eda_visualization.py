import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv("data/Crop_recommendation.csv")

# Create folder to save charts
os.makedirs("outputs", exist_ok=True)

# Check required columns
expected_columns = [
    "N", "P", "K", "temperature",
    "humidity", "ph", "rainfall", "label"
]

missing_columns = set(expected_columns) - set(data.columns)

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

# Set visual style
sns.set_theme(style="whitegrid")

# 1. Crop distribution
plt.figure(figsize=(12, 6))
sns.countplot(
    data=data,
    y="label",
    order=data["label"].value_counts().index
)
plt.title("Number of Records for Each Crop")
plt.xlabel("Number of Records")
plt.ylabel("Crop")
plt.tight_layout()
plt.savefig("outputs/crop_distribution.png", dpi=300)
plt.show()

# 2. Correlation heatmap
numeric_data = data[expected_columns[:-1]]

plt.figure(figsize=(10, 7))
sns.heatmap(
    numeric_data.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    vmin=-1,
    vmax=1
)
plt.title("Correlation Between Input Features")
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png", dpi=300)
plt.show()

# 3. Feature distributions
numeric_data.hist(figsize=(12, 9), bins=25)
plt.suptitle("Distribution of Agricultural Features")
plt.tight_layout()
plt.savefig("outputs/feature_distributions.png", dpi=300)
plt.show()

print("EDA charts saved in the outputs folder.")
