
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve

BASE_DIR = Path(__file__).resolve().parent.parent

# Load the dataset and saved model
data = pd.read_csv(
    BASE_DIR / "data" / "Crop_recommendation.csv"
)

model = joblib.load(
    BASE_DIR / "models" / "crop_model.pkl"
)

features = joblib.load(
    BASE_DIR / "models" / "feature_names.pkl"
)

X = data[features]
y = data["label"]

# Recreate the original test set
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Obtain predictions and probabilities
probabilities = model.predict_proba(X_test)
predictions = model.classes_[np.argmax(probabilities, axis=1)]

confidence = probabilities.max(axis=1)
correct = (predictions == y_test.to_numpy()).astype(int)

# Group predictions by their highest class probability
results = pd.DataFrame({
    "Confidence": confidence,
    "Correct": correct
})

bins = [0, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.000001]

results["Confidence group"] = pd.cut(
    results["Confidence"],
    bins=bins,
    include_lowest=True
)

summary = results.groupby(
    "Confidence group",
    observed=False
).agg(
    Number_of_predictions=("Correct", "count"),
    Mean_confidence=("Confidence", "mean"),
    Observed_accuracy=("Correct", "mean")
)

summary["Mean_confidence"] *= 100
summary["Observed_accuracy"] *= 100

print("\nCONFIDENCE CALIBRATION ANALYSIS")
print("--------------------------------")
print(summary.round(2).to_string())

# Plot observed accuracy against mean confidence
valid = summary.dropna(
    subset=["Mean_confidence", "Observed_accuracy"]
)

plt.figure(figsize=(8, 6))

plt.plot(
    [0, 100], [0, 100],
    linestyle="--",
    label="Ideal calibration"
)

plt.plot(
    valid["Mean_confidence"],
    valid["Observed_accuracy"],
    marker="o",
    label="Random Forest"
)

plt.xlabel("Mean predicted probability (%)")
plt.ylabel("Observed accuracy (%)")
plt.title("Top-Class Confidence Calibration")
plt.xlim(0, 100)
plt.ylim(0, 105)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

output_dir = BASE_DIR / "outputs"
output_dir.mkdir(exist_ok=True)

plt.savefig(
    output_dir / "calibration_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nCalibration chart saved to outputs/calibration_plot.png")
