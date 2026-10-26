
import pandas as pd
import joblib
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.calibration import calibration_curve

import matplotlib.pyplot as plt


# Load dataset and saved model
BASE_DIR = Path(__file__).resolve().parent.parent

data = pd.read_csv(
    BASE_DIR / "data" / "Crop_recommendation.csv"
)

model = joblib.load(
    BASE_DIR / "models" / "crop_model.pkl"
)

feature_names = joblib.load(
    BASE_DIR / "models" / "feature_names.pkl"
)

X = data[feature_names]
y = data["label"]


# Recreate the original test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Predict on the complete test set
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

confidence = probabilities.max(axis=1)
correct = predictions == y_test.to_numpy()

accuracy = accuracy_score(y_test, predictions)

print("TEST SET EVALUATION")
print("-------------------")
print(f"Test examples: {len(X_test)}")
print(f"Accuracy: {accuracy * 100:.2f}%")

print(
    f"Correct predictions: {correct.sum()}"
)

print(
    f"Incorrect predictions: {(~correct).sum()}"
)


# Confidence distribution
print("\nCONFIDENCE ANALYSIS")
print("-------------------")

for threshold in [0.50, 0.75, 0.90, 0.95]:
    count = int((confidence < threshold).sum())

    print(
        f"Below {threshold * 100:.0f}%: "
        f"{count} examples"
    )


# Compare confidence for correct and incorrect predictions
print("\nAVERAGE MODEL PROBABILITY")
print("-------------------------")

print(
    f"Correct predictions: "
    f"{confidence[correct].mean() * 100:.2f}%"
)

if (~correct).any():
    print(
        f"Incorrect predictions: "
        f"{confidence[~correct].mean() * 100:.2f}%"
    )
else:
    print("No incorrect predictions in this test split.")


# Multiclass Brier score
one_hot = pd.get_dummies(
    y_test
).reindex(columns=model.classes_, fill_value=0).to_numpy()

brier_score = np.mean(
    np.sum((probabilities - one_hot) ** 2, axis=1)
)

print(
    f"\nMulticlass Brier score: {brier_score:.4f}"
)


# Save individual predictions for inspection
results = X_test.reset_index(drop=True).copy()

results["Actual crop"] = y_test.reset_index(drop=True)
results["Predicted crop"] = predictions
results["Top probability (%)"] = confidence * 100
results["Correct"] = correct

output_dir = BASE_DIR / "outputs"
output_dir.mkdir(exist_ok=True)

results.to_csv(
    output_dir / "confidence_evaluation.csv",
    index=False
)

print(
    "\nSaved detailed results to "
    "outputs/confidence_evaluation.csv"
)


# Plot confidence for correct and incorrect examples
plt.figure(figsize=(8, 5))

plt.hist(
    confidence[correct] * 100,
    bins=10,
    alpha=0.7,
    label="Correct predictions"
)

if (~correct).any():
    plt.hist(
        confidence[~correct] * 100,
        bins=10,
        alpha=0.7,
        label="Incorrect predictions"
    )

plt.xlabel("Highest class probability (%)")
plt.ylabel("Number of examples")
plt.title("Prediction Confidence on Test Data")
plt.legend()
plt.tight_layout()

plt.savefig(
    output_dir / "confidence_distribution.png",
    dpi=300
)

plt.show()
