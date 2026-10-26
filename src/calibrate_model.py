
import joblib
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import accuracy_score, log_loss
from sklearn.base import clone


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "Crop_recommendation.csv"
MODEL_PATH = BASE_DIR / "models" / "crop_model.pkl"
FEATURE_PATH = BASE_DIR / "models" / "feature_names.pkl"
OUTPUT_MODEL_PATH = BASE_DIR / "models" / "crop_model_calibrated.pkl"


# ==========================================
# 2. LOAD DATASET AND FEATURES
# ==========================================

print("Loading dataset and model...")

data = pd.read_csv(DATA_PATH)

feature_names = joblib.load(FEATURE_PATH)

X = data[feature_names]
y = data["label"]

print("Dataset shape:", data.shape)
print("Number of crop classes:", y.nunique())
print("Features:", feature_names)


# ==========================================
# 3. SPLIT DATA INTO THREE PARTITIONS
# ==========================================

# 60% training, 20% calibration, 20% final testing

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.40,
    random_state=42,
    stratify=y
)

X_cal, X_test, y_cal, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nDATA SPLIT")
print("----------")
print("Training examples:", len(X_train))
print("Calibration examples:", len(X_cal))
print("Testing examples:", len(X_test))


# ==========================================
# 4. LOAD AND REFIT ORIGINAL MODEL
# ==========================================

saved_model = joblib.load(MODEL_PATH)

# Train a fresh copy only on the training partition.
# Calibration and test records are kept separate.
base_model = clone(saved_model)

base_model.fit(X_train, y_train)

print("\nOriginal Random Forest trained on training partition.")


# ==========================================
# 5. CALIBRATE MODEL PROBABILITIES
# ==========================================

print("\nCalibrating probabilities using sigmoid method...")

calibrated_model = CalibratedClassifierCV(
    estimator=FrozenEstimator(base_model),
    method="sigmoid"
)

# Use calibration data only for fitting the calibrator.
calibrated_model.fit(X_cal, y_cal)

print("Calibration completed.")


# ==========================================
# 6. EVALUATION FUNCTION
# ==========================================

def evaluate_model(name, model, X_test, y_test):

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, predictions)

    # Multiclass Brier score:
    # Lower is better.
    classes = model.classes_

    actual_one_hot = pd.get_dummies(
        y_test
    ).reindex(
        columns=classes,
        fill_value=0
    ).to_numpy()

    brier_score = np.mean(
        np.sum(
            (probabilities - actual_one_hot) ** 2,
            axis=1
        )
    )

    # Lower log loss is better.
    loss = log_loss(
        y_test,
        probabilities,
        labels=classes
    )

    confidence = probabilities.max(axis=1)

    correct = predictions == y_test.to_numpy()

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Multiclass Brier score: {brier_score:.4f}")
    print(f"Log loss: {loss:.4f}")
    print(
        f"Mean top-class probability: "
        f"{confidence.mean() * 100:.2f}%"
    )
    print(f"Correct predictions: {correct.sum()}")
    print(f"Incorrect predictions: {(~correct).sum()}")

    return {
        "Model": name,
        "Accuracy (%)": accuracy * 100,
        "Brier score": brier_score,
        "Log loss": loss,
        "Mean confidence (%)": confidence.mean() * 100
    }


# ==========================================
# 7. COMPARE BOTH MODELS ON THE SAME TEST SET
# ==========================================

print("\nFINAL TEST SET COMPARISON")
print("=========================")

original_results = evaluate_model(
    "Original Random Forest",
    base_model,
    X_test,
    y_test
)

calibrated_results = evaluate_model(
    "Calibrated Random Forest",
    calibrated_model,
    X_test,
    y_test
)

comparison = pd.DataFrame(
    [original_results, calibrated_results]
)

print("\nMODEL COMPARISON TABLE")
print("======================")
print(comparison.round(4).to_string(index=False))


# ==========================================
# 8. SAVE COMPARISON RESULTS
# ==========================================

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

comparison.to_csv(
    OUTPUT_DIR / "calibration_comparison.csv",
    index=False
)

print("\nComparison saved to:")
print("outputs/calibration_comparison.csv")


# ==========================================
# 9. SAVE CALIBRATED MODEL
# ==========================================

MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(
    calibrated_model,
    OUTPUT_MODEL_PATH
)

print("\nCalibrated model saved successfully:")
print(OUTPUT_MODEL_PATH)

print("\nCalibration experiment completed.")
