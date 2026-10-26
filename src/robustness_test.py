import pandas as pd
import numpy as np
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "Crop_recommendation.csv"
MODEL_PATH = ROOT / "models" / "crop_model.pkl"
OUTPUT_PATH = ROOT / "outputs" / "robustness_test.csv"

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    data = pd.read_csv(DATA_PATH)
    missing = [c for c in FEATURES if c not in data.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    # Convert all input columns to floats so decimal test values are allowed.
    for column in FEATURES:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=FEATURES).reset_index(drop=True)
    if data.empty:
        raise ValueError("No valid numeric rows found in the dataset.")

    model = joblib.load(MODEL_PATH)
    X = data[FEATURES].astype(float)
    sample = X.iloc[[0]].copy()

    print("\nBASELINE INPUT")
    print(sample.to_string(index=False))
    baseline_crop = model.predict(sample)[0]
    print("\nBaseline prediction:", baseline_crop)

    baseline_confidence = np.nan
    if hasattr(model, "predict_proba"):
        baseline_prob = model.predict_proba(sample)[0]
        baseline_confidence = float(np.max(baseline_prob))
        print(f"Baseline probability: {baseline_confidence:.2%}")

    results = []
    for feature in FEATURES:
        original_value = float(sample.iloc[0][feature])
        std = float(X[feature].std())
        if not np.isfinite(std) or std == 0:
            print(f"Skipping {feature}: no variation in dataset.")
            continue

        step = std * 0.25
        low, high = float(X[feature].min()), float(X[feature].max())

        for direction in (-1, 1):
            test_value = float(np.clip(original_value + direction * step, low, high))
            test_input = sample.copy().astype(float)
            test_input.loc[test_input.index[0], feature] = test_value

            predicted_crop = model.predict(test_input)[0]
            confidence = np.nan
            if hasattr(model, "predict_proba"):
                confidence = float(np.max(model.predict_proba(test_input)[0]))

            results.append({
                "changed_feature": feature,
                "original_value": original_value,
                "tested_value": test_value,
                "direction": "decreased" if direction < 0 else "increased",
                "baseline_crop": str(baseline_crop),
                "predicted_crop": str(predicted_crop),
                "prediction_changed": bool(predicted_crop != baseline_crop),
                "baseline_confidence_percent": baseline_confidence * 100 if np.isfinite(baseline_confidence) else np.nan,
                "tested_confidence_percent": confidence * 100 if np.isfinite(confidence) else np.nan,
            })

    if not results:
        raise RuntimeError("No robustness tests were generated.")

    result_df = pd.DataFrame(results)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_PATH, index=False)

    print("\nROBUSTNESS TEST RESULTS")
    print(result_df.to_string(index=False))
    changed = int(result_df["prediction_changed"].sum())
    print("\nSUMMARY")
    print(f"Total tests: {len(result_df)}")
    print(f"Tests where prediction changed: {changed}")
    print(f"Tests where prediction stayed the same: {len(result_df) - changed}")
    print(f"Results saved to: {OUTPUT_PATH}")
    print("\nA changed prediction is not automatically an error. This test measures sensitivity, not real-world crop suitability.")


if __name__ == "__main__":
    main()
