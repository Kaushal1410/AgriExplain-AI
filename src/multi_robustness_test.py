
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# --------------------------------------------------
# 1. Project paths and feature names
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "Crop_recommendation.csv"
MODEL_PATH = ROOT / "models" / "crop_model.pkl"
OUTPUT_PATH = ROOT / "outputs" / "multi_robustness_test.csv"

FEATURES = [
    "N", "P", "K", "temperature",
    "humidity", "ph", "rainfall"
]

# Number of random tests per perturbation level
TESTS_PER_LEVEL = 100

# Reproducible random tests
RANDOM_SEED = 42


def main():
    # --------------------------------------------------
    # 2. Load dataset and trained model
    # --------------------------------------------------
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Check the CSV filename in your data folder."
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Check the model filename in your models folder."
        )

    data = pd.read_csv(DATA_PATH)

    missing = [f for f in FEATURES if f not in data.columns]
    if missing:
        raise ValueError(f"Missing dataset columns: {missing}")

    for feature in FEATURES:
        data[feature] = pd.to_numeric(
            data[feature], errors="coerce"
        )

    data = data.dropna(subset=FEATURES).reset_index(drop=True)

    if data.empty:
        raise ValueError("No valid numeric dataset rows found.")

    model = joblib.load(MODEL_PATH)

    X = data[FEATURES].astype(float)

    # Random number generator
    rng = np.random.default_rng(RANDOM_SEED)

    # Dataset boundaries for each feature
    minimum = X.min()
    maximum = X.max()
    feature_ranges = maximum - minimum

    print("=" * 60)
    print("MULTI-FEATURE ROBUSTNESS TEST")
    print("=" * 60)
    print(f"Dataset rows: {len(X)}")
    print(f"Features tested: {len(FEATURES)}")
    print(f"Tests per level: {TESTS_PER_LEVEL}")

    # --------------------------------------------------
    # 3. Select different baseline examples
    # --------------------------------------------------
    # Test up to 50 different dataset rows.
    number_of_baselines = min(50, len(X))

    selected_indices = rng.choice(
        len(X),
        size=number_of_baselines,
        replace=False
    )

    baseline_data = X.iloc[selected_indices].reset_index(drop=True)

    baseline_predictions = model.predict(baseline_data)

    print(f"Baseline examples: {number_of_baselines}")

    # --------------------------------------------------
    # 4. Test multiple features changing together
    # --------------------------------------------------
    # 5%, 10%, and 20% of each feature's dataset range.
    perturbation_levels = [0.05, 0.10, 0.20]

    results = []

    for level in perturbation_levels:
        print(f"\nTesting perturbation level: {level:.0%}")

        for trial in range(TESTS_PER_LEVEL):
            # Pick a baseline example
            row_index = int(rng.integers(number_of_baselines))

            original_input = baseline_data.iloc[
                [row_index]
            ].copy()

            original_prediction = baseline_predictions[row_index]

            # Change ALL seven features simultaneously.
            modified_input = original_input.copy().astype(float)

            for feature in FEATURES:
                change_limit = (
                    feature_ranges[feature] * level
                )

                change = rng.uniform(
                    -change_limit, change_limit
                )

                original_value = float(
                    original_input.iloc[0][feature]
                )

                new_value = original_value + change

                # Keep values within the dataset's observed range.
                new_value = float(
                    np.clip(
                        new_value,
                        minimum[feature],
                        maximum[feature]
                    )
                )

                modified_input.loc[
                    modified_input.index[0], feature
                ] = new_value

            # Predict using original and modified conditions
            modified_prediction = model.predict(
                modified_input
            )[0]

            original_confidence = np.nan
            modified_confidence = np.nan

            if hasattr(model, "predict_proba"):
                original_probabilities = model.predict_proba(
                    original_input
                )[0]

                modified_probabilities = model.predict_proba(
                    modified_input
                )[0]

                original_confidence = float(
                    np.max(original_probabilities)
                )

                modified_confidence = float(
                    np.max(modified_probabilities)
                )

            results.append({
                "perturbation_level_percent": int(level * 100),
                "trial": trial + 1,
                "baseline_row_index": int(
                    selected_indices[row_index]
                ),
                "original_crop": str(original_prediction),
                "modified_crop": str(modified_prediction),
                "prediction_changed": bool(
                    original_prediction != modified_prediction
                ),
                "original_confidence_percent": (
                    original_confidence * 100
                    if np.isfinite(original_confidence)
                    else np.nan
                ),
                "modified_confidence_percent": (
                    modified_confidence * 100
                    if np.isfinite(modified_confidence)
                    else np.nan
                ),
            })

    # --------------------------------------------------
    # 5. Save detailed results
    # --------------------------------------------------
    results_df = pd.DataFrame(results)

    OUTPUT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    results_df.to_csv(OUTPUT_PATH, index=False)

    # --------------------------------------------------
    # 6. Summarize robustness at each level
    # --------------------------------------------------
    summary = (
        results_df.groupby("perturbation_level_percent")
        .agg(
            total_tests=("prediction_changed", "size"),
            changed_predictions=("prediction_changed", "sum"),
            average_original_confidence=(
                "original_confidence_percent", "mean"
            ),
            average_modified_confidence=(
                "modified_confidence_percent", "mean"
            ),
        )
        .reset_index()
    )

    summary["unchanged_predictions"] = (
        summary["total_tests"]
        - summary["changed_predictions"]
    )

    summary["prediction_change_rate_percent"] = (
        summary["changed_predictions"]
        / summary["total_tests"]
        * 100
    )

    summary = summary[
        [
            "perturbation_level_percent",
            "total_tests",
            "changed_predictions",
            "unchanged_predictions",
            "prediction_change_rate_percent",
            "average_original_confidence",
            "average_modified_confidence",
        ]
    ]

    print("\n" + "=" * 60)
    print("ROBUSTNESS SUMMARY")
    print("=" * 60)
    print(summary.round(2).to_string(index=False))

    print(f"\nDetailed results saved to:\n{OUTPUT_PATH}")

    print(
        "\nIMPORTANT: These tests measure prediction sensitivity. "
        "They do not establish real-world crop suitability or "
        "guarantee that modified conditions are agronomically valid."
    )


if __name__ == "__main__":
    main()
