
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "Crop_recommendation.csv"
MODEL_PATH = ROOT / "models" / "crop_model.pkl"
OUTPUT_DIR = ROOT / "outputs"

FEATURES = [
    "N", "P", "K", "temperature",
    "humidity", "ph", "rainfall"
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset and trained model
    data = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)

    missing = [f for f in FEATURES if f not in data.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = data[FEATURES].apply(pd.to_numeric, errors="coerce")
    X = X.dropna().astype(float)

    # Use a reproducible sample to reduce computation
    X_sample = X.sample(
        n=min(200, len(X)),
        random_state=42
    )

    print("Dataset rows:", len(X))
    print("Samples used for SHAP:", len(X_sample))
    print("Calculating global feature importance...")

    # TreeExplainer works with Random Forest models
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Support common SHAP output formats
    if isinstance(shap_values, list):
        values = np.stack(shap_values, axis=-1)
    elif hasattr(shap_values, "values"):
        values = np.asarray(shap_values.values)
    else:
        values = np.asarray(shap_values)

    # Multiclass SHAP values are commonly:
    # samples x features x classes
    if values.ndim == 3:
        importance = np.abs(values).mean(axis=(0, 2))
    elif values.ndim == 2:
        importance = np.abs(values).mean(axis=0)
    else:
        raise ValueError(
            f"Unexpected SHAP output shape: {values.shape}"
        )

    if len(importance) != len(FEATURES):
        raise ValueError(
            "SHAP feature count does not match model input features."
        )

    # Rank features by mean absolute SHAP value
    importance_df = pd.DataFrame({
        "Feature": FEATURES,
        "Mean_Absolute_SHAP": importance
    }).sort_values(
        "Mean_Absolute_SHAP",
        ascending=False
    )

    print("\nGLOBAL FEATURE IMPORTANCE")
    print(importance_df.to_string(index=False))

    # Save numeric results
    csv_path = OUTPUT_DIR / "global_shap_importance.csv"
    importance_df.to_csv(csv_path, index=False)

    # Plot global importance
    plt.figure(figsize=(9, 5))
    plt.barh(
        importance_df["Feature"][::-1],
        importance_df["Mean_Absolute_SHAP"][::-1]
    )
    plt.xlabel("Mean absolute SHAP value")
    plt.ylabel("Agricultural feature")
    plt.title("Global SHAP Feature Importance")
    plt.tight_layout()

    plot_path = OUTPUT_DIR / "global_shap_importance.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("\nSaved importance table:", csv_path)
    print("Saved feature importance plot:", plot_path)
    print("\nGlobal explanation analysis completed.")


if __name__ == "__main__":
    main()
