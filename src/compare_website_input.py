
import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

original_model = joblib.load(
    BASE_DIR / "models" / "crop_model.pkl"
)

calibrated_model = joblib.load(
    BASE_DIR / "models" / "crop_model_calibrated.pkl"
)

features = joblib.load(
    BASE_DIR / "models" / "feature_names.pkl"
)

# Original values from your website screenshot
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

models = {
    "Original Random Forest": original_model,
    "Calibrated Random Forest": calibrated_model
}

for name, model in models.items():
    probabilities = model.predict_proba(sample)[0]
    order = probabilities.argsort()[::-1]

    print(f"\n{name}")
    print("-" * len(name))
    print("Predicted crop:", model.predict(sample)[0])
    print(
        "Top-class probability:",
        f"{probabilities[order[0]] * 100:.2f}%"
    )

    print("Top 5 predictions:")
    for i in order[:5]:
        print(
            f"{model.classes_[i]}: "
            f"{probabilities[i] * 100:.2f}%"
        )
