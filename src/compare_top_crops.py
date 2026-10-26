
import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(
    BASE_DIR / "models" / "crop_model_calibrated.pkl"
)

features = joblib.load(
    BASE_DIR / "models" / "feature_names.pkl"
)

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

probabilities = model.predict_proba(sample)[0]
classes = model.classes_

ranking = pd.DataFrame({
    "Crop": classes,
    "Probability (%)": probabilities * 100
})

ranking = ranking.sort_values(
    "Probability (%)",
    ascending=False
).head(3).reset_index(drop=True)

print("\nTOP THREE CROP CANDIDATES")
print("=========================")
print(ranking.to_string(index=False))

print("\nINPUT CONDITIONS")
print("================")
for feature, value in input_values.items():
    print(f"{feature}: {value}")

print(
    "\nNote: probabilities reflect the model's estimates, "
    "not guaranteed crop suitability."
)
