
import os
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# 1. Load dataset and saved model
data = pd.read_csv("data/Crop_recommendation.csv")

model = joblib.load("models/crop_model.pkl")
feature_names = joblib.load("models/feature_names.pkl")

X = data[feature_names]
y = data["label"]

# 2. Recreate the same test split used during training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 3. Select one test example to explain

# Select a test example the model has already seen
# during the original evaluation split
sample = X_test.iloc[[0]]
actual_crop = y_test.iloc[0]

predicted_crop = model.predict(sample)[0]
probabilities = model.predict_proba(sample)[0]

print("Input values:")
print(sample.to_string(index=False))

print("\nActual crop:", actual_crop)
print("Predicted crop:", predicted_crop)

print("\nTop 5 predictions:")
top_indices = probabilities.argsort()[::-1][:5]

for index in top_indices:
    print(
        f"{model.classes_[index]}: "
        f"{probabilities[index] * 100:.2f}%"
    )


# ==========================================
# SHAP EXPLANATION
# ==========================================

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import os

# Explain the selected test example
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(sample)

class_index = list(model.classes_).index(predicted_crop)

# Support common SHAP output formats
if isinstance(shap_values, list):
    contributions = np.asarray(
        shap_values[class_index]
    )[0]

else:
    values = np.asarray(shap_values)

    if values.ndim == 3:
        if values.shape[1] == len(feature_names):
            contributions = values[0, :, class_index]
        elif values.shape[2] == len(feature_names):
            contributions = values[class_index, 0, :]
        else:
            raise ValueError(
                f"Unexpected SHAP shape: {values.shape}"
            )

    elif values.ndim == 2:
        contributions = values[0]

    else:
        raise ValueError(
            f"Unexpected SHAP dimensions: {values.shape}"
        )

contributions = np.asarray(
    contributions, dtype=float
).reshape(-1)

if len(contributions) != len(feature_names):
    raise ValueError(
        "SHAP contribution count does not match feature count."
    )

# Create the explanation table
explanation_df = pd.DataFrame({
    "Feature": feature_names,
    "SHAP contribution": contributions
})

explanation_df["Absolute contribution"] = (
    explanation_df["SHAP contribution"].abs()
)

explanation_df = explanation_df.sort_values(
    by="Absolute contribution",
    ascending=False
)

print("\nSHAP feature contributions:")
print(
    explanation_df[
        ["Feature", "SHAP contribution"]
    ].to_string(index=False)
)

# Generate the chart
os.makedirs("outputs", exist_ok=True)

plot_df = explanation_df.sort_values(
    by="SHAP contribution",
    ascending=True
)

colors = [
    "seagreen" if value > 0 else "indianred"
    for value in plot_df["SHAP contribution"]
]

fig, ax = plt.subplots(figsize=(9, 5))

ax.barh(
    plot_df["Feature"],
    plot_df["SHAP contribution"],
    color=colors
)

ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("SHAP contribution")
ax.set_title(
    f"Why did the model predict {predicted_crop}?"
)

fig.tight_layout()

fig.savefig(
    "outputs/shap_prediction_explanation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(fig)

print("\nSHAP explanation saved successfully.")
