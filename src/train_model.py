
import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# 1. Load the dataset
data = pd.read_csv("data/Crop_recommendation.csv")

# 2. Separate inputs (X) and target (y)
X = data.drop(columns=["label"])
y = data["label"]

print("Dataset shape:", data.shape)
print("Input features:", list(X.columns))
print("Number of crop classes:", y.nunique())


# 3. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))


# 4. Define two models
models = {
    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),
}

results = []
trained_models = {}

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)


# 5. Train and evaluate both models
for name, model in models.items():

    print(f"\n{'=' * 45}")
    print("Training:", name)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, average="macro", zero_division=0
    )
    recall = recall_score(
        y_test, predictions, average="macro", zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, average="macro", zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Macro Precision": precision,
        "Macro Recall": recall,
        "Macro F1": f1,
    })

    trained_models[name] = model

    print(f"Accuracy:       {accuracy:.4f}")
    print(f"Macro Precision:{precision:.4f}")
    print(f"Macro Recall:   {recall:.4f}")
    print(f"Macro F1-score: {f1:.4f}")

    print("\nClassification report:")
    print(classification_report(
        y_test, predictions, zero_division=0
    ))

    # Save a confusion matrix for each model
    fig, ax = plt.subplots(figsize=(12, 10))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        labels=sorted(y.unique()),
        xticks_rotation=90,
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )

    ax.set_title(f"{name} - Confusion Matrix")
    fig.tight_layout()

    filename = name.lower().replace(" ", "_")
    fig.savefig(
        f"outputs/{filename}_confusion_matrix.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


# 6. Compare models
results_df = pd.DataFrame(results).sort_values(
    by="Macro F1", ascending=False
)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

results_df.to_csv(
    "outputs/model_comparison.csv", index=False
)


# 7. Save the best model according to macro F1
best_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_name]

joblib.dump(best_model, "models/crop_model.pkl")
joblib.dump(list(X.columns), "models/feature_names.pkl")

print("\nBest model:", best_name)
print("Saved model to models/crop_model.pkl")
print("Saved feature names to models/feature_names.pkl")
print("\nTraining and evaluation completed.")
