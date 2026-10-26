
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

file_path = BASE_DIR / "outputs" / "confidence_evaluation.csv"

data = pd.read_csv(file_path)

print("TOTAL TEST EXAMPLES:", len(data))

print("\nOVERALL ACCURACY:")
print(f"{data['Correct'].astype(bool).mean() * 100:.2f}%")

# Display incorrectly classified examples
errors = data[~data["Correct"].astype(bool)].copy()

print("\nINCORRECTLY CLASSIFIED EXAMPLES:")
if errors.empty:
    print("No incorrect predictions found.")
else:
    print(
        errors[
            [
                "N", "P", "K",
                "temperature", "humidity", "ph", "rainfall",
                "Actual crop", "Predicted crop",
                "Top probability (%)"
            ]
        ].to_string(index=False)
    )

# Show the distribution of predicted probabilities
print("\nCONFIDENCE SUMMARY:")
print(
    data["Top probability (%)"].describe().round(2)
)

# Show accuracy for confidence groups
data["Confidence group"] = pd.cut(
    data["Top probability (%)"],
    bins=[0, 50, 75, 90, 95, 100],
    labels=[
        "0-50%",
        "50-75%",
        "75-90%",
        "90-95%",
        "95-100%"
    ],
    include_lowest=True
)

print("\nACCURACY BY CONFIDENCE GROUP:")
print(
    data.groupby(
        "Confidence group",
        observed=False
    )["Correct"]
    .agg(["count", "mean"])
    .assign(accuracy_percent=lambda x: x["mean"] * 100)
    .round(2)
    .to_string()
)

print("\nAnalysis completed.")
