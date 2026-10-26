
import pandas as pd

# Load the dataset
data = pd.read_csv("data/Crop_recommendation.csv")

# Display the first five rows
print("\n--- First Five Rows ---")
print(data.head())

# Display the number of rows and columns
print("\n--- Dataset Shape ---")
print(data.shape)

# Display column names and data types
print("\n--- Dataset Information ---")
data.info()

# Check for missing values
print("\n--- Missing Values ---")
print(data.isnull().sum())

# Check for duplicate rows
print("\n--- Duplicate Rows ---")
print(data.duplicated().sum())

# Count examples for each crop
print("\n--- Crop Distribution ---")
print(data["label"].value_counts())

# Statistical summary
print("\n--- Statistical Summary ---")
print(data.describe())
