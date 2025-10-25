import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# Load dataset
df = pd.read_csv("data/house.csv")

# Feature Engineering
df["size_scaled"] = StandardScaler().fit_transform(df[["size_sqft"]])


# Select features
X = df[["size_scaled", "bedrooms"]]
y = df["price"]

# Save processed dataset
os.makedirs("data/processed", exist_ok=True)
X.to_csv("data/processed/X.csv", index=False)
y.to_csv("data/processed/y.csv", index=False)
