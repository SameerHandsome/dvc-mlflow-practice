import pandas as pd
import mlflow
import yaml
import os
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

# Load params
with open("params.yaml") as f:
    params = yaml.safe_load(f)["train"]

# Load processed data
X = pd.read_csv("data/processed/X.csv")
y = pd.read_csv("data/processed/y.csv")

# Start MLflow run
mlflow.start_run()
mlflow.log_params(params)

# Prepare 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold = 1
mse_scores = []

for train_idx, val_idx in kf.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Train Linear Regression on each fold
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    mse_scores.append(mse)

    # Log each fold metric to MLflow
    mlflow.log_metric("mse", mse, step=fold)
    print(f"Fold {fold}: MSE={mse:.4f}")
    fold += 1

# Log average MSE
avg_mse = sum(mse_scores) / len(mse_scores)
mlflow.log_metric("avg_mse", avg_mse)

# Save final model
os.makedirs("models", exist_ok=True)
model_path = "models/linear_model.pkl"
joblib.dump(model, model_path)
mlflow.log_artifact(model_path)

mlflow.end_run()
print(f"✅ Linear Regression model trained. Avg MSE={avg_mse:.4f} logged to MLflow.")
