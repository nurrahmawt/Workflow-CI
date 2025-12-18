import os
os.environ.pop("MLFLOW_RUN_ID", None)  # Hapus run ID lama

import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Tracking URI lokal
mlflow.set_tracking_uri("file:///tmp/mlruns") 

# Load data
df = pd.read_csv("houseprices_preprocessing/house_data_processed.csv")
X = df.drop("House_Price", axis=1)
y = df["House_Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with mlflow.start_run():
    # Log parameter
    mlflow.log_param("model_type", "LinearRegression")

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Buat signature dan input example
    input_example = X_train.head(5)
    signature = infer_signature(X_train, model.predict(X_train))

    # Log model dengan signature & input example
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        signature=signature,
        input_example=input_example
    )

    # Prediksi & evaluasi
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Log metrics
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)

    print("MSE:", mse)
    print("R2:", r2)