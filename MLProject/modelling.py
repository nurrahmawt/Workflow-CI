import os
os.environ.pop("MLFLOW_RUN_ID", None)  # bersihkan run ID lama

import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# set tracking URI lokal
mlflow.set_tracking_uri("file:///tmp/mlruns") 

# load data
df = pd.read_csv("houseprices_preprocessing/house_data_processed.csv")
X = df.drop("House_Price", axis=1)
y = df["House_Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run(): 
    mlflow.log_param("model_type", "LinearRegression")

    model = LinearRegression()
    model.fit(X_train, y_train)

    # signature + input example
    input_example = X_train.head(5)
    signature = infer_signature(X_train, model.predict(X_train))

    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        signature=signature,
        input_example=input_example
    )

    # log metrics
    y_pred = model.predict(X_test)
    mlflow.log_metric("mse", mean_squared_error(y_test, y_pred))
    mlflow.log_metric("r2", r2_score(y_test, y_pred))

    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R2:", r2_score(y_test, y_pred))