import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# JANGAN set_tracking_uri (biarin default)
# JANGAN os.environ.pop("MLFLOW_RUN_ID")

# Load data - Pastiin path ini sesuai posisi file pas di-push
df = pd.read_csv("houseprices_preprocessing/house_data_processed.csv")
X = df.drop("House_Price", axis=1)
y = df["House_Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if not mlflow.active_run():
    mlflow.start_run()

active_run = mlflow.active_run()
print(f"Running with ID: {active_run.info.run_id}")

model = LinearRegression()
model.fit(X_train, y_train)

# Signature + Metrics
signature = infer_signature(X_train, model.predict(X_train))
mlflow.sklearn.log_model(model, "model", signature=signature)

y_pred = model.predict(X_test)
mlflow.log_metric("mse", mean_squared_error(y_test, y_pred))
mlflow.log_metric("r2", r2_score(y_test, y_pred))