"""
👨‍💻 Author: Srushanth Baride

📧 Email: Srushanth.Baride@gmail.com

🏢 Organization: 🚀 Rocket ML

📅 Date: 30-June-2024

📚 Description: TODO.
"""

import os
import json
import pandas as pd  # type: ignore
from sklearn.metrics import r2_score  # type: ignore
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.linear_model import HuberRegressor  # type: ignore
import mlflow
from mlflow.client import MlflowClient

from rocketml.pipeline import Pipeline
from rocketml.pre_process import PreProcessing

# Set the MLflow server tracking URI
os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5000"

# Initialize MLflow Client
client = MlflowClient()

# Create or get experiment
EXPERIMENT_NAME = "Rohlik Orders Forecasting Challenge"
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    experiment_id = client.create_experiment(EXPERIMENT_NAME)
else:
    experiment_id = experiment.experiment_id

df = pd.read_csv("../../data/train.csv")

data_columns = [
    "warehouse",
    "date",
    "holiday_name",
    "holiday",
    "shops_closed",
    "winter_school_holidays",
    "school_holidays",
    "id",
    "orders",
]

feature_columns = [
    "warehouse",
    "date",
    "holiday_name",
    "holiday",
    "shops_closed",
    "winter_school_holidays",
    "school_holidays",
    "id",
]

target_columns = ["orders"]

df = df[data_columns]

warehouse_map = {
    "Brno_1": 1,
    "Prague_1": 2,
    "Prague_2": 3,
    "Prague_3": 4,
    "Budapest_1": 5,
    "Munich_1": 6,
    "Frankfurt_1": 7
}

pp = PreProcessing()

steps1 = [
    (pp.drop_columns, {"columns": ["id"]}),
    (pp.encode_holiday_name, {"column_name": "holiday_name"}),
    (pp.create_dummies, {"column_name": "warehouse"}),
    (pp.replace_bool, {"values": {True: 1, False: 0}}),
]

steps2 = [
    (pp.drop_columns, {"columns": ["id"]}),
    (pp.encode_holiday_name, {"column_name": "holiday_name"}),
    (pp.create_dummies, {"column_name": "warehouse"}),
    (pp.add_date_features, {"column_name": "date"}),
    (pp.replace_bool, {"values": {True: 1, False: 0}}),
]

steps3 = [
    (pp.drop_columns, {"columns": ["id"]}),
    (pp.encode_holiday_name, {"column_name": "holiday_name"}),
    (pp.create_dummies, {"column_name": "warehouse"}),
    (pp.add_date_features, {"column_name": "date"}),
    (pp.replace_bool, {"values": {True: 1, False: 0}}),
    (pp.standard_scaler, {"columns": ["day", "month", "quarter", "year", "day_of_week", "day_of_year"]}),
]

steps4 = [
    (pp.drop_columns, {"columns": ["id"]}),
    (pp.encode_holiday_name, {"column_name": "holiday_name"}),
    (pp.replace_value, {"column_name": "warehouse", "values": warehouse_map}),
    (pp.add_date_features, {"column_name": "date"}),
    (pp.replace_bool, {"values": {True: 1, False: 0}}),
    (pp.standard_scaler, {"columns": ["day", "month", "quarter", "year", "day_of_week", "day_of_year"]}),
]

steps = steps4

pipe = Pipeline()
df_processed = pipe.preprocess_pipeline(df=df, steps=steps)

# Log the pre-processing steps as JSON
preprocessing_steps = json.dumps([{"step": step.__name__, "params": params} for step, params in steps])

X = df_processed.drop(columns=["orders"])
y = df_processed["orders"]

x_train, x_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=42
)

with mlflow.start_run(experiment_id=experiment_id, log_system_metrics=True) as run:
    # Log pre-processing steps
    mlflow.log_param("preprocessing_steps", preprocessing_steps)

    regressor = HuberRegressor()
    regressor.fit(x_train, y_train)

    y_pred = regressor.predict(x_test)

    mse = mean_squared_error(y_true=y_test, y_pred=y_pred)
    print(mse)

    r2 = r2_score(y_true=y_test, y_pred=y_pred)
    print(r2)

    mape = mean_absolute_percentage_error(y_true=y_test, y_pred=y_pred)
    print(mape)

    # Log parameters and metrics
    mlflow.log_param(key="model", value="HuberRegressor")
    mlflow.log_metric(key="mse", value=mse)
    mlflow.log_metric(key="r2", value=r2)
    mlflow.log_metric(key="mape", value=mape)

    mlflow.sklearn.log_model(regressor, "model")

    df_test = pd.read_csv("../../data/test.csv")
    pipe = Pipeline()
    df_submission = pipe.preprocess_pipeline(df=df_test, steps=steps)
    res = regressor.predict(df_submission)

    # Create submission
    submission = pd.DataFrame()
    submission["id"] = df_test["id"].to_list()
    submission["orders"] = res.tolist()
    submission.to_csv("submission.csv", index=False)
