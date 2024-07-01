"""
👨‍💻 Author: Srushanth Baride

📧 Email: Srushanth.Baride@gmail.com

🏢 Organization: 🚀 Rocket ML

📅 Date: 30-June-2024

📚 Description: TODO.
"""

import pandas as pd  # type: ignore
from sklearn.metrics import r2_score  # type: ignore
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.ensemble import ExtraTreesRegressor  # type: ignore
import mlflow

from rocketml.pipeline import Pipeline
from rocketml.pre_process import PreProcessing

# Set up experiment
experiment = mlflow.set_experiment("Rohlik Orders Forecasting Challenge")

df = pd.read_csv("../data/train.csv")

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

pp = PreProcessing()

steps = [
    (pp.drop_columns, {"columns": ["id", "date"]}),
    (pp.encode_holiday_name, {"column_name": "holiday_name"}),
    (pp.create_dummies, {"column_name": "warehouse"}),
    (pp.replace_bool, {"values": {True: 1, False: 0}}),
]

pipe = Pipeline()
df_processed = pipe.preprocess_pipeline(df=df, steps=steps)

X = df_processed.drop(columns=["orders"])
y = df_processed["orders"]

x_train, x_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=42
)

with mlflow.start_run(log_system_metrics=True):
    regressor = ExtraTreesRegressor()
    regressor.fit(x_train, y_train)

    y_pred = regressor.predict(x_test)

    mse = mean_squared_error(y_true=y_test, y_pred=y_pred)
    print(mse)

    r2 = r2_score(y_true=y_test, y_pred=y_pred)
    print(r2)

    # Log parameters and metrics
    mlflow.log_param(key="model", value="ExtraTreesRegressor")
    mlflow.log_metric(key="mse", value=mse)
    mlflow.log_metric(key="r2", value=r2)

    mlflow.sklearn.log_model(regressor, "model")

    df_test = pd.read_csv("../data/test.csv")
    pipe = Pipeline()
    df_submission = pipe.preprocess_pipeline(df=df_test, steps=steps)
    res = regressor.predict(df_submission)

    # Create submission
    submission = pd.DataFrame()
    submission["id"] = df_test["id"].to_list()
    submission["orders"] = res.tolist()
    submission.to_csv("submission.csv", index=False)
