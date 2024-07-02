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
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split  # type: ignore
from tensorflow.keras.layers import Dense  # type: ignore
from tensorflow.keras.layers import Dropout  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
import mlflow  # type: ignore

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

model = Sequential()
model.add(Dense(128, input_dim=x_train.shape[1], activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(32, activation="relu"))
model.add(Dense(1, activation="linear"))

model.compile(optimizer="adam", loss="mean_absolute_percentage_error")

with mlflow.start_run(log_system_metrics=True):
    regressor = model.fit(
        x_train, y_train, epochs=1000, validation_split=0.2, verbose=1
    )

    y_pred = model.predict(x_test)

    mse = mean_squared_error(y_true=y_test, y_pred=y_pred)
    print(mse)

    r2 = r2_score(y_true=y_test, y_pred=y_pred)
    print(r2)

    mape = mean_absolute_percentage_error(y_true=y_test, y_pred=y_pred)
    print(mape)

    # Log parameters and metrics
    mlflow.log_param(key="model", value="tensorflow_2")
    mlflow.log_metric(key="mse", value=mse)
    mlflow.log_metric(key="r2", value=r2)
    mlflow.log_metric(key="mape", value=mape)

    mlflow.tensorflow.log_model(model, "model")

    df_test = pd.read_csv("../data/test.csv")
    pipe = Pipeline()
    df_submission = pipe.preprocess_pipeline(df=df_test, steps=steps)
    res = model.predict(df_submission)

    # Create submission
    submission = pd.DataFrame()
    submission["id"] = df_test["id"].to_list()
    submission["orders"] = res.ravel()
    submission.to_csv("submission.csv", index=False)
