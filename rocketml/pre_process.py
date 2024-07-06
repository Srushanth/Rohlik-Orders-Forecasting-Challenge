"""
👨‍💻 Author: Srushanth Baride

📧 Email: Srushanth.Baride@gmail.com

🏢 Organization: 🚀 Rocket ML

📅 Date: 30-June-2024

📚 Description: TODO
"""

from typing import List

import pandas as pd  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore


class PreProcessing:
    """_summary_"""

    def __init__(self):
        pd.set_option("future.no_silent_downcasting", True)

    @staticmethod
    def drop_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """_summary_

        Args:
            df (pd.DataFrame): _description_
            columns (List[str]): _description_

        Returns:
            pd.DataFrame: _description_
        """
        return df.drop(columns=columns)

    @staticmethod
    def encode_holiday_name(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        """_summary_

        Args:
            df (pd.DataFrame): _description_
            column_name (str): _description_

        Returns:
            pd.DataFrame: _description_
        """
        df[column_name] = df[column_name].apply(lambda x: 0 if pd.isna(x) else 1)
        return df

    @staticmethod
    def create_dummies(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        """_summary_

        Args:
            df (pd.DataFrame): _description_
            column_name (str): _description_

        Returns:
            pd.DataFrame: _description_
        """
        dummies: pd.DataFrame = pd.get_dummies(df[column_name], prefix=column_name)
        df = df.drop(columns=[column_name])
        return pd.concat([df, dummies], axis=1)

    @staticmethod
    def replace_bool(df: pd.DataFrame, values: dict) -> pd.DataFrame:
        """_summary_

        Args:
            df (pd.DataFrame): _description_
            values (dict): _description_

        Returns:
            pd.DataFrame: _description_
        """
        return df.replace(to_replace=values, inplace=False).infer_objects(copy=False)

    @staticmethod
    def replace_value(df: pd.DataFrame, column_name: str, values: dict) -> pd.DataFrame:
        """_summary_

        Args:
            df (pd.DataFrame): _description_
            column_name (str): _description_
            values (dict): _description_

        Returns:
            pd.DataFrame: _description_
        """
        df[column_name] = df[column_name].replace(values)

        return df

    @staticmethod
    def add_date_features(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
        """_summary_

        Args:
            df (pd.DataFrame): _description_
            column_name (str): _description_

        Returns:
            pd.DataFrame: _description_
        """
        df[column_name] = pd.to_datetime(df[column_name])
        df["day"] = df[column_name].dt.day
        df["month"] = df[column_name].dt.month
        df["quarter"] = df[column_name].dt.quarter
        df["year"] = df[column_name].dt.year
        df["day_of_week"] = df[column_name].dt.day_of_week
        df["day_of_year"] = df[column_name].dt.day_of_year
        df = df.drop(columns=[column_name])
        return df

    @staticmethod
    def standard_scaler(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """_summary_

        Args:
            df (pd.DataFrame): _description_
            columns (List[str]): _description_

        Returns:
            pd.DataFrame: _description_
        """
        # Create a StandardScaler object
        scaler = StandardScaler()

        # Fit and transform the DataFrame (assuming 'df' is your DataFrame)
        df[columns] = scaler.fit_transform(df[columns])
        return df
