"""
👨‍💻 Author: Srushanth Baride

📧 Email: Srushanth.Baride@gmail.com

🏢 Organization: 🚀 Rocket ML

📅 Date: 30-June-2024

📚 Description: TODO
"""

import pandas as pd
from typing import List


class PreProcessing:
    """_summary_"""

    def __init__(self):
        pd.set_option('future.no_silent_downcasting', True)

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
