"""
👨‍💻 Author: Srushanth Baride

📧 Email: Srushanth.Baride@gmail.com

🏢 Organization: 🚀 Rocket ML

📅 Date: 30-June-2024

📚 Description: TODO.
"""

import pandas as pd  # type: ignore


class Pipeline:
    """_summary_"""

    def __init__(self):
        pass

    @staticmethod
    def preprocess_pipeline(df: pd.DataFrame, steps: list) -> pd.DataFrame:
        """
        Applies a series of preprocessing steps to a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to be processed.
            steps (list): A list of tuples where each tuple contains a function and a dictionary of
            keyword arguments for that function. Each function should take a DataFrame as its first
            argument.

        Returns:
            pd.DataFrame: The processed DataFrame after all steps have been applied.
        """
        for step, kwargs in steps:
            # Apply each preprocessing step to the DataFrame with the provided arguments
            df = step(df, **kwargs)
        return df
