"""
Module to preprocess the data and prepare it for model building/training.
"""

from typing import Tuple

from pandas import DataFrame, read_csv
from sklearn.model_selection import train_test_split


def read_dataframe(filename: str) -> DataFrame:
    """Read raw csv file into a pandas dataframe.

    Args:
        filename (str): The filename to read (path to data).

    Returns:
        DataFrame: The dataframe read.
    """
    df = read_csv(filename)
    # remove duplicates
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def get_splits(
    df: DataFrame,
    train_size: float = 0.8,
    test_size: float = 0.2,
    random_state: int = 123,
) -> Tuple:
    """Get the splits for training and testing.

    Args:
        df (DataFrame): The dataframe.
        train_size (float, optional): Train size for split. Defaults to .8.
        test_size (float, optional): Test size for split. Defaults to .2.
        random_state (int, optional): Random state for reproducibility purpose. Defaults to 123.

    Returns:
        Tuple: The splits:  X_train, X_test, y_train, y_test.
    """
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_inter, y_train, y_inter = train_test_split(
        X,
        y,
        test_size=train_size,
        random_state=random_state,
        stratify=y,  # to keep the dataset balanced in train and test
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_inter, y_inter, test_size=test_size, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
