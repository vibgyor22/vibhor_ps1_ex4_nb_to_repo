"""Functions for loading the Titanic data."""

from pathlib import Path

import pandas as pd


def load_data(train_path: str | Path, test_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the Titanic training and test data from CSV files.
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df