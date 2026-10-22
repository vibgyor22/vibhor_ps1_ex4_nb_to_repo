"""Functions for a first inspection of a dataset: missing values, common values, unique values."""

import numpy as np
import pandas as pd


def missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise missing values in each column.
    """
    total = df.isnull().sum()
    percent = df.isnull().mean() * 100
    summary = pd.concat([total, percent], axis=1, keys=["Total", "Percent"])
    summary["Types"] = df.dtypes.astype(str)
    return summary.T


def most_frequent_values(df: pd.DataFrame) -> pd.DataFrame:
    """Find the most common value in each column and how often it appears.
    """
    total = df.count()
    items = []
    frequencies = []
    for col in df.columns:
        counts = df[col].value_counts()
        if counts.empty:  # column has no non-missing values
            items.append(np.nan)
            frequencies.append(np.nan)
        else:
            items.append(counts.index[0])
            frequencies.append(counts.iloc[0])

    summary = pd.DataFrame({"Total": total})
    summary["Most frequent item"] = items
    summary["Frequency"] = frequencies
    summary["Percent from total"] = np.round(summary["Frequency"] / total * 100, 3)
    return summary.T


def unique_values(df: pd.DataFrame) -> pd.DataFrame:
    """Count the number of distinct values in each column.
    """
    summary = pd.DataFrame({"Total": df.count(), "Uniques": df.nunique()})
    return summary.T

def quick_glimpse(df: pd.DataFrame, n_rows: int = 5) -> None:
    """Show a first look at a dataframe: its first rows, column info and summary statistics.
    """
    display(df.head(n_rows))
    df.info()
    display(df.describe())