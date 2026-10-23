"""Functions used in the Titanic exploratory analysis: combining sets and adding features."""

import pandas as pd

NAME_COLUMNS = ["Family Name", "Title", "Given Name", "Maiden Name"]


def combine_train_test(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """Stack train and test rows and label each row with a ``set`` column.

    Rows without a Survived value are marked as ``test``; the rest as ``train``.
    """
    all_df = pd.concat([train_df, test_df], axis=0)
    all_df["set"] = "train"
    all_df.loc[all_df.Survived.isna(), "set"] = "test"
    return all_df


def add_family_size(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``Family Size`` as siblings/spouse plus parents/children plus the passenger."""
    out = df.copy()
    out["Family Size"] = out["SibSp"] + out["Parch"] + 1
    return out


def add_age_interval(df: pd.DataFrame) -> pd.DataFrame:
    """Bin ``Age`` into ``Age Interval`` values 0–4.

    Bins are: 0–16, 17–32, 33–48, 49–64, and over 64. Missing ages stay 0.
    """
    out = df.copy()
    out["Age Interval"] = 0.0
    out.loc[out["Age"] <= 16, "Age Interval"] = 0
    out.loc[(out["Age"] > 16) & (out["Age"] <= 32), "Age Interval"] = 1
    out.loc[(out["Age"] > 32) & (out["Age"] <= 48), "Age Interval"] = 2
    out.loc[(out["Age"] > 48) & (out["Age"] <= 64), "Age Interval"] = 3
    out.loc[out["Age"] > 64, "Age Interval"] = 4
    return out


def add_fare_interval(df: pd.DataFrame) -> pd.DataFrame:
    """Bin ``Fare`` into ``Fare Interval`` values 0–3.

    Bins are: up to 7.91, up to 14.454, up to 31, and above 31. Missing fares stay 0.
    """
    out = df.copy()
    out["Fare Interval"] = 0.0
    out.loc[out["Fare"] <= 7.91, "Fare Interval"] = 0
    out.loc[(out["Fare"] > 7.91) & (out["Fare"] <= 14.454), "Fare Interval"] = 1
    out.loc[(out["Fare"] > 14.454) & (out["Fare"] <= 31), "Fare Interval"] = 2
    out.loc[out["Fare"] > 31, "Fare Interval"] = 3
    return out


def add_sex_pclass(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``Sex_Pclass``, a combined label such as ``M_C1`` or ``F_C3``."""
    out = df.copy()
    out["Sex_Pclass"] = out.apply(
        lambda row: row["Sex"][0].upper() + "_C" + str(row["Pclass"]),
        axis=1,
    )
    return out


def parse_names(row: pd.Series) -> pd.Series:
    """Split a passenger ``Name`` into family name, title, given name, and maiden name.

    Expected pattern: ``Family, Title. Given (Maiden)``. Maiden name is ``None`` when
    there are no parentheses. Returns a four-value Series in that order.
    """
    try:
        text = row["Name"]
        split_text = text.split(",")
        family_name = split_text[0]
        next_text = split_text[1]
        split_text = next_text.split(".")
        title = (split_text[0] + ".").lstrip().rstrip()
        next_text = split_text[1]
        if "(" in next_text:
            split_text = next_text.split("(")
            given_name = split_text[0]
            maiden_name = split_text[1].rstrip(")")
            return pd.Series([family_name, title, given_name, maiden_name])
        given_name = next_text
        return pd.Series([family_name, title, given_name, None])
    except Exception as ex:
        print(f"Exception: {ex}")


def add_parsed_names(df: pd.DataFrame) -> pd.DataFrame:
    """Add family name, title, given name, and maiden name columns from ``Name``."""
    out = df.copy()
    out[NAME_COLUMNS] = out.apply(parse_names, axis=1)
    return out
