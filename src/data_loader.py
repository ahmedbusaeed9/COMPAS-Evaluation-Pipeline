"""
data_loader.py - Loads and validates the COMPAS dataset.
"""

import pandas as pd
import os


def load_compas_data(filepath: str = "data/compas.csv") -> pd.DataFrame:
    """
    Load the COMPAS dataset from a CSV file.

    Args:
        filepath: Path to the COMPAS CSV file.

    Returns:
        A pandas DataFrame containing the COMPAS data.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing.
    """
    # Check if file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")

    # Load CSV
    df = pd.read_csv(filepath)

    # Validate required columns exist
    required_columns = [
        "age", "sex", "race", "priors_count",
        "days_b_screening_arrest", "c_jail_in", "c_jail_out",
        "is_recid", "two_year_recid", "decile_score", "score_text"
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print(f"✅ Dataset loaded successfully!")
    print(f"   Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"   Columns: {list(df.columns)}")

    return df


if __name__ == "__main__":
    df = load_compas_data()
    print("\n📊 First 5 rows:")
    print(df.head())
    print("\n📋 Data types:")
    print(df.dtypes)
    print("\n️ Missing values per column:")
    print(df.isnull().sum())