from __future__ import annotations

import pandas as pd
from pathlib import Path

def load_wide_csv(path: str | Path) -> pd.DataFrame:
    """Load and preprocess the wide CSV data file.
    
    Args:
        path: Path to the wide.csv file containing time series data.
        
    Returns:
        DataFrame with 'date' column converted to datetime and sorted chronologically.
        
    Raises:
        ValueError: If the 'date' column is missing from the CSV file.
    """
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError("Expected a 'date' column in wide.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df
