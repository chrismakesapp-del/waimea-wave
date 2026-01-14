from __future__ import annotations

import pandas as pd

def basic_time_series_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Perform basic time series cleaning on the dataframe.
    
    Forward-fills short gaps (up to 2 consecutive missing values) to handle
    minor data interruptions, while preserving longer gaps for downstream
    imputation strategies.
    
    Args:
        df: Input DataFrame with time series data.
        
    Returns:
        Cleaned DataFrame with short gaps forward-filled.
    """
    # Forward-fill short gaps; keep remaining NaNs for downstream imputation.
    out = df.copy()
    non_date_cols = [c for c in out.columns if c != "date"]
    out[non_date_cols] = out[non_date_cols].ffill(limit=2)
    return out
