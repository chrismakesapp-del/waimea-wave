from __future__ import annotations

import pandas as pd
from dataclasses import dataclass

@dataclass(frozen=True)
class FeatureSpec:
    """Specification for feature engineering parameters.
    
    Attributes:
        target_col: Name of the target column to predict.
        lags: List of lag periods (in days) to create lagged features.
        rolling_windows: List of window sizes (in days) for rolling statistics.
    """
    target_col: str
    lags: list[int]
    rolling_windows: list[int]

def build_features(df: pd.DataFrame, spec: FeatureSpec) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build engineered features for wave height forecasting.
    
    Creates a comprehensive feature set including:
    - Missing value indicators for all sensor columns
    - Lagged features for all sensor columns at specified lag periods
    - Rolling statistics (mean and std) for wave height and period columns
    - Target variable shifted forward by one day (next-day prediction)
    
    The raw target column is removed from features to avoid data leakage,
    but lagged versions of the target are retained.
    
    Args:
        df: Input DataFrame with time series data including a 'date' column.
        spec: FeatureSpec containing target column, lags, and rolling windows.
        
    Returns:
        Tuple containing:
            - X: Feature matrix with engineered features
            - y: Target series (next-day wave heights)
            - y_date: Corresponding dates for the target values
            
    Raises:
        ValueError: If 'date' column or target column is missing from the DataFrame.
    """
    if "date" not in df.columns:
        raise ValueError("DataFrame must include a 'date' column")
    if spec.target_col not in df.columns:
        raise ValueError(f"Target column not found: {spec.target_col}")

    d = df.copy()
    d = d.sort_values("date").reset_index(drop=True)

    # Missing indicators for original sensor columns
    sensor_cols = [c for c in d.columns if c != "date"]
    for c in sensor_cols:
        d[f"{c}__is_missing"] = d[c].isna().astype(int)

    # Lag features for all sensor columns (including target) to capture dynamics
    for c in sensor_cols:
        for lag in spec.lags:
            d[f"{c}__lag_{lag}"] = d[c].shift(lag)

    # Rolling stats for target and key predictors (all wave heights + periods)
    roll_cols = [c for c in sensor_cols if c.startswith("wave_height_") or "wave_period" in c]
    for c in roll_cols:
        for w in spec.rolling_windows:
            d[f"{c}__roll_mean_{w}"] = d[c].rolling(window=w, min_periods=1).mean()
            d[f"{c}__roll_std_{w}"] = d[c].rolling(window=w, min_periods=2).std()

    # Label is next-day target (t+1)
    y = d[spec.target_col].shift(-1)
    y_date = d["date"].shift(-1)

    # Drop rows where label is missing (end of series) or where features are all NaN
    feature_cols = [c for c in d.columns if c not in ["date"]]
    X = d[feature_cols]

    mask = y.notna()
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)
    y_date = y_date.loc[mask].reset_index(drop=True)

    # Drop the raw (unlagged) target itself to avoid leakage (using same-day Waimea height to predict next day is OK,
    # but we prefer lagged forms; the raw target could be considered available at prediction time as-of 'today' though).
    # For a conservative POC, remove the raw target column; keep its lagged versions.
    if spec.target_col in X.columns:
        X = X.drop(columns=[spec.target_col])

    return X, y, y_date
