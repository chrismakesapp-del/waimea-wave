from pydantic import BaseModel, Field

class ForecastConfig(BaseModel):
    """Configuration parameters for wave height forecasting.
    
    Attributes:
        target_col: Target wave height column name.
        horizon_days: Forecast horizon in days (currently only 1 day supported).
        lags: List of lag periods in days for creating lagged features.
        rolling_windows: List of rolling window sizes in days for statistics.
        threshold_m: Threshold in meters for high wave classification.
        random_state: Random seed for reproducibility.
    """
    target_col: str = Field(default="wave_height_51201h", description="Target wave height column")
    horizon_days: int = Field(default=1, ge=1, le=1, description="POC supports next-day only")
    lags: list[int] = Field(default_factory=lambda: [1, 2, 3, 5, 7], description="Lag days")
    rolling_windows: list[int] = Field(default_factory=lambda: [3, 7], description="Rolling windows (days)")
    threshold_m: float = Field(default=3.0, description="Contest threshold in meters")
    random_state: int = Field(default=42)
