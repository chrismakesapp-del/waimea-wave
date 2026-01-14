from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor

from ..config import ForecastConfig
from ..features.engineering import FeatureSpec, build_features
from ..data.preprocessing import basic_time_series_clean
from ..utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class TrainReport:
    """Training report containing model performance metrics.
    
    Attributes:
        rmse: Root mean squared error on test set.
        mae: Mean absolute error on test set.
        threshold_m: Threshold value in meters used for classification metrics.
        precision_at_threshold: Precision for predictions >= threshold.
        recall_at_threshold: Recall for predictions >= threshold.
    """
    rmse: float
    mae: float
    threshold_m: float
    precision_at_threshold: float
    recall_at_threshold: float

class WaimeaWaveForecaster:
    """Forecaster for predicting next-day wave heights at Waimea Bay.
    
    Uses a HistGradientBoostingRegressor with engineered time series features
    including lags and rolling statistics to predict wave heights.
    """
    def __init__(self, config: Optional[ForecastConfig] = None):
        """Initialize the forecaster with optional configuration.
        
        Args:
            config: ForecastConfig instance. If None, uses default configuration.
        """
        self.config = config or ForecastConfig()
        self.pipeline: Optional[Pipeline] = None
        self.feature_columns_: Optional[list[str]] = None

    def _make_pipeline(self) -> Pipeline:
        """Create the preprocessing and modeling pipeline.
        
        Constructs a scikit-learn Pipeline with:
        - ColumnTransformer for numeric feature imputation (median strategy)
        - HistGradientBoostingRegressor for regression
        
        Returns:
            Configured Pipeline ready for fitting.
        """
        # All features are numeric in this dataset
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ])
        preprocessor = ColumnTransformer(
            transformers=[("num", numeric_transformer, slice(0, None))],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        model = HistGradientBoostingRegressor(
            random_state=self.config.random_state,
            max_depth=6,
            learning_rate=0.05,
            max_iter=400,
        )
        return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

    def fit(self, df: pd.DataFrame) -> TrainReport:
        """Train the forecaster on historical data.
        
        Performs data cleaning, feature engineering, and model training using
        a time-aware train/test split (80/20). Evaluates performance on the
        test set and computes both regression and threshold-based metrics.
        
        Args:
            df: Historical time series DataFrame with wave height data.
            
        Returns:
            TrainReport containing performance metrics on the test set.
        """
        df = basic_time_series_clean(df)
        spec = FeatureSpec(
            target_col=self.config.target_col,
            lags=self.config.lags,
            rolling_windows=self.config.rolling_windows,
        )
        X, y, _ = build_features(df, spec)

        self.feature_columns_ = list(X.columns)
        self.pipeline = self._make_pipeline()

        # Time-aware split: last 20% as test
        n = len(X)
        split = int(n * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        self.pipeline.fit(X_train, y_train)
        pred = self.pipeline.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        mae = float(mean_absolute_error(y_test, pred))

        # Threshold decision metrics (≥ 3m)
        thr = self.config.threshold_m
        y_true_evt = (y_test >= thr).astype(int).to_numpy()
        y_pred_evt = (pred >= thr).astype(int)

        tp = int(((y_true_evt == 1) & (y_pred_evt == 1)).sum())
        fp = int(((y_true_evt == 0) & (y_pred_evt == 1)).sum())
        fn = int(((y_true_evt == 1) & (y_pred_evt == 0)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        report = TrainReport(
            rmse=rmse,
            mae=mae,
            threshold_m=thr,
            precision_at_threshold=float(precision),
            recall_at_threshold=float(recall),
        )
        logger.info("Train report: %s", report)
        return report

    def predict_next_day(self, df: pd.DataFrame) -> dict[str, Any]:
        """Generate a next-day wave height forecast.
        
        Uses the most recent data point to predict the wave height for the
        following day. Returns a dictionary with prediction details including
        whether the forecast meets the threshold for high waves.
        
        Args:
            df: Historical time series DataFrame up to the current date.
            
        Returns:
            Dictionary containing:
                - asof_date: Date of the most recent observation
                - forecast_for_date: Date being forecasted
                - predicted_wave_height_m: Predicted wave height in meters
                - meets_3m_threshold: Boolean indicating if prediction >= threshold
                - threshold_m: Threshold value used
                
        Raises:
            RuntimeError: If the model has not been trained or loaded.
        """
        if self.pipeline is None or self.feature_columns_ is None:
            raise RuntimeError("Model is not trained/loaded. Call fit() or load().")

        df = basic_time_series_clean(df)
        spec = FeatureSpec(
            target_col=self.config.target_col,
            lags=self.config.lags,
            rolling_windows=self.config.rolling_windows,
        )
        X, y, y_date = build_features(df, spec)

        # Use the last available feature row (as-of latest date)
        x_last = X.iloc[[-1]][self.feature_columns_]
        pred = float(self.pipeline.predict(x_last)[0])

        return {
            "asof_date": str(df["date"].max().date()),
            "forecast_for_date": str(y_date.iloc[-1].date()),
            "predicted_wave_height_m": pred,
            "meets_3m_threshold": bool(pred >= self.config.threshold_m),
            "threshold_m": self.config.threshold_m,
        }

    def save(self, path: str | Path) -> None:
        """Save the trained model to disk.
        
        Persists the model pipeline, feature columns, and configuration
        to a joblib file for later loading and inference.
        
        Args:
            path: File path where the model should be saved.
            
        Raises:
            RuntimeError: If the model has not been trained.
        """
        if self.pipeline is None or self.feature_columns_ is None:
            raise RuntimeError("Nothing to save; model is not trained.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self.config.model_dump(),
            "feature_columns": self.feature_columns_,
            "pipeline": self.pipeline,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> "WaimeaWaveForecaster":
        """Load a previously saved model from disk.
        
        Restores the model pipeline, feature columns, and configuration
        from a joblib file created by the save() method.
        
        Args:
            path: File path to the saved model artifact.
            
        Returns:
            WaimeaWaveForecaster instance with loaded model and configuration.
        """
        payload = joblib.load(path)
        config = ForecastConfig(**payload["config"])
        obj = cls(config=config)
        obj.feature_columns_ = payload["feature_columns"]
        obj.pipeline = payload["pipeline"]
        return obj
