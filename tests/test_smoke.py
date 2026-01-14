from pathlib import Path
import pandas as pd

from waimea_waves.data.loader import load_wide_csv
from waimea_waves.models.estimator import WaimeaWaveForecaster

def test_train_and_predict_smoke(tmp_path: Path):
    """Smoke test for training and prediction pipeline.
    
    Verifies that the model can be trained, saved, loaded, and used
    for prediction without errors. Checks that training produces valid
    metrics and predictions contain expected fields.
    
    Args:
        tmp_path: Temporary directory path provided by pytest for test artifacts.
    """
    df = load_wide_csv("data/raw/wide.csv")
    f = WaimeaWaveForecaster()
    report = f.fit(df)
    assert report.rmse >= 0.0

    model_path = tmp_path/"model.joblib"
    f.save(model_path)
    f2 = WaimeaWaveForecaster.load(model_path)
    out = f2.predict_next_day(df)
    assert "predicted_wave_height_m" in out
