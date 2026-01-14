from __future__ import annotations

import json
import click

from ..data.loader import load_wide_csv
from ..models.estimator import WaimeaWaveForecaster

@click.command()
@click.option("--data", "data_path", type=click.Path(exists=True, dir_okay=False), required=True,
              help="Path to wide.csv")
@click.option("--model", "model_path", type=click.Path(exists=True, dir_okay=False), required=True,
              help="Path to model artifact (joblib).")
def main(data_path: str, model_path: str) -> None:
    """Main entry point for making wave height predictions.
    
    Loads the data and trained model, then generates a next-day forecast
    for wave height. Outputs the prediction as JSON to stdout.
    
    Args:
        data_path: Path to the wide.csv data file.
        model_path: Path to the saved model artifact (joblib file).
    """
    df = load_wide_csv(data_path)
    forecaster = WaimeaWaveForecaster.load(model_path)
    result = forecaster.predict_next_day(df)
    click.echo(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
