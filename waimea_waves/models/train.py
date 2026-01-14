from __future__ import annotations

import json
from pathlib import Path

import click

from ..data.loader import load_wide_csv
from .estimator import WaimeaWaveForecaster

@click.command()
@click.option("--data", "data_path", type=click.Path(exists=True, dir_okay=False), required=True,
              help="Path to wide.csv")
@click.option("--out", "out_path", type=click.Path(dir_okay=False), required=True,
              help="Path to save model artifact (joblib).")
def main(data_path: str, out_path: str) -> None:
    """Main entry point for training the wave height forecaster.
    
    Loads data, trains the model, saves the trained model artifact,
    and writes training metrics to a JSON file.
    
    Args:
        data_path: Path to the wide.csv data file for training.
        out_path: Path where the trained model artifact should be saved.
    """
    df = load_wide_csv(data_path)
    forecaster = WaimeaWaveForecaster()
    report = forecaster.fit(df)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    forecaster.save(out_path)

    report_path = str(Path(out_path).with_suffix(".metrics.json"))
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report.__dict__, f, indent=2)

    click.echo(f"Saved model: {out_path}")
    click.echo(f"Saved metrics: {report_path}")

if __name__ == "__main__":
    main()
