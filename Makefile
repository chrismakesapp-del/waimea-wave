.PHONY: install train predict test lint

install:
	pip install -e ".[dev]"

train:
	python -m waimea_waves.models.train --data data/raw/wide.csv --out artifacts/model.joblib

predict:
	python -m waimea_waves.inference.predict --data data/raw/wide.csv --model artifacts/model.joblib

test:
	pytest -q

lint:
	ruff check waimea_waves
