# Waimea Bay Waves — Forecasting Proof of Concept

This repository contains a proof-of-concept (POC) forecasting package that predicts **next-day significant wave height** at **Waimea Bay (NDBC Station 51201)** using historical buoy observations across the Pacific.

## Problem Statement

The World Surf League (WSL) would like to schedule a potential new contest at Waimea Bay but needs better anticipation of **contest-worthy surf (≥ 3.0m wave height)**. The POC model in this repo forecasts the next day’s wave height at Waimea Bay and also reports the probability-like decision signal for a ≥ 3m threshold (derived from regression output).

## Data

The dataset is provided as CSV files under `data/raw/`:
- `wide.csv`: one row per day; columns are buoy measurements (wave height, periods, air temperature)
- `buoy-data.csv`: long format (one row per buoy per day)

### Target Definition

By default, this POC forecasts:
- **Target column:** `wave_height_51201h` (Waimea Bay)

This matches NDBC Station **51201 – Waimea Bay, HI**.

## Part 1 — Written Responses

### 1) Framing the Problem (Kickoff Questions)

**Objectives & decisioning**
- What is the operational decision: “run contest / standby / no-go”, or a probabilistic outlook?
- What lead time is required for mobilization (permits, safety, broadcast, athlete travel)?
- What is the cost of false positives (mobilize but waves underperform) vs false negatives (miss a swell)?

**Scope & success metrics**
- What forecasting horizon matters most (1-day, 3-day, 7-day)?
- What accuracy is “good enough” to change operations?
- Do we evaluate as regression (RMSE/MAE) and/or as a threshold event (≥3m) classifier (precision/recall)?

**Constraints**
- How often should forecasts update (daily vs intra-day)?
- Are there “blackout” constraints or fixed event windows?
- Any compliance/safety minimums (e.g., max wind, visibility)?

**Historical context**
- Are there historical “contest-ready” days / prior Waimea event logs for backtesting decisions?

### 2) Enrichment (Additional Data Sources)

To materially improve forecast skill, I would add:

- **Numerical wave model outputs** (e.g., NOAA WaveWatch III) for swell height/period/direction
- **Wind fields** (local winds strongly affect face quality, chop, and effective height)
- **Storm tracks and pressure systems** (upstream swell generation)
- **Swell direction** and **spectral energy** (directional buoy spectra is often predictive)
- **Bathymetry / nearshore transformation** (shoaling/refraction affects shore-facing height)
- **Climate indices** (ENSO/PDO) for seasonal/annual regime shifts

Operationally: build an ingestion job (daily) and a retraining cadence (e.g., weekly/monthly), plus monitoring for data drift and missingness.

### 3) Data Integrity (Missing Data)

Missingness is expected in buoy data (sensor outages, telemetry gaps). In this POC:

- **Short gaps** are forward-filled (time-series safe for slow-moving ocean state)
- **Remaining NaNs** are imputed with median values during training (via scikit-learn imputer)
- We also include **missing indicators** (binary flags) so the model can learn if missingness itself is informative

In a production iteration, I would:
- Use model-based multivariate imputation across correlated buoys
- Track missingness rates by buoy/metric and alert on anomalous gaps

### 4) Approaches (CEO-Level Explanation)

There is no single “perfect” forecasting model, so the recommended approach is to start with a reliable baseline and iterate:

1. **Baseline regression** (regularized linear): fast, interpretable, and strong as a first benchmark  
2. **Tree-based ML** (gradient boosting): captures nonlinear relationships between upstream buoys and Waimea  
3. **Probabilistic forecasting**: provide uncertainty bands and “chance of ≥3m” rather than only a point estimate  
4. **Hybrid with physics models**: incorporate wave-model forecasts to extend horizon and improve robustness

This creates a practical path: deliver value quickly, then invest in complexity only when it improves decisions.

### 5) Communication (Ongoing Delivery)

Recommended delivery is decision-oriented:

- A simple **dashboard**:
  - predicted next-day wave height (point + uncertainty)
  - “chance of ≥3m” and rationale (feature importance / upstream swell indicators)
- **Alerts** when conditions exceed a configurable threshold (e.g., predicted ≥3m for the next day)
- Weekly outlook memo summarizing likely windows + confidence

## Quickstart

### 1) Create an environment and install

```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -U pip
pip install -e ".[dev]"
```

### 2) Train a model

```bash
make train
# or:
python -m waimea_waves.models.train --data data/raw/wide.csv --out artifacts/model.joblib
```

### 3) Generate a forecast (as-of last date in the dataset)

```bash
make predict
# or:
python -m waimea_waves.inference.predict --data data/raw/wide.csv --model artifacts/model.joblib
```

## Design Notes / Assumptions

- Forecast horizon is **next-day (t+1)** for this POC.
- Features are derived from **lagged** and **rolling** statistics of buoy measurements.
- For true future forecasts beyond t+1, you typically need **exogenous forecasts** for upstream buoy conditions (or physics model outputs). That is an explicit next step.

## If I had a month

- Add NOAA wave-model features and wind fields
- Extend horizon (1–7 days) with proper exogenous forecasting
- Build probabilistic forecasts and calibrate “≥3m probability”
- Backtest on historical event decisions and tune thresholds for WSL’s cost function
- Productionize: scheduled pipelines, monitoring, dashboards, and alerting
