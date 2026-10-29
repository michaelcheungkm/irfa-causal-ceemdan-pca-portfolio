# Causal CEEMDAN–PCA portfolio selection

Analysis code and numerical results for the causal walk-forward Dow Jones study. The Word manuscripts and revision reports are not included.

## Contents

- Python scripts estimate causal forecasts, build portfolios, and run the diagnostics.
- `revised_outputs/tables` holds the result tables.
- `revised_outputs/figures` holds the figures.
- `revised_outputs/checkpoints` and `revised_outputs/checkpoints_hist` hold cached forecasts.
- `revised_outputs/pit_locked` holds the point-in-time locked-rule outputs.
- `revised_outputs/market_ohlc.pkl` and `revised_outputs/market_ohlc_hist.pkl` hold the price caches.
- `tables/` holds the Excel sensitivity and calibration tables.

## Reproduce

`reproduce_revised_study.py` is the main entry point. Cached prices and forecasts are already in `revised_outputs`. A fresh price download uses `yfinance`. Causal CEEMDAN imports PyEMD from a local `PyEMD-master` folder; that library is not vendored here.

```
pip install -r requirements.txt
python reproduce_revised_study.py
```
