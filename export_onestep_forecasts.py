"""Build Table 1: causal one-step-ahead price forecasts for predictors A–E."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from cost_aware_strategies import load_causal
from reproduce_revised_study import TAB, TEST_END, TEST_START, TICKERS, load_market_data
from weekly_causal_strategies import friday_dates

ROOT = Path(r"C:\Users\user\SynologyDrive\Simon\Optimal Portfolio")
OUT = ROOT / "revised_outputs"


def main() -> None:
    close, _open, _djia = load_market_data()
    close = close.loc[:TEST_END, TICKERS]
    test_index = close.index[(close.index >= TEST_START) & (close.index <= TEST_END)]
    decision_dates = test_index[:-1]
    fridays = friday_dates(decision_dates)
    causal = load_causal()

    rows = []
    for dt in decision_dates:
        loc = close.index.get_loc(dt)
        if loc + 1 >= len(close.index):
            continue
        tgt = close.index[loc + 1]
        for tkr in TICKERS:
            rows.append(
                {
                    "Decision": pd.Timestamp(dt).strftime("%Y-%m-%d"),
                    "Target": pd.Timestamp(tgt).strftime("%Y-%m-%d"),
                    "Friday": int(dt in fridays),
                    "Ticker": tkr,
                    "Close_t": float(close.loc[dt, tkr]),
                    "A": float(causal["A"].loc[dt, tkr]),
                    "B": float(causal["B"].loc[dt, tkr]),
                    "C": float(causal["C"].loc[dt, tkr]),
                    "D": float(causal["D"].loc[dt, tkr]),
                    "E": float(causal["E"].loc[dt, tkr]),
                    "Actual_t1": float(close.loc[tgt, tkr]),
                }
            )
    panel = pd.DataFrame(rows)
    for col in "ABCDE":
        panel[f"{col}_err"] = panel[col] - panel["Actual_t1"]
        panel[f"{col}_ret_%"] = 100.0 * (panel[col] / panel["Close_t"] - 1.0)
    panel["Realized_ret_%"] = 100.0 * (panel["Actual_t1"] / panel["Close_t"] - 1.0)
    panel.to_csv(TAB / "table1_onestep_forecasts_all.csv", index=False)

    rmse_rows = []
    for tkr in TICKERS:
        sub = panel.loc[panel["Ticker"] == tkr]
        rmse_rows.append(
            {
                "Ticker": tkr,
                **{L: float(((sub[L] - sub["Actual_t1"]) ** 2).mean() ** 0.5) for L in "ABCDE"},
            }
        )
    rmse = pd.DataFrame(rmse_rows)
    rmse.to_csv(TAB / "table1_mse_causal.csv", index=False)
    rmse.to_csv(TAB / "table1_rmse_causal.csv", index=False)

    last_fri = pd.Timestamp(fridays[-1]).strftime("%Y-%m-%d")
    snap = panel.loc[panel["Decision"] == last_fri, [
        "Ticker", "Close_t", "A", "B", "C", "D", "E", "Actual_t1",
        "A_ret_%", "B_ret_%", "C_ret_%", "D_ret_%", "E_ret_%", "Realized_ret_%",
    ]].copy()
    snap.to_csv(TAB / "table1_onestep_forecasts.csv", index=False)

    prices = snap[["Ticker", "Close_t", "A", "B", "C", "D", "E", "Actual_t1"]].copy()
    prices.to_csv(TAB / "table1_onestep_forecasts_prices.csv", index=False)

    rets = snap[["Ticker", "A_ret_%", "B_ret_%", "C_ret_%", "D_ret_%", "E_ret_%", "Realized_ret_%"]].copy()
    rets.to_csv(TAB / "table1_onestep_forecasts_returns.csv", index=False)

    meta = {
        "decision_date": last_fri,
        "target_date": str(panel.loc[panel["Decision"] == last_fri, "Target"].iloc[0]),
        "n_names": int(len(snap)),
        "n_decisions": int(panel["Decision"].nunique()),
        "n_fridays": int(len(fridays)),
        "rmse_mean": {L: float(rmse[L].mean()) for L in "ABCDE"},
    }
    (OUT / "table1_onestep_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(prices.to_string(index=False))


if __name__ == "__main__":
    main()
