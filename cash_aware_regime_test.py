"""Compare residual-cash share-MV (excess value) on the GFC bear and 2024-26 bull.

The proposed book is unchanged: use_excess defaults to False in apply_mean_var.
This script only writes cash-aware comparison tables.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cost_aware_strategies import apply_mean_var, apply_weights, load_causal, predicted_returns
from historical_bear_test import (
    BEAR_END,
    BEAR_START,
    load_hist_market,
    predicted_returns as hist_predicted_returns,
    run_hist_walkforward,
)
from reproduce_revised_study import (
    FIG,
    S0,
    TAB,
    TEST_END,
    TEST_START,
    TICKERS,
    load_market_data,
    performance_stats,
    run_baselines,
    series_from_result,
)
from weekly_causal_strategies import (
    LP_BUFFER,
    LP_K,
    MV_CAP,
    MV_KAPPA,
    MV_LAM,
    apply_index_membership,
    every_n_mask,
    hysteresis_topk,
    load_dow_panel,
)

OUT_META = TAB / "table_cash_mv_meta.json"


def _month_end(ser: pd.Series) -> tuple[list[str], list[float]]:
    idx = ser.index
    last = pd.Series(idx, index=idx).groupby(idx.to_period("M")).last()
    labels = [pd.Timestamp(x).strftime("%Y-%m") for x in last.values]
    vals = [round(float(ser.loc[pd.Timestamp(x)]) / 1e6, 3) for x in last.values]
    return labels, vals


def _cash_month(res: dict) -> list[float]:
    frac = np.asarray(res.get("cash_frac", []), dtype=float)
    idx = res["index"]
    if len(frac) != len(idx):
        return []
    s = pd.Series(frac, index=idx)
    last = pd.Series(idx, index=idx).groupby(idx.to_period("M")).last()
    return [round(100.0 * float(s.loc[pd.Timestamp(x)]), 1) for x in last.values]


def _row(label: str, res: dict, regime: str) -> dict:
    s = performance_stats(res)
    s["Strategy"] = label
    s["Regime"] = regime
    cf = np.asarray(res.get("cash_frac", []), dtype=float)
    s["Avg_Cash_%"] = float(100.0 * cf.mean()) if len(cf) else np.nan
    return s


def _last_book(res: dict, dates) -> dict[str, float]:
    w = res["weights"].loc[dates].iloc[-1]
    held = w[w > 1e-8].sort_values(ascending=False)
    out = {str(k): float(v) for k, v in held.items()}
    cash = max(0.0, 1.0 - float(held.sum()))
    if cash > 1e-4:
        out["CASH"] = cash
    return out


def run_bear() -> tuple[pd.DataFrame, dict]:
    close_all, open_all, djia_all = load_hist_market()
    tickers = list(close_all.columns)
    close = close_all.loc[:BEAR_END, tickers]
    open_ = open_all.reindex(close.index)[tickers].fillna(close)
    djia = djia_all.reindex(close.index).ffill()
    test_index = close.index[(close.index >= BEAR_START) & (close.index <= BEAR_END)]
    decision_dates = test_index[:-1]
    causal = run_hist_walkforward(close, decision_dates)
    rhat = hist_predicted_returns(causal["B"].reindex(decision_dates)[tickers], close, decision_dates)
    rhat_hold = rhat.rolling(20, min_periods=5).mean()
    month = every_n_mask(decision_dates, 4)
    w = hysteresis_topk(rhat_hold, decision_dates[month], LP_K, LP_BUFFER)
    results = {
        "BH_DJIA": run_baselines(close, open_, djia, decision_dates)["BH_DJIA"],
        "EW_1N": apply_weights(
            close, open_, decision_dates,
            pd.DataFrame(1.0 / len(tickers), index=decision_dates, columns=tickers),
            None, gamma=0.001,
        ),
        "Share-MV (fully invested linear term)": apply_mean_var(
            close, open_, decision_dates, rhat_hold, month, w,
            gamma=0.001, lam=MV_LAM, kappa=MV_KAPPA, max_weight=MV_CAP,
            use_excess=False,
        ),
        "Share-MV + cash (excess value)": apply_mean_var(
            close, open_, decision_dates, rhat_hold, month, w,
            gamma=0.001, lam=MV_LAM, kappa=MV_KAPPA, max_weight=MV_CAP,
            use_excess=True,
        ),
    }
    rows = [_row(k, v, "GFC bear 2007-10-09 to 2009-03-09") for k, v in results.items()]
    extra = {
        "results": results,
        "decision_dates": decision_dates,
        "last_book": _last_book(results["Share-MV + cash (excess value)"], decision_dates),
    }
    return pd.DataFrame(rows), extra


def run_bull() -> tuple[pd.DataFrame, dict]:
    close, open_, djia = load_market_data()
    close = close.loc[:TEST_END, TICKERS]
    open_ = open_.reindex(close.index)[TICKERS].fillna(close)
    test_index = close.index[(close.index >= TEST_START) & (close.index <= TEST_END)]
    decision_dates = test_index[:-1]
    month = every_n_mask(decision_dates, 4)
    causal = load_causal()
    dow_c, dow_o, dow_pred = load_dow_panel(close.index)
    close = close.copy()
    open_ = open_.copy()
    close["DOW"] = dow_c
    open_["DOW"] = dow_o
    rhat = predicted_returns(causal["B"], close[TICKERS], decision_dates)
    rhat = rhat.copy()
    dow_px = dow_c.reindex(decision_dates)
    rhat["DOW"] = dow_pred["B"].reindex(decision_dates) / dow_px - 1.0
    rhat_hold = rhat.rolling(20, min_periods=5).mean()
    rhat_pit = apply_index_membership(rhat_hold)
    w = hysteresis_topk(rhat_pit, decision_dates[month], LP_K, LP_BUFFER)
    results = {
        "BH_DJIA": run_baselines(close, open_, djia, decision_dates)["BH_DJIA"],
        "EW_1N": apply_weights(
            close, open_, decision_dates,
            pd.DataFrame(1.0 / len(TICKERS), index=decision_dates, columns=TICKERS),
            None, gamma=0.001,
        ),
        "Share-MV (fully invested linear term)": apply_mean_var(
            close, open_, decision_dates, rhat_pit, month, w,
            gamma=0.001, lam=MV_LAM, kappa=MV_KAPPA, max_weight=MV_CAP,
            use_excess=False,
        ),
        "Share-MV + cash (excess value)": apply_mean_var(
            close, open_, decision_dates, rhat_pit, month, w,
            gamma=0.001, lam=MV_LAM, kappa=MV_KAPPA, max_weight=MV_CAP,
            use_excess=True,
        ),
    }
    rows = [_row(k, v, "Bull 2024-02-26 to 2026-02-26") for k, v in results.items()]
    extra = {
        "results": results,
        "decision_dates": decision_dates,
        "last_book": _last_book(results["Share-MV + cash (excess value)"], decision_dates),
    }
    return pd.DataFrame(rows), extra


def _plot(results: dict, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    for lab in [
        "Share-MV + cash (excess value)",
        "Share-MV (fully invested linear term)",
        "EW_1N",
        "BH_DJIA",
    ]:
        ser = series_from_result(results[lab])
        ax.plot(ser.index, ser.values / 1e6, label=lab, linewidth=1.4)
    ax.set_ylabel("Wealth (USD millions)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    bear, bear_x = run_bear()
    bull, bull_x = run_bull()
    perf = pd.concat([bear, bull], ignore_index=True)
    cols = [
        "Regime", "Strategy", "Final_Value", "Total_Return_%", "Max_Drawdown_%",
        "Sharpe_Ratio", "Turnover_%", "Avg_Stocks_Held", "Avg_Cash_%",
    ]
    perf[cols].to_csv(TAB / "table_cash_mv_regimes.csv", index=False)
    print(perf[cols].to_string(index=False))

    paths = {}
    for name, extra, title, fig_name in [
        ("bear", bear_x, "GFC bear, cash-aware share-MV", "fig_cash_mv_bear.png"),
        ("bull", bull_x, "2024-26 bull, cash-aware share-MV", "fig_cash_mv_bull.png"),
    ]:
        _plot(extra["results"], title, FIG / fig_name)
        wealth = {}
        cash_m = {}
        labels = None
        for lab, res in extra["results"].items():
            ser = series_from_result(res)
            labs, vals = _month_end(ser)
            labels = labs
            wealth[lab] = vals
        cash_m["Share-MV + cash (excess value)"] = _cash_month(
            extra["results"]["Share-MV + cash (excess value)"]
        )
        cash_m["Share-MV (fully invested linear term)"] = _cash_month(
            extra["results"]["Share-MV (fully invested linear term)"]
        )
        paths[name] = {
            "months": labels,
            "wealth_usd_m": wealth,
            "cash_pct": cash_m,
            "last_book": extra["last_book"],
        }

    meta = {
        "s0": S0,
        "gamma": 0.001,
        "kappa": MV_KAPPA,
        "lam": MV_LAM,
        "cap": MV_CAP,
        "k": LP_K,
        "buffer": LP_BUFFER,
        "note": (
            "Cash-aware book uses NC'(hat p - p^o) so residual cash competes. "
            "Same k, b, kappa, 30% cap as the proposed 2024-26 book. "
            "Proposed book tables are not overwritten."
        ),
        "bear": paths["bear"],
        "bull": paths["bull"],
    }
    OUT_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Wrote", TAB / "table_cash_mv_regimes.csv")
    print("Bear last book", paths["bear"]["last_book"])
    print("Bull last book", paths["bull"]["last_book"])


if __name__ == "__main__":
    main()
