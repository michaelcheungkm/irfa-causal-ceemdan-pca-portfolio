"""Corrections around the cash-aware book: T-bill on cash, kappa grid, AIG/GM.

Does not overwrite the 2024-26 proposed-book tables.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from cost_aware_strategies import apply_mean_var, apply_weights, load_causal, predicted_returns
from historical_bear_test import (
    BEAR_END,
    BEAR_START,
    CKPT,
    DATA_START,
    load_hist_market,
    predicted_returns as hist_predicted_returns,
    run_hist_walkforward,
)
from reproduce_revised_study import (
    S0,
    TAB,
    TEST_END,
    TEST_START,
    TICKERS,
    load_market_data,
    performance_stats,
    run_baselines,
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

KAPPAS = (0.5, 1.0, 2.0, 4.0)
FAILED = ["AIG", "GM"]


def load_irx_daily(index: pd.DatetimeIndex) -> pd.Series:
    raw = yf.download("^IRX", start="2005-01-01", end="2026-03-01", auto_adjust=False, progress=False)
    px = raw["Close"]
    if isinstance(px, pd.DataFrame):
        px = px.iloc[:, 0]
    ann = px.astype(float).reindex(index).ffill() / 100.0
    return (ann / 252.0).rename("rf")


def extra_ohlc(tickers: list[str], index: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    raw = yf.download(
        tickers,
        start=DATA_START,
        end="2009-03-15",
        auto_adjust=False,
        progress=True,
        threads=True,
    )
    close_raw = raw["Close"]
    open_raw = raw["Open"]
    if isinstance(close_raw, pd.Series):
        close_raw = close_raw.to_frame(tickers[0])
        open_raw = open_raw.to_frame(tickers[0])
    keep = []
    for t in tickers:
        if t not in close_raw.columns:
            print("missing", t, flush=True)
            continue
        s = close_raw[t].reindex(index)
        bear = s.loc[BEAR_START:BEAR_END]
        if bear.isna().mean() > 0.15:
            print("sparse", t, "na", float(bear.isna().mean()), flush=True)
            continue
        keep.append(t)
    if not keep:
        return pd.DataFrame(index=index), pd.DataFrame(index=index), []
    close = close_raw[keep].reindex(index).ffill()
    open_ = open_raw[keep].reindex(index).fillna(close)
    return close.astype(float), open_.astype(float), keep


def _row(label: str, res: dict, regime: str, kappa=None) -> dict:
    s = performance_stats(res)
    s["Strategy"] = label
    s["Regime"] = regime
    s["kappa"] = kappa
    cf = np.asarray(res.get("cash_frac", []), dtype=float)
    s["Avg_Cash_%"] = float(100.0 * cf.mean()) if len(cf) else np.nan
    return s


def _mv(close, open_, dates, rhat, month, w, kappa, use_excess, rf=None):
    return apply_mean_var(
        close, open_, dates, rhat, month, w,
        gamma=0.001, lam=MV_LAM, kappa=kappa, max_weight=MV_CAP,
        use_excess=use_excess, rf=rf,
    )


def bear_panel():
    close_all, open_all, djia_all = load_hist_market()
    tickers = list(close_all.columns)
    close = close_all.loc[:BEAR_END, tickers]
    extra_c, extra_o, kept = extra_ohlc(FAILED, close.index)
    if kept:
        close = close.join(extra_c[kept], how="left")
        open_base = open_all.reindex(index=close.index, columns=tickers).fillna(close[tickers])
        open_ = open_base.join(extra_o[kept].reindex(close.index), how="left")
        close = close.ffill()
        open_ = open_.reindex(index=close.index, columns=close.columns).fillna(close)
        print("Added failed names:", ", ".join(kept), flush=True)
    else:
        open_ = open_all.reindex(close.index)[tickers].fillna(close)
        print("No failed-name history on Yahoo", flush=True)
    djia = djia_all.reindex(close.index).ffill()
    test_index = close.index[(close.index >= BEAR_START) & (close.index <= BEAR_END)]
    decision_dates = test_index[:-1]
    causal = run_hist_walkforward(close, decision_dates)
    rhat = hist_predicted_returns(causal["B"].reindex(decision_dates), close, decision_dates)
    rhat_hold = rhat.rolling(20, min_periods=5).mean()
    month = every_n_mask(decision_dates, 4)
    w = hysteresis_topk(rhat_hold, decision_dates[month], LP_K, LP_BUFFER)
    rf = load_irx_daily(close.index)
    return close, open_, djia, decision_dates, rhat_hold, month, w, rf, kept


def bull_panel():
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
    rhat["DOW"] = dow_pred["B"].reindex(decision_dates) / dow_c.reindex(decision_dates) - 1.0
    rhat_pit = apply_index_membership(rhat.rolling(20, min_periods=5).mean())
    w = hysteresis_topk(rhat_pit, decision_dates[month], LP_K, LP_BUFFER)
    rf = load_irx_daily(close.index)
    return close, open_, djia, decision_dates, rhat_pit, month, w, rf


def main() -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    rows = []

    close, open_, djia, dates, rhat, month, w, rf, kept = bear_panel()
    regime = "GFC bear 2007-10-09 to 2009-03-09"
    n = close.shape[1]
    rows.append(_row("BH_DJIA", run_baselines(close, open_, djia, dates)["BH_DJIA"], regime))
    rows.append(_row(
        "EW_1N",
        apply_weights(
            close, open_, dates,
            pd.DataFrame(1.0 / n, index=dates, columns=close.columns),
            None, gamma=0.001,
        ),
        regime,
    ))
    rows.append(_row(
        "Proposed share-MV",
        _mv(close, open_, dates, rhat, month, w, MV_KAPPA, False, None),
        regime, MV_KAPPA,
    ))
    rows.append(_row(
        "Share-MV + cash, rf=0",
        _mv(close, open_, dates, rhat, month, w, MV_KAPPA, True, None),
        regime, MV_KAPPA,
    ))
    for k in KAPPAS:
        rows.append(_row(
            f"Share-MV + cash + T-bill, kappa={k:g}",
            _mv(close, open_, dates, rhat, month, w, k, True, rf),
            regime, k,
        ))

    close, open_, djia, dates, rhat, month, w, rf = bull_panel()
    regime = "Bull 2024-02-26 to 2026-02-26"
    rows.append(_row("BH_DJIA", run_baselines(close, open_, djia, dates)["BH_DJIA"], regime))
    rows.append(_row(
        "EW_1N",
        apply_weights(
            close, open_, dates,
            pd.DataFrame(1.0 / len(TICKERS), index=dates, columns=TICKERS),
            None, gamma=0.001,
        ),
        regime,
    ))
    rows.append(_row(
        "Proposed share-MV",
        _mv(close, open_, dates, rhat, month, w, MV_KAPPA, False, None),
        regime, MV_KAPPA,
    ))
    rows.append(_row(
        "Share-MV + cash, rf=0",
        _mv(close, open_, dates, rhat, month, w, MV_KAPPA, True, None),
        regime, MV_KAPPA,
    ))
    for k in KAPPAS:
        rows.append(_row(
            f"Share-MV + cash + T-bill, kappa={k:g}",
            _mv(close, open_, dates, rhat, month, w, k, True, rf),
            regime, k,
        ))

    perf = pd.DataFrame(rows)
    cols = [
        "Regime", "Strategy", "kappa", "Final_Value", "Total_Return_%",
        "Max_Drawdown_%", "Sharpe_Ratio", "Turnover_%", "Avg_Stocks_Held", "Avg_Cash_%",
    ]
    perf[cols].to_csv(TAB / "table_cash_mv_corrections.csv", index=False)
    print(perf[cols].to_string(index=False))
    meta = {
        "s0": S0,
        "failed_added": kept,
        "kappas": list(KAPPAS),
        "note": (
            "T-bill is ^IRX/252 on idle cash. Excess-value hurdle subtracts "
            "the 21-day T-bill from predicted dollar gain. Proposed book is "
            "the fully invested linear term and is not replaced."
        ),
    }
    (TAB / "table_cash_mv_corrections_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print("Wrote", TAB / "table_cash_mv_corrections.csv")
    print("Failed names kept:", kept)


if __name__ == "__main__":
    main()
