"""Causal GFC bear test of the proposed B + hysteresis + share-MV rule.

Prices are taken from the 2006-2015 era. The evaluated bear window is
9 October 2007 through 9 March 2009 (peak-to-trough of the DJIA).
Forecasts are written to revised_outputs/checkpoints_hist so the 2024 study
cache is not touched.

The panel is a fixed set of names that existed through 2005-2009 and were
Dow members for a substantial part of 2006-2009. GM and AIG are omitted
because they failed; that is a survivorship tilt toward survivors.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cost_aware_strategies import apply_mean_var, apply_weights
from reproduce_revised_study import (
    N_JOBS,
    S0,
    _forecast_ticker_worker,
    performance_stats,
    run_baselines,
    series_from_result,
)
from weekly_causal_strategies import every_n_mask, hysteresis_topk

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "revised_outputs"
FIG = OUT / "figures"
TAB = OUT / "tables"
CKPT = OUT / "checkpoints_hist"
CACHE = OUT / "market_ohlc_hist.pkl"

# Dow-era names with continuous Yahoo history across 2005-2009.
HIST_TICKERS = [
    "AA", "AXP", "BA", "BAC", "C", "CAT", "CSCO", "CVX", "DIS", "GE",
    "HD", "HON", "HPQ", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM",
    "MRK", "MSFT", "PFE", "PG", "T", "TRV", "UNH", "VZ", "WMT", "XOM",
]
DATA_START = "2005-01-03"
DATA_END = "2015-12-31"
# Peak-to-trough DJIA bear inside 2006-2015.
BEAR_START = "2007-10-09"
BEAR_END = "2009-03-09"
LOOKBACK_START = "2006-01-03"


def _ensure() -> None:
    for p in (OUT, FIG, TAB, CKPT):
        p.mkdir(parents=True, exist_ok=True)


def load_hist_market() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    if CACHE.exists():
        obj = pd.read_pickle(CACHE)
        return obj["close"], obj["open"], obj["djia"]
    import yfinance as yf

    print("Downloading 2005-2015 DJIA-era panel and ^DJI ...", flush=True)
    raw = yf.download(
        HIST_TICKERS + ["^DJI"],
        start=DATA_START,
        end="2016-01-05",
        auto_adjust=False,
        progress=True,
        threads=True,
    )
    close_raw = raw["Close"]
    open_raw = raw["Open"]
    if isinstance(close_raw.columns, pd.MultiIndex):
        close_raw = close_raw.copy()
        open_raw = open_raw.copy()
    keep = []
    for t in HIST_TICKERS:
        if t not in close_raw.columns:
            print("missing", t, flush=True)
            continue
        s = close_raw[t]
        if s.loc[LOOKBACK_START:BEAR_END].isna().mean() > 0.02:
            print("sparse", t, flush=True)
            continue
        keep.append(t)
    close = close_raw[keep].dropna(how="any")
    open_ = open_raw[keep].reindex(close.index).fillna(close)
    if "^DJI" in close_raw.columns:
        djia = close_raw["^DJI"].reindex(close.index)
    else:
        dji = yf.download("^DJI", start=DATA_START, end="2016-01-05", auto_adjust=False, progress=False)
        djia = dji["Close"].reindex(close.index)
        if isinstance(djia, pd.DataFrame):
            djia = djia.iloc[:, 0]
    djia = djia.astype(float).ffill()
    close = close.astype(float)
    open_ = open_.astype(float)
    pd.to_pickle({"close": close, "open": open_, "djia": djia}, CACHE)
    print("Kept", len(keep), "names:", ", ".join(keep), flush=True)
    return close, open_, djia


def run_hist_walkforward(close: pd.DataFrame, decision_dates: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    _ensure()
    tickers = list(close.columns)
    results: dict[str, pd.DataFrame] = {}
    missing_by_ticker: dict[str, list[int]] = {}
    need = pd.DatetimeIndex(pd.to_datetime(decision_dates)).normalize()
    for ticker in tickers:
        path = CKPT / f"forecast_{ticker}.csv"
        have = pd.DataFrame(columns=["decision_date", "target_date", "A", "B", "C", "D", "E"])
        if path.exists():
            have = pd.read_csv(path, parse_dates=["decision_date"])
            have["decision_date"] = pd.to_datetime(have["decision_date"]).dt.normalize()
            have = have.drop_duplicates("decision_date", keep="last")
        results[ticker] = have
        have_set = set(have["decision_date"]) if len(have) else set()
        miss = [dt for dt in need if dt not in have_set]
        if miss:
            missing_by_ticker[ticker] = [int(close.index.get_loc(dt)) for dt in miss]
    if not missing_by_ticker:
        print("Historical forecasts already cover", len(need), "decision days", flush=True)
        return _assemble(results, tickers)
    n_miss = max(len(v) for v in missing_by_ticker.values())
    print(
        f"Historical CEEMDAN: {len(missing_by_ticker)} names need up to {n_miss} "
        f"days ({N_JOBS} workers).",
        flush=True,
    )
    payloads = []
    for ticker, dpos in missing_by_ticker.items():
        payloads.append(
            {
                "ticker": ticker,
                "close": close[ticker].to_numpy(dtype=float),
                "dates": close.index.astype(str).tolist(),
                "decision_pos": dpos,
            }
        )
    done = []
    if N_JOBS == 1:
        done = [_forecast_ticker_worker(p) for p in payloads]
    else:
        with ProcessPoolExecutor(max_workers=N_JOBS) as ex:
            futs = {ex.submit(_forecast_ticker_worker, p): p["ticker"] for p in payloads}
            for fut in as_completed(futs):
                ticker, df = fut.result()
                done.append((ticker, df))
                print(f"finished {ticker} (+{len(df)})", flush=True)
    for ticker, add in done:
        old = results[ticker]
        add = add.copy()
        add["decision_date"] = pd.to_datetime(add["decision_date"]).dt.normalize()
        merged = pd.concat([old, add], ignore_index=True)
        merged["decision_date"] = pd.to_datetime(merged["decision_date"]).dt.normalize()
        merged = merged.drop_duplicates("decision_date", keep="last").sort_values("decision_date")
        merged.to_csv(CKPT / f"forecast_{ticker}.csv", index=False)
        results[ticker] = merged
    out = _assemble(results, tickers)
    pd.to_pickle(out, CKPT / "causal_forecasts.pkl")
    return out


def _assemble(results: dict[str, pd.DataFrame], tickers: list[str]) -> dict[str, pd.DataFrame]:
    out = {}
    for letter in "ABCDE":
        wide = pd.DataFrame(
            {t: results[t].set_index("decision_date")[letter] for t in tickers}
        )
        wide.index = pd.to_datetime(wide.index)
        out[letter] = wide.sort_index()
    return out


def predicted_returns(pred: pd.DataFrame, close: pd.DataFrame, dates) -> pd.DataFrame:
    cols = list(close.columns)
    px = close.reindex(dates)[cols]
    hat = pred.reindex(dates)[cols]
    return hat / px - 1.0


def stats_row(label: str, res: dict) -> dict:
    s = performance_stats(res)
    s["Strategy"] = label
    return s


def main() -> None:
    _ensure()
    close_all, open_all, djia_all = load_hist_market()
    tickers = list(close_all.columns)
    close = close_all.loc[:BEAR_END, tickers]
    open_ = open_all.reindex(close.index)[tickers].fillna(close)
    djia = djia_all.reindex(close.index).ffill()
    test_index = close.index[(close.index >= BEAR_START) & (close.index <= BEAR_END)]
    decision_dates = test_index[:-1]
    print(
        f"Bear window {BEAR_START} to {BEAR_END}: {len(tickers)} names, "
        f"{len(decision_dates)} decisions",
        flush=True,
    )
    causal = run_hist_walkforward(close, decision_dates)
    pred_b = causal["B"].reindex(decision_dates)[tickers]
    rhat = predicted_returns(pred_b, close, decision_dates)
    rhat_hold = rhat.rolling(20, min_periods=5).mean()
    month = every_n_mask(decision_dates, 4)
    month_dates = decision_dates[month]
    w = hysteresis_topk(rhat_hold, month_dates, 5, 3)
    results = {}
    results["BH_DJIA"] = run_baselines(close, open_, djia, decision_dates)["BH_DJIA"]
    ew = pd.DataFrame(1.0 / len(tickers), index=decision_dates, columns=tickers)
    results["EW_1N"] = apply_weights(close, open_, decision_dates, ew, None, gamma=0.001)
    results["B monthly top-5 hysteresis EW"] = apply_weights(
        close, open_, decision_dates, w, month, gamma=0.001
    )
    results["B monthly top-5 hysteresis MV"] = apply_mean_var(
        close, open_, decision_dates, rhat_hold, month, w,
        gamma=0.001, lam=0.01, kappa=2.0, max_weight=0.30,
    )
    results["B monthly top-5 hysteresis MV 50bp"] = apply_mean_var(
        close, open_, decision_dates, rhat_hold, month, w,
        gamma=0.005, lam=0.01, kappa=2.0, max_weight=0.30,
    )
    mom = close.pct_change(20).reindex(decision_dates)
    w_mom = hysteresis_topk(mom, month_dates, 5, 3)
    results["MOM20 monthly top-5 hysteresis EW"] = apply_weights(
        close, open_, decision_dates, w_mom, month, gamma=0.001
    )

    rows = [stats_row(k, v) for k, v in results.items()]
    perf = pd.DataFrame(rows)
    perf.to_csv(TAB / "table_hist_bear_performance.csv", index=False)
    print(perf[["Strategy", "Total_Return_%", "Max_Drawdown_%", "Sharpe_Ratio", "Turnover_%"]].to_string(index=False))

    mv = results["B monthly top-5 hysteresis MV"]
    last_w = mv["weights"].loc[decision_dates].iloc[-1]
    last = last_w[last_w > 1e-8].sort_values(ascending=False)
    last.to_csv(TAB / "table_hist_bear_last_book.csv", header=["Weight"])
    print("Last book:", ", ".join(f"{n} {100*w:.1f}%" for n, w in last.items()))

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    for lab in [
        "B monthly top-5 hysteresis MV",
        "B monthly top-5 hysteresis EW",
        "MOM20 monthly top-5 hysteresis EW",
        "EW_1N",
        "BH_DJIA",
    ]:
        ser = series_from_result(results[lab])
        ax.plot(ser.index, ser.values / 1e6, label=lab, linewidth=1.4)
    ax.set_ylabel("Wealth (USD millions)")
    ax.set_title("GFC bear test, 9 Oct 2007 – 9 Mar 2009")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig_hist_bear.png", dpi=140)
    plt.close(fig)

    meta = {
        "tickers": tickers,
        "n_names": len(tickers),
        "bear_start": BEAR_START,
        "bear_end": BEAR_END,
        "n_decisions": int(len(decision_dates)),
        "note": (
            "Fixed Dow-era panel with complete 2005-2009 history. "
            "GM and AIG omitted. Same rule as the 2024-26 book: "
            "B 20d-avg, k=5, b=3, share MV kappa=2, 30% open-wealth cap."
        ),
    }
    (TAB / "table_hist_bear_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Wrote", TAB / "table_hist_bear_performance.csv")


if __name__ == "__main__":
    main()
