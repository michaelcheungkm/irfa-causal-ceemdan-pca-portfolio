# -*- coding: utf-8 -*-
"""Point-in-time Dow rerun with the trading rule locked before February 2024."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

import reproduce_revised_study as r
from cost_aware_strategies import apply_mean_var
from regime_corrections import load_irx_daily
from weekly_causal_strategies import every_n_mask, hysteresis_topk

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "revised_outputs" / "pit_locked"
OUT.mkdir(parents=True, exist_ok=True)

NVDA_IN = pd.Timestamp("2024-11-08")
SHW_IN = pd.Timestamp("2024-11-08")
AMZN_IN = pd.Timestamp("2024-02-26")
PANEL_2020 = pd.Timestamp("2020-08-31")
TRAIN_END = pd.Timestamp("2024-02-23")
TEST_START = pd.Timestamp("2024-02-26")
TEST_END = pd.Timestamp("2026-02-26")
K = 5
BUFFERS = (0, 1, 2, 3, 4, 5, 7, 9)
KAPPAS = (0.5, 1.0, 2.0, 4.0)

# Dow membership from the 31 Aug 2020 reconstitution through the session
# before Amazon replaced Walgreens.
TRAIN_NAMES = [
    "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS", "DOW",
    "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM",
    "MRK", "MSFT", "NKE", "PG",     "TRV", "UNH", "V", "VZ", "WMT",
]


def download_extra(tickers: list[str], start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False, threads=True)
    close, open_ = raw["Close"], raw["Open"]
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
        open_ = open_.to_frame(tickers[0])
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    open_.index = close.index
    return close.astype(float), open_.astype(float)


def forecast_intc(close: pd.Series, dates: pd.DatetimeIndex) -> pd.DataFrame:
    path = OUT / "forecast_INTC.csv"
    have = pd.DataFrame(columns=["decision_date", "B"])
    if path.exists():
        have = pd.read_csv(path, parse_dates=["decision_date"])
    done = set(pd.to_datetime(have["decision_date"]).dt.normalize()) if len(have) else set()
    rows = []
    idx = close.dropna().index
    for i, dt in enumerate(dates):
        if dt in done or dt not in idx:
            continue
        loc = idx.get_loc(dt)
        if isinstance(loc, slice):
            continue
        if loc < 400:
            continue
        window = close.loc[idx[: loc + 1]].iloc[-400:].to_numpy(dtype=float)
        pred = r.forecast_variants(window, seed=17)
        rows.append({"decision_date": dt, "B": pred["B"]})
        if (i + 1) % 20 == 0:
            print(f"INTC {i + 1}/{len(dates)}", flush=True)
            chk = pd.concat([have, pd.DataFrame(rows)], ignore_index=True)
            chk.to_csv(path, index=False)
    add = pd.DataFrame(rows)
    merged = pd.concat([have, add], ignore_index=True)
    if len(merged):
        merged["decision_date"] = pd.to_datetime(merged["decision_date"]).dt.normalize()
        merged = merged.drop_duplicates("decision_date", keep="last").sort_values("decision_date")
        merged.to_csv(path, index=False)
    return merged


def load_model_b(close: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    frames = []
    ckpt = ROOT / "revised_outputs" / "checkpoints"
    for ticker in close.columns:
        path = ckpt / f"forecast_{ticker}.csv"
        if ticker == "DOW":
            path = ckpt / "forecast_DOW.csv"
        if not path.exists():
            continue
        pred = pd.read_csv(path, parse_dates=["decision_date"]).set_index("decision_date")
        pred.index = pd.to_datetime(pred.index).normalize()
        px = close[ticker].reindex(dates)
        hat = pred["B"].reindex(dates)
        frames.append((hat / px - 1.0).rename(ticker))
    out = pd.concat(frames, axis=1)
    return out.rolling(20, min_periods=5).mean()


def pit_mask(scores: pd.DataFrame) -> pd.DataFrame:
    out = scores.copy()
    rules = {
        "NVDA": lambda ix: ix < NVDA_IN,
        "SHW": lambda ix: ix < SHW_IN,
        "AMZN": lambda ix: ix < AMZN_IN,
        "DOW": lambda ix: ix >= NVDA_IN,
        "INTC": lambda ix: ix >= NVDA_IN,
        "WBA": lambda ix: ix >= AMZN_IN,
    }
    for ticker, bad in rules.items():
        if ticker in out.columns:
            out.loc[bad(out.index), ticker] = np.nan
    return out


def metrics(result: dict) -> dict:
    stats = r.performance_stats(result)
    cash = np.asarray(result.get("cash_frac", []), dtype=float)
    stats["Avg_Cash_%"] = float(np.nanmean(cash) * 100.0) if len(cash) else np.nan
    return stats


def monthly_book(close, open_, dates, scores, buffer, kappa, rf, excess: bool):
    month = every_n_mask(dates, 4)
    reb = dates[month]
    w = hysteresis_topk(scores.reindex(dates), reb, K, buffer)
    res = apply_mean_var(
        close, open_, dates, scores.reindex(dates), month, w,
        lam=0.01, kappa=kappa, max_weight=0.30, use_excess=excess, rf=rf,
    )
    return res, w


def ew_path(close, open_, dates, scores, buffer):
    """Equal-weight hysteresis, next-open fills, 10 bps, cash residual earns no interest."""
    month = every_n_mask(dates, 4)
    reb = dates[month]
    w = hysteresis_topk(scores.reindex(dates), reb, K, buffer)
    # Map equal weights onto the share engine with kappa 0 and a 20% cap so five names stay equal.
    res = apply_mean_var(
        close, open_, dates, scores.reindex(dates), month, w,
        lam=0.0, kappa=0.0, max_weight=0.20, use_excess=False, rf=None,
    )
    return res


def main() -> None:
    cache = pd.read_pickle(ROOT / "revised_outputs" / "market_ohlc.pkl")
    close0, open0, djia = cache["close"], cache["open"], cache["djia"]
    extra_c, extra_o = download_extra(["INTC", "DOW"], "2017-01-01", "2026-03-01")
    close = close0.join(extra_c, how="left")
    open_ = open0.join(extra_o, how="left")
    open_ = open_.fillna(close)
    print("prices", close.index.min().date(), close.index.max().date(), flush=True)

    test_idx = close.index[(close.index >= TEST_START) & (close.index <= TEST_END)]
    test_dates = test_idx[:-1]
    intc_need = test_dates[test_dates < NVDA_IN]
    print("forecasting INTC days", len(intc_need), flush=True)
    forecast_intc(close["INTC"].dropna(), intc_need)
    intc_hat = pd.read_csv(OUT / "forecast_INTC.csv", parse_dates=["decision_date"]).set_index("decision_date")
    intc_hat.index = pd.to_datetime(intc_hat.index).normalize()

    # --- training lock on momentum, Aug 2020 panel, through 23 Feb 2024 ---
    train_idx = close.index[(close.index >= PANEL_2020) & (close.index <= TRAIN_END)]
    train_dates = train_idx[:-1]
    train_px = close.loc[:TRAIN_END, TRAIN_NAMES].dropna(how="any")
    train_open = open_.reindex(train_px.index)[TRAIN_NAMES].fillna(train_px)
    mom = train_px.pct_change(20)
    mom_scores = mom.reindex(train_dates)
    rf_train = load_irx_daily(train_px.index)
    print("training days", len(train_dates), "names", mom_scores.shape[1], flush=True)

    buffer_rows = []
    best_b, best_sh = None, -1e9
    for b in BUFFERS:
        res = ew_path(train_px, train_open, train_dates, mom_scores, b)
        st = metrics(res)
        st["buffer"] = b
        buffer_rows.append(st)
        print(f"train EW b={b} ret={st['Total_Return_%']:.2f} sharpe={st['Sharpe_Ratio']:.2f}", flush=True)
        if st["Sharpe_Ratio"] > best_sh:
            best_sh = st["Sharpe_Ratio"]
            best_b = b
    pd.DataFrame(buffer_rows).to_csv(OUT / "train_buffer.csv", index=False)

    kappa_rows = []
    best_k, best_ksh = None, -1e9
    for kappa in KAPPAS:
        res, _w = monthly_book(train_px, train_open, train_dates, mom_scores, best_b, kappa, rf_train, True)
        st = metrics(res)
        st["kappa"] = kappa
        st["buffer"] = best_b
        kappa_rows.append(st)
        print(f"train excess k={kappa} ret={st['Total_Return_%']:.2f} sharpe={st['Sharpe_Ratio']:.2f} cash={st['Avg_Cash_%']:.1f}", flush=True)
        if st["Sharpe_Ratio"] > best_ksh:
            best_ksh = st["Sharpe_Ratio"]
            best_k = kappa
    pd.DataFrame(kappa_rows).to_csv(OUT / "train_kappa.csv", index=False)

    # --- held-out PIT Model B ---
    b_scores = load_model_b(close, test_dates)
    intc_px = close["INTC"].reindex(test_dates)
    b_scores["INTC"] = (intc_hat["B"].reindex(test_dates) / intc_px - 1.0).rolling(20, min_periods=5).mean()
    b_scores = pit_mask(b_scores)
    test_names = [c for c in b_scores.columns if b_scores[c].notna().any()]
    test_px = close.loc[:TEST_END, test_names]
    test_open = open_.reindex(test_px.index)[test_names].fillna(test_px)
    rf_test = load_irx_daily(test_px.index)
    mom_test = pit_mask(test_px.pct_change(20).reindex(test_dates))

    rows = []
    books = {
        "Model B fully invested": monthly_book(test_px, test_open, test_dates, b_scores, best_b, best_k, rf_test, False)[0],
        "Model B Treasury-bill hurdle": monthly_book(test_px, test_open, test_dates, b_scores, best_b, best_k, rf_test, True)[0],
        "Momentum equal weight": ew_path(test_px, test_open, test_dates, mom_test, best_b),
        "Model B equal weight": ew_path(test_px, test_open, test_dates, b_scores, best_b),
    }
    for name, res in books.items():
        st = metrics(res)
        st["Strategy"] = name
        rows.append(st)
        print(name, round(st["Total_Return_%"], 2), "sharpe", round(st["Sharpe_Ratio"], 2), "cash", round(st["Avg_Cash_%"], 1), flush=True)

    dji = djia.reindex(test_px.index).ffill().dropna()
    bh_v = dji / dji.iloc[0] * r.S0
    bh = {"S_values": bh_v.to_numpy(), "turnover_path": np.zeros(len(bh_v) - 1), "NCs": [], "cash_frac": np.zeros(len(bh_v) - 1)}
    st = metrics(bh)
    st["Strategy"] = "Buy-and-Hold DJIA"
    rows.append(st)
    pd.DataFrame(rows).to_csv(OUT / "test_pit.csv", index=False)

    # eligibility audit
    held = books["Model B fully invested"]["weights"]
    shw = held["SHW"] if "SHW" in held.columns else pd.Series(dtype=float)
    audit = {
        "locked_buffer": int(best_b),
        "locked_kappa": float(best_k),
        "train_start": str(PANEL_2020.date()),
        "train_end": str(TRAIN_END.date()),
        "train_rule": "Maximize equal-weight 20-day momentum Sharpe over buffers, then excess-value Sharpe over kappa, on the 31 Aug 2020 Dow panel excluding Walgreens, whose prices are no longer on the price source.",
        "shw_weight_before_join": float(shw[shw.index < SHW_IN].abs().max()) if len(shw) else None,
        "intc_forecast_days": int(intc_need.size),
        "test_return_fully_invested": rows[0]["Total_Return_%"],
        "test_return_hurdle": rows[1]["Total_Return_%"],
        "test_cash_hurdle": rows[1]["Avg_Cash_%"],
    }
    (OUT / "lock_meta.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
