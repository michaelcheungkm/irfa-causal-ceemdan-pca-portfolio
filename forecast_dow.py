"""Causal CEEMDAN walk-forward for Dow Inc. (DOW), used only while it is in the DJIA."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

from reproduce_revised_study import (
    CKPT,
    OUT,
    TEST_END,
    TEST_START,
    _forecast_ticker_worker,
    load_market_data,
)

CACHE = OUT / "dow_ohlc.pkl"


def load_dow_ohlc(index: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    if CACHE.exists():
        obj = pd.read_pickle(CACHE)
        c = obj["close"].reindex(index).ffill()
        o = obj["open"].reindex(index)
        o = o.fillna(c)
        return c.astype(float), o.astype(float)
    raw = yf.download(
        "DOW",
        start="2017-01-01",
        end="2026-09-04",
        auto_adjust=False,
        progress=True,
        threads=False,
    )
    close = raw["Close"]
    open_ = raw["Open"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
        open_ = open_.iloc[:, 0]
    close = close.astype(float)
    open_ = open_.astype(float)
    pd.to_pickle({"close": close, "open": open_}, CACHE)
    c = close.reindex(index).ffill()
    o = open_.reindex(index).fillna(c)
    return c, o


def main() -> None:
    CKPT.mkdir(parents=True, exist_ok=True)
    close, _open, _djia = load_market_data()
    close = close.loc[:TEST_END]
    dow_c, _dow_o = load_dow_ohlc(close.index)
    test_index = close.index[(close.index >= TEST_START) & (close.index <= TEST_END)]
    decision_dates = test_index[:-1]
    need = pd.DatetimeIndex(pd.to_datetime(decision_dates)).normalize()

    path = CKPT / "forecast_DOW.csv"
    have = pd.DataFrame(columns=["decision_date", "target_date", "A", "B", "C", "D", "E"])
    if path.exists():
        have = pd.read_csv(path, parse_dates=["decision_date"])
        have["decision_date"] = pd.to_datetime(have["decision_date"]).dt.normalize()
        have = have.drop_duplicates("decision_date", keep="last")
    have_set = set(have["decision_date"]) if len(have) else set()
    miss = [dt for dt in need if dt not in have_set]
    if not miss:
        print("DOW forecasts already cover", len(need), "decision days")
        return
    dpos = [int(close.index.get_loc(dt)) for dt in miss]
    print(f"Walk-forward CEEMDAN for DOW: {len(dpos)} days")
    payload = {
        "ticker": "DOW",
        "close": dow_c.reindex(close.index).values,
        "dates": close.index.astype(str).tolist(),
        "decision_pos": dpos,
    }
    ticker, add = _forecast_ticker_worker(payload)
    add["decision_date"] = pd.to_datetime(add["decision_date"]).dt.normalize()
    merged = pd.concat([have, add], ignore_index=True)
    merged["decision_date"] = pd.to_datetime(merged["decision_date"]).dt.normalize()
    merged = merged.drop_duplicates("decision_date", keep="last").sort_values("decision_date")
    merged.to_csv(path, index=False)
    print("Wrote", path, "rows", len(merged))


if __name__ == "__main__":
    main()
