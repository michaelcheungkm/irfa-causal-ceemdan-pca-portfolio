# -*- coding: utf-8 -*-
"""Diagnostics on the locked fully invested book only. No invented figures."""
import json
import sys

import numpy as np
import pandas as pd

import pit_locked_rerun as p
import reproduce_revised_study as r
from ejor_referee_diagnostics import metrics_from_returns, paired_block_diff
from weekly_causal_strategies import every_n_mask
from write_pit_documents import load_panel, wealth_series

sys.stdout.reconfigure(encoding="utf-8")


def window_return(s, start, end):
    w = s[(s.index >= start) & (s.index <= end)]
    if len(w) < 2:
        return None
    return float(w.iloc[-1] / w.iloc[0] - 1.0) * 100.0


def main():
    px, op, djia, dates, scores = load_panel()
    rf = p.load_irx_daily(px.index)
    full, _w = p.monthly_book(px, op, dates, scores, 0, 0.5, rf, False)
    ew = p.ew_path(px, op, dates, scores, 0)
    wealth = wealth_series(full)
    ew_w = wealth_series(ew)
    dji = djia.reindex(px.index).ffill().dropna()
    dji = dji[(dji.index >= p.TEST_START) & (dji.index <= p.TEST_END)]
    bh = (dji / dji.iloc[0]) * r.S0

    rets = wealth.pct_change().dropna()
    bh_rets = bh.pct_change().dropna()
    common = rets.index.intersection(bh_rets.index)
    paired = paired_block_diff(rets.loc[common].to_numpy(), bh_rets.loc[common].to_numpy())

    h1_end = pd.Timestamp("2025-02-28")
    h2_start = pd.Timestamp("2025-03-03")
    halves = {
        "full_h1": window_return(wealth, p.TEST_START, h1_end),
        "full_h2": window_return(wealth, h2_start, p.TEST_END),
        "ew_h1": window_return(ew_w, p.TEST_START, h1_end),
        "ew_h2": window_return(ew_w, h2_start, p.TEST_END),
        "bh_h1": window_return(bh, p.TEST_START, h1_end),
        "bh_h2": window_return(bh, h2_start, p.TEST_END),
    }

    month = every_n_mask(dates, 4)
    reb = list(dates[month])[:-1]
    fwd = px.pct_change(20).shift(-20)
    qrows = []
    for dt in reb:
        sc = scores.loc[dt].dropna()
        if dt not in fwd.index or sc.empty:
            continue
        realized = fwd.loc[dt].reindex(sc.index).dropna()
        sc = sc.reindex(realized.index).dropna()
        realized = realized.reindex(sc.index)
        if len(sc) < 10:
            continue
        ranks = pd.qcut(sc.rank(method="first"), 5, labels=False) + 1
        for q in range(1, 6):
            qrows.append({"date": dt, "q": int(q), "ret": float(realized[ranks == q].mean())})
    qdf = pd.DataFrame(qrows)
    quint = {}
    for q, g in qdf.groupby("q"):
        quint[str(int(q))] = {"mean": float(g["ret"].mean()), "n": int(len(g))}
    spread = qdf[qdf["q"] == 5].set_index("date")["ret"] - qdf[qdf["q"] == 1].set_index("date")["ret"]
    # one-sample t on monthly Q5-Q1
    n = len(spread)
    mu = float(spread.mean())
    se = float(spread.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    tstat = mu / se if se and se > 0 else np.nan

    out = {
        "paired": paired,
        "halves": halves,
        "quintile_means": quint,
        "q5_q1_mean": mu,
        "q5_q1_t": float(tstat),
        "q5_q1_n": int(n),
        "n_common_days": int(len(common)),
    }
    path = p.OUT / "appendix_c_locked.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
