# -*- coding: utf-8 -*-
"""Holding streaks and replacement rate on the locked b=0, kappa=0.5 book."""
import json
import sys

import numpy as np
import pandas as pd

import pit_locked_rerun as p
from weekly_causal_strategies import every_n_mask
from write_pit_documents import load_panel

sys.stdout.reconfigure(encoding="utf-8")


def streaks(weights, thr=1e-4):
    rows = []
    for c in weights.columns:
        held = (weights[c] > thr).to_numpy()
        if not held.any():
            continue
        runs = []
        run = 0
        for flag in held:
            if flag:
                run += 1
            elif run:
                runs.append(run)
                run = 0
        if run:
            runs.append(run)
        w = weights.loc[weights[c] > thr, c]
        rows.append({
            "ticker": c,
            "sessions": int(held.sum()),
            "n_streaks": len(runs),
            "max_streak": int(max(runs)),
            "mean_weight": float(w.mean()),
        })
    return pd.DataFrame(rows).sort_values("sessions", ascending=False)


def replacement(weights, reb_dates, thr=1e-4):
    dates = [dt for dt in reb_dates if dt in weights.index]
    exits = []
    purges = 0
    for prev, cur in zip(dates, dates[1:]):
        a = set(weights.columns[weights.loc[prev] > thr])
        b = set(weights.columns[weights.loc[cur] > thr])
        if not a:
            continue
        left = len(a - b) / len(a)
        exits.append(left)
        if a.isdisjoint(b):
            purges += 1
    return float(np.mean(exits)) if exits else np.nan, purges, len(exits)


def main():
    px, op, djia, dates, scores = load_panel()
    rf = p.load_irx_daily(px.index)
    full, _w = p.monthly_book(px, op, dates, scores, 0, 0.5, rf, False)
    wt = full["weights"]
    month = every_n_mask(dates, 4)
    reb = list(dates[month])
    st = streaks(wt)
    rate, purges, n = replacement(wt, reb)
    st.to_csv(p.OUT / "locked_streaks.csv", index=False)
    # rebalance names
    rows = []
    for dt in reb:
        if dt not in wt.index:
            continue
        held = wt.loc[dt]
        held = held[held > 1e-4].sort_values(ascending=False)
        rows.append({
            "date": pd.Timestamp(dt).strftime("%Y-%m-%d"),
            "names": ", ".join(f"{k} {v:.1%}" for k, v in held.items()),
        })
    pd.DataFrame(rows).to_csv(p.OUT / "locked_rebalance.csv", index=False)
    out = {
        "replace_rate": rate,
        "purges": purges,
        "n_rebalances_compared": n,
        "top": st.head(8).to_dict(orient="records"),
    }
    (p.OUT / "locked_streaks.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
