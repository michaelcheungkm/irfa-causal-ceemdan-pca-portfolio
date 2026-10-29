"""Test whether a looser-than-1/k share LP can beat EW on the same candidate set."""
from __future__ import annotations

import pandas as pd

from compare_constructions import _churn_stats, _row, _setup
from cost_aware_strategies import apply_share_lp, apply_weights
from reproduce_revised_study import TAB
from weekly_causal_strategies import hysteresis_topk


def run():
    close, open_, djia, decision_dates, month, month_dates, rhats_pit, hat_p = _setup()
    specs = [
        ("EW k=5", "EW", 5, 0.20, None, 3, None),
        ("LP k=5, 25% cap", "LP", 5, 0.25, 0.25, 3, None),
        ("EW k=6", "EW", 6, 1.0 / 6.0, None, 3, None),
        ("LP k=6, 1/6 cap", "LP", 6, 1.0 / 6.0, 1.0 / 6.0, 3, None),
        ("LP k=6, 25% cap", "LP", 6, 0.25, 0.25, 3, None),
        ("EW k=8", "EW", 8, 0.125, None, 3, None),
        ("LP k=8, 1/6 cap", "LP", 8, 1.0 / 6.0, 1.0 / 6.0, 3, None),
        ("LP k=8, 20% cap", "LP", 8, 0.20, 0.20, 3, None),
        ("EW k=10", "EW", 10, 0.10, None, 3, None),
        ("LP k=10, 1/6 cap", "LP", 10, 1.0 / 6.0, 1.0 / 6.0, 3, None),
        ("EW k=8 max-replace 3", "EW", 8, 0.125, None, 3, 3),
        ("LP k=8 1/6 max-replace 3", "LP", 8, 1.0 / 6.0, 1.0 / 6.0, 3, 3),
        ("EW k=6 max-replace 3", "EW", 6, 1.0 / 6.0, None, 3, 3),
        ("LP k=6 25% max-replace 3", "LP", 6, 0.25, 0.25, 3, 3),
    ]
    rows = []
    for label, method, k, cap, lp_cap, buf, mx in specs:
        w = hysteresis_topk(rhats_pit, month_dates, k, buf, max_replace=mx)
        for g in (0.001, 0.005):
            if method == "EW":
                res = apply_weights(close, open_, decision_dates, w, month, gamma=g)
                res = dict(res)
                res["weights"] = w
            else:
                res = apply_share_lp(
                    close, open_, decision_dates, hat_p, month,
                    select=w, gamma=g, max_weight=lp_cap,
                )
            row, _ = _row(label, method, k, cap, res, month_dates, g)
            rows.append(row)
        print("done", label)

    df = pd.DataFrame(rows)
    df.to_csv(TAB / "table_lp_vs_ew.csv", index=False)
    ten = df[df["gamma"] == 0.001]
    print("\n10 bp")
    print(ten[["Strategy", "Return_%", "Max_DD_%", "Avg_names", "Mean_replace_%",
               "Full_purges", "Turnover_%", "Sharpe"]].to_string(
        index=False, float_format=lambda x: f"{x:.2f}"))
    print("\n50 bp")
    print(df[df["gamma"] == 0.005][["Strategy", "Return_%", "Max_DD_%"]].to_string(
        index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    run()
