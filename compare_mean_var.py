"""Sweep Tsang-style mean-variance-turnover weights on the PIT B hysteresis set."""
from __future__ import annotations

import pandas as pd

from compare_constructions import _row, _setup
from cost_aware_strategies import apply_mean_var, apply_share_lp, apply_weights
from reproduce_revised_study import TAB
from weekly_causal_strategies import hysteresis_topk, _implied_prices


def run():
    close, open_, djia, decision_dates, month, month_dates, rhats_pit, hat_p = _setup()
    rows = []
    grids = [
        (5, 3, None, 0.01, 2.0, 1.0, "MV k=5 kappa=2"),
        (5, 3, None, 0.01, 4.0, 1.0, "MV k=5 kappa=4"),
        (5, 3, None, 0.01, 6.0, 1.0, "MV k=5 kappa=6"),
        (6, 3, None, 0.01, 4.0, 1.0, "MV k=6 kappa=4"),
        (6, 3, None, 0.01, 6.0, 1.0, "MV k=6 kappa=6"),
        (5, 3, None, 0.01, 1.0, 0.25, "MV k=5 kappa=1 25% cap"),
        (6, 3, None, 0.01, 1.0, 0.25, "MV k=6 kappa=1 25% cap"),
        (6, 3, None, 0.01, 2.0, 0.30, "MV k=6 kappa=2 30% cap"),
        (5, 3, None, 0.01, 2.0, 0.30, "MV k=5 kappa=2 30% cap"),
    ]
    for k, buf, mx, lam, kappa, cap, label in grids:
        w = hysteresis_topk(rhats_pit, month_dates, k, buf, max_replace=mx)
        for g in (0.001, 0.005):
            res = apply_mean_var(
                close, open_, decision_dates, rhats_pit, month, w,
                gamma=g, lam=lam, kappa=kappa, max_weight=cap,
            )
            row, _ = _row(label, "MV", k, cap, res, month_dates, g)
            rows.append(row)
        print("done", label)

    w5 = hysteresis_topk(rhats_pit, month_dates, 5, 3)
    hat_p = _implied_prices(close, rhats_pit, decision_dates)
    for g in (0.001, 0.005):
        res = apply_weights(close, open_, decision_dates, w5, month, gamma=g)
        res = dict(res)
        res["weights"] = w5
        row, _ = _row("EW k=5 hysteresis", "EW", 5, 0.2, res, month_dates, g)
        rows.append(row)
        res_lp = apply_share_lp(
            close, open_, decision_dates, hat_p, month, select=w5, gamma=g, max_weight=0.25
        )
        row, _ = _row("LP k=5 25% cap", "LP", 5, 0.25, res_lp, month_dates, g)
        rows.append(row)
    print("done EW/LP k=5")

    df = pd.DataFrame(rows)
    df.to_csv(TAB / "table_mean_var_sweep.csv", index=False)
    ten = df[df["gamma"] == 0.001]
    print("\n10 bp")
    print(
        ten[["Strategy", "Return_%", "Max_DD_%", "Avg_names", "Mean_replace_%",
             "Full_purges", "Turnover_%", "Sharpe"]].to_string(
            index=False, float_format=lambda x: f"{x:.2f}"
        )
    )
    print("\n50 bp")
    print(
        df[df["gamma"] == 0.005][["Strategy", "Return_%", "Max_DD_%"]].to_string(
            index=False, float_format=lambda x: f"{x:.2f}"
        )
    )


if __name__ == "__main__":
    run()
