"""Compare equal-weight vs share-LP books across caps and rotation limits."""
from __future__ import annotations

import numpy as np
import pandas as pd

from cost_aware_strategies import apply_share_lp, apply_weights, load_causal, predicted_returns
from reproduce_revised_study import TAB, TEST_END, TEST_START, TICKERS, load_market_data, performance_stats
from weekly_causal_strategies import (
    apply_index_membership,
    every_n_mask,
    hysteresis_topk,
    load_dow_panel,
    rebalance_churn,
    _implied_prices,
)


def _setup():
    close, open_, djia = load_market_data()
    close = close.loc[:TEST_END, TICKERS]
    open_ = open_.reindex(close.index)[TICKERS].fillna(close)
    djia = djia.reindex(close.index).ffill()
    test_index = close.index[(close.index >= TEST_START) & (close.index <= TEST_END)]
    decision_dates = test_index[:-1]
    month = every_n_mask(decision_dates, 4)
    month_dates = decision_dates[month]
    causal = load_causal()
    dow_c, dow_o, dow_pred = load_dow_panel(close.index)
    close = close.copy()
    open_ = open_.copy()
    close["DOW"] = dow_c
    open_["DOW"] = dow_o
    rhats = {L: predicted_returns(pred, close[TICKERS], decision_dates) for L, pred in causal.items()}
    dow_px = dow_c.reindex(decision_dates)
    for L in ["A", "B", "C", "D", "E"]:
        rhats[L] = rhats[L].copy()
        rhats[L]["DOW"] = dow_pred[L].reindex(decision_dates) / dow_px - 1.0
    rhats_hold = {L: r.rolling(20, min_periods=5).mean() for L, r in rhats.items()}
    rhats_pit = apply_index_membership(rhats_hold["B"])
    hat_p = _implied_prices(close, rhats_pit, decision_dates)
    return close, open_, djia, decision_dates, month, month_dates, rhats_pit, hat_p


def _churn_stats(weights, month_dates):
    ch = rebalance_churn(weights, month_dates)
    filled = ch.dropna(subset=["Replace_%"])
    return {
        "Mean_replace_%": float(filled["Replace_%"].mean()) if len(filled) else float("nan"),
        "Full_purges": int(filled["Full_purge"].sum()) if len(filled) else 0,
        "Max_replace_%": float(filled["Replace_%"].max()) if len(filled) else float("nan"),
        "churn": ch,
    }


def _row(label, method, k, cap, res, month_dates, gamma, extra=None):
    stats = performance_stats(res)
    w = res.get("weights")
    if w is None:
        # equal-weight target; reconstruct from last positive NCs is messy — use avg names
        ch = {"Mean_replace_%": float("nan"), "Full_purges": 0, "Max_replace_%": float("nan"), "churn": None}
        avg_names = float(stats.get("Avg_Stocks_Held", float("nan")))
    else:
        ch = _churn_stats(w, month_dates)
        avg_names = float((w.gt(0.01).sum(axis=1)).replace(0, np.nan).mean())
    out = {
        "Strategy": label,
        "Method": method,
        "k": k,
        "Cap": cap,
        "gamma": gamma,
        "Return_%": float(stats["Total_Return_%"]),
        "Sharpe": float(stats["Sharpe_Ratio"]),
        "Max_DD_%": float(stats["Max_Drawdown_%"]),
        "Turnover_%": float(stats["Turnover_%"]),
        "Avg_names": avg_names,
        "Mean_replace_%": ch["Mean_replace_%"],
        "Full_purges": ch["Full_purges"],
        "Max_replace_%": ch["Max_replace_%"],
        "Final_value": float(stats["Final_Value"]),
    }
    if extra:
        out.update(extra)
    return out, ch["churn"]


def run():
    close, open_, djia, decision_dates, month, month_dates, rhats_pit, hat_p = _setup()
    rows = []
    churns = {}

    constructions = [
        ("EW k=4", "EW", 4, 0.25, None, 3, 0, None),
        ("LP k=4, 25% cap", "LP", 4, 0.25, 0.25, 3, 0, None),
        ("EW k=5", "EW", 5, 0.20, None, 3, 0, None),
        ("LP k=5, 25% cap", "LP", 5, 0.25, 0.25, 3, 0, None),
        ("LP k=5, 20% cap", "LP", 5, 0.20, 0.20, 3, 0, None),
        ("EW k=6", "EW", 6, 1.0 / 6.0, None, 3, 0, None),
        ("LP k=6, 1/6 cap", "LP", 6, 1.0 / 6.0, 1.0 / 6.0, 3, 0, None),
        ("LP k=6, 20% cap", "LP", 6, 0.20, 0.20, 3, 0, None),
        ("LP k=6, 25% cap", "LP", 6, 0.25, 0.25, 3, 0, None),
        ("EW k=8", "EW", 8, 0.125, None, 3, 0, None),
        ("LP k=8, 15% cap", "LP", 8, 0.15, 0.15, 3, 0, None),
        ("LP k=8, 1/8 cap", "LP", 8, 0.125, 0.125, 3, 0, None),
        ("EW k=6 max-replace 1", "EW", 6, 1.0 / 6.0, None, 3, 0, 1),
        ("LP k=6 1/6 max-replace 1", "LP", 6, 1.0 / 6.0, 1.0 / 6.0, 3, 0, 1),
        ("EW k=6 max-replace 2", "EW", 6, 1.0 / 6.0, None, 3, 0, 2),
        ("LP k=6 1/6 max-replace 2", "LP", 6, 1.0 / 6.0, 1.0 / 6.0, 3, 0, 2),
        ("EW k=6 max-replace 3", "EW", 6, 1.0 / 6.0, None, 3, 0, 3),
        ("LP k=6 1/6 max-replace 3", "LP", 6, 1.0 / 6.0, 1.0 / 6.0, 3, 0, 3),
        ("EW k=6 min-hold 2", "EW", 6, 1.0 / 6.0, None, 3, 2, None),
        ("LP k=6 1/6 min-hold 2", "LP", 6, 1.0 / 6.0, 1.0 / 6.0, 3, 2, None),
        ("EW k=6 min-hold 3", "EW", 6, 1.0 / 6.0, None, 3, 3, None),
        ("LP k=6 1/6 min-hold 3", "LP", 6, 1.0 / 6.0, 1.0 / 6.0, 3, 3, None),
        ("EW k=6 b=7", "EW", 6, 1.0 / 6.0, None, 7, 0, None),
        ("LP k=6 1/6 b=7", "LP", 6, 1.0 / 6.0, 1.0 / 6.0, 7, 0, None),
        ("EW k=6 max-replace 2 b=5", "EW", 6, 1.0 / 6.0, None, 5, 0, 2),
        ("LP k=6 1/6 max-replace 2 b=5", "LP", 6, 1.0 / 6.0, 1.0 / 6.0, 5, 0, 2),
    ]

    for label, method, k, cap, lp_cap, buf, mh, mx in constructions:
        w = hysteresis_topk(rhats_pit, month_dates, k, buf, min_holds=mh, max_replace=mx)
        extra = {"buffer": buf, "min_holds": mh, "max_replace": -1 if mx is None else mx}
        for g in (0.001, 0.005):
            if method == "EW":
                res = apply_weights(close, open_, decision_dates, w, month, gamma=g)
                # attach target weights so churn is on names, not LP zeros
                res = dict(res)
                res["weights"] = w
            else:
                res = apply_share_lp(
                    close, open_, decision_dates, hat_p, month, select=w, gamma=g, max_weight=lp_cap
                )
            row, ch = _row(label, method, k, cap, res, month_dates, g, extra)
            rows.append(row)
            if g == 0.001:
                churns[label] = ch
        print(f"done {label}")

    from reproduce_revised_study import run_baselines

    ew = pd.DataFrame(1.0 / 30.0, index=decision_dates, columns=TICKERS)
    for g in (0.001, 0.005):
        res = apply_weights(close[TICKERS], open_[TICKERS], decision_dates, ew, None, gamma=g)
        stats = performance_stats(res)
        rows.append({
            "Strategy": "1/N",
            "Method": "EW",
            "k": 30,
            "Cap": 1.0 / 30.0,
            "gamma": g,
            "Return_%": float(stats["Total_Return_%"]),
            "Sharpe": float(stats["Sharpe_Ratio"]),
            "Max_DD_%": float(stats["Max_Drawdown_%"]),
            "Turnover_%": float(stats["Turnover_%"]),
            "Avg_names": 30.0,
            "Mean_replace_%": 0.0,
            "Full_purges": 0,
            "Max_replace_%": 0.0,
            "Final_value": float(stats["Final_Value"]),
            "buffer": 0,
            "min_holds": 0,
            "max_replace": 0,
        })
    base = run_baselines(close[TICKERS], open_[TICKERS], djia, decision_dates)
    for name in ("BH_DJIA",):
        stats = performance_stats(base[name])
        rows.append({
            "Strategy": "Buy-and-Hold DJIA",
            "Method": "BH",
            "k": 0,
            "Cap": 1.0,
            "gamma": 0.001,
            "Return_%": float(stats["Total_Return_%"]),
            "Sharpe": float(stats["Sharpe_Ratio"]),
            "Max_DD_%": float(stats["Max_Drawdown_%"]),
            "Turnover_%": float(stats["Turnover_%"]),
            "Avg_names": 0.0,
            "Mean_replace_%": 0.0,
            "Full_purges": 0,
            "Max_replace_%": 0.0,
            "Final_value": float(stats["Final_Value"]),
            "buffer": 0,
            "min_holds": 0,
            "max_replace": 0,
        })

    df = pd.DataFrame(rows)
    df.to_csv(TAB / "table_construction_compare.csv", index=False)
    print("\n=== 10 bp ===")
    ten = df[df["gamma"] == 0.001][
        ["Strategy", "Method", "k", "Cap", "Return_%", "Max_DD_%", "Turnover_%",
         "Avg_names", "Mean_replace_%", "Full_purges", "Sharpe"]
    ]
    print(ten.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\n=== 50 bp ===")
    fifty = df[df["gamma"] == 0.005][
        ["Strategy", "Return_%", "Max_DD_%", "Turnover_%"]
    ]
    print(fifty.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    return df, churns


if __name__ == "__main__":
    run()
