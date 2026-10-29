"""
Causal weekly allocation.

The leaked daily LP and the causal daily share LP are not used here.
Forecasts stay walk-forward (prefix 1…t only). Trades wait for the next open.
The book is formed from predicted returns and rebalanced on Fridays (or slower).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cost_aware_strategies import (
    CKPT,
    apply_mean_var,
    apply_share_lp,
    apply_weights,
    load_causal,
    predicted_returns,
    weekly_mask,
)
from reproduce_revised_study import (
    FIG,
    OUT,
    TAB,
    TEST_END,
    TEST_START,
    TICKERS,
    load_market_data,
    performance_stats,
    plot_drawdown,
    plot_equity,
    run_baselines,
    series_from_result,
    aligned_streak_table,
    aligned_value_table,
)


def friday_dates(decision_dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    first = decision_dates[0]
    fridays = [dt for dt in decision_dates if dt.weekday() == 4]
    if first not in fridays:
        fridays = [first] + fridays
    return pd.DatetimeIndex(fridays)


def every_n_mask(dates: pd.DatetimeIndex, n: int) -> np.ndarray:
    mask = np.zeros(len(dates), dtype=bool)
    mask[0] = True
    count = 0
    for i, dt in enumerate(dates):
        if dt.weekday() == 4:
            if count % n == 0:
                mask[i] = True
            count += 1
    return mask


def topk_from_scores(scores: pd.DataFrame, k: int) -> pd.DataFrame:
    w = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
    for dt, row in scores.iterrows():
        picks = row.nlargest(k).index
        w.loc[dt, picks] = 1.0 / k
    return w


def hysteresis_topk(
    scores: pd.DataFrame,
    rebalance_dates,
    k: int,
    buffer: int,
    min_holds: int = 0,
    max_replace: int | None = None,
) -> pd.DataFrame:
    """Keep a name while it stays inside the top k+buffer, and optionally
    until it has been held for min_holds monthly dates. If max_replace is
    set, at most that many still-eligible names may leave. Names that leave
    the live universe (NaN score) are dropped immediately."""
    w = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
    held: list[str] = []
    age: dict[str, int] = {}
    for dt in rebalance_dates:
        if dt not in scores.index:
            continue
        ranked = scores.loc[dt].dropna().sort_values(ascending=False)
        if ranked.empty:
            continue
        eligible = set(ranked.index)
        if not held:
            held = list(ranked.head(k).index)
            age = {t: 1 for t in held}
        else:
            prev = [t for t in held if t in eligible]
            protect = set(ranked.head(k + buffer).index)
            forced = {t for t in prev if age.get(t, 0) < min_holds}
            keep = [t for t in prev if t in protect or t in forced]
            if max_replace is not None:
                must_keep = max(0, len(prev) - int(max_replace))
                if len(keep) < must_keep:
                    rest = [t for t in prev if t not in keep]
                    extra = list(
                        ranked.reindex(rest).sort_values(ascending=False).head(must_keep - len(keep)).index
                    )
                    keep = keep + extra
            if len(keep) > k:
                keep_scores = ranked.reindex(keep)
                forced_keep = [t for t in keep if t in forced]
                if len(forced_keep) >= k:
                    keep = list(
                        keep_scores.reindex(forced_keep)
                        .sort_values(ascending=False)
                        .head(k)
                        .index
                    )
                else:
                    rest = [t for t in keep if t not in forced]
                    rest_sorted = list(
                        keep_scores.reindex(rest).sort_values(ascending=False).index
                    )
                    keep = forced_keep + rest_sorted[: k - len(forced_keep)]
            extras = [t for t in ranked.index if t not in keep]
            held = (keep + extras)[:k]
            age = {t: (age[t] + 1 if t in age else 1) for t in held}
        w.loc[dt, held] = 1.0 / k
    return w


def rebalance_churn(weights: pd.DataFrame, rebalance_dates) -> pd.DataFrame:
    """Name replacement at each monthly date (empty first book is skipped)."""
    rows = []
    prev: set[str] = set()
    for dt in rebalance_dates:
        if dt not in weights.index:
            continue
        held = {t for t in weights.columns if float(weights.loc[dt, t]) > 1e-8}
        if not held:
            rows.append(
                {
                    "Decision": pd.Timestamp(dt).strftime("%Y-%m-%d"),
                    "Names": "",
                    "N_in": 0,
                    "N_out": 0,
                    "Replace_%": float("nan"),
                    "Full_purge": 0,
                }
            )
            prev = set()
            continue
        if not prev:
            replace = float("nan")
            n_in = n_out = 0
            purge = 0
        else:
            n_out = len(prev - held)
            n_in = len(held - prev)
            replace = 100.0 * n_out / len(prev)
            purge = int(n_out == len(prev))
        rows.append(
            {
                "Decision": pd.Timestamp(dt).strftime("%Y-%m-%d"),
                "Names": ",".join(sorted(held)),
                "N_in": n_in,
                "N_out": n_out,
                "Replace_%": replace,
                "Full_purge": purge,
            }
        )
        prev = held
    return pd.DataFrame(rows)


NVDA_DJIA_START = pd.Timestamp("2024-11-08")


def mask_until(scores: pd.DataFrame, ticker: str, start: pd.Timestamp) -> pd.DataFrame:
    """Set a name to missing before it is a live index member."""
    out = scores.copy()
    if ticker in out.columns:
        out.loc[out.index < start, ticker] = np.nan
    return out


def apply_index_membership(scores: pd.DataFrame) -> pd.DataFrame:
    """Live DJIA eligibility: DOW until 8 Nov 2024, NVDA from that date."""
    out = scores.copy()
    if "NVDA" in out.columns:
        out.loc[out.index < NVDA_DJIA_START, "NVDA"] = np.nan
    if "DOW" in out.columns:
        out.loc[out.index >= NVDA_DJIA_START, "DOW"] = np.nan
    return out


def load_dow_panel(index: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    from forecast_dow import load_dow_ohlc

    dow_c, dow_o = load_dow_ohlc(index)
    path = CKPT / "forecast_DOW.csv"
    pred = pd.read_csv(path, parse_dates=["decision_date"]).set_index("decision_date")
    pred.index = pd.to_datetime(pred.index).normalize()
    return dow_c, dow_o, pred


def invvol_topk(scores: pd.DataFrame, close: pd.DataFrame, k: int, vol_win: int = 20) -> pd.DataFrame:
    vol = close.pct_change().rolling(vol_win, min_periods=5).std()
    w = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
    for dt, row in scores.iterrows():
        picks = list(row.nlargest(k).index)
        if dt not in vol.index:
            w.loc[dt, picks] = 1.0 / k
            continue
        iv = 1.0 / vol.loc[dt, picks].clip(lower=1e-4).astype(float)
        iv = iv.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        w.loc[dt, picks] = (iv / iv.sum()).values
    return w


def momentum_confirmed_topk(
    scores: pd.DataFrame, close: pd.DataFrame, k: int, mom_win: int = 20
) -> pd.DataFrame:
    mom = close.pct_change(mom_win)
    w = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
    for dt, row in scores.iterrows():
        if dt in mom.index:
            ok = mom.loc[dt] > 0
            elig = row[ok]
            if len(elig) >= k:
                picks = elig.nlargest(k).index
            else:
                rest = row[~ok].nlargest(k - len(elig)).index
                picks = list(elig.nlargest(len(elig)).index) + list(rest)
        else:
            picks = row.nlargest(k).index
        w.loc[dt, list(picks)] = 1.0 / k
    return w


def blend_ew(w: pd.DataFrame, active: float) -> pd.DataFrame:
    """Mix an active long-only book with 1/N. Cuts idiosyncratic drawdown."""
    n = w.shape[1]
    out = active * w + (1.0 - active) * (1.0 / n)
    return out.div(out.sum(axis=1).replace(0, np.nan), axis=0).fillna(1.0 / n)


def exclude_tickers(scores: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    out = scores.copy()
    for n in names:
        if n in out.columns:
            out[n] = np.nan
    return out


def ensemble_rank(rhats: dict[str, pd.DataFrame], letters: list[str]) -> pd.DataFrame:
    ranks = [rhats[L].rank(axis=1, method="average") for L in letters]
    return sum(ranks) / len(ranks)


def holding_skill(
    scores: pd.DataFrame, close: pd.DataFrame, rebalance_dates: pd.DatetimeIndex
) -> dict:
    """Spearman of the rebalance-day score versus the holding-period close-to-close return."""
    rhos = []
    spread = []
    dates = [dt for dt in rebalance_dates if dt in scores.index and dt in close.index]
    for i, dt in enumerate(dates[:-1]):
        nxt = dates[i + 1]
        if nxt not in close.index:
            continue
        cols = [c for c in scores.columns if c in close.columns]
        realized = close.loc[nxt, cols] / close.loc[dt, cols] - 1.0
        s = scores.loc[dt, cols]
        rho = s.corr(realized, method="spearman")
        if pd.notna(rho):
            rhos.append(float(rho))
        top = s.nlargest(5).index
        bot = s.nsmallest(5).index
        spread.append(float(realized[top].mean() - realized[bot].mean()))
    n = len(rhos)
    mean_rho = float(np.mean(rhos)) if n else float("nan")
    se = float(np.std(rhos, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    tstat = mean_rho / se if se and se > 0 else float("nan")
    # two-sided normal p-value; N is small, so this is only a guide.
    from math import erfc

    pval = float(erfc(abs(tstat) / np.sqrt(2.0))) if np.isfinite(tstat) else float("nan")
    return {
        "mean_holding_spearman": mean_rho,
        "spearman_se": se,
        "spearman_t": tstat,
        "spearman_p": pval,
        "mean_top5_minus_bottom5": float(np.mean(spread)) * 100.0 if spread else np.nan,
        "n_holds": int(n),
    }


def weekly_skill(scores: pd.DataFrame, close: pd.DataFrame, fridays: pd.DatetimeIndex) -> dict:
    """Cross-sectional Spearman of Friday scores vs next-week close-to-close return."""
    rhos = []
    spread = []
    for i, dt in enumerate(fridays[:-1]):
        nxt = fridays[i + 1]
        if dt not in scores.index or dt not in close.index or nxt not in close.index:
            continue
        cols = [c for c in scores.columns if c in close.columns]
        realized = close.loc[nxt, cols] / close.loc[dt, cols] - 1.0
        s = scores.loc[dt, cols]
        rho = s.corr(realized, method="spearman")
        if pd.notna(rho):
            rhos.append(float(rho))
        top = s.nlargest(5).index
        bot = s.nsmallest(5).index
        spread.append(float(realized[top].mean() - realized[bot].mean()))
    return {
        "mean_weekly_spearman": float(np.mean(rhos)) if rhos else np.nan,
        "mean_top5_minus_bottom5": float(np.mean(spread)) * 100.0 if spread else np.nan,
        "n_weeks": int(len(rhos)),
    }


def _next_session(index: pd.DatetimeIndex, dt) -> pd.Timestamp | None:
    if dt not in index:
        return None
    loc = index.get_loc(dt)
    if loc + 1 >= len(index):
        return None
    return index[loc + 1]


def write_optimal_portfolio(
    close: pd.DataFrame,
    open_: pd.DataFrame,
    decision_dates: pd.DatetimeIndex,
    rebalance_dates: pd.DatetimeIndex,
    rhats: dict[str, pd.DataFrame],
    rec_res: dict,
    recommended: str = "B monthly top-5 hysteresis (20d-avg)",
    k: int = 5,
    weights: pd.DataFrame | None = None,
    score_key: str = "B",
) -> dict:
    """Last Friday implemented book, plus the Friday sequence of held names."""
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    score_key = score_key if score_key in rhats else "B"
    names = [c for c in close.columns if c in rhats[score_key].columns]
    usable = [dt for dt in rebalance_dates if dt in rhats[score_key].index and dt in decision_dates]
    last_fri = pd.Timestamp(usable[-1])
    fill_dt = _next_session(close.index, last_fri)
    mark_dt = rec_res["index"][-1]
    wealth = float(rec_res["S_values"][-1])

    score = rhats[score_key].loc[last_fri, names]
    rank_names = [c for c in names if c in rhats["A"].columns]
    rank_a = rhats["A"].loc[last_fri, rank_names].rank(method="average", ascending=False)
    rank_b = rhats["B"].loc[last_fri, rank_names].rank(method="average", ascending=False)
    rank_e = rhats["E"].loc[last_fri, rank_names].rank(method="average", ascending=False)

    # Shares after the fill of last_fri, else the last implemented NC.
    dec_list = list(decision_dates)
    if last_fri in dec_list:
        nc = np.asarray(rec_res["NCs"][dec_list.index(last_fri)], dtype=float)
    else:
        nc = np.asarray(rec_res["NCs"][-1], dtype=float)
    nc = pd.Series(nc, index=close.columns)
    if weights is not None and last_fri in weights.index:
        picks = [t for t in names if t in weights.columns and float(weights.loc[last_fri, t]) > 0.01]
        picks = sorted(picks, key=lambda t: -float(weights.loc[last_fri, t]))
    else:
        picks = [t for t in names if t in nc.index and float(nc[t]) > 1e-8]
    if not picks:
        picks = list(score.nlargest(k).index)

    mark_px = close.loc[mark_dt, names]
    fill_px = open_.loc[fill_dt, names] if fill_dt is not None else mark_px
    n_held = max(len(picks), 1)
    w_last = None
    if weights is not None and last_fri in weights.index:
        w_last = weights.loc[last_fri]

    ranking_rows = []
    for tkr in rank_names:
        ranking_rows.append(
            {
                "Ticker": tkr,
                "Score": float(score[tkr]) if tkr in score.index else float("nan"),
                "Rank": float(score.rank(method="average", ascending=False)[tkr]) if tkr in score.index else float("nan"),
                "A_rank": float(rank_a[tkr]),
                "B_rank": float(rank_b[tkr]),
                "E_rank": float(rank_e[tkr]),
                "A_pred_ret_%": float(rhats["A"].loc[last_fri, tkr] * 100.0),
                "B_pred_ret_%": float(rhats["B"].loc[last_fri, tkr] * 100.0),
                "E_pred_ret_%": float(rhats["E"].loc[last_fri, tkr] * 100.0),
                "Selected": "Yes" if tkr in picks else "",
            }
        )
    ranking = pd.DataFrame(ranking_rows).sort_values("Rank")
    ranking.to_csv(TAB / "table_optimal_ranking.csv", index=False)

    book_rows = []
    for tkr in picks:
        shares = float(nc[tkr])
        px = float(mark_px[tkr])
        value = shares * px
        if w_last is not None and tkr in w_last.index:
            tw = float(w_last[tkr])
        else:
            tw = 1.0 / n_held
        book_rows.append(
            {
                "Ticker": tkr,
                "Target_weight": tw,
                "Score": float(score[tkr]),
                "Rank": float(score.rank(method="average", ascending=False)[tkr]),
                "A_pred_ret_%": float(rhats["A"].loc[last_fri, tkr] * 100.0),
                "B_pred_ret_%": float(rhats["B"].loc[last_fri, tkr] * 100.0),
                "E_pred_ret_%": float(rhats["E"].loc[last_fri, tkr] * 100.0),
                "Fill_open": float(fill_px[tkr]),
                "Shares": shares,
                "Mark_price": px,
                "Mark_value": value,
                "Portfolio_%": 100.0 * value / wealth if wealth else 0.0,
            }
        )
    book = pd.DataFrame(book_rows)
    book.to_csv(TAB / "table_optimal_portfolio.csv", index=False)

    friday_rows = []
    for dt in usable:
        if weights is not None and dt in weights.index:
            held_names = [
                t for t in names
                if t in weights.columns and float(weights.loc[dt, t]) > 0.01
            ]
            held_names = sorted(held_names, key=lambda t: -float(weights.loc[dt, t]))
        else:
            held_names = list(rhats[score_key].loc[dt, names].dropna().nlargest(k).index)
        nxt = _next_session(close.index, dt)
        row = {
            "Decision": pd.Timestamp(dt).strftime("%Y-%m-%d"),
            "Fill": nxt.strftime("%Y-%m-%d") if nxt is not None else "",
        }
        for j, name in enumerate(held_names, start=1):
            row[f"Name_{j}"] = name
            if weights is not None and dt in weights.index:
                row[f"W_{j}"] = float(weights.loc[dt, name])
        friday_rows.append(row)
    friday_books = pd.DataFrame(friday_rows)
    friday_books.to_csv(TAB / "table_friday_optimal_books.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
              "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC", "#4C78A8"]
    values = book["Mark_value"].to_numpy()
    labels = [
        f"{t}\n{w:.1f}%"
        for t, w in zip(book["Ticker"], book["Portfolio_%"])
    ]
    ax.bar(labels, values, color=colors[: len(book)], width=0.55)
    ax.set_ylabel("Mark value (USD)")
    ax.set_title(
        f"Optimal portfolio — {recommended}\n"
        f"Decision {last_fri.strftime('%d %b %Y')}, mark {pd.Timestamp(mark_dt).strftime('%d %b %Y')}"
    )
    ax.yaxis.set_major_formatter(lambda x, _pos: f"{x/1e3:,.0f}k")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "fig10_optimal_portfolio.png", dpi=160)
    plt.close(fig)

    meta = {
        "recommended": recommended,
        "k": k,
        "decision_date": last_fri.strftime("%Y-%m-%d"),
        "fill_date": fill_dt.strftime("%Y-%m-%d") if fill_dt is not None else "",
        "mark_date": pd.Timestamp(mark_dt).strftime("%Y-%m-%d"),
        "tickers": picks,
        "target_weight": {t: float(w_last[t]) for t in picks} if w_last is not None else 1.0 / k,
        "wealth": wealth,
        "n_fridays": int(len(usable)),
    }
    (OUT / "optimal_portfolio_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print("Optimal portfolio:", ", ".join(picks), "on", meta["decision_date"])
    print("Wrote", TAB / "table_optimal_portfolio.csv")
    return meta


PREFERRED_BOOK = "B monthly top-5 hysteresis MV (20d-avg, PIT NVDA)"
LP_K = 5
LP_BUFFER = 3
LP_CAP = 0.25
LP_MAX_REPLACE = None
MV_KAPPA = 2.0
MV_LAM = 0.01
MV_CAP = 0.30


def _implied_prices(close: pd.DataFrame, rhat: pd.DataFrame, dates) -> pd.DataFrame:
    px = close.reindex(dates)
    rr = rhat.reindex(index=dates, columns=close.columns)
    return px * (1.0 + rr)


def export_optimal_portfolio() -> dict:
    """Rebuild the proposed B-hysteresis monthly book and write the optimal-portfolio tables."""
    close, open_, _djia = load_market_data()
    close = close.loc[:TEST_END, TICKERS]
    open_ = open_.reindex(close.index)[TICKERS].fillna(close)
    test_index = close.index[(close.index >= TEST_START) & (close.index <= TEST_END)]
    decision_dates = test_index[:-1]
    month = every_n_mask(decision_dates, 4)
    rebalance_dates = decision_dates[month]
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
    rhats["ABE"] = ensemble_rank(rhats, ["A", "B", "E"])
    rhats_hold = {L: r.rolling(20, min_periods=5).mean() for L, r in rhats.items()}
    rhats_pit = apply_index_membership(rhats_hold["B"])
    w = hysteresis_topk(
        rhats_pit, rebalance_dates, LP_K, LP_BUFFER, max_replace=LP_MAX_REPLACE
    )
    rec_res = apply_mean_var(
        close, open_, decision_dates, rhats_pit, month, w,
        lam=MV_LAM, kappa=MV_KAPPA, max_weight=MV_CAP,
    )
    return write_optimal_portfolio(
        close,
        open_,
        decision_dates,
        rebalance_dates,
        {**rhats, **{f"{L}_hold": rhats_hold[L] for L in rhats_hold}, "B": rhats_pit},
        rec_res,
        PREFERRED_BOOK,
        k=LP_K,
        weights=rec_res["weights"],
        score_key="B",
    )


def main() -> None:
    close, open_, djia = load_market_data()
    close = close.loc[:TEST_END, TICKERS]
    open_ = open_.reindex(close.index)[TICKERS].fillna(close)
    djia = djia.reindex(close.index).ffill()
    test_index = close.index[(close.index >= TEST_START) & (close.index <= TEST_END)]
    decision_dates = test_index[:-1]
    causal = load_causal()
    baselines = run_baselines(close, open_, djia, decision_dates)
    dow_c, dow_o, dow_pred = load_dow_panel(close.index)
    close = close.copy()
    open_ = open_.copy()
    close["DOW"] = dow_c
    open_["DOW"] = dow_o
    week = weekly_mask(decision_dates)
    biweek = every_n_mask(decision_dates, 2)
    month = every_n_mask(decision_dates, 4)
    fridays = friday_dates(decision_dates)

    rhats = {L: predicted_returns(pred, close[TICKERS], decision_dates) for L, pred in causal.items()}
    rhats_smooth = {L: r.rolling(5, min_periods=1).mean() for L, r in rhats.items()}
    rhats["ABE"] = ensemble_rank(rhats, ["A", "B", "E"])
    rhats["ALL"] = ensemble_rank(rhats, ["A", "B", "C", "D", "E"])
    rhats_smooth["ABE"] = ensemble_rank(rhats_smooth, ["A", "B", "E"])
    rhats_smooth["ALL"] = ensemble_rank(rhats_smooth, ["A", "B", "C", "D", "E"])

    skill_rows = []
    for name, sc in list(rhats.items()) + [(f"{L} 5d-avg", rhats_smooth[L]) for L in rhats_smooth]:
        row = weekly_skill(sc, close, fridays)
        row["Signal"] = name
        skill_rows.append(row)
    skill = pd.DataFrame(skill_rows)
    skill.to_csv(TAB / "weekly_cross_section_skill.csv", index=False)

    month_dates = decision_dates[month]

    results = dict(baselines)
    configs: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}

    # Primary family: weekly equal-weight top-k on the raw Friday predicted return.
    for L in ["A", "B", "C", "D", "E", "ABE", "ALL"]:
        for k in (3, 5, 8):
            configs[f"{L} weekly top-{k}"] = (topk_from_scores(rhats[L], k), week)

    # Weekly hold, but the Friday score is the last five causal one-step forecasts.
    for L in ["B", "ABE", "ALL"]:
        configs[f"{L} weekly top-5 (5d-avg signal)"] = (
            topk_from_scores(rhats_smooth[L], 5),
            week,
        )

    # Keep a name until it falls out of the top-(k+buffer). Cuts turnover.
    for L in ["B", "ABE"]:
        configs[f"{L} weekly top-5 hysteresis+3"] = (
            hysteresis_topk(rhats[L], fridays, 5, 3),
            week,
        )

    # Inverse-volatility weights inside the same top-5.
    for L in ["B", "ABE"]:
        configs[f"{L} weekly top-5 inv-vol"] = (invvol_topk(rhats[L], close, 5), week)

    # Require 20-day price momentum to agree with the forecast rank.
    for L in ["B", "ABE"]:
        configs[f"{L} weekly top-5 mom-confirm"] = (
            momentum_confirmed_topk(rhats[L], close, 5),
            week,
        )

    # Slower calendars. Monthly k in {5,8,10}; quarterly is every eighth Friday.
    quarter = every_n_mask(decision_dates, 8)
    rhats_hold = {L: r.rolling(20, min_periods=5).mean() for L, r in rhats.items()}
    dow_px = dow_c.reindex(decision_dates)
    rhats_hold_pit = {}
    for L in rhats_hold:
        sc = rhats_hold[L].copy()
        if L in ["A", "B", "C", "D", "E"]:
            dow_r = dow_pred[L].reindex(decision_dates) / dow_px - 1.0
            sc["DOW"] = dow_r.rolling(20, min_periods=5).mean()
        rhats_hold_pit[L] = apply_index_membership(sc)
    mom20 = close.pct_change(20).reindex(decision_dates)

    hold_rows = []
    for name, sc in [
        ("A", rhats["A"]),
        ("B", rhats["B"]),
        ("C", rhats["C"]),
        ("D", rhats["D"]),
        ("E", rhats["E"]),
        ("ABE", rhats["ABE"]),
        ("B 20d-avg", rhats_hold["B"]),
        ("ABE 20d-avg", rhats_hold["ABE"]),
        ("B 20d-avg PIT", rhats_hold_pit["B"]),
        ("ABE 20d-avg PIT", rhats_hold_pit["ABE"]),
    ]:
        row = holding_skill(sc, close, month_dates)
        row["Signal"] = name
        hold_rows.append(row)
    hold_skill = pd.DataFrame(hold_rows)
    hold_skill.to_csv(TAB / "monthly_holding_skill.csv", index=False)

    # Implemented-signal IC: 20-day mean versus the horizon that is actually traded.
    impl_rows = []
    for name, sc in [
        ("B 20d-avg (weekly)", rhats_hold["B"]),
        ("B 20d-avg PIT (weekly)", rhats_hold_pit["B"]),
        ("ABE 20d-avg (weekly)", rhats_hold["ABE"]),
    ]:
        row = weekly_skill(sc, close, fridays)
        row["Signal"] = name
        impl_rows.append(row)
    for name, sc in [
        ("B 20d-avg (monthly hold)", rhats_hold["B"]),
        ("B 20d-avg PIT (monthly hold)", rhats_hold_pit["B"]),
        ("ABE 20d-avg (monthly hold)", rhats_hold["ABE"]),
        ("MOM20 (monthly hold)", mom20[TICKERS]),
    ]:
        row = holding_skill(sc, close, month_dates)
        row["Signal"] = name
        impl_rows.append(row)
    pd.DataFrame(impl_rows).to_csv(TAB / "table_implemented_signal_skill.csv", index=False)

    extra_weekly = []
    for name, sc in [("B 20d-avg", rhats_hold["B"]), ("B 20d-avg PIT", rhats_hold_pit["B"])]:
        row = weekly_skill(sc, close, fridays)
        row["Signal"] = name
        extra_weekly.append(row)
    skill = pd.concat([skill, pd.DataFrame(extra_weekly)], ignore_index=True)
    skill.to_csv(TAB / "weekly_cross_section_skill.csv", index=False)

    for L in ["B", "ABE"]:
        configs[f"{L} biweekly top-5"] = (topk_from_scores(rhats[L], 5), biweek)
        for k in (5, 8, 10):
            configs[f"{L} monthly top-{k}"] = (topk_from_scores(rhats[L], k), month)
        configs[f"{L} quarterly top-5"] = (topk_from_scores(rhats[L], 5), quarter)
        configs[f"{L} monthly top-5 (20d-avg signal)"] = (
            topk_from_scores(rhats_hold[L], 5),
            month,
        )
        configs[f"{L} monthly top-8 (20d-avg signal)"] = (
            topk_from_scores(rhats_hold[L], 8),
            month,
        )
        configs[f"{L} monthly top-5 inv-vol (20d-avg)"] = (
            invvol_topk(rhats_hold[L], close, 5),
            month,
        )
        configs[f"{L} monthly top-8 inv-vol (20d-avg)"] = (
            invvol_topk(rhats_hold[L], close, 8),
            month,
        )
        configs[f"{L} monthly top-5 hysteresis (20d-avg)"] = (
            hysteresis_topk(rhats_hold[L], month_dates, 5, 3),
            month,
        )
        configs[f"{L} monthly top-5 60/40 1N (20d-avg)"] = (
            blend_ew(topk_from_scores(rhats_hold[L], 5), 0.60),
            month,
        )
        configs[f"{L} monthly top-8 hysteresis+3"] = (
            hysteresis_topk(rhats[L], month_dates, 8, 3),
            month,
        )

    # Point-in-time membership: NVDA is untradable before 8 November 2024.
    configs["B monthly top-5 hysteresis (20d-avg, PIT NVDA)"] = (
        hysteresis_topk(rhats_hold_pit["B"], month_dates, 5, 3),
        month,
    )
    configs["B monthly top-5 (20d-avg, PIT NVDA)"] = (
        topk_from_scores(rhats_hold_pit["B"], 5),
        month,
    )
    b_never = exclude_tickers(rhats_hold["B"], ["NVDA"])
    configs["B monthly top-5 hysteresis (20d-avg, no NVDA)"] = (
        hysteresis_topk(b_never, month_dates, 5, 3),
        month,
    )
    for b in (0, 1, 2, 3, 4, 5, 7, 9):
        configs[f"B monthly top-5 hysteresis b={b} (20d-avg, PIT NVDA)"] = (
            hysteresis_topk(rhats_hold_pit["B"], month_dates, 5, b),
            month,
        )
    for mh in (3, 4, 5):
        configs[f"B monthly top-5 hysteresis min-hold {mh} (20d-avg, PIT NVDA)"] = (
            hysteresis_topk(rhats_hold_pit["B"], month_dates, 5, 3, min_holds=mh),
            month,
        )
    configs["B monthly top-5 hysteresis b=7 min-hold 4 (20d-avg, PIT NVDA)"] = (
        hysteresis_topk(rhats_hold_pit["B"], month_dates, 5, 7, min_holds=4),
        month,
    )
    configs["B monthly top-6 hysteresis (20d-avg, PIT NVDA)"] = (
        hysteresis_topk(rhats_hold_pit["B"], month_dates, 6, 3),
        month,
    )
    configs["B monthly top-4 hysteresis (20d-avg, PIT NVDA)"] = (
        hysteresis_topk(rhats_hold_pit["B"], month_dates, 4, 3),
        month,
    )
    configs["B monthly top-8 hysteresis (20d-avg, PIT NVDA)"] = (
        hysteresis_topk(rhats_hold_pit["B"], month_dates, 8, 3),
        month,
    )
    for mx in (1, 2, 3):
        configs[f"B monthly top-6 max-replace {mx} (20d-avg, PIT NVDA)"] = (
            hysteresis_topk(rhats_hold_pit["B"], month_dates, 6, 3, max_replace=mx),
            month,
        )
    configs["B monthly top-6 min-hold 2 (20d-avg, PIT NVDA)"] = (
        hysteresis_topk(rhats_hold_pit["B"], month_dates, 6, 3, min_holds=2),
        month,
    )
    for L in ["A", "C", "D", "E"]:
        configs[f"{L} monthly top-5 hysteresis (20d-avg, PIT NVDA)"] = (
            hysteresis_topk(rhats_hold_pit[L], month_dates, 5, 3),
            month,
        )

    # Survivorship-safe panel: drop NVDA for the whole window.
    abe_ex = exclude_tickers(rhats_hold["ABE"], ["NVDA"])
    configs["ABE monthly top-5 (20d-avg, no NVDA)"] = (topk_from_scores(abe_ex, 5), month)
    configs["ABE monthly top-8 (20d-avg, no NVDA)"] = (topk_from_scores(abe_ex, 8), month)
    configs["ABE monthly top-5 inv-vol (20d-avg, no NVDA)"] = (
        invvol_topk(abe_ex, close, 5),
        month,
    )
    configs["ABE monthly top-5 60/40 1N (20d-avg, no NVDA)"] = (
        blend_ew(topk_from_scores(abe_ex, 5), 0.60),
        month,
    )

    # Price-momentum control at the same monthly calendar (horizon-matched naive sort).
    configs["MOM20 monthly top-5"] = (topk_from_scores(mom20[TICKERS], 5), month)
    configs["MOM20 monthly top-8"] = (topk_from_scores(mom20[TICKERS], 8), month)

    # Fade controls: if concentration alone were the story, the bottom names would also win.
    configs["ABE weekly bottom-3 (fade)"] = (topk_from_scores(-rhats["ABE"], 3), week)
    configs["B weekly bottom-5 (fade)"] = (topk_from_scores(-rhats["B"], 5), week)

    for label, (w, mask) in configs.items():
        results[label] = apply_weights(close, open_, decision_dates, w, mask)

    hat_p_pit = _implied_prices(close, rhats_hold_pit["B"], decision_dates)
    w_hyst_pit = hysteresis_topk(rhats_hold_pit["B"], month_dates, 5, 3)
    w_hyst6_pit = hysteresis_topk(rhats_hold_pit["B"], month_dates, 6, 3)
    w_pref_pit = hysteresis_topk(rhats_hold_pit["B"], month_dates, LP_K, LP_BUFFER)
    results["B monthly top-5 hysteresis LP (20d-avg, PIT NVDA)"] = apply_share_lp(
        close, open_, decision_dates, hat_p_pit, month, select=w_hyst_pit, max_weight=0.25
    )
    results["B monthly top-6 hysteresis LP (20d-avg, PIT NVDA)"] = apply_share_lp(
        close, open_, decision_dates, hat_p_pit, month, select=w_hyst6_pit, max_weight=1.0 / 6.0
    )
    results[PREFERRED_BOOK] = apply_mean_var(
        close, open_, decision_dates, rhats_hold_pit["B"], month, w_pref_pit,
        lam=MV_LAM, kappa=MV_KAPPA, max_weight=MV_CAP,
    )
    results["B monthly top-8 hysteresis LP (20d-avg, PIT NVDA)"] = apply_share_lp(
        close, open_, decision_dates, hat_p_pit, month,
        select=hysteresis_topk(rhats_hold_pit["B"], month_dates, 8, 3),
        max_weight=0.15,
    )
    results["B monthly share-LP (20d-avg, PIT NVDA)"] = apply_share_lp(
        close, open_, decision_dates, hat_p_pit, month, select=None, max_weight=0.15
    )

    stress_names = [
        PREFERRED_BOOK,
        "B monthly top-6 hysteresis LP (20d-avg, PIT NVDA)",
        "B monthly top-8 hysteresis LP (20d-avg, PIT NVDA)",
        "B monthly top-5 hysteresis LP (20d-avg, PIT NVDA)",
        "B monthly share-LP (20d-avg, PIT NVDA)",
        "ABE monthly top-5",
        "ABE monthly top-8",
        "ABE monthly top-10",
        "ABE monthly top-5 (20d-avg signal)",
        "ABE monthly top-8 (20d-avg signal)",
        "ABE monthly top-5 inv-vol (20d-avg)",
        "ABE monthly top-8 inv-vol (20d-avg)",
        "ABE monthly top-5 hysteresis (20d-avg)",
        "ABE monthly top-5 60/40 1N (20d-avg)",
        "ABE monthly top-5 (20d-avg, no NVDA)",
        "ABE monthly top-8 (20d-avg, no NVDA)",
        "ABE monthly top-5 inv-vol (20d-avg, no NVDA)",
        "ABE monthly top-5 60/40 1N (20d-avg, no NVDA)",
        "B monthly top-5 (20d-avg signal)",
        "B monthly top-5 inv-vol (20d-avg)",
        "B monthly top-5 hysteresis (20d-avg)",
        "B monthly top-5 hysteresis (20d-avg, PIT NVDA)",
        "B monthly top-5 hysteresis (20d-avg, no NVDA)",
        "B monthly top-5 (20d-avg, PIT NVDA)",
        "B monthly top-5 hysteresis b=0 (20d-avg, PIT NVDA)",
        "B monthly top-5 hysteresis b=1 (20d-avg, PIT NVDA)",
        "B monthly top-5 hysteresis b=2 (20d-avg, PIT NVDA)",
        "B monthly top-5 hysteresis b=3 (20d-avg, PIT NVDA)",
        "B monthly top-5 hysteresis b=4 (20d-avg, PIT NVDA)",
        "B monthly top-5 hysteresis b=5 (20d-avg, PIT NVDA)",
        "B monthly top-5 hysteresis b=7 (20d-avg, PIT NVDA)",
        "B monthly top-5 hysteresis b=9 (20d-avg, PIT NVDA)",
        "B monthly top-5 hysteresis min-hold 3 (20d-avg, PIT NVDA)",
        "B monthly top-5 hysteresis min-hold 4 (20d-avg, PIT NVDA)",
        "B monthly top-5 hysteresis min-hold 5 (20d-avg, PIT NVDA)",
        "B monthly top-5 hysteresis b=7 min-hold 4 (20d-avg, PIT NVDA)",
        "B monthly top-6 hysteresis (20d-avg, PIT NVDA)",
        "B monthly top-4 hysteresis (20d-avg, PIT NVDA)",
        "B monthly top-8 hysteresis (20d-avg, PIT NVDA)",
        "B monthly top-6 max-replace 1 (20d-avg, PIT NVDA)",
        "B monthly top-6 max-replace 2 (20d-avg, PIT NVDA)",
        "B monthly top-6 max-replace 3 (20d-avg, PIT NVDA)",
        "B monthly top-6 min-hold 2 (20d-avg, PIT NVDA)",
        "A monthly top-5 hysteresis (20d-avg, PIT NVDA)",
        "C monthly top-5 hysteresis (20d-avg, PIT NVDA)",
        "D monthly top-5 hysteresis (20d-avg, PIT NVDA)",
        "E monthly top-5 hysteresis (20d-avg, PIT NVDA)",
        "B monthly top-5 60/40 1N (20d-avg)",
        "ABE quarterly top-5",
        "MOM20 monthly top-5",
        "BH_DJIA",
        "EW_1N",
    ]
    stress_rows = []
    for g in (0.001, 0.005, 0.010):
        for label in stress_names:
            if label == "BH_DJIA":
                res = results["BH_DJIA"]
            elif label == "EW_1N":
                ew = pd.DataFrame(1.0 / 30.0, index=decision_dates, columns=TICKERS)
                res = apply_weights(close, open_, decision_dates, ew, None, gamma=g)
            elif label == "B monthly top-5 hysteresis LP (20d-avg, PIT NVDA)":
                res = apply_share_lp(
                    close, open_, decision_dates, hat_p_pit, month,
                    select=w_hyst_pit, gamma=g, max_weight=0.25,
                )
            elif label == "B monthly top-6 hysteresis LP (20d-avg, PIT NVDA)":
                res = apply_share_lp(
                    close, open_, decision_dates, hat_p_pit, month,
                    select=w_hyst6_pit, gamma=g, max_weight=1.0 / 6.0,
                )
            elif label == PREFERRED_BOOK:
                res = apply_mean_var(
                    close, open_, decision_dates, rhats_hold_pit["B"], month, w_pref_pit,
                    gamma=g, lam=MV_LAM, kappa=MV_KAPPA, max_weight=MV_CAP,
                )
            elif label == "B monthly top-8 hysteresis LP (20d-avg, PIT NVDA)":
                res = apply_share_lp(
                    close, open_, decision_dates, hat_p_pit, month,
                    select=hysteresis_topk(rhats_hold_pit["B"], month_dates, 8, 3),
                    gamma=g, max_weight=0.15,
                )
            elif label == "B monthly share-LP (20d-avg, PIT NVDA)":
                res = apply_share_lp(
                    close, open_, decision_dates, hat_p_pit, month,
                    select=None, gamma=g, max_weight=0.15,
                )
            elif label in configs:
                w, mask = configs[label]
                res = apply_weights(close, open_, decision_dates, w, mask, gamma=g)
            else:
                continue
            row = performance_stats(res)
            row["Strategy"] = label
            row["gamma"] = g
            stress_rows.append(row)
    pd.DataFrame(stress_rows).to_csv(TAB / "table_cost_stress.csv", index=False)

    rows = []
    for name, res in results.items():
        row = performance_stats(res)
        row["Strategy"] = name
        rows.append(row)
    metrics = pd.DataFrame(rows).sort_values("Total_Return_%", ascending=False)
    metrics.to_csv(TAB / "weekly_causal_performance.csv", index=False)

    bh = float(metrics.loc[metrics["Strategy"] == "BH_DJIA", "Total_Return_%"].iloc[0])
    beat = metrics[metrics["Total_Return_%"] > bh + 1e-9]
    forecast_only = metrics[~metrics["Strategy"].isin(["BH_DJIA", "EW_1N", "EG", "ONS"])]
    best = forecast_only.iloc[0]
    stress = pd.read_csv(TAB / "table_cost_stress.csv")
    ew50 = float(stress.loc[(stress["Strategy"] == "EW_1N") & (stress["gamma"] == 0.005), "Total_Return_%"].iloc[0])
    bh50 = float(stress.loc[(stress["Strategy"] == "BH_DJIA") & (stress["gamma"] == 0.005), "Total_Return_%"].iloc[0])
    dd10 = stress[stress["gamma"] == 0.001][["Strategy", "Max_Drawdown_%", "Turnover_%"]].rename(
        columns={"Max_Drawdown_%": "DD10", "Turnover_%": "TO10"}
    )
    cand = stress[
        (stress["gamma"] == 0.005)
        & (~stress["Strategy"].isin(["BH_DJIA", "EW_1N"]))
        & (stress["Strategy"].str.contains("20d-avg"))
    ].merge(dd10, on="Strategy", how="left")
    recommended = PREFERRED_BOOK
    report_names = [
        recommended,
        "B monthly top-5 hysteresis LP (20d-avg, PIT NVDA)",
        "B monthly top-5 hysteresis (20d-avg, PIT NVDA)",
        "B monthly top-6 hysteresis LP (20d-avg, PIT NVDA)",
        "B monthly share-LP (20d-avg, PIT NVDA)",
        "B monthly top-5 hysteresis (20d-avg)",
        "B monthly top-5 hysteresis (20d-avg, no NVDA)",
        "B monthly top-5 (20d-avg, PIT NVDA)",
        "C monthly top-5 hysteresis (20d-avg, PIT NVDA)",
        "ABE monthly top-5 (20d-avg signal)",
        "MOM20 monthly top-5",
        "BH_DJIA",
        "EW_1N",
    ]
    sweep = metrics[
        metrics["Strategy"].str.match(r"B monthly top-5 hysteresis b=\d+ \(20d-avg, PIT NVDA\)$")
    ].copy()
    sweep["_b"] = sweep["Strategy"].str.extract(r"b=(\d+)").astype(int)
    sweep = sweep.sort_values("_b").drop(columns="_b")
    sweep.to_csv(TAB / "table_hysteresis_sweep.csv", index=False)
    variant_cmp = metrics[
        metrics["Strategy"].isin(
            [
                "A monthly top-5 hysteresis (20d-avg, PIT NVDA)",
                "B monthly top-5 hysteresis (20d-avg, PIT NVDA)",
                "C monthly top-5 hysteresis (20d-avg, PIT NVDA)",
                "D monthly top-5 hysteresis (20d-avg, PIT NVDA)",
                "E monthly top-5 hysteresis (20d-avg, PIT NVDA)",
            ]
        )
    ].copy()
    variant_cmp.to_csv(TAB / "table_predictor_hysteresis.csv", index=False)
    churn_labels = [
        "B monthly top-6 max-replace 1 (20d-avg, PIT NVDA)",
        "B monthly top-6 max-replace 2 (20d-avg, PIT NVDA)",
        "B monthly top-6 max-replace 3 (20d-avg, PIT NVDA)",
        "B monthly top-6 min-hold 2 (20d-avg, PIT NVDA)",
        "B monthly top-6 hysteresis (20d-avg, PIT NVDA)",
        "B monthly top-5 hysteresis (20d-avg, PIT NVDA)",
        "B monthly top-5 hysteresis b=7 (20d-avg, PIT NVDA)",
        "B monthly top-5 hysteresis min-hold 4 (20d-avg, PIT NVDA)",
        "B monthly top-5 hysteresis min-hold 5 (20d-avg, PIT NVDA)",
        "B monthly top-5 hysteresis b=7 min-hold 4 (20d-avg, PIT NVDA)",
    ]
    churn_rows = []
    for lab in churn_labels:
        if lab not in configs:
            continue
        ww, _ = configs[lab]
        ch = rebalance_churn(ww, month_dates)
        filled = ch.dropna(subset=["Replace_%"])
        churn_rows.append(
            {
                "Strategy": lab,
                "N_rebalances": int(filled.shape[0]),
                "Mean_replace_%": float(filled["Replace_%"].mean()) if len(filled) else float("nan"),
                "Full_purges": int(filled["Full_purge"].sum()) if len(filled) else 0,
                "Pct_full_purge": float(100.0 * filled["Full_purge"].mean()) if len(filled) else float("nan"),
            }
        )
        if lab == PREFERRED_BOOK:
            ch.to_csv(TAB / "table_rebalance_churn.csv", index=False)
    for lab in (
        PREFERRED_BOOK,
        "B monthly top-6 hysteresis LP (20d-avg, PIT NVDA)",
        "B monthly top-8 hysteresis LP (20d-avg, PIT NVDA)",
        "B monthly top-5 hysteresis LP (20d-avg, PIT NVDA)",
    ):
        res = results.get(lab)
        if not res or "weights" not in res:
            continue
        ch = rebalance_churn(res["weights"], month_dates)
        filled = ch.dropna(subset=["Replace_%"])
        churn_rows.append(
            {
                "Strategy": lab,
                "N_rebalances": int(filled.shape[0]),
                "Mean_replace_%": float(filled["Replace_%"].mean()) if len(filled) else float("nan"),
                "Full_purges": int(filled["Full_purge"].sum()) if len(filled) else 0,
                "Pct_full_purge": float(100.0 * filled["Full_purge"].mean()) if len(filled) else float("nan"),
            }
        )
        if lab == PREFERRED_BOOK:
            ch.to_csv(TAB / "table_rebalance_churn.csv", index=False)
    churn_sum = pd.DataFrame(churn_rows)
    overlay_rows = []
    overlay_order = [PREFERRED_BOOK] + [
        x for x in list(churn_sum["Strategy"]) if x != PREFERRED_BOOK
    ]
    for lab in overlay_order:
        if lab not in metrics["Strategy"].values:
            continue
        m = metrics[metrics["Strategy"] == lab].iloc[0]
        st = stress[stress["Strategy"] == lab]
        r50 = st[st["gamma"].round(3) == 0.005]
        ch = churn_sum[churn_sum["Strategy"] == lab]
        overlay_rows.append(
            {
                "Strategy": lab,
                "Return_10bp_%": float(m["Total_Return_%"]),
                "Return_50bp_%": float(r50["Total_Return_%"].iloc[0]) if len(r50) else float("nan"),
                "Max_Drawdown_%": float(m["Max_Drawdown_%"]),
                "Turnover_%": float(m["Turnover_%"]),
                "Mean_replace_%": float(ch["Mean_replace_%"].iloc[0]) if len(ch) else float("nan"),
                "Full_purges": int(ch["Full_purges"].iloc[0]) if len(ch) else 0,
            }
        )
    pd.DataFrame(overlay_rows).to_csv(TAB / "table_stability_overlay.csv", index=False)
    churn_sum.to_csv(TAB / "table_stability_churn.csv", index=False)
    report_names = list(dict.fromkeys(report_names))
    report = metrics[metrics["Strategy"].isin(report_names)].copy()
    report["_ord"] = report["Strategy"].map({n: i for i, n in enumerate(report_names)})
    report = report.sort_values("_ord").drop(columns="_ord")
    report.to_csv(TAB / "weekly_causal_report_table.csv", index=False)

    print("Weekly cross-sectional skill")
    print(skill.to_string(index=False))
    print("\nMonthly holding-period skill (score on rebalance vs next-month return)")
    print(hold_skill.to_string(index=False))
    print(f"\nBuy-and-Hold DJIA = {bh:.2f}%")
    print(f"Best forecast book: {best['Strategy']}  {best['Total_Return_%']:.2f}%")
    print(f"Recommended book: {recommended}")
    print("\nBeat BH:")
    print(
        beat[["Strategy", "Total_Return_%", "Max_Drawdown_%", "Sharpe_Ratio", "Turnover_%"]]
        .to_string(index=False)
    )
    print("\nAll")
    print(
        metrics[["Strategy", "Total_Return_%", "Max_Drawdown_%", "Sharpe_Ratio", "Turnover_%"]]
        .to_string(index=False)
        .encode("ascii", "replace")
        .decode("ascii")
    )

    series = {
        "BH_DJIA": series_from_result(results["BH_DJIA"]),
        "EW_1N": series_from_result(results["EW_1N"]),
        recommended: series_from_result(results[recommended]),
        "B monthly top-5 hysteresis (20d-avg)": series_from_result(
            results["B monthly top-5 hysteresis (20d-avg)"]
        ),
        "MOM20 monthly top-5": series_from_result(results["MOM20 monthly top-5"]),
    }
    plot_equity(
        series,
        FIG / "fig8_weekly_causal.png",
        "Causal predicted-return books versus Buy-and-Hold DJIA and 1/N",
    )
    plot_drawdown(
        series,
        FIG / "fig9_weekly_drawdown.png",
        "Drawdowns of the causal predicted-return books",
    )

    rec_res = results[recommended]
    aligned_streak_table(rec_res, close, list(close.columns)).to_csv(
        TAB / "table_weekly_streaks.csv", index=False
    )
    aligned_value_table(rec_res, close, list(close.columns)).to_csv(
        TAB / "table_weekly_value.csv", index=False
    )
    rec_k = (
        10 if "top-10" in recommended
        else 8 if "top-8" in recommended
        else 6 if "top-6" in recommended
        else 5
    )
    rec_mask = quarter if "quarterly" in recommended else month
    rec_letter = "B" if recommended.startswith("B ") else "ABE"
    if "PIT" in recommended:
        rec_scores = rhats_hold_pit[rec_letter]
    elif "20d-avg" in recommended:
        rec_scores = rhats_hold[rec_letter]
    else:
        rec_scores = rhats[rec_letter]
    rec_w = configs[recommended][0] if recommended in configs else rec_res.get("weights")
    write_optimal_portfolio(
        close,
        open_,
        decision_dates,
        decision_dates[rec_mask],
        {**rhats, rec_letter: rec_scores},
        rec_res,
        recommended,
        k=rec_k,
        weights=rec_w,
        score_key=rec_letter,
    )

    (OUT / "weekly_causal_meta.json").write_text(
        json.dumps(
            {
                "bh_return_pct": bh,
                "best_strategy": str(best["Strategy"]),
                "best_return_pct": float(best["Total_Return_%"]),
                "recommended": recommended,
                "recommended_return_pct": float(
                    metrics.loc[metrics["Strategy"] == recommended, "Total_Return_%"].iloc[0]
                ),
                "n_beat_bh": int(len(beat)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\nWrote", TAB / "weekly_causal_performance.csv")
    print("Wrote", FIG / "fig8_weekly_causal.png")


if __name__ == "__main__":
    import sys

    if "--optimal-only" in sys.argv:
        export_optimal_portfolio()
    else:
        main()
