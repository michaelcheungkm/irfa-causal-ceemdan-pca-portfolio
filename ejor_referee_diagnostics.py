"""Referee diagnostics for the EJOR manuscript.

Adds the tests requested against the non-significant panel Spearman IC:
  1. Quintile monotonicity (Q1-Q5) of the traded PIT B 20-day score
  2. Top-5 / I_t hit rate (precision@5) vs the cross-sectional median
  3. Stationary block-bootstrap CIs for return, Sharpe and max drawdown
  4. Random 5-name hysteresis Monte Carlo vs equal-weight top-5 vs MV
  5. Hansen SPA / White Reality Check across the A-E and buffer family
"""
from __future__ import annotations

import json
from math import erf, erfc, sqrt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from cost_aware_strategies import apply_mean_var, apply_weights, load_causal, predicted_returns
from reproduce_revised_study import (
    FIG,
    OUT,
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
    PREFERRED_BOOK,
    apply_index_membership,
    ensemble_rank,
    every_n_mask,
    hysteresis_topk,
    load_dow_panel,
)

RNG_SEED = 20260907
N_BOOT = 2000
N_MC = 1000
BLOCK_MEAN = 21
N_SPA_BOOT = 2000


def _norm_sf(z: float) -> float:
    return 0.5 * erfc(abs(z) / sqrt(2.0))


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def wealth_returns(result: dict) -> np.ndarray:
    v = np.asarray(result["S_values"], dtype=float)
    r = np.diff(v) / np.maximum(v[:-1], 1e-12)
    return r[np.isfinite(r)]


def total_return_pct(result: dict) -> float:
    v = np.asarray(result["S_values"], dtype=float)
    return float(v[-1] / v[0] - 1.0) * 100.0


def metrics_from_returns(rets: np.ndarray, ppy: float = 252.0) -> dict:
    rets = np.asarray(rets, dtype=float)
    rets = rets[np.isfinite(rets)]
    n = max(len(rets), 1)
    wealth = np.cumprod(1.0 + rets)
    wealth = np.concatenate([[1.0], wealth])
    total = float(wealth[-1] - 1.0)
    peak = np.maximum.accumulate(wealth)
    mdd = float((wealth / np.maximum(peak, 1e-12) - 1.0).min())
    mu = float(np.mean(rets)) if n else 0.0
    sig = float(np.std(rets, ddof=1)) if n > 1 else 0.0
    sharpe = (mu / sig * np.sqrt(ppy)) if sig > 0 else 0.0
    return {
        "Total_Return_%": total * 100.0,
        "Sharpe_Ratio": float(sharpe),
        "Max_Drawdown_%": mdd * 100.0,
    }


def stationary_block_indices(n: int, block_mean: int, rng: np.random.Generator) -> np.ndarray:
    """Politis & Romano (1994) stationary bootstrap with circular wrap."""
    p = 1.0 / float(block_mean)
    out = np.empty(n, dtype=int)
    i = 0
    while i < n:
        start = int(rng.integers(0, n))
        length = int(rng.geometric(p))
        for j in range(length):
            if i >= n:
                break
            out[i] = (start + j) % n
            i += 1
    return out


def block_bootstrap_metrics(
    rets: np.ndarray,
    n_boot: int = N_BOOT,
    block_mean: int = BLOCK_MEAN,
    seed: int = RNG_SEED,
) -> dict:
    rng = np.random.default_rng(seed)
    rets = np.asarray(rets, dtype=float)
    n = len(rets)
    rec = np.empty((n_boot, 3), dtype=float)
    for b in range(n_boot):
        idx = stationary_block_indices(n, block_mean, rng)
        m = metrics_from_returns(rets[idx])
        rec[b, 0] = m["Total_Return_%"]
        rec[b, 1] = m["Sharpe_Ratio"]
        rec[b, 2] = m["Max_Drawdown_%"]
    lo, hi = np.quantile(rec, [0.025, 0.975], axis=0)
    return {
        "n_boot": n_boot,
        "block_mean": block_mean,
        "return_mean": float(rec[:, 0].mean()),
        "return_lo": float(lo[0]),
        "return_hi": float(hi[0]),
        "sharpe_mean": float(rec[:, 1].mean()),
        "sharpe_lo": float(lo[1]),
        "sharpe_hi": float(hi[1]),
        "mdd_mean": float(rec[:, 2].mean()),
        "mdd_lo": float(lo[2]),
        "mdd_hi": float(hi[2]),
    }


def paired_block_diff(
    rets_a: np.ndarray,
    rets_b: np.ndarray,
    n_boot: int = N_BOOT,
    block_mean: int = BLOCK_MEAN,
    seed: int = RNG_SEED + 1,
) -> dict:
    rng = np.random.default_rng(seed)
    n = min(len(rets_a), len(rets_b))
    a = np.asarray(rets_a[:n], dtype=float)
    b = np.asarray(rets_b[:n], dtype=float)
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = stationary_block_indices(n, block_mean, rng)
        ra = metrics_from_returns(a[idx])["Total_Return_%"]
        rb = metrics_from_returns(b[idx])["Total_Return_%"]
        diffs[i] = ra - rb
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {
        "diff_mean": float(diffs.mean()),
        "diff_lo": float(lo),
        "diff_hi": float(hi),
        "p_le_zero": float(np.mean(diffs <= 0.0)),
    }


def load_panel() -> dict:
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
    rhats["ABE"] = ensemble_rank(rhats, ["A", "B", "E"])
    rhats_hold = {L: r.rolling(20, min_periods=5).mean() for L, r in rhats.items()}
    rhats_pit = {L: apply_index_membership(rhats_hold[L]) for L in rhats_hold}
    baselines = run_baselines(close, open_, djia, decision_dates)
    return {
        "close": close,
        "open": open_,
        "djia": djia,
        "decision_dates": decision_dates,
        "month": month,
        "month_dates": month_dates,
        "rhats_pit": rhats_pit,
        "baselines": baselines,
    }


def next_trading_date(index: pd.DatetimeIndex, dt, horizon: int) -> pd.Timestamp | None:
    if dt not in index:
        return None
    loc = int(index.get_loc(dt))
    nxt = loc + int(horizon)
    if nxt >= len(index):
        return None
    return index[nxt]


def quintile_and_hitrate(panel: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    close = panel["close"]
    djia = panel["djia"]
    scores = panel["rhats_pit"]["B"]
    month_dates = [dt for dt in panel["month_dates"] if dt in scores.index and dt in close.index]
    w_hyst = hysteresis_topk(scores, month_dates, LP_K, LP_BUFFER)

    q_rows = []
    hit_hyst = []
    hit_top5 = []
    q_monthly = {f"Q{q}": [] for q in range(1, 6)}
    mkt_monthly = []
    djia_monthly = []

    for i, dt in enumerate(month_dates[:-1]):
        nxt = month_dates[i + 1]
        if nxt not in close.index:
            continue
        s = scores.loc[dt].dropna()
        if s.size < 10:
            continue
        cols = [c for c in s.index if c in close.columns]
        realized = (close.loc[nxt, cols] / close.loc[dt, cols] - 1.0).astype(float)
        realized = realized.replace([np.inf, -np.inf], np.nan).dropna()
        s = s.reindex(realized.index).dropna()
        realized = realized.reindex(s.index)
        if s.size < 10:
            continue

        # 20-session close-to-close when that date exists; else the monthly hold.
        dt20 = next_trading_date(close.index, dt, 20)
        if dt20 is not None:
            realized20 = (close.loc[dt20, s.index] / close.loc[dt, s.index] - 1.0).astype(float)
            realized20 = realized20.replace([np.inf, -np.inf], np.nan)
        else:
            realized20 = realized

        ranks = s.rank(method="first", ascending=True)
        try:
            labels = pd.qcut(ranks, 5, labels=[1, 2, 3, 4, 5])
        except ValueError:
            labels = pd.Series(
                pd.qcut(np.arange(len(s)), 5, labels=[1, 2, 3, 4, 5], duplicates="drop").astype(int),
                index=s.sort_values().index,
            ).reindex(s.index)

        row = {"Decision": pd.Timestamp(dt).strftime("%Y-%m-%d"), "N": int(s.size)}
        for q in range(1, 6):
            members = s.index[labels == q]
            mu = float(realized.reindex(members).mean()) if len(members) else np.nan
            row[f"Q{q}_%"] = mu * 100.0
            if np.isfinite(mu):
                q_monthly[f"Q{q}"].append(mu)
        mkt = float(realized.mean())
        row["EW_univ_%"] = mkt * 100.0
        mkt_monthly.append(mkt)
        if dt in djia.index and nxt in djia.index and float(djia.loc[dt]) > 0:
            dj = float(djia.loc[nxt] / djia.loc[dt] - 1.0)
        else:
            dj = mkt
        row["DJIA_%"] = dj * 100.0
        djia_monthly.append(dj)
        q_rows.append(row)

        med20 = float(realized20.median())
        top5 = list(s.nlargest(5).index)
        held = [t for t in w_hyst.columns if t in s.index and float(w_hyst.loc[dt, t]) > 1e-8]
        if not held:
            held = top5
        for tkr in top5:
            if tkr in realized20.index and np.isfinite(realized20[tkr]):
                hit_top5.append(int(float(realized20[tkr]) > med20))
        for tkr in held:
            if tkr in realized20.index and np.isfinite(realized20[tkr]):
                hit_hyst.append(int(float(realized20[tkr]) > med20))

    q_df = pd.DataFrame(q_rows)
    summary_rows = []
    q5 = np.asarray(q_monthly["Q5"], dtype=float)
    q1 = np.asarray(q_monthly["Q1"], dtype=float)
    mkt = np.asarray(mkt_monthly, dtype=float)
    dj = np.asarray(djia_monthly, dtype=float)
    spread = q5 - q1
    q5_mkt = q5 - mkt
    q5_dj = q5 - dj
    t_spread = stats.ttest_1samp(spread, 0.0, alternative="greater")
    t_mkt = stats.ttest_1samp(q5_mkt, 0.0, alternative="greater")
    t_dj = stats.ttest_1samp(q5_dj, 0.0, alternative="greater")

    for q in range(1, 6):
        xs = np.asarray(q_monthly[f"Q{q}"], dtype=float)
        se = float(xs.std(ddof=1) / np.sqrt(len(xs))) if len(xs) > 1 else np.nan
        tstat = float(xs.mean() / se) if se and se > 0 else np.nan
        pval = float(2.0 * _norm_sf(tstat)) if np.isfinite(tstat) else np.nan
        summary_rows.append(
            {
                "Quintile": f"Q{q}",
                "Mean monthly %": float(xs.mean() * 100.0),
                "Ann. mean %": float(xs.mean() * 12.0 * 100.0),
                "SE monthly %": se * 100.0,
                "t": tstat,
                "p": pval,
                "Holds": int(len(xs)),
            }
        )
    q_sum = pd.DataFrame(summary_rows)

    def _hit_pack(hits: list[int], label: str) -> dict:
        n = len(hits)
        k = int(np.sum(hits)) if n else 0
        rate = k / n if n else float("nan")
        bt = stats.binomtest(k, n, 0.5, alternative="greater") if n else None
        lo, hi = stats.binomtest(k, n, 0.5).proportion_ci(0.95) if n else (np.nan, np.nan)
        return {
            "Rule": label,
            "Hit rate": float(rate),
            "Hits": k,
            "Picks": n,
            "Binomial p (H0=50%)": float(bt.pvalue) if bt is not None else np.nan,
            "CI 2.5%": float(lo),
            "CI 97.5%": float(hi),
        }

    hit_df = pd.DataFrame(
        [
            _hit_pack(hit_top5, "Raw top-5 of $\\bar{r}^{(B)}$ (Precision@5)"),
            _hit_pack(hit_hyst, "Hysteresis set $I_t$ ($k=5$, $b=3$)"),
        ]
    )
    extra = {
        "q5_minus_q1_monthly_%": float(spread.mean() * 100.0),
        "q5_minus_q1_ann_%": float(spread.mean() * 12.0 * 100.0),
        "q5_minus_q1_t": float(t_spread.statistic),
        "q5_minus_q1_p": float(t_spread.pvalue),
        "q5_minus_ew_monthly_%": float(q5_mkt.mean() * 100.0),
        "q5_minus_ew_t": float(t_mkt.statistic),
        "q5_minus_ew_p": float(t_mkt.pvalue),
        "q5_minus_djia_monthly_%": float(q5_dj.mean() * 100.0),
        "q5_minus_djia_t": float(t_dj.statistic),
        "q5_minus_djia_p": float(t_dj.pvalue),
        "n_holds": int(len(spread)),
        "hit_top5": float(np.mean(hit_top5)) if hit_top5 else float("nan"),
        "hit_hyst": float(np.mean(hit_hyst)) if hit_hyst else float("nan"),
        "hit_top5_p": float(hit_df.iloc[0]["Binomial p (H0=50%)"]),
        "hit_hyst_p": float(hit_df.iloc[1]["Binomial p (H0=50%)"]),
        "hit_top5_n": int(hit_df.iloc[0]["Picks"]),
        "hit_hyst_n": int(hit_df.iloc[1]["Picks"]),
    }
    return q_sum, hit_df, extra


def plot_quintiles(q_sum: pd.DataFrame, extra: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    xs = np.arange(len(q_sum))
    vals = q_sum["Ann. mean %"].to_numpy()
    colors = ["#8C1D18" if v < 0 else "#1B4F72" for v in vals]
    ax.bar(xs, vals, color=colors, width=0.62, edgecolor="none")
    ax.axhline(0.0, color="#333333", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(q_sum["Quintile"].tolist())
    ax.set_ylabel("Annualized mean holding return (%)")
    ax.set_xlabel("Score quintile of $\\bar{r}^{(B)}_{i,t}$ (PIT), Q5 = right tail")
    ax.set_title("Monthly right-tail sort: subsequent-month returns")
    note = (
        f"Q5−Q1 = {extra['q5_minus_q1_ann_%']:.1f}% ann. "
        f"(t={extra['q5_minus_q1_t']:.2f}, one-sided p={extra['q5_minus_q1_p']:.3f})"
    )
    ax.text(0.02, 0.95, note, transform=ax.transAxes, va="top", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _scores_like(template: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = template.copy()
    vals = out.to_numpy(dtype=float, copy=True)
    mask = np.isfinite(vals)
    noise = rng.standard_normal(vals.shape)
    vals[mask] = noise[mask]
    return pd.DataFrame(vals, index=out.index, columns=out.columns)


def random_hysteresis_mc(panel: dict, n_mc: int = N_MC, seed: int = RNG_SEED + 7) -> dict:
    rng = np.random.default_rng(seed)
    scores = panel["rhats_pit"]["B"]
    month_dates = panel["month_dates"]
    close = panel["close"]
    open_ = panel["open"]
    decision_dates = panel["decision_dates"]
    month = panel["month"]
    rets = np.empty(n_mc, dtype=float)
    sharpes = np.empty(n_mc, dtype=float)
    for i in range(n_mc):
        fake = _scores_like(scores, rng)
        w = hysteresis_topk(fake, month_dates, LP_K, LP_BUFFER)
        res = apply_weights(close, open_, decision_dates, w, month)
        st = performance_stats(res)
        rets[i] = float(st["Total_Return_%"])
        sharpes[i] = float(st["Sharpe_Ratio"])
        if (i + 1) % 100 == 0:
            print(f"  random hysteresis {i + 1}/{n_mc}", flush=True)
    return {
        "n_mc": n_mc,
        "mean_return_%": float(rets.mean()),
        "p05_return_%": float(np.quantile(rets, 0.05)),
        "p50_return_%": float(np.quantile(rets, 0.50)),
        "p95_return_%": float(np.quantile(rets, 0.95)),
        "mean_sharpe": float(sharpes.mean()),
        "paths": rets.tolist(),
    }


def hansen_spa(
    d: np.ndarray,
    n_boot: int = N_SPA_BOOT,
    block_mean: int = BLOCK_MEAN,
    seed: int = RNG_SEED + 11,
) -> dict:
    """Hansen (2005) SPA and White (2000) Reality Check on loss differentials.

    d has shape (T, K); column k is r_{k,t} - r_{bench,t}.
    H0: max_k E[d_{k}] <= 0 (no model beats the benchmark).
    """
    d = np.asarray(d, dtype=float)
    t_len, k = d.shape
    dbar = d.mean(axis=0)
    omega = d.std(axis=0, ddof=1)
    omega = np.where(omega <= 1e-12, 1e-12, omega)
    tstat = np.sqrt(t_len) * dbar / omega
    spa_stat = float(max(np.max(tstat), 0.0))
    rc_stat = float(np.max(np.sqrt(t_len) * dbar))
    a_n = np.sqrt(2.0 * np.log(np.log(max(t_len, 3))))
    recenter = dbar * (np.sqrt(t_len) * dbar / omega > -a_n).astype(float)

    rng = np.random.default_rng(seed)
    spa_star = np.empty(n_boot, dtype=float)
    rc_star = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = stationary_block_indices(t_len, block_mean, rng)
        db = d[idx]
        dbar_b = db.mean(axis=0)
        omega_b = db.std(axis=0, ddof=1)
        omega_b = np.where(omega_b <= 1e-12, 1e-12, omega_b)
        t_spa = np.sqrt(t_len) * (dbar_b - recenter) / omega_b
        t_rc = np.sqrt(t_len) * (dbar_b - np.maximum(dbar, 0.0))
        spa_star[b] = max(float(np.max(t_spa)), 0.0)
        rc_star[b] = float(np.max(t_rc))
    return {
        "best_k": int(np.argmax(dbar)),
        "best_mean_excess": float(dbar.max()),
        "spa_stat": spa_stat,
        "spa_p": float(np.mean(spa_star >= spa_stat - 1e-12)),
        "rc_stat": rc_stat,
        "rc_p": float(np.mean(rc_star >= rc_stat - 1e-12)),
        "n_models": int(k),
        "n_obs": int(t_len),
        "model_mean_excess": dbar.tolist(),
        "model_t": tstat.tolist(),
    }


def run_family(panel: dict) -> dict[str, dict]:
    close = panel["close"]
    open_ = panel["open"]
    decision_dates = panel["decision_dates"]
    month = panel["month"]
    month_dates = panel["month_dates"]
    results = {}
    for L in ["A", "B", "C", "D", "E"]:
        w = hysteresis_topk(panel["rhats_pit"][L], month_dates, LP_K, LP_BUFFER)
        results[f"{L} EW k=5 b=3"] = apply_weights(close, open_, decision_dates, w, month)
    for b in (0, 1, 2, 3, 4, 5, 7, 9):
        w = hysteresis_topk(panel["rhats_pit"]["B"], month_dates, LP_K, b)
        results[f"B EW k=5 b={b}"] = apply_weights(close, open_, decision_dates, w, month)
    w_pref = hysteresis_topk(panel["rhats_pit"]["B"], month_dates, LP_K, LP_BUFFER)
    results[PREFERRED_BOOK] = apply_mean_var(
        close,
        open_,
        decision_dates,
        panel["rhats_pit"]["B"],
        month,
        w_pref,
        lam=MV_LAM,
        kappa=MV_KAPPA,
        max_weight=MV_CAP,
    )
    results["BH_DJIA"] = panel["baselines"]["BH_DJIA"]
    results["EW_1N"] = panel["baselines"]["EW_1N"]
    return results


def main() -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    print("Loading panel...", flush=True)
    panel = load_panel()
    print("Quintiles and hit rate...", flush=True)
    q_sum, hit_df, q_extra = quintile_and_hitrate(panel)
    q_sum.to_csv(TAB / "table_quintile_sort.csv", index=False)
    hit_df.to_csv(TAB / "table_precision_at_5.csv", index=False)
    plot_quintiles(q_sum, q_extra, FIG / "fig_quintile_sort.png")

    print("Rebuilding strategy family...", flush=True)
    family = run_family(panel)
    rets = {name: wealth_returns(res) for name, res in family.items()}
    stats_map = {name: performance_stats(res) for name, res in family.items()}

    print("Block bootstrap...", flush=True)
    boot_rows = []
    boot_detail = {}
    for name in [PREFERRED_BOOK, "B EW k=5 b=3", "BH_DJIA", "EW_1N"]:
        boot = block_bootstrap_metrics(rets[name])
        boot_detail[name] = boot
        st = stats_map[name]
        boot_rows.append(
            {
                "Strategy": name,
                "Point return %": float(st["Total_Return_%"]),
                "Return 2.5%": boot["return_lo"],
                "Return 97.5%": boot["return_hi"],
                "Point Sharpe": float(st["Sharpe_Ratio"]),
                "Sharpe 2.5%": boot["sharpe_lo"],
                "Sharpe 97.5%": boot["sharpe_hi"],
                "Point MDD %": float(st["Max_Drawdown_%"]),
                "MDD 2.5%": boot["mdd_lo"],
                "MDD 97.5%": boot["mdd_hi"],
            }
        )
    boot_df = pd.DataFrame(boot_rows)
    boot_df.to_csv(TAB / "table_block_bootstrap.csv", index=False)

    mv_vs_bh = paired_block_diff(rets[PREFERRED_BOOK], rets["BH_DJIA"])
    mv_vs_ew = paired_block_diff(rets[PREFERRED_BOOK], rets["EW_1N"])
    ew5_vs_1n = paired_block_diff(rets["B EW k=5 b=3"], rets["EW_1N"])

    print(f"Random hysteresis Monte Carlo ({N_MC} paths)...", flush=True)
    mc = random_hysteresis_mc(panel, n_mc=N_MC)
    mc_out = {k: v for k, v in mc.items() if k != "paths"}
    pd.DataFrame({"Total_Return_%": mc["paths"]}).to_csv(
        TAB / "table_random_hysteresis_paths.csv", index=False
    )
    ew5 = float(stats_map["B EW k=5 b=3"]["Total_Return_%"])
    mv = float(stats_map[PREFERRED_BOOK]["Total_Return_%"])
    ew1n = float(stats_map["EW_1N"]["Total_Return_%"])
    bh = float(stats_map["BH_DJIA"]["Total_Return_%"])
    mc_p = float(np.mean(np.asarray(mc["paths"]) >= ew5))
    attr = pd.DataFrame(
        [
            {
                "Layer": "Random 5-name hysteresis",
                "Rule": "1,000 paths; identical $k=5$, $b=3$; random scores",
                "Return %": mc["mean_return_%"],
                "Sharpe": mc["mean_sharpe"],
            },
            {
                "Layer": "Equal-weight top-5 Model B",
                "Rule": "PIT $\\bar{r}^{(B)}$, $k=5$, $b=3$, $1/5$ weights",
                "Return %": ew5,
                "Sharpe": float(stats_map["B EW k=5 b=3"]["Sharpe_Ratio"]),
            },
            {
                "Layer": "Mean–variance–turnover top-5 Model B",
                "Rule": "Tsang et al. (2025) share programme on the same $I_t$",
                "Return %": mv,
                "Sharpe": float(stats_map[PREFERRED_BOOK]["Sharpe_Ratio"]),
            },
        ]
    )
    attr.to_csv(TAB / "table_performance_attribution.csv", index=False)

    print("Hansen SPA / White Reality Check...", flush=True)
    spa_names = [f"{L} EW k=5 b=3" for L in "ABCDE"] + [
        f"B EW k=5 b={b}" for b in (0, 1, 2, 3, 4, 5, 7, 9)
    ]
    # Unique names: B EW k=5 b=3 appears twice
    spa_names = list(dict.fromkeys(spa_names))
    bench = rets["BH_DJIA"]
    n = min(len(bench), min(len(rets[n]) for n in spa_names))
    d = np.column_stack([rets[name][:n] - bench[:n] for name in spa_names])
    spa = hansen_spa(d)
    spa_rows = []
    for i, name in enumerate(spa_names):
        spa_rows.append(
            {
                "Strategy": name,
                "Mean daily excess vs BH": spa["model_mean_excess"][i],
                "Studentized t": spa["model_t"][i],
                "Point return %": float(stats_map[name]["Total_Return_%"]),
            }
        )
    spa_df = pd.DataFrame(spa_rows).sort_values("Mean daily excess vs BH", ascending=False)
    spa_df.to_csv(TAB / "table_spa_family.csv", index=False)

    # SPA on the five-model family only (cleaner snooping set)
    five = [f"{L} EW k=5 b=3" for L in "ABCDE"]
    d5 = np.column_stack([rets[name][:n] - bench[:n] for name in five])
    spa5 = hansen_spa(d5, seed=RNG_SEED + 17)

    summary = {
        "quintiles": q_extra,
        "quintile_table": q_sum.to_dict(orient="records"),
        "hit_rate": {
            "precision_at_5": q_extra["hit_top5"],
            "precision_at_5_p": q_extra["hit_top5_p"],
            "precision_at_5_n": q_extra["hit_top5_n"],
            "hysteresis_hit": q_extra["hit_hyst"],
            "hysteresis_hit_p": q_extra["hit_hyst_p"],
            "hysteresis_hit_n": q_extra["hit_hyst_n"],
        },
        "bootstrap": {
            "B": N_BOOT,
            "block_mean": BLOCK_MEAN,
            "mv": boot_detail[PREFERRED_BOOK],
            "ew5": boot_detail["B EW k=5 b=3"],
            "bh": boot_detail["BH_DJIA"],
            "ew1n": boot_detail["EW_1N"],
            "mv_vs_bh": mv_vs_bh,
            "mv_vs_ew1n": mv_vs_ew,
            "ew5_vs_ew1n": ew5_vs_1n,
        },
        "attribution": {
            "random_mean_%": mc["mean_return_%"],
            "random_p05_%": mc["p05_return_%"],
            "random_p50_%": mc["p50_return_%"],
            "random_p95_%": mc["p95_return_%"],
            "random_sharpe": mc["mean_sharpe"],
            "ew_top5_%": ew5,
            "mv_top5_%": mv,
            "ew_1n_%": ew1n,
            "bh_%": bh,
            "signal_alpha_%": ew5 - mc["mean_return_%"],
            "optimizer_alpha_%": mv - ew5,
            "signal_vs_1n_%": ew5 - ew1n,
            "p_random_ge_ew5": mc_p,
        },
        "spa": {
            "family": spa_names,
            "best": spa_names[spa["best_k"]],
            "spa_p": spa["spa_p"],
            "rc_p": spa["rc_p"],
            "best_mean_excess": spa["best_mean_excess"],
            "n_models": spa["n_models"],
            "five_best": five[spa5["best_k"]],
            "five_spa_p": spa5["spa_p"],
            "five_rc_p": spa5["rc_p"],
        },
        "point": {
            "mv_sharpe": float(stats_map[PREFERRED_BOOK]["Sharpe_Ratio"]),
            "mv_mdd": float(stats_map[PREFERRED_BOOK]["Max_Drawdown_%"]),
            "bh_sharpe": float(stats_map["BH_DJIA"]["Sharpe_Ratio"]),
            "ew1n_sharpe": float(stats_map["EW_1N"]["Sharpe_Ratio"]),
            "ew5_sharpe": float(stats_map["B EW k=5 b=3"]["Sharpe_Ratio"]),
        },
    }
    (TAB / "ejor_referee_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: summary[k] for k in ("quintiles", "hit_rate", "attribution", "spa")}, indent=2))
    print("Wrote", TAB / "ejor_referee_summary.json")


if __name__ == "__main__":
    main()
