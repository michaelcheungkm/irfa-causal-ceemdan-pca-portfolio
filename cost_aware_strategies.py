"""
Causal forecasts, implementable allocation.

The original LP maximizes shares * predicted price and rebalances daily.
That is not a return forecast and it burns gamma. Here the same walk-forward
CEEMDAN-PCA-Taylor prefixes are converted to predicted returns and traded
with a no-trade region, weekly rebalancing, or a 1/N blend.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reproduce_revised_study import (
    FIG,
    OUT,
    S0,
    TAB,
    TEST_END,
    TEST_START,
    TICKERS,
    VARIANT_LABELS,
    GAMMA,
    MAX_WEIGHT,
    load_market_data,
    optimize_shares,
    performance_stats,
    plot_equity,
    run_baselines,
    series_from_result,
    style_axes,
)

CKPT = OUT / "checkpoints"


def load_causal() -> dict[str, pd.DataFrame]:
    pkl = CKPT / "causal_forecasts.pkl"
    if pkl.exists():
        return pd.read_pickle(pkl)
    out = {}
    for letter in VARIANT_LABELS:
        parts = []
        for t in TICKERS:
            df = pd.read_csv(CKPT / f"forecast_{t}.csv", parse_dates=["decision_date"])
            parts.append(df.set_index("decision_date")[letter].rename(t))
        out[letter] = pd.concat(parts, axis=1).sort_index()
    return out


def apply_weights(
    close: pd.DataFrame,
    open_: pd.DataFrame,
    decision_dates: pd.DatetimeIndex,
    target_w: pd.DataFrame,
    rebalance_mask: np.ndarray | None = None,
    gamma: float | None = None,
) -> dict:
    cost_rate = GAMMA if gamma is None else float(gamma)
    n = close.shape[1]
    prev_nc = np.zeros(n)
    cash = S0
    values = [S0]
    turns = []
    exec_dates = []
    ncs = []
    for i, dt in enumerate(decision_dates):
        loc = close.index.get_loc(dt)
        if loc + 1 >= len(close.index):
            break
        nxt = close.index[loc + 1]
        p_open = open_.loc[nxt].values
        s_open = float(prev_nc @ p_open) + cash
        do_reb = True if rebalance_mask is None else bool(rebalance_mask[i])
        if do_reb and dt in target_w.index:
            aligned = target_w.reindex(columns=close.columns).loc[dt].fillna(0.0)
            w = np.maximum(aligned.values.astype(float), 0.0)
            if w.sum() <= 0:
                # Warm-up or empty ranker: keep the previous book (cash at t0).
                nc = prev_nc.copy()
            else:
                w = w / w.sum()
                nc = (w * s_open) / np.maximum(p_open, 0.01)
        else:
            nc = prev_nc.copy()
        traded = np.abs(nc - prev_nc)
        cost = cost_rate * float(traded @ p_open)
        if float(nc @ p_open) + cost > s_open > 0:
            scale = max(0.0, (s_open * 0.999) / (float(nc @ p_open) + cost))
            nc = nc * scale
            traded = np.abs(nc - prev_nc)
            cost = cost_rate * float(traded @ p_open)
        cash = s_open - cost - float(nc @ p_open)
        wealth = float(nc @ close.loc[nxt].values) + cash
        values.append(wealth)
        turns.append(float(traded @ p_open) / max(s_open, 1e-9))
        ncs.append(nc)
        exec_dates.append(nxt)
        prev_nc = nc
    return {
        "NCs": ncs,
        "S_values": np.array(values),
        "turnover_path": np.array(turns),
        "index": pd.DatetimeIndex(exec_dates),
        "value_index": pd.DatetimeIndex([decision_dates[0]] + exec_dates),
    }


def apply_share_lp(
    close: pd.DataFrame,
    open_: pd.DataFrame,
    decision_dates: pd.DatetimeIndex,
    pred_price: pd.DataFrame,
    rebalance_mask: np.ndarray | None = None,
    select: pd.DataFrame | None = None,
    gamma: float | None = None,
    max_weight: float | None = None,
) -> dict:
    """Monthly (or masked) share LP from A strategy for portfolio allocations.docx.

    pred_price is the predicted next price used in the objective. NaN means the
    name is ineligible. If select is provided, only strictly positive entries
    may receive new shares; other names are sold.
    """
    cost_rate = GAMMA if gamma is None else float(gamma)
    cap = MAX_WEIGHT if max_weight is None else float(max_weight)
    n = close.shape[1]
    cols = list(close.columns)
    prev_nc = np.zeros(n)
    cash = S0
    values = [S0]
    turns = []
    exec_dates = []
    ncs = []
    weights = pd.DataFrame(0.0, index=decision_dates, columns=cols)
    pred_price = pred_price.reindex(index=decision_dates, columns=cols)
    if select is not None:
        select = select.reindex(index=decision_dates, columns=cols).fillna(0.0)
    for i, dt in enumerate(decision_dates):
        loc = close.index.get_loc(dt)
        if loc + 1 >= len(close.index):
            break
        nxt = close.index[loc + 1]
        p_open = open_.loc[nxt].to_numpy(dtype=float)
        s_open = float(prev_nc @ p_open) + cash
        do_reb = True if rebalance_mask is None else bool(rebalance_mask[i])
        nc = prev_nc.copy()
        if do_reb:
            hat = pred_price.loc[dt]
            ok = hat.notna().to_numpy() & np.isfinite(p_open)
            if select is not None:
                ok = ok & (select.loc[dt].to_numpy() > 1e-8)
            idx = np.flatnonzero(ok)
            if len(idx) == 0 or s_open <= 0:
                nc = np.zeros(n)
            else:
                nc_e = optimize_shares(
                    prev_nc[idx],
                    p_open[idx],
                    hat.to_numpy(dtype=float)[idx],
                    s_open,
                    gamma=cost_rate,
                    max_weight=cap,
                )
                nc = np.zeros(n)
                nc[idx] = nc_e
        traded = np.abs(nc - prev_nc)
        cost = cost_rate * float(traded @ p_open)
        if float(nc @ p_open) + cost > s_open > 0:
            scale = max(0.0, (s_open * 0.999) / (float(nc @ p_open) + cost))
            nc = nc * scale
            traded = np.abs(nc - prev_nc)
            cost = cost_rate * float(traded @ p_open)
        cash = s_open - cost - float(nc @ p_open)
        wealth = float(nc @ close.loc[nxt].to_numpy(dtype=float)) + cash
        mark = np.maximum(nc * close.loc[nxt].to_numpy(dtype=float), 0.0)
        if wealth > 0 and mark.sum() > 0:
            weights.loc[dt] = mark / wealth
        values.append(wealth)
        turns.append(float(traded @ p_open) / max(s_open, 1e-9))
        ncs.append(nc)
        exec_dates.append(nxt)
        prev_nc = nc
    return {
        "NCs": ncs,
        "S_values": np.array(values),
        "turnover_path": np.array(turns),
        "index": pd.DatetimeIndex(exec_dates),
        "value_index": pd.DatetimeIndex([decision_dates[0]] + exec_dates),
        "weights": weights,
    }


def optimize_mean_var_turnover(
    mu: np.ndarray,
    sigma: np.ndarray,
    b_prev: np.ndarray,
    lam: float = 0.01,
    kappa: float = 1.0,
    max_weight: float = 1.0,
) -> np.ndarray:
    """Weight-space fallback used only if the share programme fails."""
    from scipy.optimize import minimize

    mu = np.asarray(mu, dtype=float).reshape(-1)
    n = mu.size
    sigma = np.asarray(sigma, dtype=float).reshape(n, n)
    sigma = 0.5 * (sigma + sigma.T) + 1e-8 * np.eye(n)
    b_prev = np.asarray(b_prev, dtype=float).reshape(-1)
    if b_prev.size != n:
        b_prev = np.zeros(n)
    b_prev = np.maximum(b_prev, 0.0)
    if b_prev.sum() > 0:
        b_prev = b_prev / b_prev.sum()
    cap = 1.0 if max_weight is None else float(max_weight)
    cap = min(max(cap, 1.0 / n), 1.0)

    x0 = np.concatenate([np.ones(n) / n, np.abs(np.ones(n) / n - b_prev)])

    def objective(x):
        b = x[:n]
        u = x[n:]
        var = float(max(b @ sigma @ b, 0.0))
        return -(float(b @ mu) - lam * float(u.sum()) - kappa * np.sqrt(var + 1e-16))

    bounds = [(0.0, cap)] * n + [(0.0, None)] * n
    cons = [
        {"type": "eq", "fun": lambda x: float(x[:n].sum() - 1.0)},
        {"type": "ineq", "fun": lambda x: x[n:] - (x[:n] - b_prev)},
        {"type": "ineq", "fun": lambda x: x[n:] - (b_prev - x[:n])},
    ]
    res = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 400, "ftol": 1e-10, "disp": False},
    )
    if res.success:
        b = np.maximum(res.x[:n], 0.0)
        if b.sum() > 0:
            return b / b.sum()
    return np.ones(n) / n


def optimize_shares_mean_var(
    prev_nc: np.ndarray,
    price: np.ndarray,
    pred: np.ndarray,
    wealth: float,
    sigma: np.ndarray,
    gamma: float = 0.001,
    lam: float = 0.01,
    kappa: float = 2.0,
    max_weight: float = 0.30,
    use_excess: bool = False,
) -> np.ndarray:
    """Share-space mean–risk–turnover programme.

    Decision variables are share counts NC, not portfolio weights.

    max_{NC,u}  NC' alpha - lam * price' u - kappa * sqrt((NC odot price)' Sigma (NC odot price))
    s.t.  price' NC + gamma * price' u <= wealth
          |NC - prev_nc| <= u
          price_i NC_i <= max_weight * wealth
          NC, u >= 0

    alpha is predicted price (default) or predicted minus open price when
    use_excess is True. Excess value lets residual cash compete, because cash
    contributes 0 to NC'(pred - price) while a fully invested book no longer
    gets a ~wealth bonus from NC'pred.
    """
    from scipy.optimize import minimize

    price = np.asarray(price, dtype=float).reshape(-1)
    pred = np.asarray(pred, dtype=float).reshape(-1)
    prev_nc = np.asarray(prev_nc, dtype=float).reshape(-1)
    n = price.size
    if wealth <= 0 or n == 0:
        return np.zeros(n)
    price = np.maximum(price, 0.01)
    pred = np.where(np.isfinite(pred), pred, price)
    prev_nc = np.maximum(prev_nc, 0.0)
    if prev_nc.size != n:
        prev_nc = np.zeros(n)
    sigma = np.asarray(sigma, dtype=float).reshape(n, n)
    sigma = 0.5 * (sigma + sigma.T) + 1e-8 * np.eye(n)
    cap = 1.0 if max_weight is None else float(max_weight)
    cap = min(max(cap, 1.0 / n), 1.0)
    nc_max = cap * wealth / price
    excess = pred - price
    if use_excess and not np.any(excess > 0):
        nc0 = np.zeros(n)
    else:
        dollar0 = np.minimum(np.full(n, wealth / n), cap * wealth * 0.999)
        nc0 = np.minimum(dollar0 / price, nc_max)
    u0 = np.abs(nc0 - prev_nc)
    x0 = np.concatenate([nc0, u0])
    scale = max(wealth, 1.0)
    alpha = excess if use_excess else pred

    def objective(x):
        nc = x[:n]
        u = x[n:]
        dollar = nc * price
        var = float(max(dollar @ sigma @ dollar, 0.0))
        return -(
            float(nc @ alpha) / scale
            - lam * float(price @ u) / scale
            - kappa * np.sqrt(var + 1e-16) / scale
        )

    bounds = [(0.0, float(nc_max[i])) for i in range(n)] + [(0.0, None)] * n
    cons = [
        {
            "type": "ineq",
            "fun": lambda x: float(wealth - price @ x[:n] - gamma * (price @ x[n:])),
        },
        {"type": "ineq", "fun": lambda x: x[n:] - (x[:n] - prev_nc)},
        {"type": "ineq", "fun": lambda x: x[n:] - (prev_nc - x[:n])},
    ]
    res = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 500, "ftol": 1e-10, "disp": False},
    )
    if res.success:
        nc = np.maximum(res.x[:n], 0.0)
        spent = float(nc @ price)
        if spent > wealth > 0:
            nc = nc * (wealth * 0.999 / spent)
        return nc
    if use_excess:
        return np.zeros(n)
    return optimize_shares(prev_nc, price, pred, wealth, gamma=gamma, max_weight=cap)


def apply_mean_var(
    close: pd.DataFrame,
    open_: pd.DataFrame,
    decision_dates: pd.DatetimeIndex,
    rhat: pd.DataFrame,
    rebalance_mask: np.ndarray,
    select: pd.DataFrame,
    gamma: float | None = None,
    lam: float = 0.01,
    kappa: float = 1.0,
    cov_window: int = 60,
    hold_days: int = 21,
    max_weight: float = 1.0,
    use_excess: bool = False,
    rf: pd.Series | None = None,
) -> dict:
    """Monthly share-space mean–variance–turnover on the hysteresis set.

    If rf is a daily simple rate aligned to calendar dates, idle cash earns
    that rate between marks, and the excess-value hurdle is the 21-day T-bill
    rather than a zero nominal return.
    """
    cost_rate = GAMMA if gamma is None else float(gamma)
    cols = list(close.columns)
    n = len(cols)
    prev_nc = np.zeros(n)
    cash = S0
    values = [S0]
    turns = []
    exec_dates = []
    ncs = []
    cash_fracs = []
    weights = pd.DataFrame(0.0, index=decision_dates, columns=cols)
    rhat = rhat.reindex(index=decision_dates, columns=cols)
    select = select.reindex(index=decision_dates, columns=cols).fillna(0.0)
    rets = close[cols].pct_change()
    rf_d = None
    if rf is not None:
        rf_d = pd.to_numeric(rf, errors="coerce").astype(float).ffill()

    for i, dt in enumerate(decision_dates):
        loc = close.index.get_loc(dt)
        if loc + 1 >= len(close.index):
            break
        nxt = close.index[loc + 1]
        p_open = open_.loc[nxt].to_numpy(dtype=float)
        s_open = float(prev_nc @ p_open) + cash
        do_reb = bool(rebalance_mask[i])
        nc = prev_nc.copy()
        if do_reb and s_open > 0:
            chosen = [c for c in cols if float(select.loc[dt, c]) > 1e-8 and pd.notna(rhat.loc[dt, c])]
            if not chosen:
                nc = np.zeros(n)
            else:
                idx = [cols.index(c) for c in chosen]
                hist = rets.loc[:dt, chosen].tail(cov_window).dropna(how="any")
                if hist.shape[0] < 10:
                    sigma = np.eye(len(chosen)) * (0.01**2)
                else:
                    sigma = hist.cov().to_numpy(dtype=float)
                sigma = sigma * hold_days
                p_dec = close.loc[dt, chosen].to_numpy(dtype=float)
                rr = rhat.loc[dt, chosen].to_numpy(dtype=float)
                rf_hold = 0.0
                if use_excess and rf_d is not None and dt in rf_d.index and pd.notna(rf_d.loc[dt]):
                    rf_hold = float(rf_d.loc[dt]) * hold_days
                if use_excess:
                    pred = p_open[idx] * (1.0 + (rr * hold_days) - rf_hold)
                else:
                    pred = p_dec * (1.0 + rr * hold_days)
                nc_e = optimize_shares_mean_var(
                    prev_nc[idx],
                    p_open[idx],
                    pred,
                    s_open,
                    sigma,
                    gamma=cost_rate,
                    lam=lam,
                    kappa=kappa,
                    max_weight=max_weight,
                    use_excess=use_excess,
                )
                nc = np.zeros(n)
                nc[idx] = nc_e
        traded = np.abs(nc - prev_nc)
        cost = cost_rate * float(traded @ p_open)
        if float(nc @ p_open) + cost > s_open > 0:
            scale = max(0.0, (s_open * 0.999) / (float(nc @ p_open) + cost))
            nc = nc * scale
            traded = np.abs(nc - prev_nc)
            cost = cost_rate * float(traded @ p_open)
        cash = s_open - cost - float(nc @ p_open)
        if rf_d is not None and nxt in rf_d.index and pd.notna(rf_d.loc[nxt]):
            cash = cash * (1.0 + float(rf_d.loc[nxt]))
        wealth = float(nc @ close.loc[nxt].to_numpy(dtype=float)) + cash
        mark = np.maximum(nc * close.loc[nxt].to_numpy(dtype=float), 0.0)
        if wealth > 0 and mark.sum() > 0:
            weights.loc[dt] = mark / wealth
        values.append(wealth)
        turns.append(float(traded @ p_open) / max(s_open, 1e-9))
        ncs.append(nc)
        cash_fracs.append(float(cash / wealth) if wealth > 0 else 1.0)
        exec_dates.append(nxt)
        prev_nc = nc
    return {
        "NCs": ncs,
        "S_values": np.array(values),
        "turnover_path": np.array(turns),
        "index": pd.DatetimeIndex(exec_dates),
        "value_index": pd.DatetimeIndex([decision_dates[0]] + exec_dates),
        "weights": weights,
        "cash_frac": np.array(cash_fracs),
    }


def predicted_returns(pred: pd.DataFrame, close: pd.DataFrame, decision_dates) -> pd.DataFrame:
    px = close.reindex(decision_dates)[TICKERS]
    hat = pred.reindex(decision_dates)[TICKERS]
    return hat / px - 1.0


def topk_weights(rhat: pd.DataFrame, k: int) -> pd.DataFrame:
    w = pd.DataFrame(0.0, index=rhat.index, columns=rhat.columns)
    for dt, row in rhat.iterrows():
        picks = row.nlargest(k).index
        w.loc[dt, picks] = 1.0 / k
    return w


def long_only_tilt(rhat: pd.DataFrame, hurdle: float) -> pd.DataFrame:
    x = rhat.clip(lower=0) - hurdle
    x = x.clip(lower=0)
    s = x.sum(axis=1)
    w = x.div(s.replace(0, np.nan), axis=0)
    eq = 1.0 / x.shape[1]
    return w.fillna(eq)


def blend(w_a: pd.DataFrame, w_b: pd.DataFrame, lam: float) -> pd.DataFrame:
    out = (1.0 - lam) * w_a + lam * w_b
    return out.div(out.sum(axis=1), axis=0)


def weekly_mask(dates: pd.DatetimeIndex) -> np.ndarray:
    # rebalance on first decision and every Friday
    mask = np.zeros(len(dates), dtype=bool)
    mask[0] = True
    for i, dt in enumerate(dates):
        if dt.weekday() == 4:
            mask[i] = True
    return mask


def forecast_skill(rhat: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in rhat.columns:
        hits, ics, n = [], [], 0
        for dt, rh in rhat[col].items():
            loc = close.index.get_loc(dt)
            if loc + 1 >= len(close.index):
                continue
            nxt = close.index[loc + 1]
            realized = float(close.loc[nxt, col] / close.loc[dt, col] - 1.0)
            hits.append(float((rh > 0) == (realized > 0)))
            ics.append((rh, realized))
            n += 1
        if not ics:
            continue
        a = np.array(ics)
        ic = float(np.corrcoef(a[:, 0], a[:, 1])[0, 1]) if len(a) > 2 else np.nan
        rows.append({"ticker": col, "hit_rate": float(np.mean(hits)), "IC": ic, "n": n})
    df = pd.DataFrame(rows)
    return df


def main() -> None:
    close, open_, djia = load_market_data()
    close = close.loc[:TEST_END, TICKERS]
    open_ = open_.reindex(close.index)[TICKERS].fillna(close)
    djia = djia.reindex(close.index).ffill()
    test_index = close.index[(close.index >= TEST_START) & (close.index <= TEST_END)]
    decision_dates = test_index[:-1]
    causal = load_causal()
    baselines = run_baselines(close, open_, djia, decision_dates)
    week = weekly_mask(decision_dates)
    eq = pd.DataFrame(1.0 / len(TICKERS), index=decision_dates, columns=TICKERS)

    skill_rows = []
    results = {}
    for name, res in baselines.items():
        results[name] = res

    for letter, pred in causal.items():
        rhat = predicted_returns(pred, close, decision_dates)
        sk = forecast_skill(rhat, close)
        skill_rows.append(
            {
                "Predictor": letter,
                "Mean_hit_rate": float(sk["hit_rate"].mean()),
                "Mean_IC": float(sk["IC"].mean()),
                "Median_IC": float(sk["IC"].median()),
            }
        )
        configs = {
            f"{letter} weekly top-5": (topk_weights(rhat, 5), week),
            f"{letter} weekly top-8": (topk_weights(rhat, 8), week),
            f"{letter} weekly bottom-5 (fade)": (topk_weights(-rhat, 5), week),
            f"{letter} weekly long-tilt": (long_only_tilt(rhat, 2 * GAMMA), week),
            f"{letter} 70/30 1N+top5 weekly": (blend(eq, topk_weights(rhat, 5), 0.30), week),
        }
        # daily top-5 with hurdle: names below hurdle get 0; if fewer than 5, renormalize
        w_h = pd.DataFrame(0.0, index=rhat.index, columns=rhat.columns)
        for dt, row in rhat.iterrows():
            elig = row[row > 2 * GAMMA]
            if len(elig) == 0:
                w_h.loc[dt] = eq.loc[dt]
            else:
                picks = elig.nlargest(min(5, len(elig))).index
                w_h.loc[dt, picks] = 1.0 / len(picks)
        configs[f"{letter} daily top-5 if r>2g"] = (w_h, None)

        for label, (w, mask) in configs.items():
            results[label] = apply_weights(close, open_, decision_dates, w, mask)

    skill = pd.DataFrame(skill_rows)
    skill.to_csv(TAB / "forecast_skill_causal.csv", index=False)

    rows = []
    for name, res in results.items():
        row = performance_stats(res)
        row["Strategy"] = name
        rows.append(row)
    metrics = pd.DataFrame(rows).sort_values("Total_Return_%", ascending=False)
    metrics.to_csv(TAB / "cost_aware_performance.csv", index=False)

    bh = float(metrics.loc[metrics["Strategy"] == "BH_DJIA", "Total_Return_%"].iloc[0])
    beat = metrics[metrics["Total_Return_%"] > bh + 1e-9]
    print("Forecast skill")
    print(skill.to_string(index=False))
    print(f"\nBuy-and-Hold DJIA return = {bh:.2f}%")
    print("\nStrategies that beat BH:")
    print(beat[["Strategy", "Final_Value", "Total_Return_%", "Max_Drawdown_%", "Sharpe_Ratio", "Turnover_%"]].to_string(index=False))
    print("\nAll strategies")
    print(
        metrics[["Strategy", "Total_Return_%", "Max_Drawdown_%", "Sharpe_Ratio", "Turnover_%"]]
        .to_string(index=False)
        .encode("ascii", "replace")
        .decode("ascii")
    )

    # equity plot: BH, 1/N, and the best two forecast strategies if any
    series = {
        "BH_DJIA": series_from_result(results["BH_DJIA"]),
        "EW_1N": series_from_result(results["EW_1N"]),
    }
    forecast_names = [n for n in metrics["Strategy"] if n not in ("BH_DJIA", "EW_1N", "EG", "ONS")]
    top_forecast = metrics[metrics["Strategy"].isin(forecast_names)].head(3)["Strategy"].tolist()
    for name in top_forecast:
        series[name] = series_from_result(results[name])
    plot_equity(
        series,
        FIG / "fig7_cost_aware.png",
        "Causal forecasts with weekly / hurdled allocation versus Buy-and-Hold",
    )
    (OUT / "cost_aware_meta.json").write_text(
        json.dumps({"bh_return_pct": bh, "n_beat_bh": int(len(beat))}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
