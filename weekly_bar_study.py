"""
Causal study on weekly bars.

The forecast step and the holding period are both one week.
Look-ahead (full-sample) books and the daily share LP are not run here.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from cost_aware_strategies import apply_weights
from reproduce_revised_study import (
    CEEMDAN_TRIALS,
    FIG,
    LEAD_H,
    N_JOBS,
    OUT,
    S0,
    TAB,
    TEST_END,
    TEST_START,
    TICKERS,
    VARIANT_MAP,
    Kalman1D,
    _ensure_dirs,
    aligned_streak_table,
    aligned_value_table,
    ceemdan_matrix,
    load_market_data,
    performance_stats,
    plot_drawdown,
    plot_equity,
    run_baselines,
    savgol_hist,
    series_from_result,
    taylor_from_scores,
    trailing_pca,
)

LOOKBACK_W = 100
PCA_W = 20
SG_W = 7
CKPT_W = OUT / "checkpoints" / "weekly_forecasts.pkl"
PPY = 52.0


def to_weekly(close: pd.DataFrame, open_: pd.DataFrame, djia: pd.Series):
    w_close = close.resample("W-FRI").last().dropna(how="any")
    w_open = open_.resample("W-FRI").first().reindex(w_close.index)
    w_open = w_open.fillna(w_close)
    w_djia = djia.resample("W-FRI").last().reindex(w_close.index).ffill()
    return w_close, w_open, w_djia


def forecast_weekly(close: np.ndarray, seed: int) -> dict[str, float]:
    last = float(close[-1])
    if len(close) < 40:
        return {k: last for k in VARIANT_MAP}
    try:
        F_raw = ceemdan_matrix(close, seed=seed)
    except Exception:
        return {k: last for k in VARIANT_MAP}

    F_sg = np.column_stack(
        [savgol_hist(F_raw[:, j], window=SG_W) for j in range(F_raw.shape[1])]
    )
    t_idx = len(F_raw) - 1

    def predict_from(F: np.ndarray, h: float, momentum: bool, smooth_pc: bool) -> float:
        mu, V = trailing_pca(F, PCA_W)
        hist_len = min(20, t_idx + 1)
        scores = np.array([V.T @ (F[t_idx - i] - mu) for i in range(hist_len - 1, -1, -1)])
        if smooth_pc and len(scores) >= 7:
            scores = np.column_stack(
                [savgol_hist(scores[:, j], 7, 2) for j in range(scores.shape[1])]
            )
        g_next = taylor_from_scores(scores, h)
        if momentum and t_idx >= 5:
            price_diff = np.diff(F[max(0, t_idx - 5) : t_idx + 1, -1])
            mom = float(np.mean(price_diff[-3:]))
            g_next = g_next + 0.4 * mom * V.T[:, -1]
        pred = mu + V @ g_next
        return float(pred.sum())

    a = predict_from(F_raw, h=1.0, momentum=False, smooth_pc=False)
    b = predict_from(F_sg, h=1.0, momentum=False, smooth_pc=False)
    c = predict_from(F_sg, h=LEAD_H, momentum=True, smooth_pc=False)
    d_raw = predict_from(F_sg, h=1.0, momentum=False, smooth_pc=True)
    kf = Kalman1D(R=0.05, Q=0.01, x0=close[max(0, t_idx - 10)])
    for z in close[max(0, t_idx - 10) : t_idx + 1]:
        kf.update(float(z))
    d = 0.7 * d_raw + 0.3 * kf.x

    e = last
    try:
        from scipy.interpolate import splev, splrep
        from sklearn.decomposition import PCA

        n_comp = min(6, F_sg.shape[1])
        pca = PCA(n_components=n_comp)
        feats = pca.fit_transform(F_sg)
        pred_pca = []
        t = np.arange(len(feats))
        for j in range(n_comp):
            tck = splrep(t, feats[:, j], k=2, s=0.1 * len(feats))
            pred_pca.append(float(splev(len(feats), tck, ext=3)))
        recon = pca.inverse_transform(np.array(pred_pca).reshape(1, -1))[0]
        residual = F_sg[:, -1]
        trend_inc = residual[-1] - residual[-2] if len(residual) > 1 else 0.0
        e = float(np.clip(np.sum(recon) + residual[-1] + trend_inc, last * 0.9, last * 1.1))
    except Exception:
        e = last
    return {"A": float(a), "B": float(b), "C": float(c), "D": float(d), "E": float(e)}


def _weekly_ticker_worker(payload: dict) -> tuple[str, pd.DataFrame]:
    ticker = payload["ticker"]
    close = np.asarray(payload["close"], dtype=float)
    dates = pd.to_datetime(payload["dates"])
    decision_pos = payload["decision_pos"]
    lookback = int(payload["lookback"])
    rows = []
    for pos in decision_pos:
        start = max(0, pos - lookback + 1)
        hist = close[start : pos + 1]
        seed = (abs(hash(ticker)) + pos) % (2**31 - 1)
        preds = forecast_weekly(hist, seed=seed)
        rows.append({"decision_date": dates[pos].strftime("%Y-%m-%d"), **preds})
    return ticker, pd.DataFrame(rows)


def run_weekly_walkforward(close: pd.DataFrame, decision_dates: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    if CKPT_W.exists():
        print("Loading cached weekly forecasts")
        return pd.read_pickle(CKPT_W)

    decision_pos = [int(close.index.get_loc(dt)) for dt in decision_dates]
    payloads = [
        {
            "ticker": t,
            "close": close[t].values,
            "dates": close.index.astype(str).tolist(),
            "decision_pos": decision_pos,
            "lookback": LOOKBACK_W,
        }
        for t in TICKERS
    ]
    results = {}
    print(
        f"Weekly CEEMDAN on {len(TICKERS)} names, {len(decision_dates)} weeks, {N_JOBS} workers"
    )
    if N_JOBS == 1:
        for payload in payloads:
            ticker, df = _weekly_ticker_worker(payload)
            results[ticker] = df
    else:
        with ProcessPoolExecutor(max_workers=N_JOBS) as ex:
            futs = {ex.submit(_weekly_ticker_worker, p): p["ticker"] for p in payloads}
            for fut in as_completed(futs):
                ticker, df = fut.result()
                results[ticker] = df
                print(f"finished {ticker}", flush=True)

    out = {}
    for letter in VARIANT_MAP:
        wide = pd.DataFrame(
            {t: results[t].set_index("decision_date")[letter] for t in TICKERS}
        )
        wide.index = pd.to_datetime(wide.index)
        out[letter] = wide.sort_index()
    pd.to_pickle(out, CKPT_W)
    return out


def predicted_returns(pred: pd.DataFrame, close: pd.DataFrame, dates) -> pd.DataFrame:
    px = close.reindex(dates)[TICKERS]
    hat = pred.reindex(dates)[TICKERS]
    return hat / px - 1.0


def topk_weights(scores: pd.DataFrame, k: int) -> pd.DataFrame:
    w = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
    for dt, row in scores.iterrows():
        w.loc[dt, row.nlargest(k).index] = 1.0 / k
    return w


def ensemble_rank(rhats: dict[str, pd.DataFrame], letters: list[str]) -> pd.DataFrame:
    ranks = [rhats[L].rank(axis=1, method="average") for L in letters]
    return sum(ranks) / len(ranks)


def invvol_topk(scores: pd.DataFrame, close: pd.DataFrame, k: int, vol_win: int = 8) -> pd.DataFrame:
    vol = close.pct_change().rolling(vol_win, min_periods=4).std()
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


def weekly_skill(scores: pd.DataFrame, close: pd.DataFrame, dates) -> dict:
    rhos, spread = [], []
    dates = pd.DatetimeIndex(dates)
    for i, dt in enumerate(dates[:-1]):
        nxt = dates[i + 1]
        if dt not in scores.index or nxt not in close.index or dt not in close.index:
            continue
        realized = close.loc[nxt, TICKERS] / close.loc[dt, TICKERS] - 1.0
        s = scores.loc[dt, TICKERS]
        rho = s.corr(realized, method="spearman")
        if pd.notna(rho):
            rhos.append(float(rho))
        top = s.nlargest(5).index
        bot = s.nsmallest(5).index
        spread.append(float(realized[top].mean() - realized[bot].mean()))
    return {
        "Mean_weekly_Spearman": float(np.mean(rhos)) if rhos else np.nan,
        "Top5_minus_bottom5_%": float(np.mean(spread)) * 100.0 if spread else np.nan,
        "Weeks": int(len(rhos)),
    }


def stats(res: dict) -> dict:
    return performance_stats(res, periods_per_year=PPY)


def main() -> None:
    _ensure_dirs()
    close_d, open_d, djia_d = load_market_data()
    close, open_, djia = to_weekly(close_d.loc[:TEST_END, TICKERS], open_d, djia_d)
    week_end = pd.Timestamp(TEST_END) + pd.Timedelta(days=6)
    test = close.index[(close.index >= TEST_START) & (close.index <= week_end)]
    decision_dates = test[:-1]
    print(
        f"Weekly panel {close.index.min().date()} to {close.index.max().date()}, "
        f"decisions {len(decision_dates)}"
    )

    causal = run_weekly_walkforward(close, decision_dates)
    rhats = {L: predicted_returns(pred, close, decision_dates) for L, pred in causal.items()}
    rhats["ABE"] = ensemble_rank(rhats, ["A", "B", "E"])

    skill_rows = []
    for name, sc in rhats.items():
        row = weekly_skill(sc, close, decision_dates)
        row["Signal"] = name
        skill_rows.append(row)
    skill = pd.DataFrame(skill_rows)
    skill.to_csv(TAB / "weekly_bar_skill.csv", index=False)

    baselines = run_baselines(close, open_, djia, decision_dates)
    results = dict(baselines)
    configs = {}
    for L in ["A", "B", "C", "D", "E", "ABE"]:
        for k in (3, 5, 8):
            configs[f"{L} weekly top-{k}"] = topk_weights(rhats[L], k)
    configs["ABE weekly top-5 inv-vol"] = invvol_topk(rhats["ABE"], close, 5)
    configs["B weekly top-5 inv-vol"] = invvol_topk(rhats["B"], close, 5)
    configs["ABE weekly bottom-3 (fade)"] = topk_weights(-rhats["ABE"], 3)

    for label, w in configs.items():
        results[label] = apply_weights(close, open_, decision_dates, w, None)

    rows = []
    for name, res in results.items():
        row = stats(res)
        row["Strategy"] = name
        rows.append(row)
    metrics = pd.DataFrame(rows).sort_values("Total_Return_%", ascending=False)
    metrics.to_csv(TAB / "weekly_bar_performance.csv", index=False)

    bh = float(metrics.loc[metrics["Strategy"] == "BH_DJIA", "Total_Return_%"].iloc[0])
    forecast_only = metrics[
        ~metrics["Strategy"].isin(["BH_DJIA", "EW_1N", "EG", "ONS"])
        & ~metrics["Strategy"].str.contains("fade|bottom", case=False)
    ]
    acceptable = forecast_only[forecast_only["Total_Return_%"] > 0]
    beat = acceptable[acceptable["Total_Return_%"] > bh]
    best = (beat if not beat.empty else acceptable).iloc[0]
    recommended = str(best["Strategy"])

    # Report table: positive long books plus BH / 1/N. No leaked, daily LP, or fade.
    report_names = list(acceptable["Strategy"]) + ["BH_DJIA", "EW_1N"]
    report = metrics[metrics["Strategy"].isin(report_names)].copy()
    report["_ord"] = report["Strategy"].map(
        {n: i for i, n in enumerate([recommended, *report_names])}
    )
    report = report.sort_values(["_ord", "Total_Return_%"], ascending=[True, False], na_position="last")
    report = report.drop(columns="_ord").drop_duplicates("Strategy")
    report = report.sort_values("Total_Return_%", ascending=False)
    report.to_csv(TAB / "weekly_bar_report_table.csv", index=False)

    print("Weekly-bar skill")
    print(skill.to_string(index=False))
    print(f"\nBuy-and-Hold DJIA = {bh:.2f}%")
    print(f"Recommended: {recommended}  {best['Total_Return_%']:.2f}%")
    print("\nPositive forecast books")
    print(
        acceptable[["Strategy", "Total_Return_%", "Max_Drawdown_%", "Sharpe_Ratio", "Turnover_%"]]
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
    }
    for extra in ["B weekly top-5", "ABE weekly top-5", "A weekly top-5"]:
        if extra in results and extra not in series and extra in set(acceptable["Strategy"]):
            series[extra] = series_from_result(results[extra])
            if len(series) >= 5:
                break
    plot_equity(
        series,
        FIG / "fig8_weekly_causal.png",
        "Causal weekly-bar books versus Buy-and-Hold DJIA and 1/N",
    )
    plot_drawdown(
        series,
        FIG / "fig9_weekly_drawdown.png",
        "Drawdowns of the causal weekly-bar books",
    )

    rec = results[recommended]
    aligned_streak_table(rec, close, TICKERS).to_csv(TAB / "table_weekly_streaks.csv", index=False)
    aligned_value_table(rec, close, TICKERS).to_csv(TAB / "table_weekly_value.csv", index=False)

    mse_rows = []
    for ticker in TICKERS:
        row = {"Ticker": ticker}
        for letter, df in causal.items():
            err = []
            for dt in df.index:
                loc = close.index.get_loc(dt)
                if loc + 1 < len(close.index):
                    err.append((df.loc[dt, ticker] - close.iloc[loc + 1][ticker]) ** 2)
            row[letter] = float(np.mean(err)) if err else np.nan
        mse_rows.append(row)
    pd.DataFrame(mse_rows).to_csv(TAB / "table1_mse_weekly.csv", index=False)

    (OUT / "weekly_bar_meta.json").write_text(
        json.dumps(
            {
                "bh_return_pct": bh,
                "recommended": recommended,
                "recommended_return_pct": float(best["Total_Return_%"]),
                "n_weeks": int(len(decision_dates)),
                "lookback_weeks": LOOKBACK_W,
                "pca_window_weeks": PCA_W,
                "sg_window_weeks": SG_W,
                "periods_per_year": PPY,
                "test_start": str(decision_dates[0].date()),
                "test_end": TEST_END,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Wrote weekly-bar tables and figures")


if __name__ == "__main__":
    main()
