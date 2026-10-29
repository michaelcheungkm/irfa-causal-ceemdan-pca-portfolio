"""
Causal walk-forward reproduction of Introduction2.

Fixes look-ahead in CEEMDAN / Savitzky-Golay / PCA, executes at next-day Open,
benchmarks against B&H DJIA, 1/N, EG and ONS, and writes aligned tables.
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy.interpolate import splrep, splev
from scipy.linalg import eigh
from scipy.optimize import linprog
from scipy.signal import savgol_filter

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
PYEMD_PATH = Path(r"C:\Users\user\SynologyDrive\Simon\Stock Selections\PyEMD-master")
PRED_DIR = Path(r"C:\Users\user\SynologyDrive\Simon\Stock Selections")
OUT = ROOT / "revised_outputs"
FIG = OUT / "figures"
TAB = OUT / "tables"
CKPT = OUT / "checkpoints"

TICKERS = [
    "AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS",
    "GS", "HD", "HON", "IBM", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK",
    "MSFT", "NKE", "NVDA", "PG", "SHW", "TRV", "UNH", "V", "VZ", "WMT",
]
VARIANT_MAP = {
    "A": "predicted",
    "B": "predicted_smooth",
    "C": "predicted_lead",
    "D": "predicted_ultrasmooth",
    "E": "bs_predicted",
}
VARIANT_LABELS = {
    "A": "A  PCA-Taylor",
    "B": "B  SG + PCA-Taylor",
    "C": "C  Phase-lead + momentum",
    "D": "D  Ultrasmooth (Kalman)",
    "E": "E  B-spline",
}
TEST_START = "2024-02-26"
TEST_END = "2026-02-26"
LOOKBACK = 400
CEEMDAN_TRIALS = 10
MAX_IMF = 8
SG_WINDOW = 15
SG_POLY = 3
PCA_WINDOW = 50
LEAD_H = 1.7
S0 = 1_000_000.0
GAMMA = 0.001
MAX_WEIGHT = 0.15
N_JOBS = max(1, min(8, os.cpu_count() or 4))


def _ensure_dirs() -> None:
    for p in (OUT, FIG, TAB, CKPT):
        p.mkdir(parents=True, exist_ok=True)


def _import_ceemdan():
    if str(PYEMD_PATH) not in sys.path:
        sys.path.insert(0, str(PYEMD_PATH))
    from PyEMD import CEEMDAN  # type: ignore

    return CEEMDAN


def ceemdan_matrix(close: np.ndarray, trials: int = CEEMDAN_TRIALS, seed: int = 0) -> np.ndarray:
    CEEMDAN = _import_ceemdan()
    x = np.asarray(close, dtype=float)
    ce = CEEMDAN(trials=trials, max_imf=MAX_IMF)
    ce.noise_seed(int(seed) % (2**31 - 1))
    imfs = np.atleast_2d(ce(x))
    n_imf = imfs.shape[0]
    if n_imf < MAX_IMF:
        pad = np.zeros((MAX_IMF - n_imf, len(x)))
        imfs = np.vstack([imfs, pad])
    residual = x - imfs.sum(axis=0)
    return np.vstack([imfs[:MAX_IMF], residual[None, :]]).T


def savgol_hist(x: np.ndarray, window: int = SG_WINDOW, poly: int = SG_POLY) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    w = window if window % 2 == 1 else window + 1
    if len(x) < w:
        w = len(x) if len(x) % 2 == 1 else max(poly + 2, len(x) - (1 - len(x) % 2))
    if w <= poly or w < 5:
        return x.copy()
    return savgol_filter(x, w, min(poly, w - 1), mode="interp")


class Kalman1D:
    def __init__(self, R: float = 0.05, Q: float = 0.01, x0: float = 0.0):
        self.R = R
        self.Q = Q
        self.x = float(x0)
        self.P = 1.0

    def update(self, z: float) -> float:
        p_pred = self.P + self.Q
        k = p_pred / (p_pred + self.R)
        self.x = self.x + k * (z - self.x)
        self.P = (1.0 - k) * p_pred
        return self.x


def trailing_pca(F: np.ndarray, window: int = PCA_WINDOW):
    end = len(F)
    start = max(0, end - window)
    local = F[start:end]
    mu = local.mean(axis=0)
    centered = local - mu
    dim = centered.shape[1]
    sigma = np.cov(centered.T) + 1e-8 * np.eye(dim)
    evals, evecs = eigh(sigma)
    V = evecs[:, ::-1]
    return mu, V


def taylor_from_scores(g_hist: np.ndarray, h: float) -> np.ndarray:
    g_t = g_hist[-1]
    if len(g_hist) >= 3:
        g_tm1, g_tm2 = g_hist[-2], g_hist[-3]
        g_a = 1.5 * (g_t - g_tm1) - 0.5 * (g_tm1 - g_tm2)
        g_aa = g_t - 2.0 * g_tm1 + g_tm2
    elif len(g_hist) >= 2:
        g_a = g_t - g_hist[-2]
        g_aa = np.zeros_like(g_t)
    else:
        g_a = np.zeros_like(g_t)
        g_aa = np.zeros_like(g_t)
    return g_t + g_a * h + 0.5 * g_aa * (h**2)


def forecast_variants(close: np.ndarray, seed: int) -> dict[str, float]:
    last = float(close[-1])
    if len(close) < 80:
        return {k: last for k in VARIANT_MAP}
    try:
        F_raw = ceemdan_matrix(close, seed=seed)
    except Exception:
        return {k: last for k in VARIANT_MAP}

    F_sg = np.column_stack([savgol_hist(F_raw[:, j]) for j in range(F_raw.shape[1])])
    t_idx = len(F_raw) - 1

    def predict_from(F: np.ndarray, h: float, momentum: bool, smooth_pc: bool) -> float:
        mu, V = trailing_pca(F, PCA_WINDOW)
        hist_len = min(30, t_idx + 1)
        scores = np.array([V.T @ (F[t_idx - i] - mu) for i in range(hist_len - 1, -1, -1)])
        if smooth_pc and len(scores) >= 7:
            scores = np.column_stack([savgol_hist(scores[:, j], 7, 2) for j in range(scores.shape[1])])
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
        n_comp = min(6, F_sg.shape[1])
        from sklearn.decomposition import PCA

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
        e = float(np.sum(recon) + residual[-1] + trend_inc)
        e = float(np.clip(e, last * 0.9, last * 1.1))
    except Exception:
        e = last

    return {"A": float(a), "B": float(b), "C": float(c), "D": float(d), "E": float(e)}


def _forecast_ticker_worker(payload: dict) -> tuple[str, pd.DataFrame]:
    ticker = payload["ticker"]
    close = np.asarray(payload["close"], dtype=float)
    dates = pd.to_datetime(payload["dates"])
    decision_pos = payload["decision_pos"]
    rows = []
    for k, pos in enumerate(decision_pos):
        start = max(0, pos - LOOKBACK + 1)
        hist = close[start : pos + 1]
        seed = (abs(hash(ticker)) + pos) % (2**31 - 1)
        preds = forecast_variants(hist, seed=seed)
        rows.append(
            {
                "decision_date": dates[pos].strftime("%Y-%m-%d"),
                "target_date": dates[pos + 1].strftime("%Y-%m-%d") if pos + 1 < len(dates) else None,
                **preds,
            }
        )
        if (k + 1) % 20 == 0:
            print(f"  {ticker}: {k + 1}/{len(decision_pos)}", flush=True)
    return ticker, pd.DataFrame(rows)


def load_market_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    cache = OUT / "market_ohlc.pkl"
    if cache.exists():
        obj = pd.read_pickle(cache)
        close0 = obj["close"]
        if close0.index.max() >= pd.Timestamp(TEST_END):
            return obj["close"], obj["open"], obj["djia"]

    import yfinance as yf

    print("Downloading DJIA constituents and ^DJI ...")
    raw = yf.download(
        TICKERS + ["^DJI"],
        start="2017-01-01",
        end="2026-09-04",
        auto_adjust=False,
        progress=True,
        threads=True,
    )
    px_close = raw["Close"]
    px_open = raw["Open"]
    if isinstance(px_close.columns, pd.MultiIndex):
        px_close = px_close.copy()
        px_open = px_open.copy()
    close = px_close[TICKERS].dropna(how="any")
    open_ = px_open[TICKERS].reindex(close.index)
    if "^DJI" in px_close.columns:
        djia = px_close["^DJI"].reindex(close.index)
    else:
        dji_raw = yf.download("^DJI", start="2017-01-01", end="2026-09-04", auto_adjust=False, progress=False)
        djia = dji_raw["Close"].reindex(close.index)
        if isinstance(djia, pd.DataFrame):
            djia = djia.iloc[:, 0]
    djia = djia.astype(float)
    open_ = open_.astype(float)
    close = close.astype(float)
    # fill rare missing opens with close
    open_ = open_.fillna(close)
    pd.to_pickle({"close": close, "open": open_, "djia": djia}, cache)
    return close, open_, djia


def load_leaky_predictions(index: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    frames = {k: [] for k in VARIANT_MAP}
    actuals = []
    for ticker in TICKERS:
        path = PRED_DIR / f"pred_{ticker}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
        actuals.append(df["actual"].rename(ticker))
        for letter, col in VARIANT_MAP.items():
            frames[letter].append(df[col].rename(ticker))
    out = {}
    for letter, parts in frames.items():
        pred = pd.concat(parts, axis=1).sort_index()
        pred = pred.loc[(pred.index >= "2025-07-25") & (pred.index <= TEST_END), TICKERS]
        pred = pred[~pred.index.duplicated()].copy()
        out[letter] = pred
    return out


def optimize_shares(
    prev_nc: np.ndarray,
    price: np.ndarray,
    pred: np.ndarray,
    wealth: float,
    gamma: float | None = None,
    max_weight: float | None = None,
) -> np.ndarray:
    """Share LP from A strategy for portfolio allocations.docx.

    max NC' pred  s.t.  price' NC + gamma price' u <= wealth,
    |NC - prev_nc| <= u,  price_i NC_i <= max_weight * wealth,  NC,u >= 0.
    """
    cost_rate = GAMMA if gamma is None else float(gamma)
    cap = MAX_WEIGHT if max_weight is None else float(max_weight)
    price = np.asarray(price, dtype=float).reshape(-1)
    pred = np.asarray(pred, dtype=float).reshape(-1)
    prev_nc = np.asarray(prev_nc, dtype=float).reshape(-1)
    if wealth <= 0:
        return np.zeros_like(price)
    n = len(price)
    price = np.maximum(price, 0.01)
    c = np.concatenate([-pred, np.zeros(n)])
    a_ineq = [np.concatenate([price, cost_rate * price])]
    b_ineq = [wealth]
    for i in range(n):
        r1 = np.zeros(2 * n)
        r1[i] = 1.0
        r1[n + i] = -1.0
        a_ineq.append(r1)
        b_ineq.append(prev_nc[i])
        r2 = np.zeros(2 * n)
        r2[i] = -1.0
        r2[n + i] = -1.0
        a_ineq.append(r2)
        b_ineq.append(-prev_nc[i])
        if 0 < cap < 1.0 - 1e-12:
            r3 = np.zeros(2 * n)
            r3[i] = price[i]
            a_ineq.append(r3)
            b_ineq.append(cap * wealth)
    bounds = [(0, None)] * (2 * n)
    res = linprog(c, A_ub=np.array(a_ineq), b_ub=np.array(b_ineq), bounds=bounds, method="highs")
    if res.success:
        return np.maximum(res.x[:n], 0.0)
    return prev_nc.copy()


def simulate_close_to_close(close: pd.DataFrame, pred: pd.DataFrame) -> dict:
    """Original protocol: decide and mark on closes. pred is indexed by target date."""
    dates = pred.index
    plan_close = close.reindex(dates).shift(1)
    first_plan = dates[0] - pd.tseries.offsets.BDay(1)
    # use previous available close
    aligned_plan = close.shift(1).reindex(dates)
    mark = close.reindex(dates)
    ncs, cashes, values, turns = [], [], [S0], []
    prev_nc = np.zeros(close.shape[1])
    wealth = S0
    cash = S0
    for dt in dates:
        p_plan = aligned_plan.loc[dt].values
        p_mark = mark.loc[dt].values
        if np.isnan(p_plan).any():
            p_plan = p_mark
        nc = optimize_shares(prev_nc, p_plan, pred.loc[dt].values, wealth)
        traded = np.abs(nc - prev_nc)
        cost = GAMMA * float(traded @ p_plan)
        cash = wealth - cost - float(nc @ p_plan)
        wealth_next = float(nc @ p_mark) + cash
        ncs.append(nc)
        cashes.append(cash)
        turns.append(float(traded @ p_plan) / max(wealth, 1e-9))
        values.append(wealth_next)
        wealth = wealth_next
        prev_nc = nc
    idx = [aligned_plan.index[0] if False else (dates[0] - pd.Timedelta(days=1))]
    # value dates: start + each mark date
    value_index = [dates[0] - pd.tseries.offsets.BDay(0)]
    # keep simple: start dummy + mark dates
    value_index = pd.Index([dates[0] - pd.Timedelta(days=1)]).append(dates)
    return {
        "NCs": ncs,
        "cashes": cashes,
        "S_values": np.array(values),
        "turnover_path": np.array(turns),
        "index": dates,
        "value_index": value_index,
    }


def simulate_next_open(
    close: pd.DataFrame,
    open_: pd.DataFrame,
    pred: pd.DataFrame,
    decision_dates: pd.DatetimeIndex,
) -> dict:
    """
    Decision after close of decision_dates[t] using pred for next close.
    Execute at open of the next trading day; mark at that day's close.
    """
    ncs, cashes, values, turns = [], [], [S0], []
    prev_nc = np.zeros(close.shape[1])
    cash = S0
    exec_dates = []
    for dt in decision_dates:
        loc = close.index.get_loc(dt)
        if loc + 1 >= len(close.index):
            break
        nxt = close.index[loc + 1]
        p_open = open_.loc[nxt].values
        p_close_next = close.loc[nxt].values
        s_open = float(prev_nc @ p_open) + cash
        if dt in pred.index:
            hat = pred.loc[dt].values
        else:
            # pred indexed by target date
            hat = pred.loc[nxt].values if nxt in pred.index else close.loc[dt].values
        nc = optimize_shares(prev_nc, p_open, hat, s_open)
        traded = np.abs(nc - prev_nc)
        cost = GAMMA * float(traded @ p_open)
        if float(nc @ p_open) + cost > s_open > 0:
            scale = max(0.0, (s_open * 0.999) / (float(nc @ p_open) + cost))
            nc = nc * scale
            traded = np.abs(nc - prev_nc)
            cost = GAMMA * float(traded @ p_open)
        cash = s_open - cost - float(nc @ p_open)
        wealth = float(nc @ p_close_next) + cash
        ncs.append(nc)
        cashes.append(cash)
        turns.append(float(traded @ p_open) / max(s_open, 1e-9))
        values.append(wealth)
        exec_dates.append(nxt)
        prev_nc = nc
    return {
        "NCs": ncs,
        "cashes": cashes,
        "S_values": np.array(values),
        "turnover_path": np.array(turns),
        "index": pd.DatetimeIndex(exec_dates),
        "value_index": pd.DatetimeIndex([decision_dates[0]] + exec_dates),
    }


def weights_to_open_path(
    close: pd.DataFrame,
    open_: pd.DataFrame,
    weight_fn,
    decision_dates: pd.DatetimeIndex,
) -> dict:
    n = close.shape[1]
    prev_w = np.ones(n) / n
    prev_nc = np.zeros(n)
    cash = S0
    values = [S0]
    turns = []
    exec_dates = []
    ncs = []
    for dt in decision_dates:
        loc = close.index.get_loc(dt)
        if loc + 1 >= len(close.index):
            break
        nxt = close.index[loc + 1]
        p_open = open_.loc[nxt].values
        s_open = float(prev_nc @ p_open) + cash
        w = np.asarray(weight_fn(dt, prev_w), dtype=float)
        w = np.maximum(w, 0.0)
        if w.sum() <= 0:
            w = np.ones(n) / n
        w = w / w.sum()
        target = (w * s_open) / np.maximum(p_open, 0.01)
        traded = np.abs(target - prev_nc)
        cost = GAMMA * float(traded @ p_open)
        if float(target @ p_open) + cost > s_open > 0:
            scale = max(0.0, (s_open * 0.999) / (float(target @ p_open) + cost))
            target = target * scale
            traded = np.abs(target - prev_nc)
            cost = GAMMA * float(traded @ p_open)
        cash = s_open - cost - float(target @ p_open)
        wealth = float(target @ close.loc[nxt].values) + cash
        values.append(wealth)
        turns.append(float(traded @ p_open) / max(s_open, 1e-9))
        ncs.append(target)
        exec_dates.append(nxt)
        prev_nc = target
        prev_w = w
    return {
        "NCs": ncs,
        "cashes": [],
        "S_values": np.array(values),
        "turnover_path": np.array(turns),
        "index": pd.DatetimeIndex(exec_dates),
        "value_index": pd.DatetimeIndex([decision_dates[0]] + exec_dates),
    }


def simplex_project(v: np.ndarray) -> np.ndarray:
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    cond = u * np.arange(1, n + 1) > (cssv - 1)
    rho = int(np.nonzero(cond)[0][-1])
    theta = (cssv[rho] - 1.0) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def run_baselines(close: pd.DataFrame, open_: pd.DataFrame, djia: pd.Series, decision_dates: pd.DatetimeIndex) -> dict:
    n = close.shape[1]
    first = decision_dates[0]
    last_exec = close.index[close.index.get_loc(decision_dates[-1]) + 1]
    # Buy-and-hold DJIA: buy at first next open, hold
    first_next = close.index[close.index.get_loc(first) + 1]
    djia_open0 = float(djia.loc[first_next]) if first_next in djia.index else float(djia.loc[first])
    bh_idx = close.loc[first_next:last_exec].index
    bh_vals = S0 * djia.reindex(bh_idx).ffill().values / max(djia_open0, 1e-9)
    bh = {
        "NCs": [],
        "S_values": np.concatenate([[S0], bh_vals]),
        "turnover_path": np.array([1.0] + [0.0] * (len(bh_vals) - 1)),
        "index": bh_idx,
        "value_index": pd.DatetimeIndex([first] + list(bh_idx)),
    }

    def one_n(dt, prev_w):
        return np.ones(n) / n

    rel_close = close / close.shift(1)
    eg_state = {"b": np.ones(n) / n, "eta": 0.05}

    def eg_fn(dt, prev_w):
        if dt in rel_close.index and not rel_close.loc[dt].isna().any():
            x = rel_close.loc[dt].values
            b = eg_state["b"] * np.exp(eg_state["eta"] * x)
            b = np.maximum(b, 0)
            eg_state["b"] = b / b.sum()
        return eg_state["b"]

    ons_state = {"b": np.ones(n) / n, "A": np.eye(n)}

    def ons_fn(dt, prev_w):
        if dt in rel_close.index and not rel_close.loc[dt].isna().any():
            x = rel_close.loc[dt].values
            denom = max(float(ons_state["b"] @ x), 1e-12)
            grad = -x / denom
            ons_state["A"] = ons_state["A"] + np.outer(grad, grad)
            try:
                ainv = np.linalg.inv(ons_state["A"])
            except np.linalg.LinAlgError:
                ainv = np.linalg.pinv(ons_state["A"])
            b = simplex_project(ons_state["b"] - ainv @ grad)
            if b.sum() <= 0:
                b = np.ones(n) / n
            ons_state["b"] = b / b.sum()
        return ons_state["b"]

    return {
        "BH_DJIA": bh,
        "EW_1N": weights_to_open_path(close, open_, one_n, decision_dates),
        "EG": weights_to_open_path(close, open_, eg_fn, decision_dates),
        "ONS": weights_to_open_path(close, open_, ons_fn, decision_dates),
    }


def performance_stats(result: dict, trading_days: int | None = None, periods_per_year: float = 252.0) -> dict:
    v = np.asarray(result["S_values"], dtype=float)
    rets = np.diff(v) / np.maximum(v[:-1], 1e-12)
    rets = rets[np.isfinite(rets)]
    total = float(v[-1] / v[0] - 1.0)
    peak = np.maximum.accumulate(v)
    dd = v / np.maximum(peak, 1e-12) - 1.0
    mdd = float(dd.min())
    n = max(len(rets), 1)
    ppy = float(trading_days or periods_per_year)
    mu = float(np.mean(rets)) if n else 0.0
    sig = float(np.std(rets, ddof=1)) if n > 1 else 0.0
    sharpe = (mu / sig * np.sqrt(ppy)) if sig > 0 else 0.0
    downside = rets[rets < 0]
    dsig = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0
    sortino = (mu / dsig * np.sqrt(ppy)) if dsig > 0 else 0.0
    years = n / ppy if n else 1.0
    ann = (1.0 + total) ** (1.0 / max(years, 1e-9)) - 1.0 if years > 0 else total
    calmar = (ann / abs(mdd)) if abs(mdd) > 1e-12 else np.nan
    turns = np.asarray(result.get("turnover_path", []), dtype=float)
    turnover = float(turns.sum() * 100.0)
    ncs = result.get("NCs") or []
    avg_held = float(np.mean([(np.asarray(nc) > 1e-8).sum() for nc in ncs])) if ncs else 0.0
    return {
        "Final_Value": float(v[-1]),
        "Total_Return_%": total * 100.0,
        "Max_Drawdown_%": mdd * 100.0,
        "Calmar_Ratio": float(calmar) if np.isfinite(calmar) else np.nan,
        "Sharpe_Ratio": float(sharpe),
        "Sortino_Ratio": float(sortino),
        "Turnover_%": turnover,
        "Avg_Stocks_Held": avg_held,
    }


def aligned_value_table(result: dict, close: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    ncs = np.asarray(result["NCs"], dtype=float)
    prices = close.reindex(result["index"]).values
    avg_price = np.nanmean(prices, axis=0)
    total_value = (ncs * avg_price).sum(axis=0)
    periods = (ncs > 1e-8).sum(axis=0)
    df = pd.DataFrame(
        {
            "Stock": tickers,
            "Total_Shares": ncs.sum(axis=0),
            "Avg_Price": avg_price,
            "Total_Value": total_value,
            "Avg_Value_Per_Period": ncs.mean(axis=0) * avg_price,
            "Periods_Held": periods,
            "Max_Shares": ncs.max(axis=0),
        }
    )
    wealth = np.asarray(result["S_values"][1:], dtype=float)
    mark_w = np.divide(
        ncs * prices,
        np.maximum(wealth.reshape(-1, 1), 1e-12),
        out=np.zeros_like(ncs),
        where=wealth.reshape(-1, 1) > 0,
    )
    held = ncs > 1e-8
    mean_w = np.array(
        [float(mark_w[held[:, i], i].mean()) if held[:, i].any() else 0.0 for i in range(len(tickers))]
    )
    df = df.sort_values("Total_Value", ascending=False).reset_index(drop=True)
    df["Value_Percent"] = 100.0 * df["Total_Value"] / max(df["Total_Value"].sum(), 1e-12)
    wmap = {tickers[i]: 100.0 * mean_w[i] for i in range(len(tickers))}
    df["Mean_Weight_%"] = df["Stock"].map(wmap)
    return df


def aligned_streak_table(result: dict, close: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    ncs = np.asarray(result["NCs"], dtype=float)
    prices = close.reindex(result["index"]).values
    avg_price = np.nanmean(prices, axis=0)
    total_value = {tickers[i]: float((ncs[:, i] * avg_price[i]).sum()) for i in range(len(tickers))}
    rows = []
    for i, name in enumerate(tickers):
        h = ncs[:, i]
        held = h > 1e-8
        if not held.any():
            continue
        max_streak = cur = 0
        streaks = 0
        prev = False
        for flag in held:
            if flag:
                cur += 1
                max_streak = max(max_streak, cur)
                if not prev:
                    streaks += 1
            else:
                cur = 0
            prev = bool(flag)
        periods = int(held.sum())
        rows.append(
            {
                "Stock": name,
                "Max_Streak": int(max_streak),
                "Avg_Streak": periods / streaks if streaks else 0.0,
                "Total_Value": total_value[name],
                "Total_Periods_Held": periods,
                "Number_of_Streaks": int(streaks),
            }
        )
    return pd.DataFrame(rows).sort_values("Total_Value", ascending=False).reset_index(drop=True)


def mse_table(pred: dict[str, pd.DataFrame], close: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker in TICKERS:
        row = {"Ticker": ticker}
        actual = close[ticker]
        for letter, df in pred.items():
            common = df.index.intersection(actual.index)
            err = (df.loc[common, ticker] - actual.loc[common]) ** 2
            row[letter] = float(err.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def style_axes(ax, ylabel: str, title: str) -> None:
    ax.set_title(title, fontsize=12, pad=8)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(frameon=False, fontsize=8)


def plot_equity(series: dict[str, pd.Series], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    colors = {
        "A": "#4C78A8",
        "B": "#F58518",
        "C": "#54A24B",
        "D": "#B279A2",
        "E": "#E45756",
        "BH_DJIA": "#8C6D31",
        "EW_1N": "#72B7B2",
        "EG": "#FF9DA6",
        "ONS": "#9D755D",
        "DJIA": "#8C6D31",
        "ABE weekly top-3": "#4C78A8",
        "ABE monthly top-5": "#4C78A8",
        "B weekly top-5": "#F58518",
        "B biweekly top-5": "#F58518",
        "B monthly top-5": "#54A24B",
    }
    for name, s in series.items():
        ax.plot(s.index, s.values, label=name, lw=1.8, color=colors.get(name, None))
    style_axes(ax, "Portfolio value (USD)", title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_drawdown(series: dict[str, pd.Series], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    for name, s in series.items():
        peak = s.cummax()
        ax.plot(s.index, 100.0 * (s / peak - 1.0), label=name, lw=1.5)
    style_axes(ax, "Drawdown (%)", title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_mse_bars(mse: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.8))
    x = np.arange(len(mse))
    width = 0.16
    for i, letter in enumerate(["A", "B", "C", "D", "E"]):
        ax.bar(x + (i - 2) * width, mse[letter].values, width, label=letter)
    ax.set_xticks(x)
    ax.set_xticklabels(mse["Ticker"], rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("RMSE (price points)")
    ax.set_title("One-step-ahead close-price RMSE by ticker and predictor")
    ax.legend(frameon=False, ncol=5)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _assemble_causal(results: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out = {}
    for letter in VARIANT_MAP:
        wide = pd.DataFrame(
            {t: results[t].set_index("decision_date")[letter] for t in TICKERS}
        )
        wide.index = pd.to_datetime(wide.index)
        out[letter] = wide.sort_index()
    return out


def run_walkforward(close: pd.DataFrame, decision_dates: pd.DatetimeIndex) -> dict[str, pd.DataFrame]:
    """Causal one-step forecasts; appends any missing decision dates to the CSV cache."""
    _ensure_dirs()
    results: dict[str, pd.DataFrame] = {}
    missing_by_ticker: dict[str, list[int]] = {}
    need = pd.DatetimeIndex(pd.to_datetime(decision_dates)).normalize()

    for ticker in TICKERS:
        path = CKPT / f"forecast_{ticker}.csv"
        have = pd.DataFrame(columns=["decision_date", "target_date", "A", "B", "C", "D", "E"])
        if path.exists():
            have = pd.read_csv(path, parse_dates=["decision_date"])
            have["decision_date"] = pd.to_datetime(have["decision_date"]).dt.normalize()
            have = have.drop_duplicates("decision_date", keep="last")
        results[ticker] = have
        have_set = set(have["decision_date"]) if len(have) else set()
        miss = [dt for dt in need if dt not in have_set]
        if miss:
            missing_by_ticker[ticker] = [int(close.index.get_loc(dt)) for dt in miss]

    if not missing_by_ticker:
        print("Causal forecasts already cover", len(need), "decision days")
        out = _assemble_causal(results)
        pd.to_pickle(out, CKPT / "causal_forecasts.pkl")
        return out

    n_miss = max(len(v) for v in missing_by_ticker.values())
    print(
        f"Walk-forward CEEMDAN: {len(missing_by_ticker)} names need up to {n_miss} "
        f"new days ({N_JOBS} workers). Existing cache kept."
    )
    payloads = []
    for ticker, dpos in missing_by_ticker.items():
        payloads.append(
            {
                "ticker": ticker,
                "close": close[ticker].values,
                "dates": close.index.astype(str).tolist(),
                "decision_pos": dpos,
            }
        )
    if N_JOBS == 1:
        done = [_forecast_ticker_worker(p) for p in payloads]
    else:
        done = []
        with ProcessPoolExecutor(max_workers=N_JOBS) as ex:
            futs = {ex.submit(_forecast_ticker_worker, p): p["ticker"] for p in payloads}
            for fut in as_completed(futs):
                ticker, df = fut.result()
                done.append((ticker, df))
                print(f"finished {ticker} (+{len(df)})", flush=True)

    for ticker, add in done:
        old = results[ticker]
        add = add.copy()
        add["decision_date"] = pd.to_datetime(add["decision_date"]).dt.normalize()
        merged = pd.concat([old, add], ignore_index=True)
        merged["decision_date"] = pd.to_datetime(merged["decision_date"]).dt.normalize()
        merged = merged.drop_duplicates("decision_date", keep="last").sort_values("decision_date")
        merged.to_csv(CKPT / f"forecast_{ticker}.csv", index=False)
        results[ticker] = merged

    out = _assemble_causal(results)
    pd.to_pickle(out, CKPT / "causal_forecasts.pkl")
    return out


def series_from_result(result: dict) -> pd.Series:
    return pd.Series(result["S_values"], index=pd.to_datetime(result["value_index"]))


def save_table(df: pd.DataFrame, name: str) -> Path:
    path = TAB / f"{name}.csv"
    df.to_csv(path, index=False)
    return path


def main() -> None:
    _ensure_dirs()
    close, open_, djia = load_market_data()
    close = close.loc[:TEST_END, TICKERS]
    open_ = open_.reindex(close.index)[TICKERS].fillna(close)
    djia = djia.reindex(close.index).ffill()

    test_mask = (close.index >= TEST_START) & (close.index <= TEST_END)
    test_index = close.index[test_mask]
    # last decision day must have a next session
    decision_dates = test_index[:-1]
    print(f"Market panel {close.index.min().date()} to {close.index.max().date()}, decisions {len(decision_dates)}")

    leaky_target = load_leaky_predictions(close.index)
    # original close-to-close on leaked forecasts
    leaky_cc = {}
    leaky_open = {}
    for letter, pred in leaky_target.items():
        print(f"Original-protocol replay {letter}")
        leaky_cc[letter] = simulate_close_to_close(close, pred)
        # map target-date preds onto decision dates (previous session)
        pred_on_decision = pred.copy()
        pred_on_decision.index = [close.index[close.index.get_loc(d) - 1] if d in close.index else d for d in pred.index]
        pred_on_decision = pred_on_decision[~pred_on_decision.index.duplicated()]
        leaky_open[letter] = simulate_next_open(close, open_, pred_on_decision, decision_dates.intersection(pred_on_decision.index))

    causal = run_walkforward(close, decision_dates)
    causal_open = {}
    for letter, pred in causal.items():
        print(f"Causal open-execution {letter}")
        causal_open[letter] = simulate_next_open(close, open_, pred, decision_dates.intersection(pred.index))

    baselines = run_baselines(close, open_, djia, decision_dates)

    # Metrics
    rows = []
    for letter, res in leaky_cc.items():
        row = performance_stats(res)
        row["Strategy"] = f"{letter} leaky close-to-close"
        row["Protocol"] = "leaky_cc"
        rows.append(row)
    for letter, res in leaky_open.items():
        row = performance_stats(res)
        row["Strategy"] = f"{letter} leaky next-open"
        row["Protocol"] = "leaky_open"
        rows.append(row)
    for letter, res in causal_open.items():
        row = performance_stats(res)
        row["Strategy"] = f"{letter} causal next-open"
        row["Protocol"] = "causal_open"
        rows.append(row)
    for name, res in baselines.items():
        row = performance_stats(res)
        row["Strategy"] = name
        row["Protocol"] = "baseline_open"
        rows.append(row)
    metrics = pd.DataFrame(rows)
    save_table(metrics, "performance_all")

    table2 = metrics[metrics["Protocol"].isin(["causal_open", "baseline_open"])].copy()
    save_table(table2, "table2_walkforward_baselines")

    table2_orig = metrics[metrics["Protocol"] == "leaky_cc"].copy()
    save_table(table2_orig, "table2_original_leaky")

    mse_leaky = mse_table(leaky_target, close)
    mse_causal = mse_table(
        {k: v.shift(-1, freq=None) for k, v in causal.items()}
        if False
        else {k: v.set_axis(pd.to_datetime(v.index)) for k, v in causal.items()},
        close.shift(-1),
    )
    # causal preds are on decision date for next close
    mse_c_rows = []
    for ticker in TICKERS:
        row = {"Ticker": ticker}
        for letter, df in causal.items():
            targets, actuals = [], []
            for dt in df.index:
                loc = close.index.get_loc(dt)
                if loc + 1 < len(close.index):
                    targets.append(df.loc[dt, ticker])
                    actuals.append(close.iloc[loc + 1][ticker])
            row[letter] = float(np.mean((np.array(targets) - np.array(actuals)) ** 2))
        mse_c_rows.append(row)
    mse_causal = pd.DataFrame(mse_c_rows)
    save_table(mse_leaky, "table1_mse_leaky")
    save_table(mse_causal, "table1_mse_causal")

    best = max(causal_open, key=lambda k: causal_open[k]["S_values"][-1])
    t3 = aligned_streak_table(causal_open[best], close, TICKERS)
    t4 = aligned_value_table(causal_open[best], close, TICKERS)
    save_table(t3, "table3_streaks_aligned")
    save_table(t4, "table4_value_aligned")

    # Corrected leaky tables (original D, aligned by ticker)
    t3_leaky = aligned_streak_table(leaky_cc["D"], close, TICKERS)
    t4_leaky = aligned_value_table(leaky_cc["D"], close, TICKERS)
    save_table(t3_leaky, "table3_leaky_aligned")
    save_table(t4_leaky, "table4_leaky_aligned")

    # Figures
    def pack(res_map, labels=None):
        out = {}
        for k, res in res_map.items():
            name = labels.get(k, k) if labels else k
            out[name] = series_from_result(res)
        return out

    plot_equity(
        pack(leaky_cc, {k: f"{k} leaky" for k in leaky_cc})
        | {"DJIA": pd.Series(S0 * djia.reindex(leaky_cc["D"]["index"]).values / float(djia.reindex(leaky_cc["D"]["index"]).iloc[0]), index=leaky_cc["D"]["index"])},
        FIG / "fig1_leaky_close_to_close.png",
        "Original close-to-close protocol (full-sample CEEMDAN / SG)",
    )
    causal_series = pack(causal_open, VARIANT_LABELS)
    for name, res in baselines.items():
        causal_series[name] = series_from_result(res)
    plot_equity(causal_series, FIG / "fig2_walkforward_open.png", "Walk-forward forecasts, next-day Open execution")
    plot_drawdown(
        {k: causal_series[k] for k in list(causal_series)[:8]},
        FIG / "fig3_walkforward_drawdown.png",
        "Walk-forward drawdowns (next-day Open execution)",
    )
    plot_mse_bars(mse_causal, FIG / "fig4_mse_causal.png")
    plot_mse_bars(mse_leaky, FIG / "fig5_mse_leaky.png")

    # AAPL path example
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    aapl = close.loc[decision_dates[0] : TEST_END, "AAPL"]
    ax.plot(aapl.index, aapl.values, color="#333333", lw=1.6, label="AAPL close")
    if "AAPL" in causal["D"].columns:
        dec = causal["D"].index.intersection(decision_dates)
        tgt_dates = [close.index[close.index.get_loc(d) + 1] for d in dec if close.index.get_loc(d) + 1 < len(close.index)]
        ax.plot(tgt_dates, causal["D"].loc[dec[: len(tgt_dates)], "AAPL"].values, lw=1.3, label="D causal forecast")
    if "AAPL" in leaky_target["D"].columns:
        ax.plot(leaky_target["D"].index, leaky_target["D"]["AAPL"], lw=1.2, ls="--", label="D leaked forecast")
    style_axes(ax, "Price (USD)", "AAPL close versus leaked and causal D forecasts")
    fig.tight_layout()
    fig.savefig(FIG / "fig6_aapl_forecasts.png", dpi=160)
    plt.close(fig)

    meta = {
        "best_causal": best,
        "n_decision_days": int(len(decision_dates)),
        "test_start": str(decision_dates[0].date()),
        "test_end": TEST_END,
        "lookback": LOOKBACK,
        "ceemdan_trials": CEEMDAN_TRIALS,
        "gamma": GAMMA,
        "S0": S0,
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False))
    print("Wrote outputs to", OUT)


if __name__ == "__main__":
    main()
