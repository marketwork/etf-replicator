"""
ETF replication optimizer.

Two selection methods:
  lasso  – L1 regression to find a sparse weight vector, then QP refine
  greedy – Iterative correlation-with-residuals selection, then QP refine
  hybrid – LASSO pre-filters 4× candidates, greedy selects final n

All methods end with a constrained QP:
  min_w  ||S·w - e||²   s.t.  sum(w)=1,  w≥0
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from typing import Callable, Optional


# ---------- core QP -----------------------------------------------------------

def _qp_weights(etf_r: np.ndarray, stock_r: np.ndarray) -> np.ndarray:
    n = stock_r.shape[1]

    def obj(w):
        return float(np.sum((stock_r @ w - etf_r) ** 2))

    def grad(w):
        return 2.0 * stock_r.T @ (stock_r @ w - etf_r)

    cons = [{"type": "eq",
             "fun": lambda w: w.sum() - 1.0,
             "jac": lambda w: np.ones(n)}]
    bounds = [(0.0, 1.0)] * n

    res = minimize(obj, np.full(n, 1.0 / n), jac=grad, method="SLSQP",
                   bounds=bounds, constraints=cons,
                   options={"ftol": 1e-10, "maxiter": 2000})
    w = np.maximum(res.x, 0.0)
    s = w.sum()
    return w / s if s > 0 else np.full(n, 1.0 / n)


def _ann_te(etf_r: np.ndarray, port_r: np.ndarray) -> float:
    return float(np.std(port_r - etf_r, ddof=1) * np.sqrt(252))


# ---------- selection methods -------------------------------------------------

def _select_lasso(etf_r: pd.Series, stock_r: pd.DataFrame, n: int) -> list:
    """Binary search on LASSO alpha to isolate ~n non-zero coefficients."""
    y = etf_r.values
    X = stock_r.values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    lo, hi = 1e-7, 5.0
    best_coef = np.zeros(X.shape[1])
    best_diff = n + 1

    for _ in range(64):
        alpha = (lo + hi) / 2.0
        model = Lasso(alpha=alpha, positive=True, max_iter=50_000,
                      tol=1e-6, fit_intercept=False)
        model.fit(Xs, y)
        nnz = int((model.coef_ > 1e-7).sum())
        diff = abs(nnz - n)

        if diff < best_diff:
            best_diff = diff
            best_coef = model.coef_.copy()

        if nnz == n:
            break
        elif nnz > n:
            lo = alpha
        else:
            hi = alpha

    # Guarantee exactly n stocks via top-n by coefficient magnitude
    top_idx = np.argsort(best_coef)[::-1][:n]
    return stock_r.columns[top_idx].tolist()


def _select_greedy(etf_r: pd.Series, stock_r: pd.DataFrame, n: int,
                   preselect: int = 80,
                   progress_cb: Optional[Callable] = None) -> list:
    """
    Greedy forward selection using correlation with current residuals.
    Residuals are updated after each pick using non-negative OLS.
    """
    # Pre-filter to the most correlated stocks to keep inner loop fast
    pool = (stock_r.corrwith(etf_r)
                   .fillna(0)
                   .nlargest(min(preselect, len(stock_r.columns)))
                   .index.tolist())
    pool_r = stock_r[pool]

    selected: list = []
    residuals = etf_r.values.copy()

    for step in range(n):
        if progress_cb:
            progress_cb(step / n)

        remaining = [s for s in pool if s not in selected]
        if not remaining:
            break

        resid_s = pd.Series(residuals, index=etf_r.index)
        best = pool_r[remaining].corrwith(resid_s).fillna(0).idxmax()
        selected.append(best)

        # Update residuals with non-negative OLS on all selected so far
        X = pool_r[selected].values
        w_ols, *_ = np.linalg.lstsq(X, etf_r.values, rcond=None)
        w_ols = np.maximum(w_ols, 0.0)
        s = w_ols.sum()
        if s > 0:
            w_ols /= s
        residuals = etf_r.values - X @ w_ols

    if progress_cb:
        progress_cb(1.0)

    return selected


# ---------- main API ----------------------------------------------------------

def replicate(
    etf_r: pd.Series,
    stock_r: pd.DataFrame,
    n_stocks: int = 25,
    method: str = "lasso",
    train_frac: float = 0.7,
    progress_cb: Optional[Callable] = None,
) -> dict:
    """
    Select `n_stocks` from `stock_r` and compute optimal replication weights.

    Parameters
    ----------
    etf_r      : daily return series for the target ETF
    stock_r    : daily return frame for the candidate universe
    n_stocks   : desired number of replication stocks (20-30)
    method     : 'lasso' | 'greedy' | 'hybrid'
    train_frac : fraction of data used for optimisation (rest is OOS evaluation)
    progress_cb: optional callable(float 0→1) for progress updates

    Returns
    -------
    dict with keys: tickers, weights, etf_returns, port_returns, etf_cum,
    port_cum, rolling_te, tracking_error, correlation, r_squared,
    oos_tracking_error, oos_correlation, n_stocks
    """
    # Align
    idx = etf_r.index.intersection(stock_r.index)
    e_all = etf_r.loc[idx]
    s_all = stock_r.loc[idx].dropna(thresh=int(len(idx) * 0.85), axis=1)
    s_all = s_all.ffill().fillna(0.0)

    n = min(n_stocks, len(s_all.columns))

    # Train / test split
    split = int(len(idx) * train_frac)
    e_train = e_all.iloc[:split]
    s_train = s_all.iloc[:split]

    # ----- Stock selection (on training data) -----
    if method == "lasso":
        selected = _select_lasso(e_train, s_train, n)
    elif method == "greedy":
        selected = _select_greedy(e_train, s_train, n, progress_cb=progress_cb)
    else:  # hybrid
        candidates = _select_lasso(e_train, s_train, min(n * 4, len(s_train.columns)))
        selected = _select_greedy(e_train, s_train[candidates], n,
                                  preselect=len(candidates),
                                  progress_cb=progress_cb)

    # ----- Weight optimisation (on training data) -----
    w = _qp_weights(e_train.values, s_train[selected].values)

    # Drop near-zero weights
    mask = w > 5e-4
    selected = [t for t, m in zip(selected, mask) if m]
    w = w[mask]
    w /= w.sum()

    # ----- Full-period performance -----
    port_all = pd.Series(s_all[selected].values @ w, index=idx, name="Replica")
    etf_all = e_all.rename("ETF")

    te_is = _ann_te(e_train.values, (s_train[selected].values @ w))
    corr_is = float(np.corrcoef(e_train.values,
                                s_train[selected].values @ w)[0, 1])
    ss_res = float(np.sum((e_train.values - s_train[selected].values @ w) ** 2))
    ss_tot = float(np.sum((e_train.values - e_train.mean()) ** 2))
    r2_is = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Out-of-sample
    e_oos = e_all.iloc[split:]
    p_oos = port_all.iloc[split:]
    if len(e_oos) > 20:
        te_oos = _ann_te(e_oos.values, p_oos.values)
        corr_oos = float(np.corrcoef(e_oos.values, p_oos.values)[0, 1])
    else:
        te_oos = te_is
        corr_oos = corr_is

    return {
        "tickers": selected,
        "weights": w,
        "etf_returns": etf_all,
        "port_returns": port_all,
        "etf_cum": (1 + etf_all).cumprod(),
        "port_cum": (1 + port_all).cumprod(),
        "rolling_te": (port_all - etf_all).rolling(63).std() * np.sqrt(252),
        "tracking_error": te_is,
        "correlation": corr_is,
        "r_squared": r2_is,
        "oos_tracking_error": te_oos,
        "oos_correlation": corr_oos,
        "n_stocks": len(selected),
        "train_cutoff": idx[split - 1],
    }


def evaluate(
    etf_r: pd.Series,
    stock_r: pd.DataFrame,
    weights_dict: dict,
    train_frac: float = 0.70,
) -> dict:
    """
    Evaluate a user-supplied portfolio against an ETF.

    Parameters
    ----------
    etf_r       : daily return series for the target ETF
    stock_r     : daily return frame containing at least the portfolio tickers
    weights_dict: {ticker: weight} — weights will be normalised to sum to 1
    train_frac  : fraction of history labelled "in-sample"

    Returns same key schema as replicate() for drop-in rendering.
    """
    idx = etf_r.index.intersection(stock_r.index)
    e_all = etf_r.loc[idx]
    s_all = stock_r.loc[idx].ffill().fillna(0.0)

    # Keep only tickers present in price data
    tickers = [t for t in weights_dict if t in s_all.columns]
    missing = [t for t in weights_dict if t not in s_all.columns]
    if not tickers:
        raise ValueError(
            f"None of the portfolio tickers were found in price data. "
            f"Missing: {list(weights_dict)[:10]}"
        )

    raw_w = np.array([weights_dict[t] for t in tickers], dtype=float)
    w = raw_w / raw_w.sum()

    port_all = pd.Series(s_all[tickers].values @ w, index=idx, name="Portfolio")
    etf_all  = e_all.rename("ETF")

    split = int(len(idx) * train_frac)
    e_is, p_is = e_all.iloc[:split], port_all.iloc[:split]
    e_oos, p_oos = e_all.iloc[split:], port_all.iloc[split:]

    def _metrics(e, p):
        te   = _ann_te(e.values, p.values)
        corr = float(np.corrcoef(e.values, p.values)[0, 1]) if len(e) > 2 else 0.0
        ss_r = float(np.sum((e.values - p.values) ** 2))
        ss_t = float(np.sum((e.values - e.mean()) ** 2))
        r2   = 1.0 - ss_r / ss_t if ss_t > 0 else 0.0
        return te, corr, r2

    te_is,  corr_is,  r2_is  = _metrics(e_is,  p_is)
    te_oos, corr_oos, _      = _metrics(e_oos, p_oos) if len(e_oos) > 20 else (te_is, corr_is, None)

    return {
        "tickers":           tickers,
        "weights":           w,
        "missing":           missing,
        "etf_returns":       etf_all,
        "port_returns":      port_all,
        "etf_cum":           (1 + etf_all).cumprod(),
        "port_cum":          (1 + port_all).cumprod(),
        "rolling_te":        (port_all - etf_all).rolling(63).std() * np.sqrt(252),
        "tracking_error":    te_is,
        "correlation":       corr_is,
        "r_squared":         r2_is,
        "oos_tracking_error": te_oos,
        "oos_correlation":   corr_oos,
        "n_stocks":          len(tickers),
        "train_cutoff":      idx[split - 1],
    }
