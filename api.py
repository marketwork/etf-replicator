"""FastAPI backend for ETF Replicator."""

import asyncio
import concurrent.futures
import json
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from data import fetch_prices, to_returns, get_universe, get_stock_meta, normalize_ticker
from optimizer import replicate, evaluate

app = FastAPI(title="ETF Replicator", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory="static"), name="static")

_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)


# ── Request schema ─────────────────────────────────────────────────────────────

class OptimizeRequest(BaseModel):
    etf: str
    start: str
    end: str
    n_stocks: int = Field(25, ge=5, le=60)
    method: str = Field("lasso", pattern="^(lasso|greedy|hybrid)$")
    train_frac: float = Field(0.70, ge=0.40, le=0.95)
    min_coverage: float = Field(0.80, ge=0.50, le=1.0)
    custom_universe: list[str] = Field(default_factory=list)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return Path("static/index.html").read_text(encoding="utf-8")


@app.post("/api/optimize")
async def optimize(req: OptimizeRequest):
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def _push(msg: dict):
        asyncio.run_coroutine_threadsafe(queue.put(msg), loop)

    def _work():
        try:
            _push({"type": "progress", "pct": 5, "text": "Fetching constituent universe…"})

            etf_norm = normalize_ticker(req.etf)
            custom = [normalize_ticker(t) for t in req.custom_universe if t.strip()]
            universe, label = get_universe(etf_norm, custom or None)

            if not universe:
                _push({"type": "error",
                       "message": (
                           f"Could not find constituents for {req.etf}. "
                           "Try pasting tickers manually in the Custom Universe box."
                       )})
                return

            _push({"type": "progress", "pct": 12,
                   "text": f"Universe: {label}. Downloading prices…"})

            all_tickers = sorted(set([etf_norm] + universe))
            prices = fetch_prices(all_tickers, req.start, req.end)

            etf_up = etf_norm
            if etf_up not in prices.columns:
                _push({"type": "error",
                       "message": f"No price data found for {etf_up}. Check the ticker and date range."})
                return

            _push({"type": "progress", "pct": 35, "text": "Computing returns…"})

            returns = to_returns(prices, req.min_coverage)
            etf_r = returns[etf_up]
            stock_r = returns.drop(columns=[etf_up], errors="ignore")
            stock_r = stock_r.loc[:, stock_r.abs().sum() > 0]

            n_avail = len(stock_r.columns)
            _push({"type": "progress", "pct": 42,
                   "text": f"{n_avail} stocks available. Running optimiser…"})

            def _cb(frac: float):
                _push({"type": "progress",
                       "pct": int(42 + frac * 50),
                       "text": f"Optimising… {int(frac * 100)}%"})

            result = replicate(
                etf_r=etf_r,
                stock_r=stock_r,
                n_stocks=req.n_stocks,
                method=req.method,
                train_frac=req.train_frac,
                progress_cb=_cb,
            )

            _push({"type": "progress", "pct": 95, "text": "Fetching stock metadata…"})
            meta = get_stock_meta(result["tickers"])

            _push({"type": "done", "result": _serialize(result, meta, label)})

        except Exception as exc:
            import traceback
            _push({"type": "error",
                   "message": str(exc),
                   "detail": traceback.format_exc()})

    loop.run_in_executor(_pool, _work)

    async def _stream():
        while True:
            msg = await queue.get()
            yield f"data: {json.dumps(msg)}\n\n"
            if msg["type"] in ("done", "error"):
                break

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── /api/evaluate ─────────────────────────────────────────────────────────────

class EvaluateRequest(BaseModel):
    etf: str
    portfolio: dict  # {ticker: weight}
    start: str
    end: str
    train_frac: float = Field(0.70, ge=0.40, le=0.95)


@app.post("/api/evaluate")
async def evaluate_endpoint(req: EvaluateRequest):
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def _push(msg: dict):
        asyncio.run_coroutine_threadsafe(queue.put(msg), loop)

    def _work():
        try:
            etf_norm = normalize_ticker(req.etf)
            port_norm = {normalize_ticker(t): float(w)
                         for t, w in req.portfolio.items() if w > 0}

            if not port_norm:
                _push({"type": "error", "message": "Portfolio is empty."})
                return

            all_tickers = sorted(set([etf_norm] + list(port_norm.keys())))
            _push({"type": "progress", "pct": 10,
                   "text": f"Downloading prices for {len(all_tickers)} tickers…"})

            prices = fetch_prices(all_tickers, req.start, req.end)

            if etf_norm not in prices.columns:
                _push({"type": "error",
                       "message": f"No price data for {etf_norm}."})
                return

            _push({"type": "progress", "pct": 55, "text": "Computing metrics…"})

            returns = to_returns(prices, min_coverage=0.50)
            etf_r   = returns[etf_norm]
            stock_r = returns.drop(columns=[etf_norm], errors="ignore")

            result = evaluate(
                etf_r=etf_r,
                stock_r=stock_r,
                weights_dict=port_norm,
                train_frac=req.train_frac,
            )

            _push({"type": "progress", "pct": 85, "text": "Fetching metadata…"})
            meta = get_stock_meta(result["tickers"])

            payload = _serialize(result, meta, f"user portfolio ({len(port_norm)} stocks)")
            payload["missing"] = result.get("missing", [])
            payload["mode"] = "evaluate"
            _push({"type": "done", "result": payload})

        except Exception as exc:
            import traceback
            _push({"type": "error", "message": str(exc),
                   "detail": traceback.format_exc()})

    loop.run_in_executor(_pool, _work)

    async def _stream():
        while True:
            msg = await queue.get()
            yield f"data: {json.dumps(msg)}\n\n"
            if msg["type"] in ("done", "error"):
                break

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Serialisation ──────────────────────────────────────────────────────────────

def _ann_ret(r: pd.Series) -> float:
    n = len(r) / 252
    return float((1 + r).prod() ** (1 / n) - 1) if n > 0.1 else 0.0

def _ann_vol(r: pd.Series) -> float:
    return float(r.std(ddof=1) * np.sqrt(252))

def _max_dd(r: pd.Series) -> float:
    cum = (1 + r).cumprod()
    return float((cum / cum.cummax() - 1).min())


def _serialize(result: dict, meta: pd.DataFrame, universe_label: str) -> dict:
    er = result["etf_returns"]
    pr = result["port_returns"]

    meta_dict = {}
    for tk in result["tickers"]:
        if not meta.empty and tk in meta.index:
            row = meta.loc[tk]
            meta_dict[tk] = {
                "name": str(row.get("name", tk)),
                "sector": str(row.get("sector", "—")),
                "market_cap_B": float(row.get("market_cap_B", 0)),
            }
        else:
            meta_dict[tk] = {"name": tk, "sector": "—", "market_cap_B": 0.0}

    return {
        "etf": er.name if hasattr(er, "name") else "ETF",
        "universe_label": universe_label,
        "tickers": result["tickers"],
        "weights": result["weights"].tolist(),
        "n_stocks": result["n_stocks"],
        "tracking_error": result["tracking_error"],
        "oos_tracking_error": result["oos_tracking_error"],
        "correlation": result["correlation"],
        "oos_correlation": result["oos_correlation"],
        "r_squared": result["r_squared"],
        "train_cutoff": str(result["train_cutoff"].date()),
        "dates": [str(d.date()) for d in er.index],
        "etf_returns": er.tolist(),
        "port_returns": pr.tolist(),
        "etf_cum": result["etf_cum"].tolist(),
        "port_cum": result["port_cum"].tolist(),
        "rolling_te": result["rolling_te"].fillna(0).tolist(),
        "meta": meta_dict,
        "perf": {
            "etf":  {"ann_ret": _ann_ret(er), "ann_vol": _ann_vol(er), "max_dd": _max_dd(er)},
            "port": {"ann_ret": _ann_ret(pr), "ann_vol": _ann_vol(pr), "max_dd": _max_dd(pr)},
        },
    }
