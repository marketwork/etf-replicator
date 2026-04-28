"""Data fetching: ETF constituent universes and price history."""

import io
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import requests
import yfinance as yf
from functools import lru_cache

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def _wiki_html(url: str) -> str:
    r = requests.get(url, headers=_HEADERS, timeout=20)
    r.raise_for_status()
    return r.text


# ── Index constituent scrapers ────────────────────────────────────────────────

@lru_cache(maxsize=4)
def _sp500_tickers() -> list:
    html = _wiki_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    df = pd.read_html(io.StringIO(html))[0]
    col = next((c for c in df.columns if "symbol" in str(c).lower()), df.columns[0])
    return [str(t).replace(".", "-").strip() for t in df[col].dropna().tolist()]


@lru_cache(maxsize=4)
def _nasdaq100_tickers() -> list:
    html = _wiki_html("https://en.wikipedia.org/wiki/Nasdaq-100")
    for tbl in pd.read_html(io.StringIO(html)):
        for col in tbl.columns:
            if "ticker" in str(col).lower() or "symbol" in str(col).lower():
                vals = [str(v).strip() for v in tbl[col].dropna()
                        if str(v).strip() not in ("nan", "")]
                if len(vals) >= 90:
                    return vals
    return []


@lru_cache(maxsize=4)
def _dow30_tickers() -> list:
    html = _wiki_html("https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average")
    for tbl in pd.read_html(io.StringIO(html)):
        for col in tbl.columns:
            if "symbol" in str(col).lower() or "ticker" in str(col).lower():
                vals = [str(v).strip() for v in tbl[col].dropna()
                        if str(v).strip() not in ("nan", "")]
                if 25 <= len(vals) <= 35:
                    return vals
    return []


# ── iShares CSV (covers IWM, XL* sector, EEM, EFA, TLT, HYG …) ───────────────
# iShares product page IDs: https://www.ishares.com/us/products/{id}
_ISHARES_IDS = {
    "IWM":  "239710",   # Russell 2000
    "EEM":  "239637",   # Emerging Markets
    "EFA":  "239623",   # MSCI EAFE
    "TLT":  "239454",   # 20+ Year Treasury
    "HYG":  "239565",   # High Yield Corp Bond
    "LQD":  "239566",   # Investment Grade Corp Bond
    "XLK":  "239726",   # Technology (SPDR — see below, handled separately)
}

# SPDR sector ETFs → State Street XLSX/CSV
_SPDR_IDS = {
    "XLK": "XLK", "XLF": "XLF", "XLV": "XLV", "XLE": "XLE",
    "XLI": "XLI", "XLC": "XLC", "XLP": "XLP", "XLU": "XLU",
    "XLB": "XLB", "XLRE": "XLRE",
}


def _spdr_holdings(etf: str) -> list:
    """Fetch SPDR sector ETF holdings from State Street."""
    url = (
        f"https://www.ssga.com/us/en/intermediary/etfs/library-content/products/"
        f"fund-data/etfs/us/holdings-daily-us-en-{etf.lower()}.xlsx"
    )
    try:
        r = requests.get(url, headers=_HEADERS, timeout=20)
        r.raise_for_status()
        df = pd.read_excel(io.BytesIO(r.content), header=None, skiprows=4)
        # Ticker is usually in column 1 or 2; name in column 0
        for col in df.columns[1:4]:
            vals = df[col].dropna().tolist()
            vals = [str(v).strip() for v in vals
                    if str(v).strip() not in ("nan", "—", "-", "", "Ticker")]
            if len(vals) >= 10:
                return vals
    except Exception:
        pass
    return []


def _ishares_holdings(product_id: str) -> list:
    """Fetch iShares ETF holdings CSV."""
    url = (
        f"https://www.ishares.com/us/products/{product_id}/"
        f"1467271812596.ajax?fileType=csv&dataType=fund"
    )
    try:
        r = requests.get(url, headers=_HEADERS, timeout=20)
        r.raise_for_status()
        # CSV has a few header rows before the actual data
        text = r.text
        lines = text.splitlines()
        # Find the row that contains "Ticker" header
        start = next((i for i, l in enumerate(lines) if "Ticker" in l), None)
        if start is None:
            return []
        df = pd.read_csv(io.StringIO("\n".join(lines[start:])))
        col = next((c for c in df.columns if "ticker" in c.lower()), None)
        if col is None:
            return []
        vals = [str(v).strip() for v in df[col].dropna()
                if str(v).strip() not in ("nan", "—", "-", "")]
        return vals
    except Exception:
        return []


def _yf_holdings(etf: str) -> list:
    """Fallback: yfinance top holdings (usually top 25)."""
    t = yf.Ticker(etf)
    try:
        h = t.funds_data.top_holdings
        if h is not None and not h.empty:
            return h.index.tolist()
    except Exception:
        pass
    return []


# ── ETF → preset map ──────────────────────────────────────────────────────────

_SP500_ETFs    = {"SPY", "IVV", "VOO", "SPLG", "VTI", "RSP"}
_NASDAQ_ETFs   = {"QQQ", "QQQM", "TQQQ", "ONEQ"}
_DOW_ETFs      = {"DIA"}
_SP600_USE_SP500 = {"IJR", "SPSM"}   # Use S&P 500 as proxy universe


def get_universe(etf: str, custom: list = None) -> tuple:
    """
    Return (ticker_list, label).  custom overrides everything.
    """
    if custom:
        return custom, f"custom ({len(custom)} tickers)"

    key = etf.upper()

    # ── Known index ETFs ──
    if key in _SP500_ETFs:
        try:
            tks = _sp500_tickers()
            if len(tks) >= 400:
                return tks, f"S&P 500 ({len(tks)} stocks)"
        except Exception:
            pass

    if key in _NASDAQ_ETFs:
        try:
            tks = _nasdaq100_tickers()
            if len(tks) >= 90:
                return tks, f"NASDAQ-100 ({len(tks)} stocks)"
        except Exception:
            pass
        # Fall through to S&P 500 if NASDAQ scrape fails
        try:
            tks = _sp500_tickers()
            if len(tks) >= 400:
                return tks, f"S&P 500 (NASDAQ fallback, {len(tks)} stocks)"
        except Exception:
            pass

    if key in _DOW_ETFs:
        try:
            tks = _dow30_tickers()
            if len(tks) >= 25:
                return tks, f"DJIA ({len(tks)} stocks)"
        except Exception:
            pass

    # ── SPDR sector ETFs ──
    if key in _SPDR_IDS:
        tks = _spdr_holdings(key)
        if len(tks) >= 15:
            return tks, f"SPDR {key} holdings ({len(tks)} stocks)"

    # ── iShares ETFs ──
    if key in _ISHARES_IDS:
        tks = _ishares_holdings(_ISHARES_IDS[key])
        if len(tks) >= 15:
            return tks, f"iShares {key} holdings ({len(tks)} stocks)"

    # ── Generic: try yfinance ──
    tks = _yf_holdings(key)
    if len(tks) >= 15:
        return tks, f"yfinance top holdings ({len(tks)} stocks)"

    # ── Last resort: S&P 500 as broad universe ──
    try:
        tks = _sp500_tickers()
        if len(tks) >= 400:
            return tks, f"S&P 500 (broad fallback, {len(tks)} stocks)"
    except Exception:
        pass

    return [], "no universe found"


# ── Price / return helpers ────────────────────────────────────────────────────

def fetch_prices(tickers: list, start: str, end: str) -> pd.DataFrame:
    raw = yf.download(
        tickers, start=start, end=end,
        auto_adjust=True, progress=False,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])
    return prices


def to_returns(prices: pd.DataFrame, min_coverage: float = 0.80) -> pd.DataFrame:
    rets = prices.pct_change().iloc[1:]
    min_obs = int(len(rets) * min_coverage)
    rets = rets.dropna(thresh=min_obs, axis=1)
    return rets.ffill().fillna(0.0)


def get_stock_meta(tickers: list) -> pd.DataFrame:
    rows = []
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            rows.append({
                "ticker": t,
                "name": (info.get("shortName") or t)[:32],
                "sector": info.get("sector") or "—",
                "market_cap_B": round((info.get("marketCap") or 0) / 1e9, 1),
            })
        except Exception:
            rows.append({"ticker": t, "name": t, "sector": "—", "market_cap_B": 0.0})
    return pd.DataFrame(rows).set_index("ticker")
