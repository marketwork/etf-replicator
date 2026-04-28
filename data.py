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


# ── Ticker normaliser ─────────────────────────────────────────────────────────
# Converts Bloomberg-style "2828 HK" or "SPY US" → yfinance ticker

_EXCHANGE_SUFFIX = {
    "US": "",
    "HK": ".HK",
    "JP": ".T",
    "KS": ".KS",
    "KQ": ".KQ",
    "SS": ".SS",
    "SZ": ".SZ",
    "LN": ".L",
    "GR": ".DE",
    "FP": ".PA",
    "AU": ".AX",
    "CN": ".TO",
}


def normalize_ticker(raw: str) -> str:
    """'2828 HK' → '2828.HK',  'SPY US' → 'SPY',  'SPY' → 'SPY'."""
    raw = raw.strip().upper()
    parts = raw.split()
    if len(parts) == 2 and parts[1] in _EXCHANGE_SUFFIX:
        base, exch = parts
        suffix = _EXCHANGE_SUFFIX[exch]
        # Zero-pad numeric HK codes to 4 digits
        if exch == "HK" and base.isdigit():
            base = base.zfill(4)
        return base + suffix
    return raw


# ── Hardcoded fallback universes ──────────────────────────────────────────────

# Hang Seng Index constituents (82 stocks as of 2025)
_HANG_SENG = [
    "0001.HK","0002.HK","0003.HK","0005.HK","0006.HK","0011.HK","0012.HK",
    "0016.HK","0017.HK","0019.HK","0027.HK","0066.HK","0101.HK","0151.HK",
    "0175.HK","0241.HK","0267.HK","0268.HK","0288.HK","0291.HK","0316.HK",
    "0384.HK","0386.HK","0388.HK","0522.HK","0669.HK","0688.HK","0700.HK",
    "0762.HK","0823.HK","0857.HK","0868.HK","0883.HK","0939.HK","0941.HK",
    "0960.HK","0968.HK","0981.HK","0992.HK","1038.HK","1044.HK","1093.HK",
    "1109.HK","1113.HK","1177.HK","1209.HK","1211.HK","1299.HK","1378.HK",
    "1398.HK","1810.HK","1876.HK","1928.HK","1929.HK","2007.HK","2018.HK",
    "2020.HK","2269.HK","2313.HK","2318.HK","2319.HK","2331.HK","2382.HK",
    "2388.HK","2628.HK","3328.HK","3690.HK","3988.HK","6098.HK","6862.HK",
    "9618.HK","9633.HK","9888.HK","9961.HK","9988.HK","9999.HK",
]

# Top CSI 300 A-share stocks by market cap (proxy for 3033.HK / 3147.HK)
_CSI300_TOP = [
    "600519.SS","300750.SZ","601318.SS","000858.SZ","600036.SS","601166.SS",
    "000333.SZ","600900.SS","600276.SS","000568.SZ","002594.SZ","601888.SS",
    "600690.SS","002415.SZ","600809.SS","601012.SS","300059.SZ","600309.SS",
    "002304.SZ","601919.SS","600031.SS","601601.SS","002352.SZ","601390.SS",
    "000651.SZ","600048.SS","601628.SS","601225.SS","600999.SS","002714.SZ",
    "000776.SZ","601328.SS","600150.SS","601229.SS","002027.SZ","600585.SS",
    "002049.SZ","601211.SS","603259.SS","601816.SS","600011.SS","601699.SS",
    "002475.SZ","600196.SS","601006.SS","601688.SS","000725.SZ","002241.SZ",
    "603986.SS","601668.SS","600886.SS","601989.SS","601766.SS","000001.SZ",
    "601998.SS","600050.SS","601800.SS","601186.SS","601336.SS","600104.SS",
    "601985.SS","601111.SS","601088.SS","601939.SS","600028.SS","601857.SS",
    "601898.SS","600941.SS","601020.SS","002736.SZ","000002.SZ","600000.SS",
    "601236.SS","002230.SZ","002311.SZ","601872.SS","600029.SS","000538.SZ",
    "600346.SS","002371.SZ","300124.SZ","601138.SS","600测试","300433.SZ",
    "002142.SZ","601169.SS","000063.SZ","002572.SZ","600905.SS","601117.SS",
]

# iShares MSCI South Korea ETF top holdings (EWY) — KOSPI tickers
_EWY_FALLBACK = [
    "005930.KS","000660.KS","373220.KS","207940.KS","005380.KS","068270.KS",
    "005490.KS","035420.KS","000270.KS","028260.KS","012330.KS","066570.KS",
    "096770.KS","055550.KS","017670.KS","105560.KS","032830.KS","003550.KS",
    "034730.KS","015760.KS","009150.KS","035720.KS","033780.KS","003490.KS",
    "024110.KS","086790.KS","030200.KS","011200.KS","000810.KS","036570.KS",
    "009830.KS","010130.KS","051910.KS","006400.KS","090430.KS","003670.KS",
    "004020.KS","007070.KS","018260.KS","010950.KS","267250.KS","316140.KS",
    "042660.KS","259960.KS","352820.KS","000100.KS","047050.KS","011170.KS",
    "326030.KS","377300.KS",
]


# ── Wikipedia scrapers ────────────────────────────────────────────────────────

def _wiki_html(url: str) -> str:
    r = requests.get(url, headers=_HEADERS, timeout=20)
    r.raise_for_status()
    return r.text


@lru_cache(maxsize=4)
def _sp500_tickers() -> list:
    html = _wiki_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    df = pd.read_html(io.StringIO(html))[0]
    col = next((c for c in df.columns if "symbol" in str(c).lower()), df.columns[0])
    return [str(t).replace(".", "-").strip() for t in df[col].dropna()]


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


@lru_cache(maxsize=4)
def _hang_seng_tickers() -> list:
    """Scrape Hang Seng Index constituents from Wikipedia."""
    html = _wiki_html("https://en.wikipedia.org/wiki/Hang_Seng_Index")
    for tbl in pd.read_html(io.StringIO(html)):
        for col in tbl.columns:
            if "code" in str(col).lower() or "stock" in str(col).lower():
                vals = tbl[col].dropna().tolist()
                tickers = []
                for v in vals:
                    try:
                        code = str(int(float(str(v)))).zfill(4) + ".HK"
                        tickers.append(code)
                    except Exception:
                        pass
                if len(tickers) >= 50:
                    return tickers
    return []


# ── ETF provider CSV scrapers ─────────────────────────────────────────────────

_ISHARES_IDS = {
    "IWM":  "239710",
    "EEM":  "239637",
    "EFA":  "239623",
    "TLT":  "239454",
    "HYG":  "239565",
    "LQD":  "239566",
    "ACWI": "251850",
    "EWY":  "239681",
    "INDA": "253743",
    "EWT":  "239685",
    "EWJ":  "239665",
    "EWZ":  "239612",
    "GLD":  "239597",
}

_SPDR_IDS = {
    "XLK","XLF","XLV","XLE","XLI","XLC","XLP","XLU","XLB","XLRE",
}

# Country/location → yfinance exchange suffix (matches iShares CSV "Location" column)
_COUNTRY_SUFFIX = {
    "Korea (South)": ".KS", "South Korea": ".KS", "Korea": ".KS",
    "Japan": ".T",
    "Hong Kong": ".HK",
    "China": ".SS",
    "United Kingdom": ".L",
    "Germany": ".DE",
    "France": ".PA",
    "Australia": ".AX",
    "Canada": ".TO",
    "United States": "",
    "Switzerland": ".SW",
    "Netherlands": ".AS",
    "Sweden": ".ST",
    "Denmark": ".CO",
    "Spain": ".MC",
    "Italy": ".MI",
    "Brazil": ".SA",
    "India": ".NS",
    "Taiwan": ".TW",
    "Singapore": ".SI",
    "Mexico": ".MX",
    "South Africa": ".JO",
    "Indonesia": ".JK",
    "Thailand": ".BK",
    "Malaysia": ".KL",
}


def _ishares_holdings(product_id: str, ticker: str = "", top_n: int = None) -> list:
    # iShares requires the ETF ticker in the URL path
    slug = f"{ticker}/" if ticker else ""
    fname = f"&fileName={ticker}_holdings" if ticker else ""
    url = (
        f"https://www.ishares.com/us/products/{product_id}/{slug}"
        f"1467271812596.ajax?fileType=csv{fname}&dataType=fund"
    )
    try:
        r = requests.get(url, headers=_HEADERS, timeout=25)
        r.raise_for_status()
        lines = r.text.splitlines()
        start = next((i for i, l in enumerate(lines) if "Ticker" in l), None)
        if start is None:
            return []
        df = pd.read_csv(io.StringIO("\n".join(lines[start:])), on_bad_lines="skip")

        # Keep equities only
        if "Asset Class" in df.columns:
            df = df[df["Asset Class"].str.contains("Equity", na=False)]

        # Identify columns
        ticker_col  = next((c for c in df.columns if "ticker" in c.lower()), None)
        weight_col  = next((c for c in df.columns if "weight" in c.lower()), None)
        loc_col     = next((c for c in df.columns
                            if "location" in c.lower() or "country" in c.lower()), None)

        if ticker_col is None:
            return []

        # Filter by weight if requested
        if weight_col and top_n:
            df[weight_col] = pd.to_numeric(df[weight_col], errors="coerce")
            df = df.nlargest(top_n, weight_col)

        results = []
        for _, row in df.iterrows():
            tk = str(row[ticker_col]).strip()
            if not tk or tk in ("nan", "—", "-", ""):
                continue
            # If already has an exchange suffix, use as-is
            if "." in tk:
                results.append(tk)
                continue
            # Try to add suffix from location column
            suffix = ""
            if loc_col:
                loc = str(row.get(loc_col, ""))
                suffix = _COUNTRY_SUFFIX.get(loc, "")
            # Korean 6-digit numeric codes → .KS
            if not suffix and tk.isdigit() and len(tk) == 6:
                suffix = ".KS"
            results.append(tk + suffix)

        return [t for t in results if t]
    except Exception:
        return []


def _spdr_holdings(etf: str) -> list:
    url = (
        f"https://www.ssga.com/us/en/intermediary/etfs/library-content/products/"
        f"fund-data/etfs/us/holdings-daily-us-en-{etf.lower()}.xlsx"
    )
    try:
        r = requests.get(url, headers=_HEADERS, timeout=20)
        r.raise_for_status()
        df = pd.read_excel(io.BytesIO(r.content), header=None, skiprows=4)
        for col in df.columns[1:4]:
            vals = [str(v).strip() for v in df[col].dropna()
                    if str(v).strip() not in ("nan", "—", "-", "", "Ticker")]
            if len(vals) >= 10:
                return vals
    except Exception:
        pass
    return []


@lru_cache(maxsize=4)
def _acwi_tickers() -> list:
    """
    ACWI proxy = S&P 500 (US ~65%) + top EFA (developed ex-US ~25%) + top EEM (EM ~10%).
    iShares blocks direct ACWI CSV downloads; building it from components works fine.
    """
    universe = set()
    try:
        universe.update(_sp500_tickers())
    except Exception:
        pass
    for etf, pid, n in [("EFA", "239623", 100), ("EEM", "239637", 60)]:
        tks = _ishares_holdings(pid, ticker=etf, top_n=n)
        universe.update(tks)
    return list(universe)


def _yf_holdings(etf: str) -> list:
    try:
        h = yf.Ticker(etf).funds_data.top_holdings
        if h is not None and not h.empty:
            return h.index.tolist()
    except Exception:
        pass
    return []


# ── ETF → universe preset map ─────────────────────────────────────────────────

_SP500_SET   = {"SPY","IVV","VOO","SPLG","VTI","RSP"}
_NASDAQ_SET  = {"QQQ","QQQM","TQQQ","ONEQ"}
_DOW_SET     = {"DIA"}
_HK_TRACKERS = {"2828.HK"}           # Hang Seng
_CSI300_ETFs = {"3033.HK","3147.HK"} # CSI 300 / China A proxy


def get_universe(etf: str, custom: list = None) -> tuple:
    """Return (ticker_list, label). custom overrides everything."""
    if custom:
        return custom, f"custom ({len(custom)} tickers)"

    key = etf.upper()

    # ── Well-known index ETFs ──────────────────────────────────
    if key in _SP500_SET:
        try:
            tks = _sp500_tickers()
            if len(tks) >= 400:
                return tks, f"S&P 500 ({len(tks)} stocks)"
        except Exception:
            pass

    if key in _NASDAQ_SET:
        try:
            tks = _nasdaq100_tickers()
            if len(tks) >= 90:
                return tks, f"NASDAQ-100 ({len(tks)} stocks)"
        except Exception:
            pass

    if key in _DOW_SET:
        try:
            tks = _dow30_tickers()
            if len(tks) >= 25:
                return tks, f"DJIA ({len(tks)} stocks)"
        except Exception:
            pass

    # ── HK ETFs: Hang Seng ────────────────────────────────────
    if key in _HK_TRACKERS:
        try:
            tks = _hang_seng_tickers()
            if len(tks) >= 50:
                return tks, f"Hang Seng Index ({len(tks)} stocks)"
        except Exception:
            pass
        return _HANG_SENG, f"Hang Seng Index ({len(_HANG_SENG)} stocks, cached)"

    # ── HK ETFs: CSI 300 / China A ───────────────────────────
    if key in _CSI300_ETFs:
        # Filter out any bad entries from the hardcoded list
        clean = [t for t in _CSI300_TOP if "测试" not in t]
        return clean, f"CSI 300 top stocks ({len(clean)} stocks)"

    # ── SPDR sector ETFs ──────────────────────────────────────
    if key in _SPDR_IDS:
        tks = _spdr_holdings(key)
        if len(tks) >= 15:
            return tks, f"SPDR {key} holdings ({len(tks)} stocks)"

    # ── ACWI: build from SP500 + EFA + EEM components ────────
    if key == "ACWI":
        try:
            tks = _acwi_tickers()
            if len(tks) >= 200:
                return tks, f"ACWI proxy: S&P 500 + EFA + EEM ({len(tks)} stocks)"
        except Exception:
            pass

    # ── iShares ETFs (EWY, IWM, EEM, EFA …) ─────────────────
    if key in _ISHARES_IDS:
        tks = _ishares_holdings(_ISHARES_IDS[key], ticker=key)
        if len(tks) >= 15:
            return tks, f"iShares {key} holdings ({len(tks)} stocks)"

    # ── EWY fallback ──────────────────────────────────────────
    if key == "EWY":
        return _EWY_FALLBACK, f"MSCI Korea top stocks ({len(_EWY_FALLBACK)} stocks, cached)"

    # ── Generic: yfinance top holdings ───────────────────────
    tks = _yf_holdings(key)
    if len(tks) >= 15:
        return tks, f"yfinance top holdings ({len(tks)} stocks)"

    # ── Last resort: S&P 500 ──────────────────────────────────
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
