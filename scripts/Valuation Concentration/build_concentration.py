# build_concentration.py
from pathlib import Path
import pandas as pd
import yfinance as yf

HERE = Path(__file__).resolve()

def find_root():
    for p in [HERE.parent, *HERE.parents]:
        if (p / "data" / "prices").exists() or (p / "data").exists():
            return p
    raise RuntimeError("Project root not found")

ROOT = find_root()
OUT_DIR = ROOT / "data" / "concentration"
OUT_DIR.mkdir(parents=True, exist_ok=True)

START = "2010-01-01"

# --- Market universe ---
# If you already maintain a broad S&P list on disk, use it. Otherwise fall back to a small list.
SP500_TICKERS_FILE = OUT_DIR / "sp500_tickers.csv"  # optional; if present, should have a 'ticker' column

DEFAULT_SP500_TOP = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "TSLA", "UNH", "JPM",
    "LLY", "V", "XOM", "AVGO", "MA", "COST", "HD", "PG", "JNJ", "MRK",
]

def load_sp500_universe():
    if SP500_TICKERS_FILE.exists():
        df = pd.read_csv(SP500_TICKERS_FILE)
        col = "ticker" if "ticker" in df.columns else df.columns[0]
        tickers = df[col].dropna().astype(str).str.strip().unique().tolist()
        return tickers
    return DEFAULT_SP500_TOP

SP500_UNIVERSE = load_sp500_universe()

# --- Expanded semiconductor universe (static + defensible) ---
# Mix of: AI compute, foundries, equipment, memory, analog/power, mobile/RF, IDM, Japan toolchain, Korea/Taiwan leaders.
SEMIS = [
    # AI / compute / datacenter semis
    "NVDA", "AMD", "AVGO", "QCOM", "MRVL", "TXN", "ADI", "MCHP", "NXPI", "ON", "INTC",

    # Foundries / manufacturing (publicly traded)
    "TSM", "UMC", "GFS",

    # Equipment / EDA / IP
    "ASML", "AMAT", "LRCX", "KLAC", "ASX",  # ASX = ASE Technology (packaging) [note: ticker sometimes "ASX" is ASE in US ADR]
    "TER", "LSCC", "SWKS", "QRVO",

    # Memory (US + Asia)
    "MU",
    "000660.KS",   # SK Hynix
    "005930.KS",   # Samsung Electronics (proxy for memory + logic exposure)

    # Europe semis
    "STM",         # STMicro
    "IFX.DE",      # Infineon (Xetra)
    "BESI.AS",     # BE Semiconductor Industries
    "ASMI.AS",     # ASM International

    # Japan semiconductor equipment / ecosystem
    "8035.T",      # Tokyo Electron
    "6857.T",      # Advantest
    "6723.T",      # Renesas
    "6526.T",      # Socionext (optional; newer listing)
    "4063.T",      # Shin-Etsu Chemical (silicon wafers) – optional but relevant upstream
    "3436.T",      # SUMCO (wafers)

    # Taiwan / China ecosystem
    "2454.TW",     # MediaTek
    "3711.TW",     # ASE Technology (local; if ASX ADR causes issues)
    "2330.TW",     # TSMC local (optional duplicate with TSM ADR; keep one if you prefer)
]

# De-duplicate tickers (in case you keep both ADR + local)
SEMIS = list(dict.fromkeys(SEMIS))

def _safe_float(x):
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None

def get_price_monthly(ticker: str) -> pd.Series:
    tk = yf.Ticker(ticker)
    px = tk.history(start=START, interval="1mo", auto_adjust=False)
    if px.empty:
        raise RuntimeError(f"{ticker}: no price history returned")
    px = px[~px.index.duplicated(keep="last")]
    close = px["Close"].rename(ticker)
    return close

def estimate_shares_constant(tk: yf.Ticker, ticker: str, last_price: float) -> float:
    """
    Fallback: estimate sharesOutstanding using marketCap / last_price.
    Assumes shares constant through time (acceptable for a concentration diagnostic).
    """
    # Try fast_info first (often works better than info for some tickers)
    market_cap = None
    try:
        fi = getattr(tk, "fast_info", None)
        if fi:
            market_cap = _safe_float(fi.get("marketCap"))
    except Exception:
        market_cap = None

    # Try info as fallback
    if market_cap is None:
        try:
            info = tk.info
            market_cap = _safe_float(info.get("marketCap"))
        except Exception:
            market_cap = None

    if market_cap is None or last_price <= 0:
        raise RuntimeError(f"{ticker}: cannot infer shares (marketCap missing and sharesOutstanding missing)")

    return market_cap / last_price

def get_market_cap_series(ticker: str) -> pd.Series:
    tk = yf.Ticker(ticker)
    close = get_price_monthly(ticker)
    last_price = float(close.dropna().iloc[-1])

    # Preferred: sharesOutstanding
    shares = None
    try:
        info = tk.info
        shares = _safe_float(info.get("sharesOutstanding"))
    except Exception:
        shares = None

    if not shares or shares <= 0:
        shares = estimate_shares_constant(tk, ticker, last_price)

    mc = (close * shares).rename(ticker)
    return mc

def build_panel(tickers: list[str]) -> pd.DataFrame:
    series = []
    errors = {}
    for t in tickers:
        try:
            series.append(get_market_cap_series(t))
        except Exception as e:
            errors[t] = str(e)

    if errors:
        # Save diagnostics so you can see which tickers failed and why
        pd.Series(errors, name="error").to_csv(OUT_DIR / "ticker_failures.csv")
        print(f"Warning: {len(errors)} tickers failed. See data/concentration/ticker_failures.csv")

    if not series:
        raise RuntimeError("No tickers succeeded; cannot build panel")

    panel = pd.concat(series, axis=1).sort_index()
    return panel

def top_n_share(panel: pd.DataFrame, n: int) -> pd.Series:
    total = panel.sum(axis=1)
    topn = panel.apply(lambda r: r.nlargest(n).sum(), axis=1)
    return (topn / total).rename(f"top_{n}_share")

def main():
    sp = build_panel(SP500_UNIVERSE)
    semi = build_panel(SEMIS)

    out = pd.DataFrame({
        "sp500_top5": top_n_share(sp, 5),
        "sp500_top10": top_n_share(sp, 10),
        # Semiconductor-universe concentration (now meaningful because the universe is broader)
        "semi_top2": top_n_share(semi, 2),
        "semi_top3": top_n_share(semi, 3),
        "semi_top5": top_n_share(semi, 5),
        "semi_top10": top_n_share(semi, 10),
    }).dropna()

    out.to_csv(OUT_DIR / "market_concentration.csv")
    print("Saved data/concentration/market_concentration.csv")

if __name__ == "__main__":
    main()
