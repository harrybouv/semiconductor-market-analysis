from pathlib import Path
import pandas as pd
import yfinance as yf

START = "2010-01-01"
TOPS = [5, 10]

HERE = Path(__file__).resolve()

def find_root() -> Path:
    for p in [HERE.parent, *HERE.parents]:
        if (p / "data").exists():
            return p
    raise FileNotFoundError("No project root with /data found")

ROOT = find_root()
DATA_DIR = ROOT / "data" / "concentration"
DATA_DIR.mkdir(parents=True, exist_ok=True)

TICKERS_CSV = DATA_DIR / "sp500_tickers.csv"
OUT_CSV = DATA_DIR / "sp500_topn_share.csv"

def load_tickers() -> list[str]:
    df = pd.read_csv(TICKERS_CSV)
    tickers = df["ticker"].astype(str).str.strip().tolist()
    tickers = [t for t in tickers if t and t.lower() != "nan"]
    return [t.replace(".", "-") for t in tickers]

def get_shares(t: str) -> float | None:
    tk = yf.Ticker(t)
    info = getattr(tk, "info", {}) or {}
    s = info.get("sharesOutstanding")
    if s and s > 0:
        return float(s)
    fi = getattr(tk, "fast_info", {}) or {}
    s = fi.get("shares_outstanding")
    if s and s > 0:
        return float(s)
    return None

def main():
    print("ROOT =", ROOT)
    print("Using tickers file =", TICKERS_CSV)

    if not TICKERS_CSV.exists():
        raise FileNotFoundError(f"Missing {TICKERS_CSV}")

    tickers = load_tickers()
    print("Tickers loaded:", len(tickers))

    shares = {}
    for t in tickers:
        s = get_shares(t)
        if s:
            shares[t] = s

    tickers_ok = list(shares.keys())
    print("Tickers with shares:", len(tickers_ok))
    if len(tickers_ok) < max(TOPS):
        raise RuntimeError("Too few tickers have sharesOutstanding. Expand list or check yfinance.")

    # Download monthly prices
    px = yf.download(
        tickers_ok,
        start=START,
        interval="1mo",
        auto_adjust=True,
        threads=True,
        group_by="ticker",
        progress=False,
    )

    # Extract Close panel
    closes = {}
    for t in tickers_ok:
        try:
            if isinstance(px.columns, pd.MultiIndex):
                if "Close" in px.columns.get_level_values(0):
                    s = px["Close"][t]
                else:
                    s = px[t]["Close"]
            else:
                s = px["Close"]
            closes[t] = s
        except Exception:
            pass

    close_df = pd.DataFrame(closes).dropna(how="all")
    if close_df.empty:
        raise RuntimeError("Price download produced empty panel. Likely yfinance blocked/rate-limited.")

    close_df.index = pd.to_datetime(close_df.index).to_period("M").to_timestamp("M")
    close_df = close_df.sort_index()

    # Market cap proxy
    mc = close_df.copy()
    for t in mc.columns:
        mc[t] = mc[t] * shares[t]

    total = mc.sum(axis=1)

    out = pd.DataFrame(index=mc.index)
    for n in TOPS:
        topn = mc.apply(lambda r: r.nlargest(n).sum(), axis=1)
        out[f"top_{n}_share"] = topn / total

    out = out.dropna()
    out.to_csv(OUT_CSV)
    print("Saved:", OUT_CSV)

if __name__ == "__main__":
    main()
