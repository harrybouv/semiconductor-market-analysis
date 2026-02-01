from pathlib import Path
import yfinance as yf

# 1) Where to save
out_dir = Path("../data/prices")
out_dir.mkdir(parents=True, exist_ok=True)

# 2) What to download
tickers = ["NVDA", "AMD", "TSM", "ASML", "AVGO"]

for t in tickers:
    df = yf.download(t, start="1999-01-01", interval="1mo", auto_adjust=False, progress=False)

    # Clean column names (yfinance sometimes creates a MultiIndex)
    if hasattr(df.columns, "levels"):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    # Keep only what we need
    df = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].copy()

    # Save
    out_path = out_dir / f"{t}_monthly_prices.csv"
    df.to_csv(out_path)
    print(f"Saved {t}: {len(df)} rows -> {out_path}")
