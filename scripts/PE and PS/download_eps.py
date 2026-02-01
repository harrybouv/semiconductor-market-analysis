# download_eps.py
import pandas as pd
from pathlib import Path
import yfinance as yf

TICKERS = ["NVDA", "AMD", "TSM", "ASML", "AVGO"]

OUT_DIR = Path("data/fundamentals")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def get_annual_eps(ticker: str) -> pd.DataFrame:
    tk = yf.Ticker(ticker)

    # Try income statement tables first
    tbl = None
    if hasattr(tk, "income_stmt") and isinstance(tk.income_stmt, pd.DataFrame) and not tk.income_stmt.empty:
        tbl = tk.income_stmt
    elif hasattr(tk, "financials") and isinstance(tk.financials, pd.DataFrame) and not tk.financials.empty:
        tbl = tk.financials

    if tbl is None:
        raise RuntimeError(f"{ticker}: could not access financials table")

    # Pick a sensible EPS row
    row = None
    for name in ["Diluted EPS", "Basic EPS", "EPS Diluted", "EPS Basic"]:
        if name in tbl.index:
            row = name
            break
    if row is None:
        raise RuntimeError(f"{ticker}: no EPS row found in financials (looked for Diluted/Basic EPS)")

    s = tbl.loc[row].dropna()

    df = pd.DataFrame({"Date": pd.to_datetime(s.index), "EPS": s.values})
    df = df.sort_values("Date")
    return df

if __name__ == "__main__":
    for t in TICKERS:
        df = get_annual_eps(t)
        out = OUT_DIR / f"{t}_annual_eps.csv"
        df.to_csv(out, index=False)
        print(f"Wrote: {out}")
