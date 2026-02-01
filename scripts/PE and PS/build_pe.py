# build_pe.py
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf

TICKERS = ["NVDA", "AMD", "TSM", "ASML", "AVGO"]

HERE = Path(__file__).resolve()

def find_root() -> Path:
    # choose the first parent that contains your real monthly price files
    for p in [HERE.parent, *HERE.parents]:
        prices_dir = p / "data" / "prices"
        if prices_dir.exists() and any(prices_dir.glob("*_monthly_prices.csv")):
            return p
    raise FileNotFoundError("Could not find root containing data/prices/*_monthly_prices.csv")

ROOT = find_root()

PRICES_DIR = ROOT / "data" / "prices"
OUT_DIR = ROOT / "data" / "valuation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "1999-01-01"


def read_monthly_prices(ticker: str) -> pd.DataFrame:
    f = PRICES_DIR / f"{ticker}_monthly_prices.csv"
    df = pd.read_csv(f, parse_dates=["Date"]).sort_values("Date")
    df["Date"] = df["Date"].dt.to_period("M").dt.to_timestamp("M")
    df = df.drop_duplicates("Date", keep="last")

    col = "Adj Close" if "Adj Close" in df.columns else "Close"
    out = df[["Date", col]].rename(columns={col: "Price"})
    out = out[out["Date"] >= pd.to_datetime(START_DATE)]
    return out.reset_index(drop=True)


def get_annual_eps(ticker: str) -> pd.Series:
    """
    Returns annual EPS series indexed by fiscal period end date.
    Uses yfinance income statement; simple + stable.
    """
    tk = yf.Ticker(ticker)

    inc = tk.income_stmt
    if inc is None or inc.empty:
        inc = tk.financials
    if inc is None or inc.empty:
        raise ValueError(f"{ticker}: no annual income statement available in yfinance")

    inc = inc.T  # dates as rows
    eps_col = next((c for c in ["Diluted EPS", "Basic EPS", "EPS Diluted", "EPS Basic"] if c in inc.columns), None)
    if eps_col is None:
        raise ValueError(f"{ticker}: EPS column not found. Columns: {inc.columns.tolist()}")

    eps = pd.to_numeric(inc[eps_col], errors="coerce").dropna()
    eps.index = pd.to_datetime(eps.index).to_period("M").to_timestamp("M")
    eps = eps.sort_index()
    eps.name = "EPS"
    return eps


def build_pe_for_ticker(ticker: str) -> dict:
    prices = read_monthly_prices(ticker)
    eps = get_annual_eps(ticker)

    # align EPS to monthly dates via forward fill
    eps_m = eps.reindex(prices["Date"], method="ffill")

    df = prices.copy()
    df["EPS"] = eps_m.values

    # PE undefined for EPS <= 0
    df["PE"] = np.where(df["EPS"] > 0, df["Price"] / df["EPS"], np.nan)

    df = df.dropna(subset=["PE"])

    out = df[["Date", "Price", "EPS", "PE"]]
    out_path = OUT_DIR / f"{ticker}_pe_monthly.csv"
    out.to_csv(out_path, index=False)

    print(f"{ticker}: saved {len(out)} rows | eps_points={len(eps)}")

    return {
        "ticker": ticker,
        "rows_saved": len(out),
        "eps_points": int(len(eps)),
        "first_eps_date": str(eps.index.min().date()) if len(eps) else None,
        "last_eps_date": str(eps.index.max().date()) if len(eps) else None,
    }


def main():
    summary_rows = []
    for t in TICKERS:
        summary_rows.append(build_pe_for_ticker(t))

    summary = pd.DataFrame(summary_rows)
    summary_path = OUT_DIR / "pe_build_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("\nSaved summary ->", summary_path)
    print(summary)


if __name__ == "__main__":
    print("Project root:", ROOT)
    main()
