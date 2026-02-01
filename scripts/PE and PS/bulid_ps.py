from pathlib import Path
import pandas as pd
import yfinance as yf

# ---------- CONFIG ----------
TICKERS = ["NVDA", "AMD", "TSM", "ASML", "AVGO"]
START_DATE = "1999-01-01"

PRICES_DIR = Path("../data/prices")          # your existing monthly prices CSVs
OUT_DIR = Path("../data/valuation")          # output folder
OUT_DIR.mkdir(parents=True, exist_ok=True)

# If quarterly revenue has fewer than this many rows, we fall back to annual revenue
MIN_QUARTERS_REQUIRED = 12  # ~3 years of quarterly data


# ---------- HELPERS ----------
def read_monthly_prices(ticker: str) -> pd.DataFrame:
    f = PRICES_DIR / f"{ticker}_monthly_prices.csv"
    df = pd.read_csv(f, parse_dates=["Date"]).set_index("Date").sort_index()
    # safety: standardise column names
    df.columns = [c.strip() for c in df.columns]
    if "Adj Close" not in df.columns:
        raise ValueError(f"{ticker}: 'Adj Close' missing in {f}")
    # keep monthly series only
    return df[["Adj Close"]].dropna()


def get_revenue_series(ticker: str) -> tuple[pd.Series, str]:
    """
    Returns (revenue_series, mode)
    revenue_series is indexed by date, values are revenue in currency units.
    mode is "quarterly_ttm" or "annual".
    """
    tk = yf.Ticker(ticker)

    # Try quarterly first
    inc_q = tk.quarterly_income_stmt
    if inc_q is None or inc_q.empty:
        inc_q = tk.quarterly_financials  # fallback field

    if inc_q is not None and not inc_q.empty:
        inc_q = inc_q.T  # dates as rows
        rev_col = next((c for c in ["Total Revenue", "Revenue", "Net Revenue"] if c in inc_q.columns), None)
        if rev_col:
            qrev = pd.to_numeric(inc_q[rev_col], errors="coerce").dropna()
            qrev.index = pd.to_datetime(qrev.index)

            # If we have enough quarters, use TTM
            if len(qrev) >= MIN_QUARTERS_REQUIRED:
                ttm = qrev.sort_index().rolling(4).sum().dropna()
                ttm.name = "revenue_ttm"
                return ttm, "quarterly_ttm"

    # Fall back to annual revenue (always fine for long-history valuation regimes)
    inc_a = tk.income_stmt
    if inc_a is None or inc_a.empty:
        raise ValueError(f"{ticker}: No income statement data available in yfinance.")
    inc_a = inc_a.T
    rev_col = next((c for c in ["Total Revenue", "Revenue", "Net Revenue"] if c in inc_a.columns), None)
    if not rev_col:
        raise ValueError(f"{ticker}: Revenue column not found. Columns: {inc_a.columns.tolist()}")

    arev = pd.to_numeric(inc_a[rev_col], errors="coerce").dropna()
    arev.index = pd.to_datetime(arev.index)
    arev = arev.sort_index()
    arev.name = "revenue_annual"
    return arev, "annual"


def get_shares_outstanding(ticker: str, latest_price: float) -> tuple[float, str]:
    """
    Returns (shares, method).
    Tries fast_info -> info -> implied from market cap.
    """
    tk = yf.Ticker(ticker)

    # 1) fast_info (fast, but sometimes empty)
    fi = getattr(tk, "fast_info", {}) or {}
    shares = fi.get("shares_outstanding", None)
    mcap = fi.get("market_cap", None)

    if shares is not None and pd.notna(shares) and shares > 0:
        return float(shares), "fast_info_shares_outstanding"
    if mcap is not None and pd.notna(mcap) and mcap > 0:
        return float(mcap) / float(latest_price), "fast_info_implied_from_market_cap"

    # 2) info (slower, more reliable)
    info = getattr(tk, "info", {}) or {}
    shares = info.get("sharesOutstanding", None)
    mcap = info.get("marketCap", None)

    if shares is not None and pd.notna(shares) and shares > 0:
        return float(shares), "info_sharesOutstanding"
    if mcap is not None and pd.notna(mcap) and mcap > 0:
        return float(mcap) / float(latest_price), "info_implied_from_marketCap"

    # 3) last resort: raise with diagnostics
    raise ValueError(
        f"{ticker}: Could not get shares or market cap from yfinance. "
        f"fast_info keys={list(fi.keys())[:10]} info keys include sharesOutstanding/marketCap? "
        f"{'sharesOutstanding' in info}/{ 'marketCap' in info}"
    )



# ---------- MAIN ----------
def main():
    summary_rows = []

    for t in TICKERS:
        # 1) prices
        px = read_monthly_prices(t)
        px = px[px.index >= pd.to_datetime(START_DATE)]
        latest_price = float(px["Adj Close"].iloc[-1])

        # 2) revenue (quarterly TTM if possible, else annual)
        rev, rev_mode = get_revenue_series(t)

        # 3) align revenue onto monthly dates via forward-fill
        rev_m = rev.reindex(px.index, method="ffill")

        # 4) shares (direct or implied)
        shares, shares_method = get_shares_outstanding(t, latest_price)

        # 5) market cap & P/S
        df = px.copy()
        df["revenue"] = rev_m
        df["market_cap"] = df["Adj Close"] * shares
        df["PS"] = df["market_cap"] / df["revenue"]

        # drop unusable rows (early months before first revenue observation)
        df = df.dropna(subset=["PS"])

        # 6) save
        out = df.reset_index()[["Date", "Adj Close", "revenue", "market_cap", "PS"]]
        out_path = OUT_DIR / f"{t}_ps_monthly.csv"
        out.to_csv(out_path, index=False)

        summary_rows.append({
            "ticker": t,
            "rows_saved": len(out),
            "rev_mode": rev_mode,
            "rev_points": int(len(rev)),
            "shares_method": shares_method,
            "shares_used": shares
        })

        print(f"{t}: saved {len(out)} rows | revenue={rev_mode} ({len(rev)} pts) | shares={shares_method}")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "ps_build_summary.csv", index=False)
    print("\nSaved summary ->", OUT_DIR / "ps_build_summary.csv")
    print(summary)


if __name__ == "__main__":
    main()
