from pathlib import Path
import pandas as pd

in_dir = Path("../data/prices")
out_dir = Path("../data/processed")
out_dir.mkdir(parents=True, exist_ok=True)

tickers = ["NVDA", "AMD", "TSM", "ASML", "AVGO"]

all_series = []

for t in tickers:
    f = in_dir / f"{t}_monthly_prices.csv"
    df = pd.read_csv(f, parse_dates=["Date"])

    df = df.sort_values("Date").set_index("Date")

    # Use Adj Close for splits/dividends
    px = df["Adj Close"].rename(t)

    # Monthly returns
    ret = px.pct_change().rename(f"{t}_ret")

    # Drawdown
    running_max = px.cummax()
    dd = (px / running_max - 1.0).rename(f"{t}_drawdown")

    out = pd.concat([px, ret, dd], axis=1)
    out.to_csv(out_dir / f"{t}_processed_monthly.csv")

    all_series.append(px)

# Combined price panel (useful for plotting/normalising)
panel = pd.concat(all_series, axis=1).dropna(how="all")
panel.to_csv(out_dir / "prices_panel_adjclose.csv")

# Normalised (start=100) panel for easy comparison
norm = (panel / panel.iloc[0]) * 100
norm.to_csv(out_dir / "prices_panel_normalised_100.csv")

print("Saved processed files to:", out_dir)
print("Created:", "prices_panel_adjclose.csv", "prices_panel_normalised_100.csv")
