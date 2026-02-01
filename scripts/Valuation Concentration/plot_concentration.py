# plot_concentration.py
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve()

def find_root():
    for p in [HERE.parent, *HERE.parents]:
        if (p / "data" / "concentration").exists():
            return p
    raise RuntimeError("Project root not found (expected data/concentration/)")

ROOT = find_root()
DATA = ROOT / "data" / "concentration" / "market_concentration.csv"
FIG_DIR = ROOT / "figures" / "concentration"
FIG_DIR.mkdir(parents=True, exist_ok=True)

def pct(x):
    return x * 100

def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Missing {DATA} (run build_concentration.py first)")

    df = pd.read_csv(DATA, index_col=0, parse_dates=True).sort_index()

    # ---- Plot 1: S&P 500 concentration (top 5 / top 10) ----
    plt.figure(figsize=(12, 6))
    if "sp500_top5" in df.columns:
        plt.plot(df.index, pct(df["sp500_top5"]), label="S&P 500 Top 5")
    if "sp500_top10" in df.columns:
        plt.plot(df.index, pct(df["sp500_top10"]), label="S&P 500 Top 10")

    plt.title("S&P 500 Market Capitalisation Concentration Over Time")
    plt.ylabel("Share of Total Market Cap (%)")
    plt.xlabel("Year")
    plt.legend()
    plt.tight_layout()
    out1 = FIG_DIR / "sp500_concentration.png"
    plt.savefig(out1, dpi=200)
    plt.close()
    print(f"Saved {out1}")

    # ---- Plot 2: Semiconductor universe concentration (top 2/3/5/10) ----
    plt.figure(figsize=(12, 6))
    for col, name in [
        ("semi_top2", "Semis Top 2"),
        ("semi_top3", "Semis Top 3"),
        ("semi_top5", "Semis Top 5"),
        ("semi_top10", "Semis Top 10"),
    ]:
        if col in df.columns:
            plt.plot(df.index, pct(df[col]), label=name)

    plt.title("Semiconductor Universe Market Capitalisation Concentration Over Time")
    plt.ylabel("Share of Total Market Cap (%)")
    plt.xlabel("Year")
    plt.legend()
    plt.tight_layout()
    out2 = FIG_DIR / "semis_concentration.png"
    plt.savefig(out2, dpi=200)
    plt.close()
    print(f"Saved {out2}")

if __name__ == "__main__":
    main()
