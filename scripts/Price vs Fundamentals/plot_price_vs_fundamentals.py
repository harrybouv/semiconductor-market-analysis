from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_project_root() -> Path:
    # this file lives in <root>/Price vs Fundamentals/
    return Path(__file__).resolve().parent.parent


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Endpoint plot (keep as-is, but add sanity)
# -----------------------------
def plot_endpoint(summary_csv: Path, out_png: Path) -> None:
    df = pd.read_csv(summary_csv).sort_values("ticker").reset_index(drop=True)

    x = np.arange(len(df))
    eps = df["share_eps_pct"].astype(float)
    mult = df["share_multiple_pct"].astype(float)

    fig = plt.figure()
    plt.bar(x - 0.2, eps, width=0.4, label="Fundamentals (EPS) share %")
    plt.bar(x + 0.2, mult, width=0.4, label="Multiple (P/E) share %")
    plt.axhline(0)

    plt.xticks(x, df["ticker"])
    plt.ylabel("Share of total log price return (%)")
    plt.title("Price vs Fundamentals Decomposition (Endpoint)")
    plt.legend()
    plt.tight_layout()

    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# -----------------------------
# Rolling plots (NEW: plot stable log contributions, not % shares)
# -----------------------------
def _load_timeseries(timeseries_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(timeseries_csv)
    df["start"] = pd.to_datetime(df["start"], errors="coerce")
    df["end"] = pd.to_datetime(df["end"], errors="coerce")

    # Ensure numeric
    for c in ["log_price", "log_eps", "log_multiple", "share_eps_pct", "share_multiple_pct"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # stable_for_shares may not exist if you haven't updated build script yet
    if "stable_for_shares" in df.columns:
        # accept 0/1 or True/False strings
        df["stable_for_shares"] = df["stable_for_shares"].astype(str).str.lower().isin(["true", "1", "yes"])

    return df


def plot_rolling_contributions_all(df: pd.DataFrame, out_png: Path) -> None:
    """
    For each ticker, plot the rolling log contribution from multiple:
      log_multiple = ln(PE_t / PE_{t-window})
    This is stable and interpretable (tailwind>0, headwind<0).
    """
    if df.empty:
        raise ValueError("No rolling decomposition rows found.")

    fig = plt.figure()
    for tkr in sorted(df["ticker"].dropna().unique()):
        sub = df[df["ticker"] == tkr].sort_values("end")
        plt.plot(sub["end"], sub["log_multiple"], label=tkr)

    plt.axhline(0)
    plt.ylabel("Rolling valuation contribution: ln(PE_t / PE_{t-window})")
    plt.title("Rolling Valuation Contribution (All tickers)")
    plt.legend()
    plt.tight_layout()

    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_rolling_contributions_one(df: pd.DataFrame, out_png: Path, ticker: str) -> None:
    """
    For a single ticker, plot rolling log contributions:
      - log_eps (fundamentals)
      - log_multiple (valuation)
      - log_price (sum; sanity check)
    """
    sub = df[df["ticker"].str.upper() == ticker.upper()].copy()
    if sub.empty:
        raise ValueError(f"No rolling rows found for ticker={ticker}.")

    sub = sub.sort_values("end")

    fig = plt.figure()
    plt.plot(sub["end"], sub["log_eps"], label="Fundamentals: ln(EPS_t / EPS_{t-window})")
    plt.plot(sub["end"], sub["log_multiple"], label="Valuation: ln(PE_t / PE_{t-window})")
    plt.plot(sub["end"], sub["log_price"], label="Total: ln(P_t / P_{t-window})")
    plt.axhline(0)
    plt.ylabel("Rolling log contribution")
    plt.title(f"Rolling Decomposition (Contributions) — {ticker.upper()}")
    plt.legend()
    plt.tight_layout()

    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_rolling_shares_diagnostic_all(df: pd.DataFrame, out_png: Path) -> None:
    """
    Diagnostic-only: plot multiple-share % for windows marked stable_for_shares.
    If stable_for_shares doesn't exist, fall back to dropping extreme NaNs only.
    """
    if df.empty:
        raise ValueError("No rolling decomposition rows found.")

    fig = plt.figure()

    has_stable = "stable_for_shares" in df.columns
    for tkr in sorted(df["ticker"].dropna().unique()):
        sub = df[df["ticker"] == tkr].sort_values("end").copy()

        if has_stable:
            sub.loc[~sub["stable_for_shares"], "share_multiple_pct"] = np.nan

        plt.plot(sub["end"], sub["share_multiple_pct"], label=tkr)

    plt.axhline(0)
    plt.ylabel("Multiple share of log return (%)")
    plt.title("Rolling Multiple Share (%) — Diagnostic (stable windows only)")
    plt.legend()
    plt.tight_layout()

    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot price vs fundamentals decomposition outputs.")
    parser.add_argument(
        "--decomp_dir",
        default="Price vs Fundamentals/data/decomposition",
        help="Relative to project root.",
    )
    parser.add_argument(
        "--fig_dir",
        default="Price vs Fundamentals/figures",
        help="Relative to project root.",
    )
    parser.add_argument(
        "--ticker",
        default=None,
        help="Optional ticker for a per-ticker rolling plot (e.g., NVDA).",
    )
    parser.add_argument(
        "--diagnostic_shares",
        action="store_true",
        help="Also write a diagnostic rolling % share plot (stable windows only). Not for headline use.",
    )
    args = parser.parse_args()

    root = get_project_root()
    decomp_dir = (root / args.decomp_dir).resolve()
    fig_dir = (root / args.fig_dir).resolve()
    ensure_dir(fig_dir)

    summary_csv = decomp_dir / "decomposition_summary.csv"
    ts_csv = decomp_dir / "decomposition_timeseries.csv"

    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing {summary_csv} (run build script first).")
    if not ts_csv.exists():
        raise FileNotFoundError(f"Missing {ts_csv} (run build script first).")

    # 1) Endpoint plot
    plot_endpoint(summary_csv, fig_dir / "decomposition_endpoint.png")

    # 2) Rolling contributions (headline-safe)
    ts = _load_timeseries(ts_csv)
    plot_rolling_contributions_all(ts, fig_dir / "decomposition_rolling_valuation_contribution.png")

    # 3) Optional per-ticker rolling contributions (headline-safe)
    if args.ticker:
        plot_rolling_contributions_one(ts, fig_dir / f"decomposition_rolling_{args.ticker.upper()}_contrib.png", args.ticker)

    # 4) Optional diagnostic shares plot (only stable windows)
    if args.diagnostic_shares:
        plot_rolling_shares_diagnostic_all(ts, fig_dir / "decomposition_rolling_multiple_share_DIAGNOSTIC.png")

    print(f"[OK] Wrote figures to: {fig_dir}")


if __name__ == "__main__":
    main()
