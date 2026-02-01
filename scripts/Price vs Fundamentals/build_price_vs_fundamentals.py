from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd


# -----------------------------
# Robust project-root resolution
# -----------------------------
def get_project_root() -> Path:
    """
    This script lives at: <root>/Price vs Fundamentals/build_price_vs_fundamentals.py
    So project root is two levels up from this file.
    """
    return Path(__file__).resolve().parent.parent


# -----------------------------
# Column helpers
# -----------------------------
def pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Pick first matching column (case-insensitive)."""
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def to_month_start(x: pd.Series) -> pd.Series:
    return pd.to_datetime(x, errors="coerce").dt.to_period("M").dt.to_timestamp()


def safe_pos(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    return x.where(x > 0, np.nan)


# -----------------------------
# Load PE monthly and normalize
# -----------------------------
def load_pe_monthly(path: Path) -> pd.DataFrame:
    """
    Normalizes to: date, price, pe, eps_implied
    where eps_implied = price / pe (only when price>0 and pe>0).
    """
    df = pd.read_csv(path)

    date_col = pick_col(df, ["date", "month", "timestamp", "time"])
    if not date_col:
        raise ValueError(f"{path.name}: couldn't find a Date column.")

    df["date"] = to_month_start(df[date_col])

    price_col = pick_col(df, ["adj close", "adj_close", "adjclose", "price", "close"])
    if not price_col:
        raise ValueError(f"{path.name}: couldn't find Adj Close / price column.")
    df["price"] = safe_pos(df[price_col])

    pe_col = pick_col(df, ["pe", "p/e", "pe_ratio", "trailing_pe", "pe_ttm"])
    if not pe_col:
        raise ValueError(f"{path.name}: couldn't find PE column.")
    df["pe"] = safe_pos(df[pe_col])

    out = df[["date", "price", "pe"]].copy()
    out = out.dropna(subset=["date"]).sort_values("date")

    # monthly de-dup (keep last observation in month)
    out = out.groupby("date", as_index=False).last()

    # implied EPS (positive only)
    out["eps"] = safe_pos(out["price"] / out["pe"])

    return out


# -----------------------------
# Decomposition
# -----------------------------
def decompose_endpoints(panel: pd.DataFrame) -> Dict[str, Any]:
    """
    Identity: Price = PE * EPS (EPS implied from Price/PE).
    ln(P1/P0) = ln(EPS1/EPS0) + ln(PE1/PE0)
    """
    p = panel.dropna(subset=["price", "pe", "eps"]).copy()
    if len(p) < 2:
        raise ValueError("Not enough valid rows after cleaning (need positive price, pe, eps).")

    s = p.iloc[0]
    e = p.iloc[-1]

    p0, p1 = float(s["price"]), float(e["price"])
    eps0, eps1 = float(s["eps"]), float(e["eps"])
    pe0, pe1 = float(s["pe"]), float(e["pe"])

    log_price = float(np.log(p1 / p0))
    log_eps = float(np.log(eps1 / eps0))
    log_mult = float(np.log(pe1 / pe0))

    # Shares can be unstable when log_price ~ 0. Keep them, but caller should decide whether to use.
    denom = log_price if abs(log_price) > 1e-12 else np.nan
    share_eps = (log_eps / denom) if np.isfinite(denom) else np.nan
    share_mult = (log_mult / denom) if np.isfinite(denom) else np.nan

    return {
        "start": pd.Timestamp(s["date"]),
        "end": pd.Timestamp(e["date"]),
        "log_price": log_price,
        "log_eps": log_eps,
        "log_multiple": log_mult,
        "share_eps_pct": float(100 * share_eps) if pd.notna(share_eps) else np.nan,
        "share_multiple_pct": float(100 * share_mult) if pd.notna(share_mult) else np.nan,
    }


def rolling_decomposition(
    panel: pd.DataFrame,
    ticker: str,
    window_months: int,
    min_abs_log_price: float,
    share_cap_pct: float,
) -> pd.DataFrame:
    """
    Produces a rolling time series that is *safe*:
    - Always outputs raw log components (stable).
    - Outputs shares only when |log_price| >= min_abs_log_price (otherwise shares are NaN).
    - Caps shares to +/- share_cap_pct (for safety when denom is small but not filtered).
    """
    p = panel.dropna(subset=["price", "pe", "eps"]).copy().sort_values("date").reset_index(drop=True)
    rows = []

    for i in range(window_months, len(p)):
        sub = p.iloc[i - window_months : i + 1]
        try:
            d = decompose_endpoints(sub)
        except Exception:
            continue

        log_price = d["log_price"]
        log_eps = d["log_eps"]
        log_mult = d["log_multiple"]

        stable = np.isfinite(log_price) and (abs(log_price) >= min_abs_log_price)

        share_eps_pct = np.nan
        share_mult_pct = np.nan
        if stable:
            share_eps_pct = float(d["share_eps_pct"])
            share_mult_pct = float(d["share_multiple_pct"])

            # hard cap (prevents ugly explosions in plots)
            share_eps_pct = float(np.clip(share_eps_pct, -share_cap_pct, share_cap_pct))
            share_mult_pct = float(np.clip(share_mult_pct, -share_cap_pct, share_cap_pct))

        rows.append(
            {
                "ticker": ticker,
                "start": d["start"],
                "end": d["end"],
                "log_price": log_price,
                "log_eps": log_eps,
                "log_multiple": log_mult,
                "stable_for_shares": bool(stable),
                "share_eps_pct": share_eps_pct,
                "share_multiple_pct": share_mult_pct,
            }
        )

    return pd.DataFrame(rows)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Price vs Fundamentals decomposition using existing data/valuation/<TICKER>_pe_monthly.csv files."
    )
    parser.add_argument("--tickers", nargs="*", default=["NVDA", "AMD", "TSM", "ASML", "AVGO"])
    parser.add_argument(
        "--valuation_dir",
        default="data/valuation",
        help="Relative to project root. Contains <TICKER>_pe_monthly.csv",
    )
    parser.add_argument(
        "--out_dir",
        default="Price vs Fundamentals/data/decomposition",
        help="Relative to project root. Output folder.",
    )
    parser.add_argument("--start", default=None, help="Optional start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="Optional end date YYYY-MM-DD")

    # Rolling options
    parser.add_argument("--rolling_months", type=int, default=24, help="Rolling window length in months")
    parser.add_argument(
        "--min_abs_log_price",
        type=float,
        default=0.05,
        help="Minimum |log_price| in a rolling window to compute % shares. "
        "0.05 ~ about 5% log return (~5.1% simple return).",
    )
    parser.add_argument(
        "--share_cap_pct",
        type=float,
        default=200.0,
        help="Caps share percentages to +/- this value (safety for plots).",
    )

    # Debug / exports
    parser.add_argument(
        "--write_panels",
        action="store_true",
        help="Write cleaned per-ticker panel (date, price, pe, eps_implied).",
    )
    parser.add_argument(
        "--write_diagnostics",
        action="store_true",
        help="Write per-ticker diagnostics (how many rows dropped, date coverage).",
    )

    args = parser.parse_args()

    root = get_project_root()
    val_dir = (root / args.valuation_dir).resolve()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Project root: {root}")
    print(f"[INFO] Using valuation_dir: {val_dir}")
    print(f"[INFO] Writing outputs to: {out_dir}")

    start_dt = pd.to_datetime(args.start) if args.start else None
    end_dt = pd.to_datetime(args.end) if args.end else None

    summary_rows = []
    rolling_frames = []
    diagnostics_rows = []

    for t in args.tickers:
        pe_path = val_dir / f"{t}_pe_monthly.csv"
        if not pe_path.exists():
            print(f"[WARN] {t}: missing {pe_path}")
            continue

        try:
            raw = pd.read_csv(pe_path)
            panel = load_pe_monthly(pe_path)
        except Exception as e:
            print(f"[WARN] {t}: failed to load/normalize: {e}")
            continue

        if start_dt is not None:
            panel = panel[panel["date"] >= start_dt]
        if end_dt is not None:
            panel = panel[panel["date"] <= end_dt]

        # diagnostics
        if args.write_diagnostics:
            n_raw = len(raw)
            n_panel = len(panel)
            n_valid = int(panel.dropna(subset=["price", "pe", "eps"]).shape[0])
            d0 = panel["date"].min()
            d1 = panel["date"].max()
            diagnostics_rows.append(
                {
                    "ticker": t,
                    "raw_rows": n_raw,
                    "monthly_rows": n_panel,
                    "valid_rows_price_pe_eps": n_valid,
                    "start_date": d0.date().isoformat() if pd.notna(d0) else None,
                    "end_date": d1.date().isoformat() if pd.notna(d1) else None,
                }
            )

        try:
            d = decompose_endpoints(panel)
        except Exception as e:
            print(f"[WARN] {t}: endpoint decomposition failed: {e}")
            continue

        summary_rows.append(
            {
                "ticker": t,
                "start": d["start"].date().isoformat(),
                "end": d["end"].date().isoformat(),
                "log_price": d["log_price"],
                "log_eps": d["log_eps"],
                "log_multiple": d["log_multiple"],
                # shares (can be NaN if log_price ~ 0, but endpoint is usually fine)
                "share_eps_pct": d["share_eps_pct"],
                "share_multiple_pct": d["share_multiple_pct"],
            }
        )

        # Print with guard for NaN formatting
        eps_share = d["share_eps_pct"]
        mult_share = d["share_multiple_pct"]
        eps_share_s = f"{eps_share:.1f}%" if np.isfinite(eps_share) else "NaN"
        mult_share_s = f"{mult_share:.1f}%" if np.isfinite(mult_share) else "NaN"

        print(
            f"[OK] {t}: {d['start'].date()} -> {d['end'].date()} | "
            f"EPS share={eps_share_s} | Multiple share={mult_share_s}"
        )

        roll = rolling_decomposition(
            panel=panel,
            ticker=t,
            window_months=args.rolling_months,
            min_abs_log_price=args.min_abs_log_price,
            share_cap_pct=args.share_cap_pct,
        )
        if not roll.empty:
            rolling_frames.append(roll)

        if args.write_panels:
            panel.to_csv(out_dir / f"{t}_panel_clean.csv", index=False)

    if not summary_rows:
        print("\n[ERROR] No tickers processed.")
        print("        Check that data/valuation contains <TICKER>_pe_monthly.csv")
        return

    # Write endpoint summary
    summary = pd.DataFrame(summary_rows).sort_values("ticker")
    summary_path = out_dir / "decomposition_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nWrote: {summary_path}")

    # Write rolling time series (always includes raw log components; shares only when stable_for_shares=True)
    if rolling_frames:
        rolling = pd.concat(rolling_frames, ignore_index=True)
        rolling_path = out_dir / "decomposition_timeseries.csv"
        rolling.to_csv(rolling_path, index=False)
        print(f"Wrote: {rolling_path}")
        print(
            f"[INFO] Rolling shares computed only when |log_price| >= {args.min_abs_log_price} "
            f"(see stable_for_shares column)."
        )

    # Optional diagnostics
    if args.write_diagnostics and diagnostics_rows:
        diag = pd.DataFrame(diagnostics_rows).sort_values("ticker")
        diag_path = out_dir / "decomposition_diagnostics.csv"
        diag.to_csv(diag_path, index=False)
        print(f"Wrote: {diag_path}")


if __name__ == "__main__":
    main()
