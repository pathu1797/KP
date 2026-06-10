"""
NASA Exoplanet Archive — Kepler Objects of Interest (KOI) Dataset
=================================================================
Downloads the KOI table from the NASA Exoplanet Archive TAP service,
loads it into a Pandas DataFrame, prints the first 5 rows, and
generates a comprehensive missing-value summary report.

API docs: https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html
"""

import io
import sys
import textwrap
from pathlib import Path

import pandas as pd
import requests

# ── Configuration ────────────────────────────────────────────────────────────
TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

# Query the cumulative KOI table.  Change the table name to "ps" if you'd
# rather pull the Planetary Systems (confirmed planets) table instead.
QUERY = "SELECT * FROM cumulative"

DATA_DIR = Path(__file__).parent / "data"
CSV_CACHE = DATA_DIR / "koi_cumulative.csv"


# ── Helpers ──────────────────────────────────────────────────────────────────
def download_koi_dataset() -> pd.DataFrame:
    """Download the KOI cumulative table via TAP and return a DataFrame.

    The raw CSV is cached locally so subsequent runs skip the download.
    """
    if CSV_CACHE.exists():
        print(f"[OK] Using cached dataset: {CSV_CACHE}")
        return pd.read_csv(CSV_CACHE, comment="#")

    print("[..] Downloading KOI dataset from NASA Exoplanet Archive ...")
    params = {
        "query": QUERY,
        "format": "csv",
    }

    try:
        resp = requests.get(TAP_URL, params=params, timeout=120)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"[!!] Download failed: {exc}", file=sys.stderr)
        sys.exit(1)

    # Persist to disk
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CSV_CACHE.write_bytes(resp.content)
    print(f"[OK] Saved {len(resp.content):,} bytes -> {CSV_CACHE}")

    # The TAP CSV sometimes starts with comment lines (#); skip them.
    return pd.read_csv(io.StringIO(resp.text), comment="#")


def print_header(title: str, width: int = 72) -> None:
    """Pretty-print a section header."""
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Build a summary DataFrame of missing values per column."""
    total = len(df)
    missing = df.isnull().sum()
    pct = (missing / total * 100).round(2)

    report = pd.DataFrame({
        "Column": df.columns,
        "Dtype": [str(df[c].dtype) for c in df.columns],
        "Non-Null Count": df.notnull().sum().values,
        "Missing Count": missing.values,
        "Missing %": pct.values,
    })

    # Sort so the most-missing columns appear first
    report = report.sort_values("Missing %", ascending=False).reset_index(drop=True)
    return report


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    df = download_koi_dataset()

    # ── 1. Shape overview ────────────────────────────────────────────────
    print_header("Dataset Overview")
    print(f"  Rows   : {df.shape[0]:,}")
    print(f"  Columns: {df.shape[1]:,}")

    # ── 2. First 5 rows ─────────────────────────────────────────────────
    print_header("First 5 Rows")
    # Use to_string so wide tables aren't truncated in the terminal
    with pd.option_context(
        "display.max_columns", None,
        "display.width", 200,
        "display.max_colwidth", 30,
    ):
        print(df.head().to_string(index=False))

    # ── 3. Column dtypes ─────────────────────────────────────────────────
    print_header("Column Data Types")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  {str(dtype):>12s} : {count}")

    # ── 4. Missing-value report ──────────────────────────────────────────
    print_header("Missing Values Summary")
    report = missing_value_report(df)

    # Columns with zero missing values
    complete = report[report["Missing Count"] == 0]
    incomplete = report[report["Missing Count"] > 0]

    print(f"\n  Columns with NO missing values : {len(complete)}")
    print(f"  Columns WITH missing values   : {len(incomplete)}")
    print(f"  Total cells in dataset        : {df.shape[0] * df.shape[1]:,}")
    total_missing = int(report["Missing Count"].sum())
    pct_total = round(total_missing / (df.shape[0] * df.shape[1]) * 100, 2)
    print(f"  Total missing cells           : {total_missing:,} ({pct_total}%)")

    # Print the full table
    print_header("Per-Column Missing Values (sorted by % missing)")
    with pd.option_context("display.max_rows", None, "display.width", 140):
        print(report.to_string(index=False))

    # ── 5. Save report to CSV ────────────────────────────────────────────
    report_path = DATA_DIR / "missing_values_report.csv"
    report.to_csv(report_path, index=False)
    print(f"\n[OK] Report saved -> {report_path}")


if __name__ == "__main__":
    main()
