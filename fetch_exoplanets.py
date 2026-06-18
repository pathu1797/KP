import io
import sys
import textwrap
from pathlib import Path
import pandas as pd
import requests
TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
QUERY = "SELECT * FROM cumulative"
DATA_DIR = Path(__file__).parent / "data"
CSV_CACHE = DATA_DIR / "koi_cumulative.csv"
def download_koi_dataset() -> pd.DataFrame:
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
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CSV_CACHE.write_bytes(resp.content)
    print(f"[OK] Saved {len(resp.content):,} bytes -> {CSV_CACHE}")
    return pd.read_csv(io.StringIO(resp.text), comment="#")
def print_header(title: str, width: int = 72) -> None:
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)
def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
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
    report = report.sort_values("Missing %", ascending=False).reset_index(drop=True)
    return report
def main() -> None:
    df = download_koi_dataset()
    print_header("Dataset Overview")
    print(f"  Rows   : {df.shape[0]:,}")
    print(f"  Columns: {df.shape[1]:,}")
    print_header("First 5 Rows")
    with pd.option_context(
        "display.max_columns", None,
        "display.width", 200,
        "display.max_colwidth", 30,
    ):
        print(df.head().to_string(index=False))
    print_header("Column Data Types")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"  {str(dtype):>12s} : {count}")
    print_header("Missing Values Summary")
    report = missing_value_report(df)
    complete = report[report["Missing Count"] == 0]
    incomplete = report[report["Missing Count"] > 0]
    print(f"\n  Columns with NO missing values : {len(complete)}")
    print(f"  Columns WITH missing values   : {len(incomplete)}")
    print(f"  Total cells in dataset        : {df.shape[0] * df.shape[1]:,}")
    total_missing = int(report["Missing Count"].sum())
    pct_total = round(total_missing / (df.shape[0] * df.shape[1]) * 100, 2)
    print(f"  Total missing cells           : {total_missing:,} ({pct_total}%)")
    print_header("Per-Column Missing Values (sorted by % missing)")
    with pd.option_context("display.max_rows", None, "display.width", 140):
        print(report.to_string(index=False))
    report_path = DATA_DIR / "missing_values_report.csv"
    report.to_csv(report_path, index=False)
    print(f"\n[OK] Report saved -> {report_path}")
if __name__ == "__main__":
    main()