from pathlib import Path
import pandas as pd
DATA_DIR = Path(__file__).parent / "data"
RAW_CSV = DATA_DIR / "koi_cumulative.csv"
CLEAN_CSV = DATA_DIR / "cleaned_data.csv"
MISSING_THRESHOLD = 0.50  

def main() -> None:
    df = pd.read_csv(RAW_CSV, comment="#")
    print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]:,} columns")
    missing_frac = df.isnull().sum() / len(df)
    drop_cols = missing_frac[missing_frac > MISSING_THRESHOLD].index.tolist()
    keep_cols = missing_frac[missing_frac <= MISSING_THRESHOLD].index.tolist()
    print(f"\nThreshold     : >{MISSING_THRESHOLD*100:.0f}% missing")
    print(f"Columns dropped: {len(drop_cols)}")
    print(f"Columns kept   : {len(keep_cols)}")
    if drop_cols:
        print("\nDropped columns:")
        for col in drop_cols:
            pct = missing_frac[col] * 100
            print(f"  - {col}  ({pct:.1f}% missing)")
    df_clean = df[keep_cols]
    df_clean.to_csv(CLEAN_CSV, index=False)
    print(f"\nSaved cleaned data -> {CLEAN_CSV}")
    print(f"Final shape: {df_clean.shape[0]:,} rows x {df_clean.shape[1]:,} columns")
if __name__ == "__main__":
    main()
