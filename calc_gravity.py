"""
Estimate exoplanet surface gravity from the Kepler KOI dataset.

Uses a piecewise mass–radius relation (Chen & Kipping 2017) to estimate
planet mass from koi_prad, then computes surface gravity:

    g = (M / R^2) * g_Earth

All intermediate quantities are in Earth units; the final surface gravity
column is in m/s^2.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
CLEAN_CSV = DATA_DIR / "cleaned_data.csv"
OUT_CSV = DATA_DIR / "gravity_data.csv"

# ── Physical constants (Earth units) ─────────────────────────────────────────
G_EARTH = 9.81  # m/s^2

# Chen & Kipping (2017) power-law break-points (in Earth radii)
R_TERRAN_MAX = 1.23   # Terran  -> Neptunian transition
R_NEPTUNE_MAX = 14.26  # Neptunian -> Jovian transition

# Terran exponent
ALPHA_TERRAN = 3.58

# Neptunian exponent
ALPHA_NEPTUNE = 1.70

# Jovian exponent (mass nearly flat with radius)
ALPHA_JOVIAN = 0.01

# Ensure continuity at each break-point
_C_NEPTUNE = (R_TERRAN_MAX ** ALPHA_TERRAN) / (R_TERRAN_MAX ** ALPHA_NEPTUNE)
_M_AT_NJ = _C_NEPTUNE * (R_NEPTUNE_MAX ** ALPHA_NEPTUNE)
_C_JOVIAN = _M_AT_NJ / (R_NEPTUNE_MAX ** ALPHA_JOVIAN)


# ── Mass–radius relation ────────────────────────────────────────────────────
def estimate_mass_earth(radius_earth: float) -> float:
    """Return estimated mass in Earth masses from radius in Earth radii.

    Piecewise power-law (Chen & Kipping 2017, ApJ 834 17):
      Terran   (R < 1.23 R_E) : M = R^3.58
      Neptunian (1.23-14.26)  : M = C1 * R^1.70   (C1 for continuity)
      Jovian   (R >= 14.26)   : M = C2 * R^0.01   (C2 for continuity)
    """
    if np.isnan(radius_earth) or radius_earth <= 0:
        return np.nan

    if radius_earth < R_TERRAN_MAX:
        return radius_earth ** ALPHA_TERRAN
    elif radius_earth < R_NEPTUNE_MAX:
        return _C_NEPTUNE * radius_earth ** ALPHA_NEPTUNE
    else:
        return _C_JOVIAN * radius_earth ** ALPHA_JOVIAN


def surface_gravity(mass_earth: float, radius_earth: float) -> float:
    """Return surface gravity in m/s^2.

    g = (M/M_E) / (R/R_E)^2  *  g_Earth
    """
    if np.isnan(mass_earth) or np.isnan(radius_earth) or radius_earth <= 0:
        return np.nan
    return (mass_earth / radius_earth ** 2) * G_EARTH


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    df = pd.read_csv(CLEAN_CSV)
    print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]:,} columns")

    # Estimate mass (Earth masses)
    df["est_mass_earth"] = df["koi_prad"].apply(estimate_mass_earth)

    # Compute surface gravity (m/s^2)
    df["surface_gravity_ms2"] = df.apply(
        lambda row: surface_gravity(row["est_mass_earth"], row["koi_prad"]),
        axis=1,
    )

    # Round for readability
    df["est_mass_earth"] = df["est_mass_earth"].round(4)
    df["surface_gravity_ms2"] = df["surface_gravity_ms2"].round(4)

    # Save
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved -> {OUT_CSV}")
    print(f"Final shape: {df.shape[0]:,} rows x {df.shape[1]:,} columns")

    # Show first 5 rows for the key columns
    cols = ["kepid", "kepoi_name", "koi_prad", "est_mass_earth", "surface_gravity_ms2"]
    print(f"\nFirst 5 rows (key columns):\n")
    print(df[cols].head().to_string(index=False))

    # Quick stats on the new column
    sg = df["surface_gravity_ms2"].dropna()
    print(f"\nSurface gravity stats (m/s^2):")
    print(f"  Count  : {len(sg):,}")
    print(f"  Min    : {sg.min():.2f}")
    print(f"  Median : {sg.median():.2f}")
    print(f"  Mean   : {sg.mean():.2f}")
    print(f"  Max    : {sg.max():.2f}")
    print(f"  NaN    : {df['surface_gravity_ms2'].isna().sum():,}")


if __name__ == "__main__":
    main()
