"""
predict_planet.py — Exoplanet Disposition Predictor
=====================================================
Loads the trained Random Forest model and predicts whether a planet
candidate is CONFIRMED, CANDIDATE, or FALSE POSITIVE.

Usage:
    from predict_planet import predict_planet

    result = predict_planet(
        koi_prad=1.0,          # Planet radius  (Earth radii)
        koi_teq=255,           # Equilibrium temperature (K)
        koi_period=365.25,     # Orbital period (days)
        koi_depth=84,          # Transit depth  (ppm)
        ...
    )
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np

MODEL_DIR = Path(__file__).parent / "data" / "model"

# ── Load model artifacts (once at import time) ──────────────────────────────
_clf = joblib.load(MODEL_DIR / "rf_classifier.joblib")
_imputer = joblib.load(MODEL_DIR / "imputer.joblib")
_le = joblib.load(MODEL_DIR / "label_encoder.joblib")
_feature_names = joblib.load(MODEL_DIR / "feature_names.joblib")
_medians = joblib.load(MODEL_DIR / "feature_medians.joblib")


# ── Chen & Kipping (2017) mass–radius relation ─────────────────────────────
R_TERRAN_MAX = 1.23
R_NEPTUNE_MAX = 14.26
ALPHA_TERRAN = 3.58
ALPHA_NEPTUNE = 1.70
ALPHA_JOVIAN = 0.01
_C_NEPTUNE = (R_TERRAN_MAX ** ALPHA_TERRAN) / (R_TERRAN_MAX ** ALPHA_NEPTUNE)
_M_AT_NJ = _C_NEPTUNE * (R_NEPTUNE_MAX ** ALPHA_NEPTUNE)
_C_JOVIAN = _M_AT_NJ / (R_NEPTUNE_MAX ** ALPHA_JOVIAN)
G_EARTH = 9.81  # m/s^2


def _estimate_mass(radius_earth: float) -> float:
    """Planet mass in Earth masses from radius in Earth radii."""
    if radius_earth <= 0 or np.isnan(radius_earth):
        return np.nan
    if radius_earth < R_TERRAN_MAX:
        return radius_earth ** ALPHA_TERRAN
    elif radius_earth < R_NEPTUNE_MAX:
        return _C_NEPTUNE * radius_earth ** ALPHA_NEPTUNE
    else:
        return _C_JOVIAN * radius_earth ** ALPHA_JOVIAN


def _surface_gravity(mass_earth: float, radius_earth: float) -> float:
    """Surface gravity in m/s^2."""
    if radius_earth <= 0 or np.isnan(radius_earth) or np.isnan(mass_earth):
        return np.nan
    return (mass_earth / radius_earth ** 2) * G_EARTH


# ── Main prediction function ────────────────────────────────────────────────
def predict_planet(**kwargs: Any) -> dict:
    """Predict the disposition of a planet candidate.

    Supply any subset of the model's 110 features as keyword arguments.
    Unspecified features default to the training-set median.

    Key parameters you'll typically want to set:
        koi_prad          – Planet radius (Earth radii)
        koi_teq           – Equilibrium temperature (K)
        koi_period        – Orbital period (days)
        koi_depth         – Transit depth (ppm)
        koi_duration      – Transit duration (hours)
        koi_impact        – Impact parameter
        koi_model_snr     – Model signal-to-noise ratio
        koi_score         – Disposition score (0–1)
        koi_steff         – Stellar effective temp (K)
        koi_slogg         – Stellar log(g) (cgs)
        koi_srad          – Stellar radius (solar radii)
        koi_smass         – Stellar mass  (solar masses)
        koi_fpflag_nt     – Not-transit-like flag (0 or 1)
        koi_fpflag_ss     – Stellar eclipse flag (0 or 1)
        koi_fpflag_co     – Centroid offset flag (0 or 1)
        koi_fpflag_ec     – Ephemeris match flag (0 or 1)

    Returns a dict with:
        prediction   – "CONFIRMED", "CANDIDATE", or "FALSE POSITIVE"
        probabilities – dict of class -> probability
        est_mass     – estimated mass (Earth masses)
        surface_grav – estimated surface gravity (m/s^2)
    """

    # Start with training medians as defaults
    row = dict(_medians)

    # Override with user-supplied values
    for key, val in kwargs.items():
        if key in row:
            row[key] = float(val)
        elif key not in ("est_mass_earth", "surface_gravity_ms2"):
            print(f"  [warning] Unknown feature '{key}' — ignored.")

    # ── Auto-compute physics columns from koi_prad ──────────────────────
    radius = row.get("koi_prad", _medians.get("koi_prad", 1.0))
    mass = _estimate_mass(radius)
    grav = _surface_gravity(mass, radius)

    row["est_mass_earth"] = mass
    row["surface_gravity_ms2"] = grav

    # ── Build feature vector in the correct column order ────────────────
    feature_vector = np.array([[row[f] for f in _feature_names]])

    # Predict
    pred_idx = _clf.predict(feature_vector)[0]
    probas = _clf.predict_proba(feature_vector)[0]
    label = _le.inverse_transform([pred_idx])[0]

    prob_dict = {cls: round(float(p), 4) for cls, p in zip(_le.classes_, probas)}

    return {
        "prediction": label,
        "probabilities": prob_dict,
        "est_mass_earth": round(mass, 4),
        "surface_gravity_ms2": round(grav, 4),
    }


# ── Pretty printer ──────────────────────────────────────────────────────────
def print_prediction(result: dict, title: str = "Prediction") -> None:
    """Print a prediction result in a clean, readable format."""
    w = 56
    print()
    print("=" * w)
    print(f"  {title}")
    print("=" * w)

    label = result["prediction"]
    if label == "CONFIRMED":
        status = "CONFIRMED PLANET"
    elif label == "CANDIDATE":
        status = "CANDIDATE"
    else:
        status = "FALSE POSITIVE"

    print(f"\n  Predicted Status : {status}")
    print(f"  Est. Mass        : {result['est_mass_earth']:.2f} Earth masses")
    print(f"  Surface Gravity  : {result['surface_gravity_ms2']:.2f} m/s^2")

    print(f"\n  Confidence breakdown:")
    for cls, prob in result["probabilities"].items():
        bar = "#" * int(prob * 30)
        print(f"    {cls:<16s}  {prob:6.2%}  {bar}")
    print("=" * w)


# ── Demo: test with Earth's real values ─────────────────────────────────────
if __name__ == "__main__":

    # ── Earth ────────────────────────────────────────────────────────────
    earth = predict_planet(
        koi_prad=1.0,           # 1 Earth radius
        koi_teq=255,            # ~255 K equilibrium temperature
        koi_period=365.25,      # 1 year
        koi_depth=84,           # ~84 ppm transit depth (Earth transiting Sun)
        koi_duration=13.0,      # ~13 hour transit duration
        koi_impact=0.0,         # central transit
        koi_model_snr=30.0,     # good SNR
        koi_score=1.0,          # high-confidence detection
        koi_steff=5778,         # Sun's effective temperature
        koi_slogg=4.44,         # Sun's log(g)
        koi_srad=1.0,           # 1 solar radius
        koi_smass=1.0,          # 1 solar mass
        koi_fpflag_nt=0,        # no false-positive flags
        koi_fpflag_ss=0,
        koi_fpflag_co=0,
        koi_fpflag_ec=0,
        koi_insol=1.0,          # 1 Earth insolation
        koi_smet=0.0,           # solar metallicity
        koi_count=1,            # 1 planet in system
    )
    print_prediction(earth, title="Earth Verification Test")

    # ── Hot Jupiter (for contrast) ───────────────────────────────────────
    hot_jup = predict_planet(
        koi_prad=12.0,          # Jupiter-sized
        koi_teq=1800,           # very hot
        koi_period=2.5,         # ultra-short period
        koi_depth=15000,        # deep transit
        koi_duration=3.0,       # short duration
        koi_impact=0.3,
        koi_model_snr=200.0,    # very strong signal
        koi_score=1.0,
        koi_steff=6200,
        koi_slogg=4.3,
        koi_srad=1.3,
        koi_smass=1.2,
        koi_fpflag_nt=0,
        koi_fpflag_ss=0,
        koi_fpflag_co=0,
        koi_fpflag_ec=0,
    )
    print_prediction(hot_jup, title="Hot Jupiter Test")

    # ── Likely false positive ────────────────────────────────────────────
    false_pos = predict_planet(
        koi_prad=35.0,          # way too large — likely a star
        koi_teq=1200,
        koi_period=1.5,
        koi_depth=50000,        # extremely deep — eclipsing binary
        koi_duration=5.0,
        koi_model_snr=500.0,
        koi_score=0.0,          # low confidence
        koi_fpflag_nt=1,        # flagged as not-transit-like
        koi_fpflag_ss=1,        # flagged as stellar eclipse
        koi_fpflag_co=0,
        koi_fpflag_ec=0,
    )
    print_prediction(false_pos, title="False Positive Test")
