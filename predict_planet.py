from pathlib import Path
from typing import Any
import joblib
import numpy as np
MODEL_DIR = Path(__file__).parent / "data" / "model"
_clf = joblib.load(MODEL_DIR / "rf_classifier.joblib")
_imputer = joblib.load(MODEL_DIR / "imputer.joblib")
_le = joblib.load(MODEL_DIR / "label_encoder.joblib")
_feature_names = joblib.load(MODEL_DIR / "feature_names.joblib")
_medians = joblib.load(MODEL_DIR / "feature_medians.joblib")
R_TERRAN_MAX = 1.23
R_NEPTUNE_MAX = 14.26
ALPHA_TERRAN = 3.58
ALPHA_NEPTUNE = 1.70
ALPHA_JOVIAN = 0.01
_C_NEPTUNE = (R_TERRAN_MAX ** ALPHA_TERRAN) / (R_TERRAN_MAX ** ALPHA_NEPTUNE)
_M_AT_NJ = _C_NEPTUNE * (R_NEPTUNE_MAX ** ALPHA_NEPTUNE)
_C_JOVIAN = _M_AT_NJ / (R_NEPTUNE_MAX ** ALPHA_JOVIAN)
G_EARTH = 9.81  
def _estimate_mass(radius_earth: float) -> float:
    if radius_earth <= 0 or np.isnan(radius_earth):
        return np.nan
    if radius_earth < R_TERRAN_MAX:
        return radius_earth ** ALPHA_TERRAN
    elif radius_earth < R_NEPTUNE_MAX:
        return _C_NEPTUNE * radius_earth ** ALPHA_NEPTUNE
    else:
        return _C_JOVIAN * radius_earth ** ALPHA_JOVIAN
def _surface_gravity(mass_earth: float, radius_earth: float) -> float:
    if radius_earth <= 0 or np.isnan(radius_earth) or np.isnan(mass_earth):
        return np.nan
    return (mass_earth / radius_earth ** 2) * G_EARTH
def predict_planet(**kwargs: Any) -> dict:
    row = dict(_medians)
    for key, val in kwargs.items():
        if key in row:
            row[key] = float(val)
        elif key not in ("est_mass_earth", "surface_gravity_ms2"):
            print(f"  [warning] Unknown feature '{key}' — ignored.")
    radius = row.get("koi_prad", _medians.get("koi_prad", 1.0))
    mass = _estimate_mass(radius)
    grav = _surface_gravity(mass, radius)
    row["est_mass_earth"] = mass
    row["surface_gravity_ms2"] = grav
    feature_vector = np.array([[row[f] for f in _feature_names]])
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
def print_prediction(result: dict, title: str = "Prediction") -> None:
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
if __name__ == "__main__":
    earth = predict_planet(
        koi_prad=1.0,           
        koi_teq=255,            
        koi_period=365.25,      
        koi_depth=84,           
        koi_duration=13.0,      
        koi_impact=0.0,         
        koi_model_snr=30.0,     
        koi_score=1.0,          
        koi_steff=5778,         
        koi_slogg=4.44,         
        koi_srad=1.0,           
        koi_smass=1.0,          
        koi_fpflag_nt=0,        
        koi_fpflag_ss=0,
        koi_fpflag_co=0,
        koi_fpflag_ec=0,
        koi_insol=1.0,          
        koi_smet=0.0,           
        koi_count=1,            
    )
    print_prediction(earth, title="Earth Verification Test")
    hot_jup = predict_planet(
        koi_prad=12.0,          
        koi_teq=1800,           
        koi_period=2.5,         
        koi_depth=15000,        
        koi_duration=3.0,       
        koi_impact=0.3,
        koi_model_snr=200.0,    
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
    false_pos = predict_planet(
        koi_prad=35.0,          
        koi_teq=1200,
        koi_period=1.5,
        koi_depth=50000,        
        koi_duration=5.0,
        koi_model_snr=500.0,
        koi_score=0.0,          
        koi_fpflag_nt=1,        
        koi_fpflag_ss=1,        
        koi_fpflag_co=0,
        koi_fpflag_ec=0,
    )
    print_prediction(false_pos, title="False Positive Test")