from pathlib import Path
import joblib
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="KOI Disposition Classifier",
    page_icon="·",
    layout="wide",
    initial_sidebar_state="expanded",
)
MODEL_DIR = Path(__file__).parent / "data" / "model"

@st.cache_resource
def load_model():
    return (
        joblib.load(MODEL_DIR / "rf_classifier.joblib"),
        joblib.load(MODEL_DIR / "label_encoder.joblib"),
        joblib.load(MODEL_DIR / "feature_names.joblib"),
        joblib.load(MODEL_DIR / "feature_medians.joblib"),
    )

clf, le, feature_names, medians = load_model()

R_T, R_N = 1.23, 14.26
A_T, A_N, A_J = 3.58, 1.70, 0.01
C_N = (R_T ** A_T) / (R_T ** A_N)
C_J = (C_N * R_N ** A_N) / (R_N ** A_J)
G_E = 9.81

def est_mass(r):
    if r <= 0: return 0.0
    if r < R_T: return r ** A_T
    if r < R_N: return C_N * r ** A_N
    return C_J * r ** A_J

def surf_grav(m, r):
    return (m / r ** 2) * G_E if r > 0 else 0.0

st.sidebar.title("KOI Classifier")
st.sidebar.subheader("Input configuration panel")

st.sidebar.markdown("### Planet")
koi_prad = st.sidebar.slider("Radius (R_Earth)", 0.1, 50.0, 1.0, 0.1)
koi_teq = st.sidebar.slider("Eq. temperature (K)", 50, 3000, 255, 5)
koi_period = st.sidebar.slider("Orbital period (d)", 0.1, 1000.0, 365.25, 0.5)

st.sidebar.markdown("### Detection")
koi_score = st.sidebar.slider("Disposition score", 0.0, 1.0, 1.0, 0.01)
koi_model_snr = st.sidebar.slider("Model SNR", 0.0, 500.0, 30.0, 1.0)
koi_depth = st.sidebar.slider("Transit depth (ppm)", 0.0, 50000.0, 84.0, 10.0)

st.sidebar.markdown("### Host star")
koi_steff = st.sidebar.slider("Stellar temp (K)", 2500, 10000, 5778, 25)
koi_srad = st.sidebar.slider("Stellar radius (R_sun)", 0.1, 10.0, 1.0, 0.05)

st.sidebar.markdown("### FP Flags")
c1, c2 = st.sidebar.columns(2)
fp_nt = c1.toggle("Not transit", False)
fp_co = c1.toggle("Centroid", False)
fp_ss = c2.toggle("Stellar ecl.", False)
fp_ec = c2.toggle("Ephemeris", False)

mass = est_mass(koi_prad)
grav = surf_grav(mass, koi_prad)
regime = "Terran" if koi_prad < R_T else ("Neptunian" if koi_prad < R_N else "Jovian")

row = dict(medians)
row.update({
    "koi_prad": koi_prad, "koi_teq": koi_teq, "koi_period": koi_period,
    "koi_score": koi_score, "koi_model_snr": koi_model_snr,
    "koi_depth": koi_depth, "koi_steff": koi_steff, "koi_srad": koi_srad,
    "koi_fpflag_nt": int(fp_nt), "koi_fpflag_ss": int(fp_ss),
    "koi_fpflag_co": int(fp_co), "koi_fpflag_ec": int(fp_ec),
    "est_mass_earth": mass, "surface_gravity_ms2": grav,
})

X = np.array([[row[f] for f in feature_names]])
pred_idx = clf.predict(X)[0]
probas = clf.predict_proba(X)[0]
label = le.inverse_transform([pred_idx])[0]
prob_dict = {c: float(p) for c, p in zip(le.classes_, probas)}
top_prob = max(prob_dict.values())

st.header("Disposition Analysis")

st.subheader(f"Verdict: {label}")
st.write(f"Confidence: {top_prob:.1%}")

flags = [n for n, v in [("NT", fp_nt), ("SS", fp_ss), ("CO", fp_co), ("EC", fp_ec)] if v]
flag_str = ", ".join(flags) if flags else "none"
st.write(f"Flags: {flag_str}")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Computed Properties")
    st.metric("Radius (R_E)", f"{koi_prad:.2f}")
    st.metric("Est. mass (M_E)", f"{mass:.4f}")
    st.metric("Surface gravity (m/s²)", f"{grav:.4f}")
    st.metric("Gravity ratio", f"{grav / G_E:.4f}x Earth")
    st.metric("Eq. temperature (K)", f"{koi_teq:,}")
    st.metric("Regime", regime)

with col2:
    st.subheader("Input Parameters")
    st.metric("Orbital period (d)", f"{koi_period:.2f}")
    st.metric("Transit depth (ppm)", f"{koi_depth:,.0f}")
    st.metric("Model SNR", f"{koi_model_snr:.1f}")
    st.metric("Disp. score", f"{koi_score:.2f}")
    st.metric("Stellar temp (K)", f"{koi_steff:,}")
    st.metric("Stellar radius (R_sun)", f"{koi_srad:.2f}")

with col3:
    st.subheader("Class Probabilities")
    for cls in ["CONFIRMED", "CANDIDATE", "FALSE POSITIVE"]:
        p = prob_dict.get(cls, 0)
        st.write(f"{cls}: {p:.1%}")

