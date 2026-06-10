"""
KOI Disposition Classifier — Enterprise Dashboard
"""

from pathlib import Path

import joblib
import numpy as np
import streamlit as st

# ── 1. Page config: wide layout, first Streamlit call ────────────────────────
st.set_page_config(
    page_title="KOI Disposition Classifier",
    page_icon="·",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Model loading ────────────────────────────────────────────────────────────
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

# ── Physics constants ────────────────────────────────────────────────────────
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


# ── 3. Injected CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg:        #0a0a0f;
    --surface:   #10111a;
    --surface-2: #14151f;
    --border:    #1e2030;
    --border-hi: #2d3748;
    --text:      #e2e4e9;
    --text-2:    #9ca3b0;
    --text-3:    #5a6170;
    --mono:      #b0b8c4;
    --blue:      #4c8dff;
    --green:     #34d399;
    --green-bg:  #0d1f17;
    --amber:     #fbbf24;
    --amber-bg:  #1a1708;
    --red:       #f87171;
    --red-bg:    #1c0f0f;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    -webkit-font-smoothing: antialiased;
}

/* layout reset */
.block-container {
    padding: 2rem 2.5rem 1rem 2.5rem !important;
    max-width: 100% !important;
}
header[data-testid="stHeader"] { background: transparent !important; }

/* sidebar */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
    width: 280px !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    padding: 1.2rem 1rem !important;
}

/* ── top bar ── */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 1.1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.6rem;
}
.topbar-left {
    display: flex;
    align-items: baseline;
    gap: 14px;
}
.topbar-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--text);
    letter-spacing: -0.01em;
}
.topbar-tag {
    font-size: 0.65rem;
    font-weight: 500;
    color: var(--text-3);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    border: 1px solid var(--border);
    padding: 2px 8px;
    border-radius: 3px;
}
.topbar-meta {
    font-size: 0.68rem;
    color: var(--text-3);
    font-family: 'JetBrains Mono', monospace;
}

/* ── verdict banner ── */
.verdict {
    display: flex;
    align-items: center;
    padding: 11px 20px;
    border: 1px solid var(--border-hi);
    border-radius: 4px;
    margin-bottom: 1.4rem;
    gap: 12px;
}
.verdict .v-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    flex-shrink: 0;
}
.verdict .v-tag {
    font-size: 0.62rem;
    color: var(--text-3);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 600;
}
.verdict .v-status {
    font-size: 0.88rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.02em;
}
.verdict .v-right {
    margin-left: auto;
    display: flex;
    align-items: center;
    gap: 16px;
}
.verdict .v-conf {
    font-size: 0.72rem;
    color: var(--text-3);
    font-family: 'JetBrains Mono', monospace;
}
.v-confirmed { border-color: #1a3a2a; }
.v-confirmed .v-dot { background: var(--green); }
.v-confirmed .v-status { color: var(--green); }
.v-candidate { border-color: #2a2510; }
.v-candidate .v-dot { background: var(--amber); }
.v-candidate .v-status { color: var(--amber); }
.v-fp { border-color: #2a1515; }
.v-fp .v-dot { background: var(--red); }
.v-fp .v-status { color: var(--red); }

/* ── section label ── */
.sec {
    font-size: 0.6rem;
    color: var(--text-3);
    text-transform: uppercase;
    letter-spacing: 1.6px;
    font-weight: 600;
    margin-bottom: 10px;
}

/* ── data table ── */
.dtable {
    width: 100%;
    border: 1px solid var(--border);
    border-radius: 4px;
    border-collapse: separate;
    border-spacing: 0;
    overflow: hidden;
    margin-bottom: 1.3rem;
}
.dtable th {
    background: var(--surface-2);
    font-size: 0.58rem;
    color: var(--text-3);
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-weight: 600;
    padding: 8px 14px;
    text-align: left;
    border-bottom: 1px solid var(--border);
}
.dtable td {
    padding: 10px 14px;
    font-size: 0.92rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
    color: var(--text);
    border-bottom: 1px solid var(--border);
    background: var(--surface);
}
.dtable tr:last-child td {
    border-bottom: none;
}
.dtable .unit {
    font-size: 0.72rem;
    color: var(--text-3);
    font-weight: 400;
    margin-left: 3px;
}

/* ── probability list ── */
.plist {
    list-style: none;
    padding: 0;
    margin: 0;
}
.plist li {
    display: flex;
    align-items: center;
    padding: 10px 0;
    border-bottom: 1px solid var(--border);
    gap: 14px;
}
.plist li:last-child { border-bottom: none; }
.plist .pl-name {
    font-size: 0.78rem;
    color: var(--text-2);
    font-weight: 500;
    width: 130px;
    flex-shrink: 0;
}
.plist .pl-track {
    flex: 1;
    height: 3px;
    background: var(--border);
    border-radius: 2px;
    overflow: hidden;
}
.plist .pl-fill {
    height: 100%;
    background: var(--text-3);
    border-radius: 2px;
    transition: width 0.4s ease;
}
.plist .pl-fill.active { background: var(--blue); }
.plist .pl-pct {
    font-size: 0.78rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 500;
    color: var(--mono);
    width: 52px;
    text-align: right;
    flex-shrink: 0;
}

/* ── notes panel ── */
.note {
    font-size: 0.72rem;
    color: var(--text-3);
    line-height: 1.7;
    border-top: 1px solid var(--border);
    padding-top: 10px;
    margin-top: 6px;
}
.note code {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: var(--text-2);
    background: var(--surface-2);
    padding: 1px 5px;
    border-radius: 3px;
}

/* sidebar labels */
.sb-label {
    font-size: 0.58rem;
    color: var(--text-3);
    text-transform: uppercase;
    letter-spacing: 1.4px;
    font-weight: 600;
    margin: 14px 0 4px 0;
}
.sb-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 2px;
}
.sb-sub {
    font-size: 0.68rem;
    color: var(--text-3);
    margin-bottom: 12px;
}
.sb-divider {
    height: 1px;
    background: var(--border);
    margin: 12px 0;
}
</style>
""", unsafe_allow_html=True)

# ── 2. Sidebar: all interactive controls ─────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sb-title">KOI Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-sub">Input configuration panel</div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-label">Planet</div>', unsafe_allow_html=True)
    koi_prad = st.slider("Radius (R_Earth)", 0.1, 50.0, 1.0, 0.1)
    koi_teq = st.slider("Eq. temperature (K)", 50, 3000, 255, 5)
    koi_period = st.slider("Orbital period (d)", 0.1, 1000.0, 365.25, 0.5)

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-label">Detection</div>', unsafe_allow_html=True)
    koi_score = st.slider("Disposition score", 0.0, 1.0, 1.0, 0.01)
    koi_model_snr = st.slider("Model SNR", 0.0, 500.0, 30.0, 1.0)
    koi_depth = st.slider("Transit depth (ppm)", 0.0, 50000.0, 84.0, 10.0)

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-label">Host star</div>', unsafe_allow_html=True)
    koi_steff = st.slider("Stellar temp (K)", 2500, 10000, 5778, 25)
    koi_srad = st.slider("Stellar radius (R_sun)", 0.1, 10.0, 1.0, 0.05)

    st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-label">FP Flags</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    fp_nt = c1.toggle("Not transit", False)
    fp_co = c1.toggle("Centroid", False)
    fp_ss = c2.toggle("Stellar ecl.", False)
    fp_ec = c2.toggle("Ephemeris", False)

# ── Compute ──────────────────────────────────────────────────────────────────
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

# ── Main panel ───────────────────────────────────────────────────────────────

# Top bar
st.markdown(f"""
<div class="topbar">
    <div class="topbar-left">
        <span class="topbar-title">Disposition Analysis</span>
        <span class="topbar-tag">Random Forest</span>
        <span class="topbar-tag">110 features</span>
    </div>
    <div class="topbar-meta">accuracy 93.05% &nbsp;&middot;&nbsp; 300 estimators &nbsp;&middot;&nbsp; balanced</div>
</div>
""", unsafe_allow_html=True)

# 5. Verdict banner
v_map = {
    "CONFIRMED":      ("v-confirmed", "CONFIRMED PLANET"),
    "CANDIDATE":      ("v-candidate", "CANDIDATE"),
    "FALSE POSITIVE": ("v-fp",        "FALSE POSITIVE"),
}
v_css, v_text = v_map[label]

flags = [n for n, v in [("NT", fp_nt), ("SS", fp_ss), ("CO", fp_co), ("EC", fp_ec)] if v]
flag_str = ", ".join(flags) if flags else "none"

st.markdown(f"""
<div class="verdict {v_css}">
    <div class="v-dot"></div>
    <span class="v-tag">Verdict</span>
    <span class="v-status">{v_text}</span>
    <div class="v-right">
        <span class="v-conf">conf {top_prob:.1%}</span>
        <span class="v-conf">flags {flag_str}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# 4. Data-dense grid — three columns
col_phys, col_input, col_prob = st.columns([2, 2, 1.5], gap="large")

with col_phys:
    st.markdown('<div class="sec">Computed Properties</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <table class="dtable">
        <tr>
            <th>Property</th><th>Value</th>
        </tr>
        <tr>
            <td style="color:var(--text-2);font-size:0.8rem;">Radius</td>
            <td>{koi_prad:.2f}<span class="unit">R_E</span></td>
        </tr>
        <tr>
            <td style="color:var(--text-2);font-size:0.8rem;">Est. mass</td>
            <td>{mass:.4f}<span class="unit">M_E</span></td>
        </tr>
        <tr>
            <td style="color:var(--text-2);font-size:0.8rem;">Surface gravity</td>
            <td>{grav:.4f}<span class="unit">m/s²</span></td>
        </tr>
        <tr>
            <td style="color:var(--text-2);font-size:0.8rem;">Gravity ratio</td>
            <td>{grav / G_E:.4f}<span class="unit">× Earth</span></td>
        </tr>
        <tr>
            <td style="color:var(--text-2);font-size:0.8rem;">Eq. temperature</td>
            <td>{koi_teq:,}<span class="unit">K</span></td>
        </tr>
        <tr>
            <td style="color:var(--text-2);font-size:0.8rem;">Regime</td>
            <td>{regime}</td>
        </tr>
    </table>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="note">
        Mass via Chen &amp; Kipping (2017) piecewise power-law:
        <code>R &lt; {R_T}</code> terran,
        <code>{R_T}–{R_N}</code> neptunian,
        <code>&gt; {R_N}</code> jovian.
        Gravity: <code>g = (M / R²) × 9.81</code>
    </div>
    """, unsafe_allow_html=True)

with col_input:
    st.markdown('<div class="sec">Input Parameters</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <table class="dtable">
        <tr>
            <th>Parameter</th><th>Value</th>
        </tr>
        <tr>
            <td style="color:var(--text-2);font-size:0.8rem;">Orbital period</td>
            <td>{koi_period:.2f}<span class="unit">d</span></td>
        </tr>
        <tr>
            <td style="color:var(--text-2);font-size:0.8rem;">Transit depth</td>
            <td>{koi_depth:,.0f}<span class="unit">ppm</span></td>
        </tr>
        <tr>
            <td style="color:var(--text-2);font-size:0.8rem;">Model SNR</td>
            <td>{koi_model_snr:.1f}</td>
        </tr>
        <tr>
            <td style="color:var(--text-2);font-size:0.8rem;">Disp. score</td>
            <td>{koi_score:.2f}</td>
        </tr>
        <tr>
            <td style="color:var(--text-2);font-size:0.8rem;">Stellar temp</td>
            <td>{koi_steff:,}<span class="unit">K</span></td>
        </tr>
        <tr>
            <td style="color:var(--text-2);font-size:0.8rem;">Stellar radius</td>
            <td>{koi_srad:.2f}<span class="unit">R☉</span></td>
        </tr>
    </table>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="note">
        Reference — Earth: 1.0 R_E, 1.0 M_E, 9.81 m/s², 255 K
        &nbsp;|&nbsp; Neptune: 3.88 R_E, 17.1 M_E
        &nbsp;|&nbsp; Jupiter: 11.2 R_E, 318 M_E
    </div>
    """, unsafe_allow_html=True)

# 6. Refined probability list
with col_prob:
    st.markdown('<div class="sec">Class Probabilities</div>', unsafe_allow_html=True)

    winner = max(prob_dict, key=prob_dict.get)
    items = ""
    for cls in ["CONFIRMED", "CANDIDATE", "FALSE POSITIVE"]:
        p = prob_dict.get(cls, 0)
        active = "active" if cls == winner else ""
        items += f"""
        <li>
            <span class="pl-name">{cls}</span>
            <div class="pl-track">
                <div class="pl-fill {active}" style="width:{max(p*100, 0.5)}%;"></div>
            </div>
            <span class="pl-pct">{p:.1%}</span>
        </li>"""

    st.markdown(f'<ul class="plist">{items}</ul>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="note" style="margin-top:16px;">
        Trained on 7,651 KOI samples with
        <code>class_weight=balanced</code>.
        Total dataset: 9,564 objects.
    </div>
    """, unsafe_allow_html=True)
