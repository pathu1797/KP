# Exoplanet Habitability Intelligence System 🌌

An end-to-end machine learning pipeline and data-dense scientific dashboard that processes telescope telemetry from the NASA Kepler Objects of Interest (KOI) catalog to classify candidate exoplanets as **Confirmed Planets**, **Candidates**, or **False Positives** with **93.05% overall accuracy**.

This project integrates non-linear empirical astrophysics formulas with an ensemble learning classifier to significantly improve prediction baseline performance, prioritizing the discovery of rare, rocky worlds situated within planetary habitable zones.

---

## 📈 Model Performance & Core Metrics

By utilizing class-balanced training constraints to overcome severe astronomical data imbalance (where non-habitable targets heavily outnumber true candidates), the Random Forest pipeline achieves highly optimized precision and recall parameters on unseen verification data.

### Per-Class Evaluation Metrics

| Classification Target | Precision | Recall | F1-Score | Evaluation Context |
| :--- | :--- | :--- | :--- | :--- |
| **CONFIRMED** | **91.26%** | **89.44%** | **0.9034** | High verification fidelity; catches 9/10 true planets |
| **FALSE POSITIVE** | **97.66%** | **99.38%** | **0.9852** | Near-perfect filtration of instrument/binary star noise |
| **CANDIDATE** | **83.85%** | **82.58%** | **0.8321** | Stable boundaries across unconfirmed planetary signals |

### Feature Engineering Impact
Rather than relying solely on default observational data, this pipeline explicitly evaluates calculated physical environments. Through predictive feature mapping, our custom engineered columns ranked inside the top 15 most decisive features across hundreds of decision tree iterations:
* **Rank 8:** `surface_gravity_ms2` (Custom calculated)
* **Rank 11:** `est_mass_earth` (Custom calculated)

---

## 🧬 Scientific Methodology & Physics Engine

Telescope transits easily provide a planet's radius based on light curves, but rarely provide mass directly. To give the AI deep physical intuition, a custom physics engine was injected using the **Chen & Kipping (2017) Piecewise Mass-Radius Relation**.

The application parses each planet's radius (`koi_prad`) and dynamically runs it through the corresponding structural regime to approximate planet mass ($M$) in Earth units:

* **Terran Regime (R < 1.23 R⊕):** Rocky composition profiling.
* **Neptunian Regime (1.23 - 14.26 R⊕):** Volatile-rich, gas/ice envelope profiling.
* **Jovian Regime (R >= 14.26 R⊕):** Gas giant compression profiling where radius scaling flatlines.

Using the estimated mass ($M$) and observed radius ($R$), the engine derives the true **Surface Gravity** ($g$) acting upon the planetary boundary:

`g = M / R^2`

This calculated surface gravity field provides a critical indicator for atmosphere retention limits, allowing the Random Forest model to instantly separate low-mass airless rocks and high-mass gas giants from true rocky candidates.

---

## 📂 Repository Structure

```text
├── data/
│   ├── koi_cumulative.csv        # Cached raw NASA Kepler telemetry dataset (~11.8 MB)
│   ├── cleaned_data.csv          # Telemetry file post-null-threshold scrubbing
│   └── gravity_data.csv          # Feature-complete dataset with calculated physics parameters
├── app.py                        # High-density, professional Streamlit dashboard application
├── calc_gravity.py               # Modules containing Chen & Kipping (2017) math framework
├── train_model.py                # Model training script with class balancing and serialization
├── requirements.txt              # Complete scientific stack dependency list
└── README.md                     # Project documentation

🛠️ Technology Stack & Dependencies
Language: Python 3.13.7

Data Manipulation: Pandas, NumPy

Machine Learning Engine: Scikit-Learn (Ensemble Framework)

Visualizations: Matplotlib, Seaborn

Interface Architecture: Streamlit, Custom Markdown CSS Injection

⚙️ Installation & Local Deployment
Follow these sequential steps to initialize the environment, clean the core archives, train the classifier weights, and run the interface dashboard locally.

1. Clone the Space Pipeline
Bash
git clone [https://github.com/your-username/exoplanet-habitability-system.git](https://github.com/your-username/exoplanet-habitability-system.git)
cd exoplanet-habitability-system
2. Configure Virtual Environment & Install Dependencies
Bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
3. Run Pipeline Diagnostics & Start the Application
To run the automated pipeline and spin up the production layout interface directly within your local workspace browser:

Bash
streamlit run app.py
The terminal will provide a local execution port link (typically http://localhost:8501). Open the URL to access the live dashboard environment.

🖥️ Production Dashboard Architecture
The production frontend interface avoids standard prototyping templates in favor of a clean, high-density scientific terminal layout:

Sidebar Inputs: Houses clear numerical input fields and slider parameters for active tracking, keeping interactive configurations isolated from analytical output displays.

Unified UI Canvas: Strips away non-essential UI features and implements custom slate-gray grid styling, utilizing minimalist layouts to frame critical classification criteria clearly.

Analytical Telemetry: Showcases live calculated data breakdowns from the physics engine alongside localized multi-class probability metrics for each query submission.
