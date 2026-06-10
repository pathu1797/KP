"""
Random Forest Classifier — Predict KOI Disposition
====================================================
Trains on gravity_data.csv with class_weight='balanced' to handle the
imbalanced target (CONFIRMED planets are rarer than FALSE POSITIVEs).

Outputs: confusion matrix, per-class precision / recall / F1, and
overall accuracy on the held-out 20% test set.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).parent / "data"
GRAVITY_CSV = DATA_DIR / "gravity_data.csv"
CM_PNG = DATA_DIR / "confusion_matrix.png"

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Columns to drop: identifiers, free-text, provenance strings, and the target
DROP_COLS = [
    "kepid",             # Kepler ID (identifier, not a feature)
    "kepoi_name",        # KOI name string
    "koi_disposition",   # ← target
    "koi_pdisposition",  # pipeline disposition (leaks the answer)
    "koi_delivname",     # delivery name
    "koi_vet_stat",      # vetting status string
    "koi_vet_date",      # vetting date
    "koi_disp_prov",     # disposition provenance
    "koi_parm_prov",     # parameter provenance
    "koi_sparprov",      # stellar parameter provenance
    "koi_trans_mod",     # transit model name
    "koi_limbdark_mod",  # limb-darkening model name
    "koi_fittype",       # fit type string
    "koi_datalink_dvs",  # data link (URL-like)
    "koi_datalink_dvr",  # data link (URL-like)
    "koi_tce_delivname", # TCE delivery name
    "koi_comment",       # free-text comment
    "ra_str",            # RA string representation
    "dec_str",           # Dec string representation
    "koi_quarters",      # bitmask string
]


def main() -> None:
    # ── 1. Load ──────────────────────────────────────────────────────────
    df = pd.read_csv(GRAVITY_CSV)
    print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]:,} columns")

    # Target
    y_raw = df["koi_disposition"]
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    print(f"Classes: {list(le.classes_)}")
    print(f"Distribution: {dict(zip(le.classes_, np.bincount(y)))}")

    # ── 2. Feature matrix ────────────────────────────────────────────────
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    X = df.drop(columns=cols_to_drop)

    # Keep only numeric columns
    X = X.select_dtypes(include=[np.number])
    feature_names = list(X.columns)
    print(f"Features used: {len(feature_names)}")

    # ── 3. Impute missing values (median strategy) ───────────────────────
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    # ── 4. Train / test split ────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,  # preserve class proportions
    )
    print(f"\nTrain: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")

    # ── 5. Train Random Forest with balanced class weights ───────────────
    print("\nTraining Random Forest (class_weight='balanced') ...")
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # ── 6. Evaluate ──────────────────────────────────────────────────────
    y_pred = clf.predict(X_test)
    accuracy = (y_pred == y_test).mean()

    print(f"\n{'='*64}")
    print(f"  Test Accuracy: {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"{'='*64}")

    # Classification report (precision, recall, F1 per class)
    print("\nClassification Report:\n")
    print(classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        digits=4,
    ))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n")
    print(pd.DataFrame(
        cm,
        index=[f"True: {c}" for c in le.classes_],
        columns=[f"Pred: {c}" for c in le.classes_],
    ).to_string())

    # ── 7. Plot confusion matrix ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0f0f23")
    ax.set_facecolor("#1a1a2e")

    disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
    disp.plot(ax=ax, cmap="YlGnBu", colorbar=True, values_format="d")

    ax.set_title(
        "Confusion Matrix — Random Forest (balanced)",
        fontsize=14, fontweight="bold", color="#ffffff", pad=14,
    )
    ax.set_xlabel("Predicted", fontsize=12, color="#e0e0e0")
    ax.set_ylabel("Actual", fontsize=12, color="#e0e0e0")
    ax.tick_params(colors="#c0c0c0")

    # Rotate x labels for readability
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")

    plt.tight_layout()
    fig.savefig(CM_PNG, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nConfusion matrix chart saved -> {CM_PNG}")

    # ── 8. Top 15 feature importances ────────────────────────────────────
    importances = clf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:15]
    print("\nTop 15 Feature Importances:")
    for rank, idx in enumerate(top_idx, 1):
        print(f"  {rank:2d}. {feature_names[idx]:<28s}  {importances[idx]:.4f}")

    # ── 9. Save model artifacts for deployment ───────────────────────────
    import joblib

    model_dir = DATA_DIR / "model"
    model_dir.mkdir(exist_ok=True)

    joblib.dump(clf, model_dir / "rf_classifier.joblib")
    joblib.dump(imputer, model_dir / "imputer.joblib")
    joblib.dump(le, model_dir / "label_encoder.joblib")
    joblib.dump(feature_names, model_dir / "feature_names.joblib")

    # Also save the training medians so the prediction script knows defaults
    medians = dict(zip(feature_names, imputer.statistics_))
    joblib.dump(medians, model_dir / "feature_medians.joblib")

    print(f"\nModel artifacts saved -> {model_dir}")


if __name__ == "__main__":
    main()
