"""
Goldilocks Zone scatter plot — Equilibrium Temperature vs Planetary Radius,
coloured by KOI disposition.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

DATA_DIR = Path(__file__).parent / "data"
GRAVITY_CSV = DATA_DIR / "gravity_data.csv"
OUT_PNG = DATA_DIR / "goldilocks_zone.png"

# ── Styling ──────────────────────────────────────────────────────────────────
# Curated palette for dispositions
PALETTE = {
    "CONFIRMED":      "#00e676",   # vivid green
    "CANDIDATE":      "#29b6f6",   # sky blue
    "FALSE POSITIVE": "#ff5252",   # coral red
    "NOT DISPOSITIONED": "#9e9e9e" # grey
}

# Plot order so confirmed planets render on top
DRAW_ORDER = ["FALSE POSITIVE", "NOT DISPOSITIONED", "CANDIDATE", "CONFIRMED"]


def main() -> None:
    df = pd.read_csv(GRAVITY_CSV)
    print(f"Loaded: {df.shape[0]:,} rows")

    # Drop rows with NaN in the columns we need
    plot_df = df.dropna(subset=["koi_teq", "koi_prad", "koi_disposition"]).copy()
    print(f"Rows with valid teq + prad + disposition: {len(plot_df):,}")

    # ── Figure setup ─────────────────────────────────────────────────────
    sns.set_style("darkgrid", {
        "axes.facecolor": "#1a1a2e",
        "figure.facecolor": "#0f0f23",
        "grid.color": "#2a2a4a",
    })

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor("#0f0f23")
    ax.set_facecolor("#1a1a2e")

    # ── Habitable zone band (rough 180–310 K) ───────────────────────────
    ax.axvspan(180, 310, alpha=0.12, color="#00e676", zorder=0)
    ax.text(
        245, 28.5, "Habitable Zone",
        fontsize=11, fontweight="bold", color="#00e676",
        ha="center", va="top", alpha=0.7,
        fontstyle="italic",
    )

    # ── Scatter by disposition (draw order controls layering) ────────────
    for disp in DRAW_ORDER:
        subset = plot_df[plot_df["koi_disposition"] == disp]
        if subset.empty:
            continue

        color = PALETTE.get(disp, "#ffffff")

        ax.scatter(
            subset["koi_teq"],
            subset["koi_prad"],
            c=color,
            label=f"{disp}  ({len(subset):,})",
            s=18,
            alpha=0.65,
            edgecolors="white",
            linewidths=0.15,
            zorder=DRAW_ORDER.index(disp) + 1,
        )

    # ── Axes limits & labels ─────────────────────────────────────────────
    ax.set_xlim(0, 2500)
    ax.set_ylim(0, 30)

    ax.set_xlabel(
        "Equilibrium Temperature  (K)",
        fontsize=13, fontweight="bold", color="#e0e0e0", labelpad=10,
    )
    ax.set_ylabel(
        "Planetary Radius  (R$_\\oplus$)",
        fontsize=13, fontweight="bold", color="#e0e0e0", labelpad=10,
    )
    ax.set_title(
        "Kepler Objects of Interest — Goldilocks Zone Overview",
        fontsize=17, fontweight="bold", color="#ffffff", pad=18,
    )

    # Tick styling
    ax.tick_params(colors="#b0b0b0", labelsize=11)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(250))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))

    # ── Reference lines for familiar planet sizes ────────────────────────
    for r, name in [(1.0, "Earth"), (3.88, "Neptune"), (11.2, "Jupiter")]:
        if r <= 30:
            ax.axhline(r, color="#ffffff", alpha=0.18, linewidth=0.8,
                        linestyle="--", zorder=0)
            ax.text(2470, r + 0.35, name, fontsize=9, color="#aaaaaa",
                    ha="right", va="bottom", fontstyle="italic")

    # ── Earth marker ─────────────────────────────────────────────────────
    ax.scatter(
        [255], [1.0], marker="*", s=220, c="#ffd740",
        edgecolors="white", linewidths=0.6, zorder=10,
    )
    ax.annotate(
        "Earth", xy=(255, 1.0), xytext=(320, 2.5),
        fontsize=10, fontweight="bold", color="#ffd740",
        arrowprops=dict(arrowstyle="->", color="#ffd740", lw=1.2),
    )

    # ── Legend ────────────────────────────────────────────────────────────
    legend = ax.legend(
        title="Disposition",
        loc="upper right",
        fontsize=10,
        title_fontsize=11,
        frameon=True,
        framealpha=0.35,
        edgecolor="#444466",
        facecolor="#1a1a2e",
        labelcolor="#e0e0e0",
        markerscale=1.5,
    )
    legend.get_title().set_color("#ffffff")

    # ── Save ─────────────────────────────────────────────────────────────
    plt.tight_layout()
    fig.savefig(OUT_PNG, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nSaved chart -> {OUT_PNG}")


if __name__ == "__main__":
    main()
