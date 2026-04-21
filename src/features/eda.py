"""
eda.py

Exploratory data analysis — runs on the ENGINEERED datasets
so plots reflect the final features the model will see.

Outputs saved to reports/eda/:
    ckd_missing.png           missing value rates
    ckd_class_dist.png        class balance bar chart
    ckd_correlation.png       feature correlation heatmap
    ckd_distributions.png     key feature histograms split by class
    thyroid_missing.png
    thyroid_class_dist.png
    thyroid_key_features.png  TSH/T3/FTI histograms by class
    thyroid_tsh_t3_ratio.png  engineered ratio by class
    eda_summary.json          machine-readable summary

Usage:
    python src/features/eda.py
"""

import json
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

PROC_DIR    = "data/processed"
REPORTS_DIR = "reports/eda"
os.makedirs(REPORTS_DIR, exist_ok=True)

# Colour palette — consistent across all plots
COLORS = {
    "ckd":     "#e07b54",
    "notckd":  "#5b8db8",
    "normal":  "#5b8db8",
    "hypo":    "#e07b54",
    "hyper":   "#7fb87f",
    "missing": "#c0392b",
    "grid":    "#eeeeee",
}

CKD_KEY_FEATURES     = ["sc", "bu", "hemo", "sod", "pot", "bgr", "pcv", "wc", "rc"]
THYROID_KEY_FEATURES = ["TSH", "T3", "TT4", "FTI"]


def _save(fig, filename: str):
    path = os.path.join(REPORTS_DIR, filename)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved %s", path)


# CKD EDA

def eda_ckd() -> dict:
    """
    Run EDA on ckd_engineered.csv.

    Produces 4 plot files and returns a summary dict.
    """
    path = os.path.join(PROC_DIR, "ckd_engineered.csv")
    df   = pd.read_csv(path)
    log.info("CKD EDA | %d rows × %d cols", len(df), len(df.columns))

    summary = {
        "rows":          len(df),
        "cols":          len(df.columns),
        "class_balance": df["classification"].value_counts().to_dict(),
        "missing_pct":   (df.isnull().mean() * 100).round(1).to_dict(),
        "feature_means_by_class": {},
    }

    #  Plot 1: missing value rates 
    missing = (df.isnull().mean() * 100).sort_values(ascending=False)
    missing = missing[missing > 0]

    fig, ax = plt.subplots(figsize=(11, 4))
    if len(missing) > 0:
        bars = ax.bar(missing.index, missing.values, color=COLORS["missing"], alpha=0.8)
        ax.axhline(20, color="black", linestyle="--", linewidth=1,
                   label="20% warning threshold")
        for bar, val in zip(bars, missing.values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    val + 0.3, f"{val:.0f}%",
                    ha="center", va="bottom", fontsize=8)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No missing values", transform=ax.transAxes,
                ha="center", va="center", fontsize=14)
    ax.set_title("CKD — Missing Value Rate per Feature", fontsize=13)
    ax.set_ylabel("% rows missing")
    ax.set_xlabel("Feature")
    plt.xticks(rotation=45, ha="right")
    ax.set_facecolor(COLORS["grid"])
    plt.tight_layout()
    _save(fig, "ckd_missing.png")

    #  Plot 2: class distribution 
    fig, ax = plt.subplots(figsize=(5, 4))
    labels  = ["Not CKD (0)", "CKD (1)"]
    counts  = [
        int((df["classification"] == 0).sum()),
        int((df["classification"] == 1).sum()),
    ]
    bars = ax.bar(labels, counts,
                  color=[COLORS["notckd"], COLORS["ckd"]], alpha=0.85)
    for bar, val in zip(bars, counts):
        pct = val / sum(counts) * 100
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + 1, f"{val}\n({pct:.0f}%)",
                ha="center", va="bottom", fontweight="bold")
    ax.set_title("CKD — Class Distribution", fontsize=13)
    ax.set_ylabel("Count")
    ax.set_facecolor(COLORS["grid"])
    plt.tight_layout()
    _save(fig, "ckd_class_dist.png")

    #  Plot 3: correlation heatmap 
    numeric = df.select_dtypes(include=np.number)
    corr    = numeric.corr()

    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    # Annotate only cells with |r| > 0.4 to keep it readable
    for i in range(len(corr)):
        for j in range(len(corr)):
            val = corr.values[i, j]
            if abs(val) > 0.4 and i != j:
                ax.text(j, i, f"{val:.2f}",
                        ha="center", va="center", fontsize=6,
                        color="white" if abs(val) > 0.7 else "black")
    ax.set_title("CKD — Feature Correlation Matrix (|r|>0.4 annotated)", fontsize=12)
    plt.tight_layout()
    _save(fig, "ckd_correlation.png")

    #  Plot 4: key feature distributions by class 
    available = [f for f in CKD_KEY_FEATURES if f in df.columns]
    ncols     = 3
    nrows     = int(np.ceil(len(available) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3.5))
    axes      = np.array(axes).flatten()

    for i, feat in enumerate(available):
        ax          = axes[i]
        not_ckd     = df[df["classification"] == 0][feat].dropna()
        ckd         = df[df["classification"] == 1][feat].dropna()
        ax.hist(not_ckd, bins=25, alpha=0.65,
                label=f"Not CKD (n={len(not_ckd)})", color=COLORS["notckd"])
        ax.hist(ckd,     bins=25, alpha=0.65,
                label=f"CKD (n={len(ckd)})",     color=COLORS["ckd"])
        ax.set_title(feat, fontsize=10)
        ax.legend(fontsize=7)
        ax.set_facecolor(COLORS["grid"])
        summary["feature_means_by_class"][feat] = {
            "ckd_mean":    round(float(ckd.mean()),     3) if len(ckd)     else None,
            "notckd_mean": round(float(not_ckd.mean()), 3) if len(not_ckd) else None,
        }

    for j in range(len(available), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("CKD — Key Feature Distributions by Class", fontsize=13, y=1.01)
    plt.tight_layout()
    _save(fig, "ckd_distributions.png")

    log.info("CKD EDA complete — 4 plots saved")
    return summary


# Thyroid EDA

def eda_thyroid() -> dict:
    """
    Run EDA on thyroid_engineered.csv.

    Produces 4 plot files and returns a summary dict.
    """
    path = os.path.join(PROC_DIR, "thyroid_engineered.csv")
    df   = pd.read_csv(path)
    log.info("Thyroid EDA | %d rows × %d cols", len(df), len(df.columns))

    label_map = {0: "Normal", 1: "Hypothyroid", 2: "Hyperthyroid"}
    label_colors = [COLORS["normal"], COLORS["hypo"], COLORS["hyper"]]

    counts_raw = df["label"].value_counts().sort_index()
    summary = {
        "rows":          len(df),
        "cols":          len(df.columns),
        "class_balance": {label_map[k]: int(v) for k, v in counts_raw.items()},
        "missing_pct":   (df.isnull().mean() * 100).round(1).to_dict(),
        "feature_means_by_class": {},
    }

    #  Plot 1: missing value rates 
    missing = (df.isnull().mean() * 100).sort_values(ascending=False)
    missing = missing[missing > 0]

    fig, ax = plt.subplots(figsize=(11, 4))
    if len(missing) > 0:
        bars = ax.bar(missing.index, missing.values, color=COLORS["missing"], alpha=0.8)
        ax.axhline(20, color="black", linestyle="--", linewidth=1,
                   label="20% threshold")
        for bar, val in zip(bars, missing.values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    val + 0.3, f"{val:.0f}%",
                    ha="center", va="bottom", fontsize=8)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No missing values", transform=ax.transAxes,
                ha="center", fontsize=14)
    ax.set_title("Thyroid — Missing Value Rate per Feature", fontsize=13)
    ax.set_ylabel("% rows missing")
    plt.xticks(rotation=45, ha="right")
    ax.set_facecolor(COLORS["grid"])
    plt.tight_layout()
    _save(fig, "thyroid_missing.png")

    #  Plot 2: class distribution 
    fig, ax = plt.subplots(figsize=(6, 4))
    names   = [label_map[k] for k in counts_raw.index]
    vals    = list(counts_raw.values)
    bars    = ax.bar(names, vals, color=label_colors[:len(names)], alpha=0.85)
    for bar, val in zip(bars, vals):
        pct = val / sum(vals) * 100
        ax.text(bar.get_x() + bar.get_width() / 2,
                val + 5, f"{val}\n({pct:.1f}%)",
                ha="center", va="bottom", fontweight="bold", fontsize=9)
    ax.set_title("Thyroid — Class Distribution (3-class)", fontsize=13)
    ax.set_ylabel("Count")
    ax.set_facecolor(COLORS["grid"])
    plt.tight_layout()
    _save(fig, "thyroid_class_dist.png")

    #  Plot 3: key lab features by class 
    available = [f for f in THYROID_KEY_FEATURES if f in df.columns]
    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 4))
    if len(available) == 1:
        axes = [axes]

    for ax, feat in zip(axes, available):
        for label_id, label_name in label_map.items():
            vals = df[df["label"] == label_id][feat].dropna()
            ax.hist(vals, bins=30, alpha=0.65,
                    label=f"{label_name} (n={len(vals)})",
                    color=label_colors[label_id])
            summary["feature_means_by_class"][feat] = \
                summary["feature_means_by_class"].get(feat, {})
            summary["feature_means_by_class"][feat][f"{label_name}_mean"] = (
                round(float(vals.mean()), 3) if len(vals) > 0 else None
            )
        ax.set_title(feat, fontsize=11)
        ax.legend(fontsize=7)
        ax.set_facecolor(COLORS["grid"])

    fig.suptitle("Thyroid — Key Lab Values by Class", fontsize=13)
    plt.tight_layout()
    _save(fig, "thyroid_key_features.png")

    #  Plot 4: engineered tsh_t3_ratio by class 
    if "tsh_t3_ratio" in df.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        for label_id, label_name in label_map.items():
            vals = df[df["label"] == label_id]["tsh_t3_ratio"].dropna()
            # Clip extreme outliers for display only
            vals_clipped = vals.clip(upper=vals.quantile(0.98))
            ax.hist(vals_clipped, bins=40, alpha=0.65,
                    label=f"{label_name} (n={len(vals)})",
                    color=label_colors[label_id])
        ax.set_title(
            "Thyroid — TSH/T3 Ratio by Class\n"
            "(primary hypo = very high ratio; secondary hypo = low ratio)",
            fontsize=10,
        )
        ax.set_xlabel("TSH / T3 ratio (clipped at 98th pct)")
        ax.legend(fontsize=8)
        ax.set_facecolor(COLORS["grid"])
        plt.tight_layout()
        _save(fig, "thyroid_tsh_t3_ratio.png")
        log.info("  tsh_t3_ratio plot saved ")

    log.info("Thyroid EDA complete — 4 plots saved")
    return summary



# Entry point

def run_eda():
    ckd_summary     = eda_ckd()
    thyroid_summary = eda_thyroid()

    combined = {"ckd": ckd_summary, "thyroid": thyroid_summary}
    out_path = os.path.join(REPORTS_DIR, "eda_summary.json")
    with open(out_path, "w") as fh:
        json.dump(combined, fh, indent=2)
    log.info("EDA summary → %s", out_path)

    print("\n=== EDA Summary ===")
    print(f"CKD:     {ckd_summary['rows']} rows | balance: {ckd_summary['class_balance']}")
    print(f"Thyroid: {thyroid_summary['rows']} rows | balance: {thyroid_summary['class_balance']}")
    print(f"\nTop missing (CKD):")
    for k, v in list(ckd_summary['missing_pct'].items())[:5]:
        if v > 0:
            print(f"  {k}: {v:.1f}%")
    print(f"\nPlots → {REPORTS_DIR}/")


if __name__ == "__main__":
    run_eda()