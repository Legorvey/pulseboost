"""
scripts/plot_fig6_and_fig7_ieee.py
================================================================================
Scientific Figures Generator for BAB III Part E (IEEE Paper Format)
Figure 6: SHAP Feature Importance for Blood Pressure (SBP and DBP, Top-5 Features)
Figure 7: SHAP Feature Importance for Oxygen Saturation (SpO2, 2 CFS Features)
Style: IEEE Transactions / Conference Standard (Clean, Minimalist, 300+ DPI)
Outputs:
  - results/fig6_shap_blood_pressure_ieee.png (Vertical 2-panel single-column IEEE, 300 DPI)
  - results/fig6_shap_blood_pressure_ieee.svg (Vector format)
  - results/fig6_shap_blood_pressure_ieee_wide.png (Horizontal 2-column option, 300 DPI)
  - results/fig6_shap_blood_pressure_ieee_wide.svg (Vector format)
  - results/fig7_shap_spo2_ieee.png (Single-column IEEE horizontal bar chart, 300 DPI)
  - results/fig7_shap_spo2_ieee.svg (Vector format)
  - results/fig7_shap_spo2_ieee_wide.png (Two-column layout option with Bar & Beeswarm, 300 DPI)
  - results/fig7_shap_spo2_ieee_wide.svg (Vector format)
================================================================================
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import shap

# Configure matplotlib for IEEE publication quality
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["xtick.direction"] = "out"
plt.rcParams["ytick.direction"] = "out"
plt.rcParams["xtick.major.size"] = 3.5
plt.rcParams["ytick.major.size"] = 3.5
plt.rcParams["xtick.major.width"] = 0.8
plt.rcParams["ytick.major.width"] = 0.8


def find_project_root() -> Path:
    cwd = Path.cwd()
    if (cwd / "KTI_XGBoost").is_dir():
        return cwd / "KTI_XGBoost"
    if cwd.name == "KTI_XGBoost":
        return cwd
    for parent in cwd.parents:
        if (parent / "KTI_XGBoost").is_dir():
            return parent / "KTI_XGBoost"
        if parent.name == "KTI_XGBoost":
            return parent
    return cwd


def generate_fig6():
    """
    Fig. 6: SHAP Feature Importance for Blood Pressure
    Two vertical panels:
      (a) Systolic Blood Pressure (SBP) - Top 5 features
      (b) Diastolic Blood Pressure (DBP) - Top 5 features
    Based on mean absolute SHAP from results/xgboost_shap_summary.txt
    """
    print("=" * 70)
    print("GENERATING FIG. 6: SHAP FEATURE IMPORTANCE FOR BLOOD PRESSURE")
    print("=" * 70)

    project_root = find_project_root()
    results_dir = project_root / "results"

    # Data from Tahap 10 SHAP summary
    sbp_features = ["apg_b_a", "apg_e_a", "max_ascending_slope", "apg_d_a", "pulse_area"]
    sbp_shap_vals = [5.2954, 2.3938, 1.9197, 1.7966, 1.7929]

    dbp_features = ["apg_b_a", "ibi_median_s", "apg_d_a", "min_descending_slope", "skewness"]
    dbp_shap_vals = [2.6419, 1.1294, 1.1181, 0.8548, 0.8256]

    # Readable display labels for IEEE paper
    feature_display_names = {
        "apg_b_a": "apg_b_a (APG b/a ratio)",
        "apg_e_a": "apg_e_a (APG e/a ratio)",
        "apg_d_a": "apg_d_a (APG d/a ratio)",
        "max_ascending_slope": "max_ascending_slope",
        "pulse_area": "pulse_area",
        "ibi_median_s": "ibi_median_s (median IBI)",
        "min_descending_slope": "min_descending_slope",
        "skewness": "skewness (pulse skewness)",
    }

    sbp_labels = [feature_display_names.get(f, f) for f in sbp_features]
    dbp_labels = [feature_display_names.get(f, f) for f in dbp_features]

    # Invert order for horizontal bar chart (top feature at top of plot)
    sbp_labels = sbp_labels[::-1]
    sbp_shap_vals = sbp_shap_vals[::-1]

    dbp_labels = dbp_labels[::-1]
    dbp_shap_vals = dbp_shap_vals[::-1]

    # =========================================================================
    # FIG 6 - OPTION A: VERTICAL 2-PANEL (SINGLE-COLUMN IEEE, 3.5 in / 89 mm)
    # =========================================================================
    fig_w, fig_h = 3.5, 5.2
    fig, axes = plt.subplots(2, 1, figsize=(fig_w, fig_h), dpi=300)

    # Panel (a) SBP
    ax1 = axes[0]
    y_pos1 = np.arange(len(sbp_labels))
    bars1 = ax1.barh(y_pos1, sbp_shap_vals, height=0.55, color="#1E6091", edgecolor="#123D5E", linewidth=0.9, zorder=3)

    for bar, val in zip(bars1, sbp_shap_vals):
        w = bar.get_width()
        ax1.text(w + 0.12, bar.get_y() + bar.get_height() / 2, f"{val:.2f}", ha="left", va="center", fontsize=7.8, fontweight="bold", color="#111111", zorder=4)

    ax1.set_title("(a) Systolic Blood Pressure (SBP)", fontsize=8.8, fontweight="bold", pad=6)
    ax1.set_xlabel("Mean |SHAP Value| (mmHg)", fontsize=8.0, fontweight="bold")
    ax1.set_yticks(y_pos1)
    ax1.set_yticklabels(sbp_labels, fontsize=7.2)
    ax1.set_xlim(0, 6.2)
    ax1.set_xticks([0, 1, 2, 3, 4, 5, 6])
    ax1.xaxis.grid(True, linestyle="--", alpha=0.4, color="#AAAAAA", zorder=0)
    ax1.set_axisbelow(True)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel (b) DBP
    ax2 = axes[1]
    y_pos2 = np.arange(len(dbp_labels))
    bars2 = ax2.barh(y_pos2, dbp_shap_vals, height=0.55, color="#237844", edgecolor="#134927", linewidth=0.9, zorder=3)

    for bar, val in zip(bars2, dbp_shap_vals):
        w = bar.get_width()
        ax2.text(w + 0.06, bar.get_y() + bar.get_height() / 2, f"{val:.2f}", ha="left", va="center", fontsize=7.8, fontweight="bold", color="#111111", zorder=4)

    ax2.set_title("(b) Diastolic Blood Pressure (DBP)", fontsize=8.8, fontweight="bold", pad=6)
    ax2.set_xlabel("Mean |SHAP Value| (mmHg)", fontsize=8.0, fontweight="bold")
    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels(dbp_labels, fontsize=7.2)
    ax2.set_xlim(0, 3.2)
    ax2.set_xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    ax2.xaxis.grid(True, linestyle="--", alpha=0.4, color="#AAAAAA", zorder=0)
    ax2.set_axisbelow(True)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout(pad=0.8)
    plt.subplots_adjust(hspace=0.42)

    out_png_1col = results_dir / "fig6_shap_blood_pressure_ieee.png"
    out_svg_1col = results_dir / "fig6_shap_blood_pressure_ieee.svg"
    fig.savefig(out_png_1col, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_svg_1col, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved Fig. 6 (Single-Column IEEE): {out_png_1col} & {out_svg_1col}")

    # =========================================================================
    # FIG 6 - OPTION B: HORIZONTAL 2-PANEL (TWO-COLUMN IEEE, 7.16 in / 182 mm)
    # =========================================================================
    fig_w2, fig_h2 = 7.16, 2.75
    fig2, axes2 = plt.subplots(1, 2, figsize=(fig_w2, fig_h2), dpi=300)

    # Panel (a) SBP
    ax1 = axes2[0]
    bars1 = ax1.barh(y_pos1, sbp_shap_vals, height=0.55, color="#1E6091", edgecolor="#123D5E", linewidth=0.9, zorder=3)
    for bar, val in zip(bars1, sbp_shap_vals):
        w = bar.get_width()
        ax1.text(w + 0.12, bar.get_y() + bar.get_height() / 2, f"{val:.2f}", ha="left", va="center", fontsize=7.8, fontweight="bold", color="#111111", zorder=4)

    ax1.set_title("(a) Systolic Blood Pressure (SBP)", fontsize=8.8, fontweight="bold", pad=6)
    ax1.set_xlabel("Mean |SHAP Value| (mmHg)", fontsize=8.0, fontweight="bold")
    ax1.set_yticks(y_pos1)
    ax1.set_yticklabels(sbp_labels, fontsize=7.2)
    ax1.set_xlim(0, 6.2)
    ax1.set_xticks([0, 1, 2, 3, 4, 5, 6])
    ax1.xaxis.grid(True, linestyle="--", alpha=0.4, color="#AAAAAA", zorder=0)
    ax1.set_axisbelow(True)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel (b) DBP
    ax2 = axes2[1]
    bars2 = ax2.barh(y_pos2, dbp_shap_vals, height=0.55, color="#237844", edgecolor="#134927", linewidth=0.9, zorder=3)
    for bar, val in zip(bars2, dbp_shap_vals):
        w = bar.get_width()
        ax2.text(w + 0.06, bar.get_y() + bar.get_height() / 2, f"{val:.2f}", ha="left", va="center", fontsize=7.8, fontweight="bold", color="#111111", zorder=4)

    ax2.set_title("(b) Diastolic Blood Pressure (DBP)", fontsize=8.8, fontweight="bold", pad=6)
    ax2.set_xlabel("Mean |SHAP Value| (mmHg)", fontsize=8.0, fontweight="bold")
    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels(dbp_labels, fontsize=7.2)
    ax2.set_xlim(0, 3.2)
    ax2.set_xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    ax2.xaxis.grid(True, linestyle="--", alpha=0.4, color="#AAAAAA", zorder=0)
    ax2.set_axisbelow(True)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout(pad=0.8)
    plt.subplots_adjust(wspace=0.35)

    out_png_2col = results_dir / "fig6_shap_blood_pressure_ieee_wide.png"
    out_svg_2col = results_dir / "fig6_shap_blood_pressure_ieee_wide.svg"
    fig2.savefig(out_png_2col, dpi=300, bbox_inches="tight", facecolor="white")
    fig2.savefig(out_svg_2col, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    print(f"Saved Fig. 6 Wide (Two-Column IEEE): {out_png_2col} & {out_svg_2col}")


def generate_fig7():
    """
    Fig. 7: SHAP Feature Importance for SpO2
    Displays the contribution of the two CFS-selected features:
      - max_ascending_slope = 0.9305 percentage points
      - rise_time_s = 0.5878 percentage points
    """
    print("=" * 70)
    print("GENERATING FIG. 7: SHAP FEATURE IMPORTANCE FOR SPO2")
    print("=" * 70)

    project_root = find_project_root()
    results_dir = project_root / "results"

    spo2_features = ["max_ascending_slope", "rise_time_s"]
    spo2_shap_vals = [0.9305, 0.5878]

    feature_display_names = {
        "max_ascending_slope": "max_ascending_slope (systolic slope)",
        "rise_time_s": "rise_time_s (pulse rise time)",
    }

    spo2_labels = [feature_display_names.get(f, f) for f in spo2_features][::-1]
    spo2_shap_vals = spo2_shap_vals[::-1]

    # =========================================================================
    # FIG 7 - PRIMARY: CLEAN HORIZONTAL BAR CHART (SINGLE-COLUMN IEEE, 3.5 in)
    # =========================================================================
    fig_w, fig_h = 3.5, 2.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)

    y_pos = np.arange(len(spo2_labels))
    bars = ax.barh(y_pos, spo2_shap_vals, height=0.48, color="#B83227", edgecolor="#791E16", linewidth=0.9, zorder=3)

    for bar, val in zip(bars, spo2_shap_vals):
        w = bar.get_width()
        ax.text(w + 0.03, bar.get_y() + bar.get_height() / 2, f"{val:.2f}", ha="left", va="center", fontsize=8.0, fontweight="bold", color="#111111", zorder=4)

    ax.set_title("Oxygen Saturation (SpO2) Feature Importance", fontsize=8.8, fontweight="bold", pad=6)
    ax.set_xlabel("Mean |SHAP Value| (% points)", fontsize=8.0, fontweight="bold")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(spo2_labels, fontsize=7.4)
    ax.set_xlim(0, 1.18)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.xaxis.grid(True, linestyle="--", alpha=0.4, color="#AAAAAA", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout(pad=0.6)

    out_png_1col = results_dir / "fig7_shap_spo2_ieee.png"
    out_svg_1col = results_dir / "fig7_shap_spo2_ieee.svg"
    fig.savefig(out_png_1col, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_svg_1col, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved Fig. 7 (Single-Column IEEE): {out_png_1col} & {out_svg_1col}")

    # =========================================================================
    # FIG 7 - OPTION B: TWO-COLUMN WIDE WITH BAR AND BEESWARM FROM REAL VAL DATA
    # =========================================================================
    # Compute real validation TreeSHAP values for the beeswarm plot
    model_path = project_root / "models" / "spo2_config_d.json"
    data_path = project_root / "results" / "osasud_features_model_ready_v1.csv"

    if model_path.is_file() and data_path.is_file():
        model = xgb.XGBRegressor()
        model.load_model(str(model_path))

        df_all = pd.read_csv(data_path)
        val_df = df_all[(df_all["split"] == "val") & (df_all["spo2_target_pct"].notna())].copy()
        cfs_feats = ["rise_time_s", "max_ascending_slope"]

        explainer = shap.TreeExplainer(model)
        shap_exp = explainer(val_df[cfs_feats])

        fig2 = plt.figure(figsize=(7.16, 2.5), dpi=300)

        # Left panel: Bar
        ax_l = plt.subplot(1, 2, 1)
        bars = ax_l.barh(y_pos, spo2_shap_vals, height=0.48, color="#B83227", edgecolor="#791E16", linewidth=0.9, zorder=3)
        for bar, val in zip(bars, spo2_shap_vals):
            w = bar.get_width()
            ax_l.text(w + 0.03, bar.get_y() + bar.get_height() / 2, f"{val:.2f}", ha="left", va="center", fontsize=8.0, fontweight="bold", color="#111111", zorder=4)

        ax_l.set_title("(a) Global Importance (Mean |SHAP|)", fontsize=8.8, fontweight="bold", pad=6)
        ax_l.set_xlabel("Mean |SHAP Value| (% points)", fontsize=8.0, fontweight="bold")
        ax_l.set_yticks(y_pos)
        ax_l.set_yticklabels(spo2_labels, fontsize=7.4)
        ax_l.set_xlim(0, 1.18)
        ax_l.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax_l.xaxis.grid(True, linestyle="--", alpha=0.4, color="#AAAAAA", zorder=0)
        ax_l.set_axisbelow(True)
        ax_l.spines["top"].set_visible(False)
        ax_l.spines["right"].set_visible(False)

        # Right panel: Beeswarm from actual validation data
        ax_r = plt.subplot(1, 2, 2)
        plt.sca(ax_r)
        shap.plots.beeswarm(shap_exp, max_display=2, show=False)
        ax_r.set_title("(b) Feature Impact Distribution (Beeswarm)", fontsize=8.8, fontweight="bold", pad=6)
        ax_r.set_xlabel("SHAP Value (% points) [Impact on SpO2]", fontsize=8.0, fontweight="bold")
        ax_r.tick_params(labelsize=7.4)

        plt.tight_layout(pad=0.8)
        plt.subplots_adjust(wspace=0.35)

        out_png_2col = results_dir / "fig7_shap_spo2_ieee_wide.png"
        out_svg_2col = results_dir / "fig7_shap_spo2_ieee_wide.svg"
        fig2.savefig(out_png_2col, dpi=300, bbox_inches="tight", facecolor="white")
        fig2.savefig(out_svg_2col, format="svg", bbox_inches="tight", facecolor="white")
        plt.close(fig2)
        print(f"Saved Fig. 7 Wide (Two-Column IEEE): {out_png_2col} & {out_svg_2col}")


if __name__ == "__main__":
    generate_fig6()
    generate_fig7()
    print("\nALL IEEE SHAP FIGURES GENERATED SUCCESSFULLY!")
