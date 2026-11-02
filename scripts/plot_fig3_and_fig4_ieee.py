"""
scripts/plot_fig3_and_fig4_ieee.py
================================================================================
Scientific Figures Generator for BAB III Part C (IEEE Paper Format)
Figure 3: TEST MAE Comparison (Mean Baseline vs Median Baseline vs XGBoost)
Figure 4: Predicted vs Reference Scatter Plots (SBP, DBP, SpO2)
Style: IEEE Transactions / Conference Standard (Clean, Minimalist, 300+ DPI)
Outputs:
  - results/fig3_test_baseline_ieee.png (Vertical 3-panel single-column IEEE, 300 DPI)
  - results/fig3_test_baseline_ieee.svg (Vector format)
  - results/fig3_test_baseline_ieee_wide.png (Horizontal 2-column IEEE option, 300 DPI)
  - results/fig3_test_baseline_ieee_wide.svg (Vector format)
  - results/fig4_predicted_vs_reference_ieee.png (Vertical 3-panel single-column IEEE, 300 DPI)
  - results/fig4_predicted_vs_reference_ieee.svg (Vector format)
  - results/fig4_predicted_vs_reference_ieee_wide.png (Horizontal 2-column IEEE option, 300 DPI)
  - results/fig4_predicted_vs_reference_ieee_wide.svg (Vector format)
================================================================================
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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


def generate_fig3():
    """
    Fig. 3: TEST MAE Comparison
    Three-panel bar chart comparing Mean baseline, Median baseline, and XGBoost.
    Values:
      - SBP: Mean = 18.1315, Median = 18.0279, XGBoost = 13.4191 mmHg
      - DBP: Mean = 8.4828,  Median = 8.3505,  XGBoost = 6.4159 mmHg
      - SpO2: Mean = 1.9218, Median = 1.8688,  XGBoost = 1.7114 percentage points
    """
    print("=" * 70)
    print("GENERATING FIG. 3: TEST MAE COMPARISON")
    print("=" * 70)

    project_root = find_project_root()
    metrics_path = project_root / "results" / "xgboost_final_test_metrics.csv"
    results_dir = project_root / "results"

    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing file: {metrics_path}")

    df_metrics = pd.read_csv(metrics_path)

    # Verify rows exist
    targets_config = [
        {
            "target": "sbp",
            "name": "Systolic Blood Pressure (SBP)",
            "panel_label": "(a) Systolic Blood Pressure (SBP)",
            "unit": "mmHg",
            "y_label": "Test MAE (mmHg)",
            "ylim": (0, 22.0),
            "yticks": [0, 5, 10, 15, 20],
        },
        {
            "target": "dbp",
            "name": "Diastolic Blood Pressure (DBP)",
            "panel_label": "(b) Diastolic Blood Pressure (DBP)",
            "unit": "mmHg",
            "y_label": "Test MAE (mmHg)",
            "ylim": (0, 10.5),
            "yticks": [0, 2, 4, 6, 8, 10],
        },
        {
            "target": "spo2",
            "name": "Oxygen Saturation (SpO2)",
            "panel_label": "(c) Oxygen Saturation (SpO2)",
            "unit": "percentage points",
            "y_label": "Test MAE (% points)",
            "ylim": (0, 2.45),
            "yticks": [0, 0.5, 1.0, 1.5, 2.0],
        },
    ]

    # Bar appearance
    color_mean = "#9EABB8"      # Light slate
    color_median = "#697C8E"    # Medium slate
    color_xgboost = "#1E6091"   # Prominent IEEE blue

    edge_mean = "#72808E"
    edge_median = "#4E5C6B"
    edge_xgboost = "#123D5E"

    bar_labels = ["Mean Baseline", "Median Baseline", "XGBoost"]

    # =========================================================================
    # FIG 3 - OPTION A: VERTICAL 3-PANEL (SINGLE-COLUMN IEEE, 3.5 in / 89 mm)
    # =========================================================================
    fig_w, fig_h = 3.5, 7.3
    fig, axes = plt.subplots(3, 1, figsize=(fig_w, fig_h), dpi=300)

    for i, t_cfg in enumerate(targets_config):
        ax = axes[i]
        t = t_cfg["target"]

        # Extract values
        m_mean = df_metrics[(df_metrics["target"] == t) & (df_metrics["model_type"].str.contains("Mean"))]["mae"].values[0]
        m_med = df_metrics[(df_metrics["target"] == t) & (df_metrics["model_type"].str.contains("Median"))]["mae"].values[0]
        m_xgb = df_metrics[(df_metrics["target"] == t) & (df_metrics["model_type"].str.contains("XGBoost"))]["mae"].values[0]

        values = [m_mean, m_med, m_xgb]
        colors = [color_mean, color_median, color_xgboost]
        edges = [edge_mean, edge_median, edge_xgboost]

        x_pos = np.arange(len(values))
        bars = ax.bar(x_pos, values, width=0.52, color=colors, edgecolor=edges, linewidth=0.9, zorder=3)

        # Value annotations on top of bars (NO asterisks)
        for j, bar in enumerate(bars):
            h = bar.get_height()
            is_xgb = (j == 2)
            font_wt = "bold" if is_xgb else "normal"
            txt_col = "#123D5E" if is_xgb else "#222222"

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + (t_cfg["ylim"][1] * 0.025),
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight=font_wt,
                color=txt_col,
                zorder=4,
            )

        ax.set_title(t_cfg["panel_label"], fontsize=9.0, fontweight="bold", pad=6)
        ax.set_ylabel(t_cfg["y_label"], fontsize=8.2, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(bar_labels, fontsize=7.8)
        ax.set_ylim(t_cfg["ylim"])
        ax.set_yticks(t_cfg["yticks"])
        ax.yaxis.grid(True, linestyle="--", alpha=0.4, color="#AAAAAA", zorder=0)
        ax.set_axisbelow(True)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout(pad=0.8)
    plt.subplots_adjust(hspace=0.45)

    out_png_1col = results_dir / "fig3_test_baseline_ieee.png"
    out_svg_1col = results_dir / "fig3_test_baseline_ieee.svg"
    fig.savefig(out_png_1col, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_svg_1col, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved Fig. 3 (Single-Column IEEE): {out_png_1col} & {out_svg_1col}")

    # =========================================================================
    # FIG 3 - OPTION B: HORIZONTAL 3-PANEL (TWO-COLUMN IEEE, 7.16 in / 182 mm)
    # =========================================================================
    fig_w2, fig_h2 = 7.16, 2.85
    fig2, axes2 = plt.subplots(1, 3, figsize=(fig_w2, fig_h2), dpi=300)

    for i, t_cfg in enumerate(targets_config):
        ax = axes2[i]
        t = t_cfg["target"]

        m_mean = df_metrics[(df_metrics["target"] == t) & (df_metrics["model_type"].str.contains("Mean"))]["mae"].values[0]
        m_med = df_metrics[(df_metrics["target"] == t) & (df_metrics["model_type"].str.contains("Median"))]["mae"].values[0]
        m_xgb = df_metrics[(df_metrics["target"] == t) & (df_metrics["model_type"].str.contains("XGBoost"))]["mae"].values[0]

        values = [m_mean, m_med, m_xgb]
        colors = [color_mean, color_median, color_xgboost]
        edges = [edge_mean, edge_median, edge_xgboost]

        x_pos = np.arange(len(values))
        bars = ax.bar(x_pos, values, width=0.55, color=colors, edgecolor=edges, linewidth=0.9, zorder=3)

        for j, bar in enumerate(bars):
            h = bar.get_height()
            is_xgb = (j == 2)
            font_wt = "bold" if is_xgb else "normal"
            txt_col = "#123D5E" if is_xgb else "#222222"

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + (t_cfg["ylim"][1] * 0.025),
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight=font_wt,
                color=txt_col,
                zorder=4,
            )

        ax.set_title(t_cfg["panel_label"], fontsize=9.2, fontweight="bold", pad=8)
        ax.set_ylabel(t_cfg["y_label"], fontsize=8.5, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(bar_labels, fontsize=7.8)
        ax.set_ylim(t_cfg["ylim"])
        ax.set_yticks(t_cfg["yticks"])
        ax.yaxis.grid(True, linestyle="--", alpha=0.4, color="#AAAAAA", zorder=0)
        ax.set_axisbelow(True)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout(pad=0.9)
    plt.subplots_adjust(wspace=0.32)

    out_png_2col = results_dir / "fig3_test_baseline_ieee_wide.png"
    out_svg_2col = results_dir / "fig3_test_baseline_ieee_wide.svg"
    fig2.savefig(out_png_2col, dpi=300, bbox_inches="tight", facecolor="white")
    fig2.savefig(out_svg_2col, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    print(f"Saved Fig. 3 Wide (Two-Column IEEE): {out_png_2col} & {out_svg_2col}")


def generate_fig4():
    """
    Fig. 4: Predicted vs Reference Scatter Plots
    Three panels:
      (a) SBP (UCI Test: 12,507 valid pairs)
      (b) DBP (UCI Test: 12,507 valid pairs)
      (c) SpO2 (OSASUD Test: 4,706 valid pairs)
    Features:
      - Coordinate matching so y = x line is exact 45-degree diagonal.
      - Transparency to show density without synthetic points or altered data.
      - Exact sample counts and performance metrics stated on each panel.
    """
    print("=" * 70)
    print("GENERATING FIG. 4: PREDICTED VERSUS REFERENCE")
    print("=" * 70)

    project_root = find_project_root()
    uci_path = project_root / "results" / "xgboost_final_predictions_uci.csv"
    osasud_path = project_root / "results" / "xgboost_final_predictions_osasud.csv"
    results_dir = project_root / "results"

    if not uci_path.is_file() or not osasud_path.is_file():
        raise FileNotFoundError(f"Missing prediction files: {uci_path} or {osasud_path}")

    # Load UCI predictions
    df_uci = pd.read_csv(uci_path).dropna(subset=["sbp_mmhg", "pred_sbp_config_c", "dbp_mmhg", "pred_dbp_config_c"])
    # Load OSASUD predictions
    df_osasud = pd.read_csv(osasud_path).dropna(subset=["spo2_target_pct", "pred_spo2_config_d"])

    n_sbp = len(df_uci)
    n_dbp = len(df_uci)
    n_spo2 = len(df_osasud)

    print(f"Actual Valid TEST Sample Sizes: SBP N={n_sbp:,}, DBP N={n_dbp:,}, SpO2 N={n_spo2:,}")

    panels_info = [
        {
            "title": "(a) Systolic Blood Pressure (SBP)",
            "ref": df_uci["sbp_mmhg"].values,
            "pred": df_uci["pred_sbp_config_c"].values,
            "unit": "mmHg",
            "x_label": "Reference Finapres SBP (mmHg)",
            "y_label": "Predicted XGBoost SBP (mmHg)",
            "lims": (65, 205),
            "ticks": [80, 100, 120, 140, 160, 180, 200],
            "color": "#1E6091",
            "alpha": 0.16,
            "size": 8,
            "mae": 13.42,
            "r": 0.616,
            "n": n_sbp,
        },
        {
            "title": "(b) Diastolic Blood Pressure (DBP)",
            "ref": df_uci["dbp_mmhg"].values,
            "pred": df_uci["pred_dbp_config_c"].values,
            "unit": "mmHg",
            "x_label": "Reference Finapres DBP (mmHg)",
            "y_label": "Predicted XGBoost DBP (mmHg)",
            "lims": (45, 135),
            "ticks": [50, 70, 90, 110, 130],
            "color": "#237844",
            "alpha": 0.16,
            "size": 8,
            "mae": 6.42,
            "r": 0.574,
            "n": n_dbp,
        },
        {
            "title": "(c) Oxygen Saturation (SpO2)",
            "ref": df_osasud["spo2_target_pct"].values,
            "pred": df_osasud["pred_spo2_config_d"].values,
            "unit": "%",
            "x_label": "Reference Oximeter SpO2 (%)",
            "y_label": "Predicted XGBoost SpO2 (%)",
            "lims": (65, 102),
            "ticks": [70, 75, 80, 85, 90, 95, 100],
            "color": "#B83227",
            "alpha": 0.22,
            "size": 10,
            "mae": 1.71,
            "r": 0.378,
            "n": n_spo2,
        },
    ]

    # =========================================================================
    # FIG 4 - OPTION A: VERTICAL 3-PANEL (SINGLE-COLUMN IEEE, 3.5 in / 89 mm)
    # =========================================================================
    fig_w, fig_h = 3.5, 8.8
    fig, axes = plt.subplots(3, 1, figsize=(fig_w, fig_h), dpi=300)

    for i, p_info in enumerate(panels_info):
        ax = axes[i]
        ref = p_info["ref"]
        pred = p_info["pred"]
        lims = p_info["lims"]

        # Scatter points (100% real data)
        ax.scatter(ref, pred, s=p_info["size"], color=p_info["color"], alpha=p_info["alpha"], edgecolors="none", zorder=2)

        # Ideal y = x line
        ax.plot(lims, lims, color="#222222", linestyle="--", linewidth=1.1, label="Ideal (y = x)", zorder=3)

        ax.set_title(p_info["title"], fontsize=9.0, fontweight="bold", pad=5)
        ax.set_xlabel(p_info["x_label"], fontsize=8.0, fontweight="bold")
        ax.set_ylabel(p_info["y_label"], fontsize=8.0, fontweight="bold")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xticks(p_info["ticks"])
        ax.set_yticks(p_info["ticks"])
        ax.set_aspect("equal", "box")

        ax.grid(True, linestyle="--", alpha=0.35, color="#AAAAAA", zorder=0)

        # Informative statistics box
        stats_text = f"N = {p_info['n']:,}\nMAE = {p_info['mae']:.2f} {p_info['unit']}\nPearson r = {p_info['r']:.3f}"
        ax.text(
            0.04,
            0.96,
            stats_text,
            transform=ax.transAxes,
            fontsize=7.2,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CCCCCC", alpha=0.92),
            zorder=5,
        )

        ax.legend(loc="lower right", fontsize=7.0, framealpha=0.92, edgecolor="#CCCCCC")

    plt.tight_layout(pad=0.6)
    plt.subplots_adjust(hspace=0.38)

    out_png_1col = results_dir / "fig4_predicted_vs_reference_ieee.png"
    out_svg_1col = results_dir / "fig4_predicted_vs_reference_ieee.svg"
    fig.savefig(out_png_1col, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_svg_1col, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved Fig. 4 (Single-Column IEEE): {out_png_1col} & {out_svg_1col}")

    # =========================================================================
    # FIG 4 - OPTION B: HORIZONTAL 3-PANEL (TWO-COLUMN IEEE, 7.16 in / 182 mm)
    # =========================================================================
    fig_w2, fig_h2 = 7.16, 2.75
    fig2, axes2 = plt.subplots(1, 3, figsize=(fig_w2, fig_h2), dpi=300)

    for i, p_info in enumerate(panels_info):
        ax = axes2[i]
        ref = p_info["ref"]
        pred = p_info["pred"]
        lims = p_info["lims"]

        ax.scatter(ref, pred, s=p_info["size"], color=p_info["color"], alpha=p_info["alpha"], edgecolors="none", zorder=2)
        ax.plot(lims, lims, color="#222222", linestyle="--", linewidth=1.1, label="Ideal (y = x)", zorder=3)

        ax.set_title(p_info["title"], fontsize=8.8, fontweight="bold", pad=6)
        ax.set_xlabel(p_info["x_label"], fontsize=7.8, fontweight="bold")
        ax.set_ylabel(p_info["y_label"], fontsize=7.8, fontweight="bold")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xticks(p_info["ticks"])
        ax.set_yticks(p_info["ticks"])
        ax.set_aspect("equal", "box")

        ax.grid(True, linestyle="--", alpha=0.35, color="#AAAAAA", zorder=0)

        stats_text = f"N = {p_info['n']:,}\nMAE = {p_info['mae']:.2f} {p_info['unit']}\nr = {p_info['r']:.3f}"
        ax.text(
            0.04,
            0.96,
            stats_text,
            transform=ax.transAxes,
            fontsize=7.0,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CCCCCC", alpha=0.92),
            zorder=5,
        )

        ax.legend(loc="lower right", fontsize=6.8, framealpha=0.92, edgecolor="#CCCCCC")

    plt.tight_layout(pad=0.8)
    plt.subplots_adjust(wspace=0.32)

    out_png_2col = results_dir / "fig4_predicted_vs_reference_ieee_wide.png"
    out_svg_2col = results_dir / "fig4_predicted_vs_reference_ieee_wide.svg"
    fig2.savefig(out_png_2col, dpi=300, bbox_inches="tight", facecolor="white")
    fig2.savefig(out_svg_2col, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    print(f"Saved Fig. 4 Wide (Two-Column IEEE): {out_png_2col} & {out_svg_2col}")


if __name__ == "__main__":
    generate_fig3()
    generate_fig4()
    print("\nALL IEEE TEST FIGURES GENERATED SUCCESSFULLY!")
