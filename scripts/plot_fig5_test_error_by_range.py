"""
scripts/plot_fig5_test_error_by_range.py
================================================================================
Figure 5 Generator: Test Prediction Errors Across Reference-Value Ranges
Style: IEEE Transactions / Conference Standard (Clean, Minimalist, 300+ DPI)
Source:
  - results/xgboost_final_predictions_uci.csv
  - results/xgboost_final_predictions_osasud.csv
Outputs:
  - results/fig5_test_error_by_range_ieee.png (Vertical 3-panel single-column IEEE, 300 DPI)
  - results/fig5_test_error_by_range_ieee.svg (Vector format)
  - results/fig5_test_error_by_range_ieee_wide.png (Horizontal 2-column IEEE option, 300 DPI)
  - results/fig5_test_error_by_range_ieee_wide.svg (Vector format)
================================================================================
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


def generate_fig5():
    print("=" * 70)
    print("GENERATING FIG. 5: TEST PREDICTION ERRORS ACROSS REFERENCE-VALUE RANGES")
    print("=" * 70)

    project_root = find_project_root()
    uci_path = project_root / "results" / "xgboost_final_predictions_uci.csv"
    osasud_path = project_root / "results" / "xgboost_final_predictions_osasud.csv"
    results_dir = project_root / "results"

    if not uci_path.is_file() or not osasud_path.is_file():
        raise FileNotFoundError(f"Missing prediction files: {uci_path} or {osasud_path}")

    # Load data
    df_u = pd.read_csv(uci_path).dropna(subset=["sbp_mmhg", "pred_sbp_config_c", "dbp_mmhg", "pred_dbp_config_c"])
    df_o = pd.read_csv(osasud_path).dropna(subset=["spo2_target_pct", "pred_spo2_config_d"])

    # Compute actual groups
    # SBP
    sbp_low = df_u[df_u["sbp_mmhg"] < 120]
    sbp_mid = df_u[(df_u["sbp_mmhg"] >= 120) & (df_u["sbp_mmhg"] < 140)]
    sbp_high = df_u[df_u["sbp_mmhg"] >= 140]

    sbp_mae_low = float(np.mean(np.abs(sbp_low["pred_sbp_config_c"] - sbp_low["sbp_mmhg"])))
    sbp_mae_mid = float(np.mean(np.abs(sbp_mid["pred_sbp_config_c"] - sbp_mid["sbp_mmhg"])))
    sbp_mae_high = float(np.mean(np.abs(sbp_high["pred_sbp_config_c"] - sbp_high["sbp_mmhg"])))

    # DBP
    dbp_low = df_u[df_u["dbp_mmhg"] < 80]
    dbp_mid = df_u[(df_u["dbp_mmhg"] >= 80) & (df_u["dbp_mmhg"] < 90)]
    dbp_high = df_u[df_u["dbp_mmhg"] >= 90]

    dbp_mae_low = float(np.mean(np.abs(dbp_low["pred_dbp_config_c"] - dbp_low["dbp_mmhg"])))
    dbp_mae_mid = float(np.mean(np.abs(dbp_mid["pred_dbp_config_c"] - dbp_mid["dbp_mmhg"])))
    dbp_mae_high = float(np.mean(np.abs(dbp_high["pred_dbp_config_c"] - dbp_high["dbp_mmhg"])))

    # SpO2
    sp_low = df_o[df_o["spo2_target_pct"] < 90]
    sp_mid = df_o[(df_o["spo2_target_pct"] >= 90) & (df_o["spo2_target_pct"] < 95)]
    sp_high = df_o[df_o["spo2_target_pct"] >= 95]

    sp_mae_low = float(np.mean(np.abs(sp_low["pred_spo2_config_d"] - sp_low["spo2_target_pct"])))
    sp_mae_mid = float(np.mean(np.abs(sp_mid["pred_spo2_config_d"] - sp_mid["spo2_target_pct"])))
    sp_mae_high = float(np.mean(np.abs(sp_high["pred_spo2_config_d"] - sp_high["spo2_target_pct"])))

    # Verification prints
    print(f"SBP: <120 (N={len(sbp_low)}, MAE={sbp_mae_low:.3f}), 120-139 (N={len(sbp_mid)}, MAE={sbp_mae_mid:.3f}), >=140 (N={len(sbp_high)}, MAE={sbp_mae_high:.3f})")
    print(f"DBP: <80 (N={len(dbp_low)}, MAE={dbp_mae_low:.3f}), 80-89 (N={len(dbp_mid)}, MAE={dbp_mae_mid:.3f}), >=90 (N={len(dbp_high)}, MAE={dbp_mae_high:.3f})")
    print(f"SpO2: <90% (N={len(sp_low)}, MAE={sp_mae_low:.3f}), 90-<95% (N={len(sp_mid)}, MAE={sp_mae_mid:.3f}), >=95% (N={len(sp_high)}, MAE={sp_mae_high:.3f})")

    panels_data = [
        {
            "panel_title": "(a) Systolic Blood Pressure (SBP)",
            "y_label": "Test MAE (mmHg)",
            "unit": "mmHg",
            "ylim": (0, 20.5),
            "yticks": [0, 5, 10, 15, 20],
            "labels": [
                f"< 120 mmHg\n(n = {len(sbp_low):,})",
                f"120–139 mmHg\n(n = {len(sbp_mid):,})",
                f"≥ 140 mmHg\n(n = {len(sbp_high):,})",
            ],
            "values": [sbp_mae_low, sbp_mae_mid, sbp_mae_high],
            "color": "#1E6091",
            "edge": "#123D5E",
            "optimal_idx": 1,  # 120-139 mmHg is the lowest MAE
        },
        {
            "panel_title": "(b) Diastolic Blood Pressure (DBP)",
            "y_label": "Test MAE (mmHg)",
            "unit": "mmHg",
            "ylim": (0, 33.0),
            "yticks": [0, 5, 10, 15, 20, 25, 30],
            "labels": [
                f"< 80 mmHg\n(n = {len(dbp_low):,})",
                f"80–89 mmHg\n(n = {len(dbp_mid):,})",
                f"≥ 90 mmHg\n(n = {len(dbp_high):,})",
            ],
            "values": [dbp_mae_low, dbp_mae_mid, dbp_mae_high],
            "color": "#237844",
            "edge": "#134927",
            "optimal_idx": 0,  # <80 mmHg is the lowest MAE
        },
        {
            "panel_title": "(c) Oxygen Saturation (SpO2)",
            "y_label": "Test MAE (% points)",
            "unit": "% points",
            "ylim": (0, 9.8),
            "yticks": [0, 2, 4, 6, 8],
            "labels": [
                f"< 90%\n(n = {len(sp_low):,})",
                f"90–<95%\n(n = {len(sp_mid):,})",
                f"≥ 95%\n(n = {len(sp_high):,})",
            ],
            "values": [sp_mae_low, sp_mae_mid, sp_mae_high],
            "color": "#B83227",
            "edge": "#791E16",
            "optimal_idx": 2,  # >=95% is the lowest MAE
        },
    ]

    # =========================================================================
    # OPTION 1: VERTICAL 3-PANEL (SINGLE-COLUMN IEEE, 3.5 in / 89 mm)
    # =========================================================================
    fig_w, fig_h = 3.5, 7.6
    fig, axes = plt.subplots(3, 1, figsize=(fig_w, fig_h), dpi=300)

    for i, p in enumerate(panels_data):
        ax = axes[i]
        vals = p["values"]
        x_pos = np.arange(len(vals))

        bars = ax.bar(x_pos, vals, width=0.52, color=p["color"], edgecolor=p["edge"], linewidth=0.9, zorder=3)

        # Labels on top of bars
        for j, bar in enumerate(bars):
            h = bar.get_height()
            is_optimal = (j == p["optimal_idx"])
            font_wt = "bold" if is_optimal else "normal"

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + (p["ylim"][1] * 0.025),
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=8.2,
                fontweight=font_wt,
                color="#111111",
                zorder=4,
            )

        ax.set_title(p["panel_title"], fontsize=9.0, fontweight="bold", pad=6)
        ax.set_ylabel(p["y_label"], fontsize=8.0, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(p["labels"], fontsize=7.5)
        ax.set_ylim(p["ylim"])
        ax.set_yticks(p["yticks"])
        ax.yaxis.grid(True, linestyle="--", alpha=0.4, color="#AAAAAA", zorder=0)
        ax.set_axisbelow(True)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout(pad=0.8)
    plt.subplots_adjust(hspace=0.45)

    out_png = results_dir / "fig5_test_error_by_range_ieee.png"
    out_svg = results_dir / "fig5_test_error_by_range_ieee.svg"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_svg, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved Fig. 5 (Single-Column IEEE): {out_png} & {out_svg}")

    # =========================================================================
    # OPTION 2: HORIZONTAL 3-PANEL (TWO-COLUMN IEEE, 7.16 in / 182 mm)
    # =========================================================================
    fig_w2, fig_h2 = 7.16, 2.85
    fig2, axes2 = plt.subplots(1, 3, figsize=(fig_w2, fig_h2), dpi=300)

    for i, p in enumerate(panels_data):
        ax = axes2[i]
        vals = p["values"]
        x_pos = np.arange(len(vals))

        bars = ax.bar(x_pos, vals, width=0.55, color=p["color"], edgecolor=p["edge"], linewidth=0.9, zorder=3)

        for j, bar in enumerate(bars):
            h = bar.get_height()
            is_optimal = (j == p["optimal_idx"])
            font_wt = "bold" if is_optimal else "normal"

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + (p["ylim"][1] * 0.025),
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=8.2,
                fontweight=font_wt,
                color="#111111",
                zorder=4,
            )

        ax.set_title(p["panel_title"], fontsize=9.2, fontweight="bold", pad=8)
        ax.set_ylabel(p["y_label"], fontsize=8.2, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(p["labels"], fontsize=7.5)
        ax.set_ylim(p["ylim"])
        ax.set_yticks(p["yticks"])
        ax.yaxis.grid(True, linestyle="--", alpha=0.4, color="#AAAAAA", zorder=0)
        ax.set_axisbelow(True)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout(pad=0.9)
    plt.subplots_adjust(wspace=0.32)

    out_png_wide = results_dir / "fig5_test_error_by_range_ieee_wide.png"
    out_svg_wide = results_dir / "fig5_test_error_by_range_ieee_wide.svg"
    fig2.savefig(out_png_wide, dpi=300, bbox_inches="tight", facecolor="white")
    fig2.savefig(out_svg_wide, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    print(f"Saved Fig. 5 Wide (Two-Column IEEE): {out_png_wide} & {out_svg_wide}")


if __name__ == "__main__":
    generate_fig5()
    print("\nFIG. 5 GENERATION COMPLETED SUCCESSFULLY!")
