"""
scripts/plot_fig2_validation_mae.py
================================================================================
Figure 2 Generator: Validation MAE across XGBoost Feature Configurations
Style: IEEE Transactions / Conference Standard (Clean, Minimalist, 300+ DPI)
Source: results/xgboost_validation_metrics.csv
Outputs:
  - results/fig2_validation_mae_ieee.png (Two-column width, 300 DPI)
  - results/fig2_validation_mae_ieee.svg (Two-column width, Vector)
  - results/fig2_validation_mae_ieee_1col.png (Single-column width, 300 DPI)
  - results/fig2_validation_mae_ieee_1col.svg (Single-column width, Vector)
================================================================================
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Configure matplotlib for publication quality
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


def generate_fig2():
    project_root = find_project_root()
    csv_path = project_root / "results" / "xgboost_validation_metrics.csv"
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.is_file():
        raise FileNotFoundError(f"Source file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Filter XGBoost configurations
    df_xgb = df[df["config"].isin(["config_a", "config_b", "config_c", "config_d"])].copy()

    # Data structures for 3 targets
    targets_info = [
        {
            "target": "sbp",
            "name": "Systolic Blood Pressure (SBP)",
            "panel_label": "(a) Systolic Blood Pressure (SBP)",
            "unit": "mmHg",
            "y_label": "Validation MAE (mmHg)",
            "ylim": (0, 17.5),
            "yticks": [0, 3, 6, 9, 12, 15],
            "feature_counts": {"config_a": 16, "config_b": 17, "config_c": 19, "config_d": 4},
        },
        {
            "target": "dbp",
            "name": "Diastolic Blood Pressure (DBP)",
            "panel_label": "(b) Diastolic Blood Pressure (DBP)",
            "unit": "mmHg",
            "y_label": "Validation MAE (mmHg)",
            "ylim": (0, 8.8),
            "yticks": [0, 2, 4, 6, 8],
            "feature_counts": {"config_a": 16, "config_b": 17, "config_c": 19, "config_d": 6},
        },
        {
            "target": "spo2",
            "name": "Oxygen Saturation (SpO2)",
            "panel_label": "(c) Oxygen Saturation (SpO2)",
            "unit": "percentage points",
            "y_label": "Validation MAE (% points)",
            "ylim": (0, 2.7),
            "yticks": [0, 0.5, 1.0, 1.5, 2.0, 2.5],
            "feature_counts": {"config_a": 16, "config_b": 17, "config_c": 19, "config_d": 2},
        },
    ]

    # Color palette
    color_regular = "#4A7BB0"       # Professional slate-blue
    edge_regular = "#2C4D6F"
    color_best = "#1B9E77"          # High-contrast emerald green for lowest MAE
    edge_best = "#0F5C45"

    # =========================================================================
    # OPTION 1: TWO-COLUMN IEEE LAYOUT (HORIZONTAL 1x3 PANELS, ~7.16 in / 182 mm)
    # =========================================================================
    fig_w, fig_h = 7.16, 2.9  # Standard IEEE 2-column width
    fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h), dpi=300)

    for i, t_info in enumerate(targets_info):
        ax = axes[i]
        sub = df_xgb[df_xgb["target"] == t_info["target"]].sort_values("config").copy()

        configs = sub["config"].tolist()
        maes = sub["val_mae"].tolist()
        min_mae = min(maes)
        best_idx = maes.index(min_mae)

        labels = [
            f"Config A\n({t_info['feature_counts']['config_a']} feats)",
            f"Config B\n({t_info['feature_counts']['config_b']} feats)",
            f"Config C\n({t_info['feature_counts']['config_c']} feats)",
            f"Config D\n({t_info['feature_counts']['config_d']} feats)",
        ]

        bar_colors = [color_best if j == best_idx else color_regular for j in range(len(configs))]
        edge_colors = [edge_best if j == best_idx else edge_regular for j in range(len(configs))]

        x_pos = np.arange(len(configs))
        bars = ax.bar(x_pos, maes, width=0.58, color=bar_colors, edgecolor=edge_colors, linewidth=0.9, zorder=3)

        # Labels on top of bars
        for j, bar in enumerate(bars):
            h = bar.get_height()
            is_best = (j == best_idx)

            if is_best:
                val_text = f"{h:.2f}*"
                font_weight = "bold"
                text_color = "#0F5C45"
            else:
                val_text = f"{h:.2f}"
                font_weight = "normal"
                text_color = "#222222"

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + (t_info["ylim"][1] * 0.025),
                val_text,
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight=font_weight,
                color=text_color,
                zorder=4,
            )

            # Badge / annotation for the best configuration
            if is_best:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + (t_info["ylim"][1] * 0.09),
                    "★ Lowest",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                    fontweight="bold",
                    color="#0F5C45",
                    zorder=4,
                )

        ax.set_title(t_info["panel_label"], fontsize=9.5, fontweight="bold", pad=8)
        ax.set_ylabel(t_info["y_label"], fontsize=8.5, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=7.8)
        ax.set_ylim(t_info["ylim"])
        ax.set_yticks(t_info["yticks"])
        ax.yaxis.grid(True, linestyle="--", alpha=0.4, color="#AAAAAA", zorder=0)
        ax.set_axisbelow(True)

        # Spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout(pad=1.0)
    plt.subplots_adjust(wspace=0.30)

    # Save 2-column IEEE outputs
    out_png_wide = results_dir / "fig2_validation_mae_ieee.png"
    out_svg_wide = results_dir / "fig2_validation_mae_ieee.svg"
    fig.savefig(out_png_wide, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_svg_wide, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved Two-Column Figure: {out_png_wide} & {out_svg_wide}")

    # =========================================================================
    # OPTION 2: SINGLE-COLUMN IEEE LAYOUT (VERTICAL 3x1 PANELS, ~3.5 in / 89 mm)
    # =========================================================================
    fig_w1, fig_h1 = 3.5, 7.2  # Standard IEEE 1-column width
    fig1, axes1 = plt.subplots(3, 1, figsize=(fig_w1, fig_h1), dpi=300)

    for i, t_info in enumerate(targets_info):
        ax = axes1[i]
        sub = df_xgb[df_xgb["target"] == t_info["target"]].sort_values("config").copy()

        configs = sub["config"].tolist()
        maes = sub["val_mae"].tolist()
        min_mae = min(maes)
        best_idx = maes.index(min_mae)

        labels = [
            f"Config A\n({t_info['feature_counts']['config_a']} feats)",
            f"Config B\n({t_info['feature_counts']['config_b']} feats)",
            f"Config C\n({t_info['feature_counts']['config_c']} feats)",
            f"Config D\n({t_info['feature_counts']['config_d']} feats)",
        ]

        bar_colors = [color_best if j == best_idx else color_regular for j in range(len(configs))]
        edge_colors = [edge_best if j == best_idx else edge_regular for j in range(len(configs))]

        x_pos = np.arange(len(configs))
        bars = ax.bar(x_pos, maes, width=0.55, color=bar_colors, edgecolor=edge_colors, linewidth=0.9, zorder=3)

        for j, bar in enumerate(bars):
            h = bar.get_height()
            is_best = (j == best_idx)

            if is_best:
                val_text = f"{h:.2f}*"
                font_weight = "bold"
                text_color = "#0F5C45"
            else:
                val_text = f"{h:.2f}"
                font_weight = "normal"
                text_color = "#222222"

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + (t_info["ylim"][1] * 0.025),
                val_text,
                ha="center",
                va="bottom",
                fontsize=8.0,
                fontweight=font_weight,
                color=text_color,
                zorder=4,
            )

            if is_best:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + (t_info["ylim"][1] * 0.10),
                    "★ Lowest",
                    ha="center",
                    va="bottom",
                    fontsize=7.2,
                    fontweight="bold",
                    color="#0F5C45",
                    zorder=4,
                )

        ax.set_title(t_info["panel_label"], fontsize=9.0, fontweight="bold", pad=6)
        ax.set_ylabel(t_info["y_label"], fontsize=8.0, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_ylim(t_info["ylim"])
        ax.set_yticks(t_info["yticks"])
        ax.yaxis.grid(True, linestyle="--", alpha=0.4, color="#AAAAAA", zorder=0)
        ax.set_axisbelow(True)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout(pad=0.8)
    plt.subplots_adjust(hspace=0.45)

    out_png_1col = results_dir / "fig2_validation_mae_ieee_1col.png"
    out_svg_1col = results_dir / "fig2_validation_mae_ieee_1col.svg"
    fig1.savefig(out_png_1col, dpi=300, bbox_inches="tight", facecolor="white")
    fig1.savefig(out_svg_1col, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig1)
    print(f"Saved Single-Column Figure: {out_png_1col} & {out_svg_1col}")


if __name__ == "__main__":
    generate_fig2()
