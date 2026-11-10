"""
scripts/plot_fig8_ppg_apg_morphology_ieee.py
================================================================================
Scientific Figure Generator for KTI Paper (PulseBoost)
Figures Generated:
  1. Figure 8 Vertical (Single-Column IEEE ~3.5 in):
     - results/fig8_ppg_preprocessing_ieee.png / .svg
     - assets/images/kti/fig8_ppg_preprocessing_ieee.png / .svg
  2. Figure 8 Horizontal (Two-Column IEEE Side-by-Side ~7.16 in):
     - assets/images/kti/fig8_ppg_preprocessing_horizontal.png / .svg
     - results/fig8_ppg_preprocessing_horizontal.png / .svg
  3. Figure 9 Vertical (Single-Column IEEE ~3.5 in):
     - results/fig9_ppg_apg_features_ieee.png / .svg
     - assets/images/kti/fig9_ppg_apg_features_ieee.png / .svg
  4. Figure 9 Horizontal (Two-Column IEEE Side-by-Side ~7.16 in):
     - assets/images/kti/fig9_ppg_apg_features_horizontal.png / .svg
     - results/fig9_ppg_apg_features_horizontal.png / .svg
  5. Archive Figures (Preserved 4-panel layouts):
     - results/fig8_ppg_apg_morphology_ieee.png / .svg (2x2)
     - results/fig8_ppg_apg_morphology_ieee_1col.png / .svg (1-column 4-panel)

Dataset: OSASUD (Patient 12, Window P12_W0011, t = 300-330 s)
Filter: Chebyshev Type II Bandpass [0.5, 8.0] Hz, Order 4, Rs = 20 dB, Zero-phase
APG: Savitzky-Golay Filter (Window Length = 9, Polyorder = 3, Deriv = 2, Fs = 80 Hz)
Style: IEEE Transactions Standard (Clean, Minimalist, 300+ DPI, Pure White Background)
================================================================================
"""

import sys
from pathlib import Path
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

# Matplotlib configuration for IEEE publication quality
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
    """Finds the root directory of the KTI_XGBoost project."""
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


def compute_signals_and_landmarks():
    """
    Loads OSASUD Patient 12 data, applies the exact continuous segment
    Chebyshev Type II zero-phase filter, extracts window P12_W0011,
    and isolates Beat 1 with its morphological parameters and APG fiducial points.
    """
    project_root = find_project_root()
    npz_path = project_root / "data" / "osasud" / "patients_npz" / "patient_12.npz"

    if not npz_path.is_file():
        raise FileNotFoundError(f"OSASUD Patient 12 data not found at: {npz_path}")

    with np.load(npz_path, allow_pickle=False) as npz:
        ppg_2d = npz["ppg_80hz"]

    fs = 80  # Hz

    # 1. Filter continuous segment (0 to 7920 s) to eliminate boundary edge artifacts
    seg_raw = ppg_2d[0:7920].flatten()

    cheby_order = 4
    cheby_rs_db = 20
    cheby_wn = [0.5, 8.0]
    cheby_sos = sig.cheby2(cheby_order, cheby_rs_db, cheby_wn, btype="bandpass", fs=fs, output="sos")
    seg_filt = sig.sosfiltfilt(cheby_sos, seg_raw)

    # 2. Savitzky-Golay 2nd derivative (APG) from final pipeline (WL=9, polyorder=3)
    savgol_wl = 9
    savgol_polyorder = 3
    seg_apg = sig.savgol_filter(seg_filt, window_length=savgol_wl, polyorder=savgol_polyorder, deriv=2, delta=1.0 / fs)

    # 3. 10-second snippet (detik 300 to 310) for raw & filtered PPG
    s_10 = 300 * fs
    e_10 = 310 * fs
    t_10 = np.arange(s_10, e_10) / fs
    raw_10 = seg_raw[s_10:e_10]
    filt_10 = seg_filt[s_10:e_10]

    # 4. Representative pulse: Beat 1 in window P12_W0011 (t = 301.5875 s to 302.8750 s)
    v_on = 24127
    p_sys = 24153
    v_off = 24230

    t_b_start = v_on / fs
    t_b_end = v_off / fs
    t_beat = (np.arange(v_on, v_off + 1) - v_on) / fs
    y_beat = seg_filt[v_on:v_off + 1]
    apg_beat = seg_apg[v_on:v_off + 1]

    # Baseline & PW50 calculation
    beat_samples = v_off - v_on + 1
    base_line = y_beat[0] + (y_beat[-1] - y_beat[0]) * (np.arange(beat_samples) / (beat_samples - 1))
    y_sub = y_beat - base_line
    rel_peak_idx = p_sys - v_on
    peak_height = y_sub[rel_peak_idx]
    thresh_50 = 0.5 * peak_height

    # Ascending & descending 50% crossings
    asc_y = y_sub[:rel_peak_idx + 1]
    k_asc = 0.0
    for k in range(len(asc_y) - 1):
        if asc_y[k] <= thresh_50 <= asc_y[k + 1]:
            k_asc = k + (thresh_50 - asc_y[k]) / (asc_y[k + 1] - asc_y[k])
            break
    t_cross_asc = k_asc / fs

    desc_y = y_sub[rel_peak_idx:]
    k_desc = 0.0
    for k in range(len(desc_y) - 1):
        if desc_y[k] >= thresh_50 >= desc_y[k + 1]:
            k_desc = rel_peak_idx + k + (desc_y[k] - thresh_50) / (desc_y[k] - desc_y[k + 1])
            break
    t_cross_desc = k_desc / fs

    pw50 = float(t_cross_desc - t_cross_asc)
    y_50_asc = y_beat[0] + (y_beat[-1] - y_beat[0]) * (k_asc / (beat_samples - 1)) + thresh_50
    y_50_desc = y_beat[0] + (y_beat[-1] - y_beat[0]) * (k_desc / (beat_samples - 1)) + thresh_50
    y_pw50 = float(0.5 * (y_50_asc + y_50_desc))

    rise_time_s = float((p_sys - v_on) / fs)
    pulse_amplitude = float(y_beat[rel_peak_idx] - y_beat[0])

    # 5. APG Fiducial Points according to Final Pipeline V2 Rules:
    idx_a = 2
    val_a = float(apg_beat[idx_a])
    idx_b = 12
    val_b = float(apg_beat[idx_b])
    idx_c = 38
    val_c = float(apg_beat[idx_c])
    idx_d = 47
    val_d = float(apg_beat[idx_d])
    idx_e = 53
    val_e = float(apg_beat[idx_e])

    fids = [
        {"name": "a", "idx": idx_a, "t": idx_a / fs, "val": val_a, "ratio": 1.0, "color": "#DC2626", "va": "bottom", "offset": 450},
        {"name": "b", "idx": idx_b, "t": idx_b / fs, "val": val_b, "ratio": val_b / val_a, "color": "#2563EB", "va": "top", "offset": -650},
        {"name": "c", "idx": idx_c, "t": idx_c / fs, "val": val_c, "ratio": val_c / val_a, "color": "#D97706", "va": "bottom", "offset": 450},
        {"name": "d", "idx": idx_d, "t": idx_d / fs, "val": val_d, "ratio": val_d / val_a, "color": "#9333EA", "va": "top", "offset": -650},
        {"name": "e", "idx": idx_e, "t": idx_e / fs, "val": val_e, "ratio": val_e / val_a, "color": "#B45309", "va": "bottom", "offset": 450},
    ]

    return {
        "t_10": t_10,
        "raw_10": raw_10,
        "filt_10": filt_10,
        "t_b_start": t_b_start,
        "t_b_end": t_b_end,
        "t_beat": t_beat,
        "y_beat": y_beat,
        "base_line": base_line,
        "rel_peak_idx": rel_peak_idx,
        "rise_time_s": rise_time_s,
        "pulse_amplitude": pulse_amplitude,
        "t_cross_asc": t_cross_asc,
        "t_cross_desc": t_cross_desc,
        "pw50": pw50,
        "y_pw50": y_pw50,
        "apg_beat": apg_beat,
        "fids": fids,
    }


def generate_figure_8_vertical(data_dict, project_root: Path):
    """
    Figure 8 Vertical: PPG Preprocessing
    Single-column IEEE format (width = 3.5 in) with two vertical panels:
      (a) Raw PPG Signal
      (b) Filtered PPG Signal
    """
    results_dir = project_root / "results"
    assets_dir = project_root / "assets" / "images" / "kti"

    fig_w, fig_h = 3.5, 3.8
    fig, axes = plt.subplots(2, 1, figsize=(fig_w, fig_h), dpi=300)

    # (a) Raw PPG Signal
    ax_a = axes[0]
    ax_a.plot(data_dict["t_10"], data_dict["raw_10"], color="#2D3748", lw=1.1, zorder=2)
    ax_a.set_title("(a) Raw PPG Signal", fontsize=9.2, fontweight="bold", pad=5)
    ax_a.set_ylabel("Amplitude (a.u.)", fontsize=8.2, fontweight="bold")
    ax_a.set_xlim(300, 310)
    ax_a.set_xticks([300, 302, 304, 306, 308, 310])
    ax_a.set_ylim(-10, 185)
    ax_a.tick_params(labelsize=7.5)
    ax_a.grid(True, linestyle="--", alpha=0.45, color="#CBD5E1", zorder=0)
    ax_a.set_axisbelow(True)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)

    # (b) Filtered PPG Signal
    ax_b = axes[1]
    ax_b.axhline(0, color="#94A3B8", linestyle=":", lw=0.8, alpha=0.7, zorder=1)
    ax_b.plot(data_dict["t_10"], data_dict["filt_10"], color="#1E6091", lw=1.1, zorder=2)
    ax_b.axvspan(data_dict["t_b_start"], data_dict["t_b_end"], color="#F59E0B", alpha=0.18, zorder=0, label="Selected pulse")
    ax_b.set_title("(b) Filtered PPG Signal", fontsize=9.2, fontweight="bold", pad=5)
    ax_b.set_xlabel("Time (s)", fontsize=8.2, fontweight="bold")
    ax_b.set_ylabel("Amplitude (a.u.)", fontsize=8.2, fontweight="bold")
    ax_b.set_xlim(300, 310)
    ax_b.set_xticks([300, 302, 304, 306, 308, 310])
    ax_b.set_ylim(-75, 75)
    ax_b.tick_params(labelsize=7.5)
    ax_b.grid(True, linestyle="--", alpha=0.45, color="#CBD5E1", zorder=0)
    ax_b.set_axisbelow(True)
    ax_b.legend(fontsize=7.0, loc="upper right", frameon=True, facecolor="white", edgecolor="#CBD5E1")
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    plt.tight_layout(pad=0.8)
    plt.subplots_adjust(hspace=0.38)

    out_png = results_dir / "fig8_ppg_preprocessing_ieee.png"
    out_svg = results_dir / "fig8_ppg_preprocessing_ieee.svg"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_svg, format="svg", bbox_inches="tight", facecolor="white")
    print(f"Saved Fig. 8 Vertical: {out_png}")
    print(f"Saved Fig. 8 Vertical: {out_svg}")

    if assets_dir.is_dir():
        asset_png = assets_dir / "fig8_ppg_preprocessing_ieee.png"
        asset_svg = assets_dir / "fig8_ppg_preprocessing_ieee.svg"
        fig.savefig(asset_png, dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(asset_svg, format="svg", bbox_inches="tight", facecolor="white")
        print(f"Saved copy to assets: {asset_png}")

    plt.close(fig)


def generate_figure_8_horizontal(data_dict, project_root: Path):
    """
    Figure 8 Horizontal: PPG Preprocessing
    Two-column IEEE side-by-side format (1 row x 2 columns, width = 7.16 in):
      Panel (a) — Raw PPG Signal
      Panel (b) — Filtered PPG Signal
    """
    results_dir = project_root / "results"
    assets_dir = project_root / "assets" / "images" / "kti"

    fig_w, fig_h = 7.16, 2.7
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h), dpi=300)

    # (a) Raw PPG Signal
    axes[0].plot(data_dict["t_10"], data_dict["raw_10"], color="#2D3748", lw=1.1, zorder=2)
    axes[0].set_title("(a) Raw PPG Signal", fontsize=9.2, fontweight="bold", pad=5)
    axes[0].set_xlabel("Time (s)", fontsize=8.2, fontweight="bold")
    axes[0].set_ylabel("Amplitude (a.u.)", fontsize=8.2, fontweight="bold")
    axes[0].set_xlim(300, 310)
    axes[0].set_xticks([300, 302, 304, 306, 308, 310])
    axes[0].set_ylim(-10, 185)
    axes[0].tick_params(labelsize=7.5)
    axes[0].grid(True, linestyle="--", alpha=0.45, color="#CBD5E1", zorder=0)
    axes[0].set_axisbelow(True)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # (b) Filtered PPG Signal
    axes[1].axhline(0, color="#94A3B8", linestyle=":", lw=0.8, alpha=0.7, zorder=1)
    axes[1].plot(data_dict["t_10"], data_dict["filt_10"], color="#1E6091", lw=1.1, zorder=2)
    axes[1].axvspan(data_dict["t_b_start"], data_dict["t_b_end"], color="#F59E0B", alpha=0.18, zorder=0, label="Selected pulse")
    axes[1].set_title("(b) Filtered PPG Signal", fontsize=9.2, fontweight="bold", pad=5)
    axes[1].set_xlabel("Time (s)", fontsize=8.2, fontweight="bold")
    axes[1].set_ylabel("Amplitude (a.u.)", fontsize=8.2, fontweight="bold")
    axes[1].set_xlim(300, 310)
    axes[1].set_xticks([300, 302, 304, 306, 308, 310])
    axes[1].set_ylim(-75, 75)
    axes[1].tick_params(labelsize=7.5)
    axes[1].grid(True, linestyle="--", alpha=0.45, color="#CBD5E1", zorder=0)
    axes[1].set_axisbelow(True)
    axes[1].legend(fontsize=7.2, loc="upper right", frameon=True, facecolor="white", edgecolor="#CBD5E1")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    plt.tight_layout(pad=0.8)
    plt.subplots_adjust(wspace=0.22)

    out_png_asset = assets_dir / "fig8_ppg_preprocessing_horizontal.png"
    out_svg_asset = assets_dir / "fig8_ppg_preprocessing_horizontal.svg"
    fig.savefig(out_png_asset, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_svg_asset, format="svg", bbox_inches="tight", facecolor="white")
    print(f"Saved Fig. 8 Horizontal: {out_png_asset}")
    print(f"Saved Fig. 8 Horizontal: {out_svg_asset}")

    # Also keep a copy in results/
    out_png_res = results_dir / "fig8_ppg_preprocessing_horizontal.png"
    out_svg_res = results_dir / "fig8_ppg_preprocessing_horizontal.svg"
    fig.savefig(out_png_res, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_svg_res, format="svg", bbox_inches="tight", facecolor="white")

    plt.close(fig)


def generate_figure_9_vertical(data_dict, project_root: Path):
    """
    Figure 9 Vertical: PPG and APG Feature Extraction
    Single-column IEEE format (width = 3.5 in) with two vertical panels:
      (a) Representative PPG Pulse Morphology
      (b) Acceleration Photoplethysmogram (APG)
    """
    results_dir = project_root / "results"
    assets_dir = project_root / "assets" / "images" / "kti"

    fig_w, fig_h = 3.5, 4.3
    fig, axes = plt.subplots(2, 1, figsize=(fig_w, fig_h), dpi=300)

    # Panel (a): Representative PPG Pulse Morphology
    ax_a = axes[0]
    t_peak = data_dict["rise_time_s"]
    y_peak = data_dict["y_beat"][data_dict["rel_peak_idx"]]

    ax_a.plot(data_dict["t_beat"], data_dict["y_beat"], color="#1E6091", lw=1.4, zorder=2)
    ax_a.plot(data_dict["t_beat"], data_dict["base_line"], color="#94A3B8", linestyle="--", lw=0.8, zorder=1, label="Baseline")

    # Onset, Peak, Offset markers
    ax_a.scatter([0], [data_dict["y_beat"][0]], color="#16A34A", s=35, zorder=5, label="Onset")
    ax_a.scatter([t_peak], [y_peak], color="#DC2626", s=35, zorder=5, label="Peak")
    ax_a.scatter([data_dict["t_beat"][-1]], [data_dict["y_beat"][-1]], color="#7C3AED", s=35, zorder=5, label="Offset")

    # PW50 Line & Label
    ax_a.plot([data_dict["t_cross_asc"], data_dict["t_cross_desc"]], [data_dict["y_pw50"], data_dict["y_pw50"]],
              color="#D97706", lw=1.2, zorder=4)
    ax_a.scatter([data_dict["t_cross_asc"], data_dict["t_cross_desc"]], [data_dict["y_pw50"], data_dict["y_pw50"]],
                 color="#D97706", s=20, zorder=5)
    ax_a.text(0.5 * (data_dict["t_cross_asc"] + data_dict["t_cross_desc"]), data_dict["y_pw50"] + 6,
              f"PW50 = {data_dict['pw50']:.3f} s", ha="center", va="bottom", fontsize=7.2, fontweight="bold", color="#B45309")

    # Rise Time Line & Label
    ax_a.annotate("", xy=(t_peak, -60), xytext=(0, -60),
                  arrowprops=dict(arrowstyle="<->", color="#2563EB", lw=1.0))
    ax_a.text(0.01, -70, f"Rise Time = {t_peak:.3f} s", ha="left", va="top", fontsize=7.2, fontweight="bold", color="#1D4ED8")

    ax_a.set_title("(a) Representative PPG Pulse Morphology", fontsize=9.0, fontweight="bold", pad=5)
    ax_a.set_xlabel("Time within Beat (s)", fontsize=8.2, fontweight="bold")
    ax_a.set_ylabel("Amplitude (a.u.)", fontsize=8.2, fontweight="bold")
    ax_a.set_xlim(-0.06, 1.35)
    ax_a.set_ylim(-86, 82)
    ax_a.tick_params(labelsize=7.5)
    ax_a.grid(True, linestyle="--", alpha=0.45, color="#CBD5E1", zorder=0)
    ax_a.set_axisbelow(True)
    ax_a.legend(fontsize=6.8, loc="upper right", frameon=True, facecolor="white", edgecolor="#CBD5E1")
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)

    # Panel (b): Acceleration Photoplethysmogram (APG)
    ax_b = axes[1]
    ax_b.axhline(0, color="#94A3B8", linestyle=":", lw=0.8, alpha=0.7, zorder=1)
    ax_b.plot(data_dict["t_beat"], data_dict["apg_beat"], color="#059669", lw=1.3, zorder=2)

    for pt in data_dict["fids"]:
        ax_b.scatter([pt["t"]], [pt["val"]], color=pt["color"], s=30, zorder=5)
        ax_b.text(pt["t"], pt["val"] + pt["offset"], pt["name"], ha="center", va=pt["va"],
                  fontsize=8.5, fontweight="bold", color=pt["color"], zorder=6)

    # Fiducial Ratios in small corner box
    ratio_str = "b/a = -0.54 | c/a = 0.29\nd/a = -0.03 | e/a = 0.06"
    ax_b.text(0.96, 0.94, ratio_str, transform=ax_b.transAxes, ha="right", va="top",
              fontsize=6.8, fontweight="bold", color="#374151",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CBD5E1", alpha=0.9))

    ax_b.set_title("(b) Acceleration Photoplethysmogram (APG)", fontsize=9.0, fontweight="bold", pad=5)
    ax_b.set_xlabel("Time within Beat (s)", fontsize=8.2, fontweight="bold")
    ax_b.set_ylabel("APG (a.u./s²)", fontsize=8.2, fontweight="bold")
    ax_b.set_xlim(-0.06, 1.35)
    ax_b.set_ylim(-7200, 11500)
    ax_b.tick_params(labelsize=7.5)
    ax_b.grid(True, linestyle="--", alpha=0.45, color="#CBD5E1", zorder=0)
    ax_b.set_axisbelow(True)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    plt.tight_layout(pad=0.8)
    plt.subplots_adjust(hspace=0.40)

    out_png = results_dir / "fig9_ppg_apg_features_ieee.png"
    out_svg = results_dir / "fig9_ppg_apg_features_ieee.svg"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_svg, format="svg", bbox_inches="tight", facecolor="white")
    print(f"Saved Fig. 9 Vertical: {out_png}")
    print(f"Saved Fig. 9 Vertical: {out_svg}")

    if assets_dir.is_dir():
        asset_png = assets_dir / "fig9_ppg_apg_features_ieee.png"
        asset_svg = assets_dir / "fig9_ppg_apg_features_ieee.svg"
        fig.savefig(asset_png, dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(asset_svg, format="svg", bbox_inches="tight", facecolor="white")
        print(f"Saved copy to assets: {asset_png}")

    plt.close(fig)


def generate_figure_9_horizontal(data_dict, project_root: Path):
    """
    Figure 9 Horizontal: PPG and APG Feature Extraction
    Two-column IEEE side-by-side format (1 row x 2 columns, width = 7.16 in):
      Panel (a) — Representative PPG Pulse Morphology
      Panel (b) — Acceleration Photoplethysmogram (APG)
    """
    results_dir = project_root / "results"
    assets_dir = project_root / "assets" / "images" / "kti"

    fig_w, fig_h = 7.16, 2.7
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h), dpi=300)

    # (a) Representative PPG Pulse Morphology
    t_peak = data_dict["rise_time_s"]
    y_peak = data_dict["y_beat"][data_dict["rel_peak_idx"]]

    axes[0].plot(data_dict["t_beat"], data_dict["y_beat"], color="#1E6091", lw=1.4, zorder=2)
    axes[0].plot(data_dict["t_beat"], data_dict["base_line"], color="#94A3B8", linestyle="--", lw=0.8, zorder=1, label="Baseline")
    axes[0].scatter([0], [data_dict["y_beat"][0]], color="#16A34A", s=35, zorder=5, label="Onset")
    axes[0].scatter([t_peak], [y_peak], color="#DC2626", s=35, zorder=5, label="Peak")
    axes[0].scatter([data_dict["t_beat"][-1]], [data_dict["y_beat"][-1]], color="#7C3AED", s=35, zorder=5, label="Offset")

    axes[0].plot([data_dict["t_cross_asc"], data_dict["t_cross_desc"]], [data_dict["y_pw50"], data_dict["y_pw50"]],
                 color="#D97706", lw=1.2, zorder=4)
    axes[0].scatter([data_dict["t_cross_asc"], data_dict["t_cross_desc"]], [data_dict["y_pw50"], data_dict["y_pw50"]],
                    color="#D97706", s=20, zorder=5)
    axes[0].text(0.5 * (data_dict["t_cross_asc"] + data_dict["t_cross_desc"]), data_dict["y_pw50"] + 6,
                 f"PW50 = {data_dict['pw50']:.3f} s", ha="center", va="bottom", fontsize=7.2, fontweight="bold", color="#B45309")

    axes[0].annotate("", xy=(t_peak, -60), xytext=(0, -60), arrowprops=dict(arrowstyle="<->", color="#2563EB", lw=1.0))
    axes[0].text(0.01, -70, f"Rise Time = {t_peak:.3f} s", ha="left", va="top", fontsize=7.2, fontweight="bold", color="#1D4ED8")

    axes[0].set_title("(a) Representative PPG Pulse Morphology", fontsize=9.2, fontweight="bold", pad=5)
    axes[0].set_xlabel("Time within Beat (s)", fontsize=8.2, fontweight="bold")
    axes[0].set_ylabel("Amplitude (a.u.)", fontsize=8.2, fontweight="bold")
    axes[0].set_xlim(-0.06, 1.35)
    axes[0].set_ylim(-86, 82)
    axes[0].tick_params(labelsize=7.5)
    axes[0].grid(True, linestyle="--", alpha=0.45, color="#CBD5E1", zorder=0)
    axes[0].set_axisbelow(True)
    axes[0].legend(fontsize=7.0, loc="upper right", frameon=True, facecolor="white", edgecolor="#CBD5E1")
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # (b) Acceleration Photoplethysmogram (APG)
    axes[1].axhline(0, color="#94A3B8", linestyle=":", lw=0.8, alpha=0.7, zorder=1)
    axes[1].plot(data_dict["t_beat"], data_dict["apg_beat"], color="#059669", lw=1.3, zorder=2)

    for pt in data_dict["fids"]:
        axes[1].scatter([pt["t"]], [pt["val"]], color=pt["color"], s=30, zorder=5)
        axes[1].text(pt["t"], pt["val"] + pt["offset"], pt["name"], ha="center", va=pt["va"],
                     fontsize=8.5, fontweight="bold", color=pt["color"], zorder=6)

    ratio_str = "b/a = -0.54 | c/a = 0.29\nd/a = -0.03 | e/a = 0.06"
    axes[1].text(0.96, 0.94, ratio_str, transform=axes[1].transAxes, ha="right", va="top",
                 fontsize=6.8, fontweight="bold", color="#374151",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CBD5E1", alpha=0.9))

    axes[1].set_title("(b) Acceleration Photoplethysmogram (APG)", fontsize=9.2, fontweight="bold", pad=5)
    axes[1].set_xlabel("Time within Beat (s)", fontsize=8.2, fontweight="bold")
    axes[1].set_ylabel("APG (a.u./s²)", fontsize=8.2, fontweight="bold")
    axes[1].set_xlim(-0.06, 1.35)
    axes[1].set_ylim(-7200, 11500)
    axes[1].tick_params(labelsize=7.5)
    axes[1].grid(True, linestyle="--", alpha=0.45, color="#CBD5E1", zorder=0)
    axes[1].set_axisbelow(True)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    plt.tight_layout(pad=0.8)
    plt.subplots_adjust(wspace=0.22)

    out_png_asset = assets_dir / "fig9_ppg_apg_features_horizontal.png"
    out_svg_asset = assets_dir / "fig9_ppg_apg_features_horizontal.svg"
    fig.savefig(out_png_asset, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_svg_asset, format="svg", bbox_inches="tight", facecolor="white")
    print(f"Saved Fig. 9 Horizontal: {out_png_asset}")
    print(f"Saved Fig. 9 Horizontal: {out_svg_asset}")

    # Also keep a copy in results/
    out_png_res = results_dir / "fig9_ppg_apg_features_horizontal.png"
    out_svg_res = results_dir / "fig9_ppg_apg_features_horizontal.svg"
    fig.savefig(out_png_res, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_svg_res, format="svg", bbox_inches="tight", facecolor="white")

    plt.close(fig)


def generate_figure_8_archive(data_dict, project_root: Path):
    """Preserves the 4-panel archive (2x2 and 1-column layouts)."""
    results_dir = project_root / "results"
    assets_dir = project_root / "assets" / "images" / "kti"

    # 1. 2x2 layout
    fig_w, fig_h = 7.16, 5.2
    fig, axes = plt.subplots(2, 2, figsize=(fig_w, fig_h), dpi=300)

    # (a) Raw PPG
    axes[0, 0].plot(data_dict["t_10"], data_dict["raw_10"], color="#2D3748", lw=1.1, zorder=2)
    axes[0, 0].set_title("(a) Raw PPG Signal", fontsize=9.2, fontweight="bold", pad=5)
    axes[0, 0].set_xlabel("Time (s)", fontsize=8.2, fontweight="bold")
    axes[0, 0].set_ylabel("Amplitude (a.u.)", fontsize=8.2, fontweight="bold")
    axes[0, 0].set_xlim(300, 310)
    axes[0, 0].set_xticks([300, 302, 304, 306, 308, 310])
    axes[0, 0].set_ylim(-10, 185)
    axes[0, 0].tick_params(labelsize=7.5)
    axes[0, 0].grid(True, linestyle="--", alpha=0.45, color="#CBD5E1", zorder=0)
    axes[0, 0].set_axisbelow(True)
    axes[0, 0].spines["top"].set_visible(False)
    axes[0, 0].spines["right"].set_visible(False)

    # (b) Filtered PPG
    axes[0, 1].axhline(0, color="#94A3B8", linestyle=":", lw=0.8, alpha=0.7, zorder=1)
    axes[0, 1].plot(data_dict["t_10"], data_dict["filt_10"], color="#1E6091", lw=1.1, zorder=2)
    axes[0, 1].axvspan(data_dict["t_b_start"], data_dict["t_b_end"], color="#F59E0B", alpha=0.18, zorder=0, label="Selected beat (c)-(d)")
    axes[0, 1].set_title("(b) Filtered PPG Signal", fontsize=9.2, fontweight="bold", pad=5)
    axes[0, 1].set_xlabel("Time (s)", fontsize=8.2, fontweight="bold")
    axes[0, 1].set_ylabel("Amplitude (a.u.)", fontsize=8.2, fontweight="bold")
    axes[0, 1].set_xlim(300, 310)
    axes[0, 1].set_xticks([300, 302, 304, 306, 308, 310])
    axes[0, 1].set_ylim(-75, 75)
    axes[0, 1].tick_params(labelsize=7.5)
    axes[0, 1].grid(True, linestyle="--", alpha=0.45, color="#CBD5E1", zorder=0)
    axes[0, 1].set_axisbelow(True)
    axes[0, 1].legend(fontsize=7.2, loc="upper right", frameon=True, facecolor="white", edgecolor="#CBD5E1")
    axes[0, 1].spines["top"].set_visible(False)
    axes[0, 1].spines["right"].set_visible(False)

    # (c) Representative PPG Pulse Morphology
    t_peak = data_dict["rise_time_s"]
    y_peak = data_dict["y_beat"][data_dict["rel_peak_idx"]]
    axes[1, 0].plot(data_dict["t_beat"], data_dict["y_beat"], color="#1E6091", lw=1.4, zorder=2)
    axes[1, 0].plot(data_dict["t_beat"], data_dict["base_line"], color="#94A3B8", linestyle="--", lw=0.8, zorder=1, label="Baseline")
    axes[1, 0].scatter([0], [data_dict["y_beat"][0]], color="#16A34A", s=35, zorder=5, label="Onset")
    axes[1, 0].scatter([t_peak], [y_peak], color="#DC2626", s=35, zorder=5, label="Systolic Peak")
    axes[1, 0].scatter([data_dict["t_beat"][-1]], [data_dict["y_beat"][-1]], color="#7C3AED", s=35, zorder=5, label="Offset")
    axes[1, 0].plot([data_dict["t_cross_asc"], data_dict["t_cross_desc"]], [data_dict["y_pw50"], data_dict["y_pw50"]],
                    color="#D97706", lw=1.2, zorder=4)
    axes[1, 0].scatter([data_dict["t_cross_asc"], data_dict["t_cross_desc"]], [data_dict["y_pw50"], data_dict["y_pw50"]],
                       color="#D97706", s=20, zorder=5)
    axes[1, 0].text(0.5 * (data_dict["t_cross_asc"] + data_dict["t_cross_desc"]), data_dict["y_pw50"] + 6,
                    f"PW50 = {data_dict['pw50']:.3f} s", ha="center", va="bottom", fontsize=7.2, fontweight="bold", color="#B45309")
    axes[1, 0].annotate("", xy=(t_peak, -60), xytext=(0, -60), arrowprops=dict(arrowstyle="<->", color="#2563EB", lw=1.0))
    axes[1, 0].text(0.01, -70, f"Rise Time = {t_peak:.3f} s", ha="left", va="top", fontsize=7.2, fontweight="bold", color="#1D4ED8")
    axes[1, 0].set_title("(c) Representative PPG Pulse Morphology", fontsize=9.2, fontweight="bold", pad=5)
    axes[1, 0].set_xlabel("Time within Beat (s)", fontsize=8.2, fontweight="bold")
    axes[1, 0].set_ylabel("Amplitude (a.u.)", fontsize=8.2, fontweight="bold")
    axes[1, 0].set_xlim(-0.06, 1.35)
    axes[1, 0].set_ylim(-86, 82)
    axes[1, 0].tick_params(labelsize=7.5)
    axes[1, 0].grid(True, linestyle="--", alpha=0.45, color="#CBD5E1", zorder=0)
    axes[1, 0].set_axisbelow(True)
    axes[1, 0].legend(fontsize=7.0, loc="upper right", frameon=True, facecolor="white", edgecolor="#CBD5E1")
    axes[1, 0].spines["top"].set_visible(False)
    axes[1, 0].spines["right"].set_visible(False)

    # (d) APG
    axes[1, 1].axhline(0, color="#94A3B8", linestyle=":", lw=0.8, alpha=0.7, zorder=1)
    axes[1, 1].plot(data_dict["t_beat"], data_dict["apg_beat"], color="#059669", lw=1.3, zorder=2)
    for pt in data_dict["fids"]:
        axes[1, 1].scatter([pt["t"]], [pt["val"]], color=pt["color"], s=30, zorder=5)
        axes[1, 1].text(pt["t"], pt["val"] + pt["offset"], pt["name"], ha="center", va=pt["va"],
                        fontsize=8.5, fontweight="bold", color=pt["color"], zorder=6)
    ratio_str = "b/a = -0.54 | c/a = 0.29\nd/a = -0.03 | e/a = 0.06"
    axes[1, 1].text(0.96, 0.94, ratio_str, transform=axes[1, 1].transAxes, ha="right", va="top",
                    fontsize=6.8, fontweight="bold", color="#374151",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CBD5E1", alpha=0.9))
    axes[1, 1].set_title("(d) Acceleration Photoplethysmogram (APG)", fontsize=9.2, fontweight="bold", pad=5)
    axes[1, 1].set_xlabel("Time within Beat (s)", fontsize=8.2, fontweight="bold")
    axes[1, 1].set_ylabel("APG (a.u./s²)", fontsize=8.2, fontweight="bold")
    axes[1, 1].set_xlim(-0.06, 1.35)
    axes[1, 1].set_ylim(-7200, 11500)
    axes[1, 1].tick_params(labelsize=7.5)
    axes[1, 1].grid(True, linestyle="--", alpha=0.45, color="#CBD5E1", zorder=0)
    axes[1, 1].set_axisbelow(True)
    axes[1, 1].spines["top"].set_visible(False)
    axes[1, 1].spines["right"].set_visible(False)

    plt.tight_layout(pad=0.8)
    plt.subplots_adjust(hspace=0.35, wspace=0.28)

    out_png_2x2 = results_dir / "fig8_ppg_apg_morphology_ieee.png"
    out_svg_2x2 = results_dir / "fig8_ppg_apg_morphology_ieee.svg"
    fig.savefig(out_png_2x2, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_svg_2x2, format="svg", bbox_inches="tight", facecolor="white")
    if assets_dir.is_dir():
        fig.savefig(assets_dir / "fig8_ppg_apg_morphology_ieee.png", dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(assets_dir / "fig8_ppg_apg_morphology_ieee.svg", format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # 2. 1-column 4-panel vertical layout
    fig1, axes1 = plt.subplots(4, 1, figsize=(3.5, 7.8), dpi=300)
    axes1[0].plot(data_dict["t_10"], data_dict["raw_10"], color="#2D3748", lw=1.1, zorder=2)
    axes1[0].set_title("(a) Raw PPG Signal", fontsize=8.8, fontweight="bold", pad=4)
    axes1[0].set_ylabel("Amplitude (a.u.)", fontsize=7.8, fontweight="bold")
    axes1[0].set_xlim(300, 310)
    axes1[0].set_ylim(-10, 185)
    axes1[0].tick_params(labelsize=7.0)
    axes1[0].grid(True, linestyle="--", alpha=0.45, color="#CBD5E1", zorder=0)
    axes1[0].spines["top"].set_visible(False)
    axes1[0].spines["right"].set_visible(False)

    axes1[1].axhline(0, color="#94A3B8", linestyle=":", lw=0.8, alpha=0.7, zorder=1)
    axes1[1].plot(data_dict["t_10"], data_dict["filt_10"], color="#1E6091", lw=1.1, zorder=2)
    axes1[1].axvspan(data_dict["t_b_start"], data_dict["t_b_end"], color="#F59E0B", alpha=0.18, zorder=0, label="Selected pulse")
    axes1[1].set_title("(b) Filtered PPG Signal", fontsize=8.8, fontweight="bold", pad=4)
    axes1[1].set_xlabel("Time (s)", fontsize=7.8, fontweight="bold")
    axes1[1].set_ylabel("Amplitude (a.u.)", fontsize=7.8, fontweight="bold")
    axes1[1].set_xlim(300, 310)
    axes1[1].set_ylim(-75, 75)
    axes1[1].tick_params(labelsize=7.0)
    axes1[1].grid(True, linestyle="--", alpha=0.45, color="#CBD5E1", zorder=0)
    axes1[1].legend(fontsize=6.5, loc="upper right", frameon=True, facecolor="white", edgecolor="#CBD5E1")
    axes1[1].spines["top"].set_visible(False)
    axes1[1].spines["right"].set_visible(False)

    axes1[2].plot(data_dict["t_beat"], data_dict["y_beat"], color="#1E6091", lw=1.3, zorder=2)
    axes1[2].plot(data_dict["t_beat"], data_dict["base_line"], color="#94A3B8", linestyle="--", lw=0.8, zorder=1, label="Baseline")
    axes1[2].scatter([0], [data_dict["y_beat"][0]], color="#16A34A", s=25, zorder=5, label="Onset")
    axes1[2].scatter([t_peak], [y_peak], color="#DC2626", s=25, zorder=5, label="Peak")
    axes1[2].scatter([data_dict["t_beat"][-1]], [data_dict["y_beat"][-1]], color="#7C3AED", s=25, zorder=5, label="Offset")
    axes1[2].plot([data_dict["t_cross_asc"], data_dict["t_cross_desc"]], [data_dict["y_pw50"], data_dict["y_pw50"]],
                 color="#D97706", lw=1.1, zorder=4)
    axes1[2].text(0.5 * (data_dict["t_cross_asc"] + data_dict["t_cross_desc"]), data_dict["y_pw50"] + 6,
                 f"PW50 = {data_dict['pw50']:.3f} s", ha="center", va="bottom", fontsize=6.8, fontweight="bold", color="#B45309")
    axes1[2].annotate("", xy=(t_peak, -60), xytext=(0, -60), arrowprops=dict(arrowstyle="<->", color="#2563EB", lw=0.9))
    axes1[2].text(0.01, -70, f"Rise = {t_peak:.3f} s", ha="left", va="top", fontsize=6.8, fontweight="bold", color="#1D4ED8")
    axes1[2].set_title("(c) Representative PPG Pulse Morphology", fontsize=8.8, fontweight="bold", pad=4)
    axes1[2].set_ylabel("Amplitude (a.u.)", fontsize=7.8, fontweight="bold")
    axes1[2].set_xlim(-0.06, 1.35)
    axes1[2].set_ylim(-86, 82)
    axes1[2].tick_params(labelsize=7.0)
    axes1[2].grid(True, linestyle="--", alpha=0.45, color="#CBD5E1", zorder=0)
    axes1[2].legend(fontsize=6.5, loc="upper right", frameon=True, facecolor="white", edgecolor="#CBD5E1")
    axes1[2].spines["top"].set_visible(False)
    axes1[2].spines["right"].set_visible(False)

    axes1[3].axhline(0, color="#94A3B8", linestyle=":", lw=0.8, alpha=0.7, zorder=1)
    axes1[3].plot(data_dict["t_beat"], data_dict["apg_beat"], color="#059669", lw=1.2, zorder=2)
    for pt in data_dict["fids"]:
        axes1[3].scatter([pt["t"]], [pt["val"]], color=pt["color"], s=25, zorder=5)
        axes1[3].text(pt["t"], pt["val"] + pt["offset"], pt["name"], ha="center", va=pt["va"],
                     fontsize=8.0, fontweight="bold", color=pt["color"], zorder=6)
    ratio_str = "b/a = -0.54 | c/a = 0.29\nd/a = -0.03 | e/a = 0.06"
    axes1[3].text(0.96, 0.94, ratio_str, transform=axes1[3].transAxes, ha="right", va="top",
                  fontsize=6.5, fontweight="bold", color="#374151",
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CBD5E1", alpha=0.9))
    axes1[3].set_title("(d) Acceleration Photoplethysmogram (APG)", fontsize=8.8, fontweight="bold", pad=4)
    axes1[3].set_xlabel("Time within Beat (s)", fontsize=7.8, fontweight="bold")
    axes1[3].set_ylabel("APG (a.u./s²)", fontsize=7.8, fontweight="bold")
    axes1[3].set_xlim(-0.06, 1.35)
    axes1[3].set_ylim(-7200, 11500)
    axes1[3].tick_params(labelsize=7.0)
    axes1[3].grid(True, linestyle="--", alpha=0.45, color="#CBD5E1", zorder=0)
    axes1[3].spines["top"].set_visible(False)
    axes1[3].spines["right"].set_visible(False)

    plt.tight_layout(pad=0.6)
    plt.subplots_adjust(hspace=0.42)

    out_png_1col = results_dir / "fig8_ppg_apg_morphology_ieee_1col.png"
    out_svg_1col = results_dir / "fig8_ppg_apg_morphology_ieee_1col.svg"
    fig1.savefig(out_png_1col, dpi=300, bbox_inches="tight", facecolor="white")
    fig1.savefig(out_svg_1col, format="svg", bbox_inches="tight", facecolor="white")
    if assets_dir.is_dir():
        fig1.savefig(assets_dir / "fig8_ppg_apg_morphology_ieee_1col.png", dpi=300, bbox_inches="tight", facecolor="white")
        fig1.savefig(assets_dir / "fig8_ppg_apg_morphology_ieee_1col.svg", format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig1)

    print("Preserved archive figures (results/ & assets/images/kti/):")
    print(f"  - {out_png_2x2.name}")
    print(f"  - {out_png_1col.name}")


def main():
    print("=" * 80)
    print("GENERATING FIGURE 8 & FIGURE 9 (IEEE FORMATS: VERTICAL & HORIZONTAL)")
    print("=" * 80)
    project_root = find_project_root()
    data_dict = compute_signals_and_landmarks()

    # 1. Fig 8 Vertical (Single-Column IEEE)
    generate_figure_8_vertical(data_dict, project_root)

    # 2. Fig 8 Horizontal (Two-Column IEEE Side-by-Side)
    generate_figure_8_horizontal(data_dict, project_root)

    # 3. Fig 9 Vertical (Single-Column IEEE)
    generate_figure_9_vertical(data_dict, project_root)

    # 4. Fig 9 Horizontal (Two-Column IEEE Side-by-Side)
    generate_figure_9_horizontal(data_dict, project_root)

    # 5. Archive 4-panel figures
    generate_figure_8_archive(data_dict, project_root)

    print("=" * 80)
    print("FIGURE GENERATION COMPLETED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    main()
