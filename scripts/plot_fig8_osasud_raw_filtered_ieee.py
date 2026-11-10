"""
scripts/plot_fig8_osasud_raw_filtered_ieee.py
================================================================================
Scientific Figure Generator for KTI Paper (IEEE Format)
Figure 8: Real PPG Signal Preprocessing (Raw vs Filtered Waveform)
Dataset: OSASUD (Patient 12, Window P12_W0011, t = 300-330 s)
Display Snippet: First 10 seconds (t = 300-310 s) of the 30-second window
Filter: Chebyshev Type II Bandpass [0.5, 8.0] Hz, Order 4, Rs = 20 dB, Zero-phase
Style: IEEE Transactions Standard (Single-Column, 300+ DPI, Minimalist)

Outputs:
  - results/fig8_osasud_raw_filtered_ieee.png
  - results/fig8_osasud_raw_filtered_ieee.svg
  - assets/images/kti/fig8_osasud_raw_filtered_ieee.png (if directory exists)
  - assets/images/kti/fig8_osasud_raw_filtered_ieee.svg (if directory exists)
================================================================================
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
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


def verify_table_features(project_root: Path):
    """
    Verifies features for patient_id=12, window_id=P12_W0011
    comparing pilot_v2 and full feature extraction tables.
    """
    print("\n" + "=" * 80)
    print("VERIFICATION OF EXTRACTED FEATURES (PATIENT 12, WINDOW P12_W0011)")
    print("=" * 80)

    pilot_v2_path = project_root / "results" / "osasud_19_features_pilot_v2.csv"
    full_path = project_root / "results" / "osasud_19_features_full.csv"

    if not pilot_v2_path.is_file():
        print(f"[ERROR] Missing file: {pilot_v2_path}")
        return None
    if not full_path.is_file():
        print(f"[ERROR] Missing file: {full_path}")
        return None

    df_pilot = pd.read_csv(pilot_v2_path)
    df_full = pd.read_csv(full_path)

    row_p = df_pilot[(df_pilot["patient_id"].astype(str) == "12") & (df_pilot["window_id"] == "P12_W0011")]
    row_f = df_full[(df_full["patient_id"].astype(str) == "12") & (df_full["window_id"] == "P12_W0011")]

    if row_p.empty or row_f.empty:
        print("[ERROR] Window P12_W0011 not found in one or both files.")
        return None

    r_p = row_p.iloc[0]
    r_f = row_f.iloc[0]

    check_features = [
        "heart_rate_bpm",
        "rise_time_s",
        "pulse_width_50_s",
        "pulse_amplitude",
        "pulse_area",
        "apg_b_a",
        "apg_c_a",
        "apg_d_a",
        "apg_e_a",
    ]

    print(f"{'Feature Name':<22} | {'Pilot V2 Value':<20} | {'Full Table Value':<20} | Status")
    print("-" * 80)
    all_matched = True
    for feat in check_features:
        vp = float(r_p[feat])
        vf = float(r_f[feat])
        matched = (vp == vf) or abs(vp - vf) < 1e-9
        if not matched:
            all_matched = False
        status_str = "MATCH" if matched else "DIFFERENCE"
        print(f"{feat:<22} | {vp:<20.8f} | {vf:<20.8f} | {status_str}")

    print("-" * 80)
    print(f"Overall Feature Consistency: {'PERFECT MATCH (100% Identical)' if all_matched else 'DIFFERENCES FOUND'}")
    print(f"Window Metadata: Start={r_f['start_timestamp']}, End={r_f['end_timestamp']}, SpO2 Target={r_f['spo2_target_pct']}%, Status={r_f['feature_extraction_status']}")
    print("=" * 80 + "\n")
    return all_matched


def generate_figure_8():
    """
    Generates Fig. 8 showing Raw PPG Signal vs Filtered PPG Signal
    for Patient 12 (Window P12_W0011, 10-second snippet: 300 to 310 s).
    """
    print("=" * 80)
    print("GENERATING FIG. 8: REAL PPG SIGNAL PREPROCESSING (OSASUD)")
    print("=" * 80)

    project_root = find_project_root()
    npz_path = project_root / "data" / "osasud" / "patients_npz" / "patient_12.npz"
    results_dir = project_root / "results"
    assets_dir = project_root / "assets" / "images" / "kti"

    results_dir.mkdir(parents=True, exist_ok=True)

    if not npz_path.is_file():
        raise FileNotFoundError(f"OSASUD Patient 12 data not found at: {npz_path}")

    # 1. Load OSASUD Patient 12 NPZ
    print(f"Loading data from: {npz_path.relative_to(project_root)}")
    with np.load(npz_path, allow_pickle=False) as npz:
        ppg_2d = npz["ppg_80hz"]
        n_total_sec = len(ppg_2d)

    fs = 80  # Hz

    # 2. Identify Continuous Valid PPG Segments (No NaNs)
    valid_sec_mask = (~np.isnan(ppg_2d)).all(axis=1)
    diff_valid = np.diff(valid_sec_mask.astype(int))
    starts = np.where(diff_valid == 1)[0] + 1
    ends = np.where(diff_valid == -1)[0] + 1
    if valid_sec_mask[0]:
        starts = np.r_[0, starts]
    if valid_sec_mask[-1]:
        ends = np.r_[ends, n_total_sec]

    segments = list(zip(starts, ends))

    # Find the segment that contains Window P12_W0011 (seconds 300 to 330)
    target_start_sec = 300
    target_end_sec = 330
    snippet_start_sec = 300
    snippet_end_sec = 310

    target_seg = None
    for s_sec, e_sec in segments:
        if s_sec <= target_start_sec and target_end_sec <= e_sec:
            target_seg = (s_sec, e_sec)
            break

    if target_seg is None:
        raise ValueError(f"Window {target_start_sec}-{target_end_sec} s not found in any continuous valid segment.")

    seg_s, seg_e = target_seg
    print(f"Window P12_W0011 (300-330 s) is contained in continuous segment: t = {seg_s} s to {seg_e} s ({seg_e - seg_s} seconds, {(seg_e - seg_s)*fs:,} samples).")

    # 3. Apply Chebyshev Type II Zero-Phase Bandpass Filter on the ENTIRE continuous segment
    # Configuration matches FINAL pipeline:
    cheby_order = 4
    cheby_rs_db = 20
    cheby_wn = [0.5, 8.0]

    print(f"Designing Chebyshev Type II filter: Order={cheby_order}, Stopband Edges={cheby_wn} Hz, Attenuation={cheby_rs_db} dB, Fs={fs} Hz")
    cheby_sos = sig.cheby2(cheby_order, cheby_rs_db, cheby_wn, btype="bandpass", fs=fs, output="sos")

    print(f"Applying zero-phase sosfiltfilt to continuous segment ({seg_e - seg_s} seconds)...")
    seg_raw = ppg_2d[seg_s:seg_e].flatten()
    seg_filt = sig.sosfiltfilt(cheby_sos, seg_raw)

    # 4. Extract the 10-Second Snippet (Detik 300-310) from the Filtered Segment
    # Boundary distortion is completely eliminated because filtering occurred on the long continuous segment.
    idx_snip_s = (snippet_start_sec - seg_s) * fs
    idx_snip_e = (snippet_end_sec - seg_s) * fs

    t_snippet = np.arange(snippet_start_sec * fs, snippet_end_sec * fs) / fs
    raw_snippet = seg_raw[idx_snip_s:idx_snip_e]
    filt_snippet = seg_filt[idx_snip_s:idx_snip_e]

    print(f"Extracted 10-second snippet (t = {snippet_start_sec} to {snippet_end_sec} s, {len(t_snippet)} samples):")
    print(f"  Raw PPG amplitude range     : [{np.min(raw_snippet):.2f}, {np.max(raw_snippet):.2f}] a.u. (Mean: {np.mean(raw_snippet):.2f})")
    print(f"  Filtered PPG amplitude range: [{np.min(filt_snippet):.2f}, {np.max(filt_snippet):.2f}] a.u. (Mean: {np.mean(filt_snippet):.2f})")

    # 5. Plotting: Vertical 2-Panel Single-Column IEEE Figure
    fig_w, fig_h = 3.5, 3.8
    fig, axes = plt.subplots(2, 1, figsize=(fig_w, fig_h), dpi=300, sharex=True)

    # Panel (a): Raw PPG Signal
    ax1 = axes[0]
    ax1.plot(t_snippet, raw_snippet, color="#2D3748", linewidth=1.1, zorder=2)
    ax1.set_title("(a) Raw PPG Signal", fontsize=9.2, fontweight="bold", pad=5)
    ax1.set_ylabel("Amplitude (a.u.)", fontsize=8.5, fontweight="bold")
    ax1.set_ylim(-10, 185)
    ax1.set_yticks([0, 50, 100, 150])
    ax1.tick_params(axis="both", labelsize=8.0)
    ax1.grid(True, linestyle="--", alpha=0.45, color="#CBD5E1", zorder=0)
    ax1.set_axisbelow(True)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel (b): Filtered PPG Signal
    ax2 = axes[1]
    ax2.axhline(0, color="#94A3B8", linestyle=":", linewidth=0.8, alpha=0.7, zorder=1)
    ax2.plot(t_snippet, filt_snippet, color="#1E6091", linewidth=1.1, zorder=2)
    ax2.set_title("(b) Filtered PPG Signal", fontsize=9.2, fontweight="bold", pad=5)
    ax2.set_xlabel("Time (s)", fontsize=8.5, fontweight="bold")
    ax2.set_ylabel("Amplitude (a.u.)", fontsize=8.5, fontweight="bold")
    ax2.set_xlim(snippet_start_sec, snippet_end_sec)
    ax2.set_xticks([300, 302, 304, 306, 308, 310])
    ax2.set_ylim(-75, 75)
    ax2.set_yticks([-60, -30, 0, 30, 60])
    ax2.tick_params(axis="both", labelsize=8.0)
    ax2.grid(True, linestyle="--", alpha=0.45, color="#CBD5E1", zorder=0)
    ax2.set_axisbelow(True)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout(pad=0.8)
    plt.subplots_adjust(hspace=0.36)

    # 6. Save Figure to results/ and assets/images/kti/
    out_png = results_dir / "fig8_osasud_raw_filtered_ieee.png"
    out_svg = results_dir / "fig8_osasud_raw_filtered_ieee.svg"

    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_svg, format="svg", bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_svg}")

    if assets_dir.is_dir():
        asset_png = assets_dir / "fig8_osasud_raw_filtered_ieee.png"
        asset_svg = assets_dir / "fig8_osasud_raw_filtered_ieee.svg"
        fig.savefig(asset_png, dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(asset_svg, format="svg", bbox_inches="tight", facecolor="white")
        print(f"Saved copy to assets: {asset_png}")
        print(f"Saved copy to assets: {asset_svg}")

    plt.close(fig)
    print("=" * 80)
    print("FIGURE 8 GENERATION COMPLETE.")
    print("=" * 80)


def main():
    project_root = find_project_root()
    verify_table_features(project_root)
    generate_figure_8()


if __name__ == "__main__":
    main()
