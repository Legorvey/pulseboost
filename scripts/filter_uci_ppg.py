import sys
from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import cheby2, sosfiltfilt

def find_paths():
    """Mencari lokasi dataset dan menyiapkan folder results."""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    data_path = project_root / "data" / "uci" / "Part_1.mat"
    if not data_path.is_file():
        candidates = [
            Path.cwd() / "data" / "uci" / "Part_1.mat",
            Path.cwd() / "KTI_XGBoost" / "data" / "uci" / "Part_1.mat",
        ]
        for c in candidates:
            if c.is_file():
                data_path = c
                break

    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    fig1_path = results_dir / "uci_part1_record1_ppg_raw_vs_filtered.png"
    fig2_path = results_dir / "uci_part1_record1_ppg_overlay.png"

    return data_path, fig1_path, fig2_path

def main():
    data_path, fig1_path, fig2_path = find_paths()

    if not data_path.is_file():
        print(f"[ERROR] Berkas dataset tidak ditemukan di: {data_path}")
        sys.exit(1)

    print(f"Membuka file dataset: {data_path}")

    # Membaca hanya rekaman pertama dari Part_1.mat (read-only)
    with h5py.File(data_path, "r") as f:
        if "Part_1" not in f:
            print("[ERROR] Key 'Part_1' tidak ditemukan dalam file HDF5.")
            sys.exit(1)

        first_ref = f["Part_1"][0, 0]
        record_obj = f[first_ref]

        fs = 125  # Hz
        t_start = 60.0
        t_end = 70.0
        idx_start = int(t_start * fs)  # 7500
        idx_end = int(t_end * fs)      # 8750

        print(f"Mengambil sinyal PPG (kolom 0) dari detik ke-{t_start:.1f} s.d. {t_end:.1f} (sampel {idx_start}-{idx_end})...")
        ppg_raw = record_obj[idx_start:idx_end, 0]

    time_axis = np.arange(idx_start, idx_end) / fs

    # Desain filter Chebyshev Type II Bandpass
    # Parameter sesuai tugas:
    # - Jenis: Bandpass
    # - Metode: Chebyshev Type II (cheby2)
    # - Orde: 4
    # - Passband: 0.5 s.d. 8.0 Hz
    # - Stopband attenuation (rs): 20 dB
    # - Zero-phase filtering: sosfiltfilt
    order = 4
    rs = 20  # dB stopband attenuation
    lowcut = 0.5
    highcut = 8.0

    sos = cheby2(order, rs, [lowcut, highcut], btype="bandpass", fs=fs, output="sos")
    ppg_filtered = sosfiltfilt(sos, ppg_raw)

    # 1. Statistik Sederhana
    raw_min, raw_max = ppg_raw.min(), ppg_raw.max()
    raw_mean, raw_std = ppg_raw.mean(), ppg_raw.std()

    filt_min, filt_max = ppg_filtered.min(), ppg_filtered.max()
    filt_mean, filt_std = ppg_filtered.mean(), ppg_filtered.std()

    print("\n=== STATISTIK SEGMEN PPG (DETIK 60 - 70) ===")
    print("Sinyal PPG Mentah:")
    print(f"  Min  : {raw_min:.4f}")
    print(f"  Max  : {raw_max:.4f}")
    print(f"  Mean : {raw_mean:.4f}")
    print(f"  Std  : {raw_std:.4f}")

    print("\nSinyal PPG Terfilter (Chebyshev Type II):")
    print(f"  Min  : {filt_min:.4f}")
    print(f"  Max  : {filt_max:.4f}")
    print(f"  Mean : {filt_mean:.4f}")
    print(f"  Std  : {filt_std:.4f}")

    # =========================================================================
    # Gambar 1: Dua Subplot Vertikal (PPG Mentah vs PPG Terfilter)
    # =========================================================================
    fig1, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig1.suptitle("Perbandingan Sinyal PPG: Mentah vs Terfilter (Chebyshev Type II, 0.5 - 8 Hz)", fontsize=13, fontweight="bold")

    # Subplot Atas: PPG Mentah
    axes[0].plot(time_axis, ppg_raw, color="#1f77b4", linewidth=1.2, label="PPG Mentah")
    axes[0].set_title("PPG Mentah (Mengandung DC Offset ~1.85 a.u. & Noise)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Amplitudo (a.u.)", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(loc="upper right")

    # Subplot Bawah: PPG Terfilter
    axes[1].plot(time_axis, ppg_filtered, color="#2ca02c", linewidth=1.2, label="PPG Terfilter (0.5 - 8 Hz)")
    axes[1].set_title("PPG Terfilter Chebyshev Type II (Zero-phase, Terpusat di 0)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Waktu (detik)", fontsize=10)
    axes[1].set_ylabel("Amplitudo (a.u.)", fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    fig1.savefig(fig1_path, dpi=300)
    plt.close(fig1)
    print(f"\n[SUKSES] Gambar 1 disimpan di:\n  {fig1_path}")

    # =========================================================================
    # Gambar 2: Overlay Sinyal Mentah dan Terfilter pada Sumbu Waktu Sama
    # =========================================================================
    fig2, ax = plt.subplots(figsize=(12, 6))
    fig2.suptitle("Overlay Sinyal PPG: Mentah vs Terfilter pada Sumbu Waktu yang Sama", fontsize=13, fontweight="bold")

    ax.plot(time_axis, ppg_raw, color="#1f77b4", alpha=0.75, linewidth=1.3, label="PPG Mentah (dengan DC baseline)")
    ax.plot(time_axis, ppg_filtered, color="#d62728", alpha=0.9, linewidth=1.3, label="PPG Terfilter Chebyshev II (baseline tereliminasi)")

    ax.set_title("Visualisasi Perubahan Offset & Reduksi Noise (Detik 60 - 70)", fontsize=11)
    ax.set_xlabel("Waktu (detik)", fontsize=10)
    ax.set_ylabel("Amplitudo (a.u.)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.9)

    # Tambahkan anotasi penjelasan sederhana pada grafik
    ax.text(60.2, 2.7, "PPG Mentah (berkisar ~1.0 s.d. 3.0)", color="#1f77b4", fontweight="bold", fontsize=9)
    ax.text(60.2, -0.6, "PPG Terfilter (terpusat di ~0.0)", color="#d62728", fontweight="bold", fontsize=9)

    plt.tight_layout()
    fig2.savefig(fig2_path, dpi=300)
    plt.close(fig2)
    print(f"[SUKSES] Gambar 2 disimpan di:\n  {fig2_path}")

if __name__ == "__main__":
    main()
