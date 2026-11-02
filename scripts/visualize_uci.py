import sys
from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt

def find_paths():
    """Mencari path dataset dan memastikan folder output results tersedia."""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    
    # Jalur dataset Part_1.mat
    data_path = project_root / "data" / "uci" / "Part_1.mat"
    if not data_path.is_file():
        # Fallback jika dijalankan dari root direktori lain
        candidates = [
            Path.cwd() / "data" / "uci" / "Part_1.mat",
            Path.cwd() / "KTI_XGBoost" / "data" / "uci" / "Part_1.mat",
        ]
        for c in candidates:
            if c.is_file():
                data_path = c
                break

    # Jalur folder results
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_image_path = results_dir / "uci_part1_record1_raw.png"
    
    return data_path, output_image_path

def visualize_raw_signals():
    data_path, output_image_path = find_paths()

    if not data_path.is_file():
        print(f"[ERROR] Berkas dataset tidak ditemukan di: {data_path}")
        sys.exit(1)

    print(f"Membuka dataset: {data_path}")

    # Buka dengan mode read-only 'r' tanpa mengubah data asli
    with h5py.File(data_path, "r") as f:
        if "Part_1" not in f:
            print("[ERROR] Key 'Part_1' tidak ditemukan dalam berkas HDF5.")
            sys.exit(1)

        # Mengambil referensi objek ke rekaman pertama
        part1_dataset = f["Part_1"]
        first_ref = part1_dataset[0, 0]
        record_obj = f[first_ref]

        print(f"Ukuran total rekaman pertama di disk: {record_obj.shape} (tipe: {record_obj.dtype})")

        # Konfigurasi waktu dan sampling
        fs = 125  # Sampling rate: 125 Hz
        t_start = 60.0  # detik ke-60
        t_end = 70.0    # detik ke-70
        
        idx_start = int(t_start * fs)  # Sampel ke-7500
        idx_end = int(t_end * fs)      # Sampel ke-8750
        
        # Validasi batas indeks
        total_samples = record_obj.shape[0]
        if idx_end > total_samples:
            print(f"[ERROR] Indeks akhir ({idx_end}) melebihi total sampel yang ada ({total_samples}).")
            sys.exit(1)

        print(f"Membaca irisan data dari sampel ke-{idx_start} sampai {idx_end} ({t_end - t_start:.1f} detik)...")
        # Hanya membaca 1250 sampel x 3 kanal ke memori RAM
        raw_slice = record_obj[idx_start:idx_end, :]

    # Sumbu waktu dalam detik
    time_axis = np.arange(idx_start, idx_end) / fs

    # Pisahkan tiap kanal sinyal
    ppg_signal = raw_slice[:, 0]
    abp_signal = raw_slice[:, 1]
    ecg_signal = raw_slice[:, 2]

    # Ringkasan statistik sinyal
    print("\nRingkasan Statistik Sinyal (Detik 60 - 70):")
    print(f"  - Kanal 0 (PPG): min={ppg_signal.min():.3f}, max={ppg_signal.max():.3f}, mean={ppg_signal.mean():.3f}")
    print(f"  - Kanal 1 (ABP): min={abp_signal.min():.3f}, max={abp_signal.max():.3f}, mean={abp_signal.mean():.3f} mmHg")
    print(f"  - Kanal 2 (ECG): min={ecg_signal.min():.3f}, max={ecg_signal.max():.3f}, mean={ecg_signal.mean():.3f}")

    # Visualisasi Matplotlib: 3 grafik vertikal sejajar
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 8), sharex=True)
    fig.suptitle("Visualisasi Sinyal Mentah UCI Part_1 (Rekaman 1 | Detik 60 - 70)", fontsize=13, fontweight="bold")

    # 1. Grafik Atas: PPG Mentah
    axes[0].plot(time_axis, ppg_signal, color="#1f77b4", linewidth=1.2, label="PPG Mentah")
    axes[0].set_title("Kanal 1: Photoplethysmogram (PPG) Mentah", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Amplitudo (a.u.)", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(loc="upper right")

    # 2. Grafik Tengah: ABP Mentah
    axes[1].plot(time_axis, abp_signal, color="#d62728", linewidth=1.2, label="ABP Mentah")
    axes[1].set_title("Kanal 2: Arterial Blood Pressure (ABP) Mentah", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Tekanan Darah (mmHg)", fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(loc="upper right")

    # 3. Grafik Bawah: ECG Mentah
    axes[2].plot(time_axis, ecg_signal, color="#2ca02c", linewidth=1.2, label="ECG Mentah")
    axes[2].set_title("Kanal 3: Electrocardiogram (ECG) Mentah", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Waktu (detik)", fontsize=10)
    axes[2].set_ylabel("Amplitudo (a.u.)", fontsize=10)
    axes[2].grid(True, linestyle="--", alpha=0.6)
    axes[2].legend(loc="upper right")

    plt.tight_layout()

    # Simpan grafik
    fig.savefig(output_image_path, dpi=300)
    plt.close(fig)

    print(f"\n[SUKSES] Grafik visualisasi berhasil disimpan di:\n  {output_image_path}")

if __name__ == "__main__":
    visualize_raw_signals()
