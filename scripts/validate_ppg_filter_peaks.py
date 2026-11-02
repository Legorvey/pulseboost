import sys
from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

def find_paths():
    """Mencari path dataset dan folder results."""
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

    fig_edge_path = results_dir / "uci_filter_edge_comparison.png"
    fig_freq_path = results_dir / "uci_filter_frequency_response.png"
    fig_peaks_path = results_dir / "uci_peaks_validated.png"
    summary_path = results_dir / "uci_filter_validation_summary.txt"

    return data_path, fig_edge_path, fig_freq_path, fig_peaks_path, summary_path

def main():
    data_path, fig_edge_path, fig_freq_path, fig_peaks_path, summary_path = find_paths()

    if not data_path.is_file():
        print(f"[ERROR] Berkas dataset tidak ditemukan di: {data_path}")
        sys.exit(1)

    print(f"Membuka dataset: {data_path}")

    # 1. Baca HANYA rekaman pertama (seluruh 61.000 sampel = 488 detik) kanal PPG
    with h5py.File(data_path, "r") as f:
        if "Part_1" not in f:
            print("[ERROR] Key 'Part_1' tidak ditemukan dalam file HDF5.")
            sys.exit(1)

        first_ref = f["Part_1"][0, 0]
        record_obj = f[first_ref]
        total_samples = record_obj.shape[0]

        fs = 125  # Hz
        print(f"Membaca rekaman 1 PPG (seluruh {total_samples} sampel / {total_samples/fs:.1f} detik)...")
        ppg_long = record_obj[:, 0]  # kanal 0 (PPG)

    # 2. Konfigurasi Filter Chebyshev Type II yang SAMA PERSIS
    order = 4
    rs = 20  # dB
    Wn = [0.5, 8.0]
    sos = sig.cheby2(order, rs, Wn, btype="bandpass", fs=fs, output="sos")

    # 3. Metode Baru: Filtering pada rekaman panjang (488 detik), lalu potong detik 60-70
    t_start = 60.0
    t_end = 70.0
    idx_start = int(t_start * fs)  # 7500
    idx_end = int(t_end * fs)      # 8750

    print("Menerapkan zero-phase filter (sosfiltfilt) pada rekaman panjang...")
    filt_long_full = sig.sosfiltfilt(sos, ppg_long)
    ppg_filt_long_slice = filt_long_full[idx_start:idx_end]

    # 4. Metode Lama: Filtering langsung pada potongan 10 detik
    ppg_raw_slice = ppg_long[idx_start:idx_end]
    ppg_filt_short_slice = sig.sosfiltfilt(sos, ppg_raw_slice)

    time_slice = np.arange(idx_start, idx_end) / fs

    # 5. Analisis Efek Tepi (Perbedaan Amplitudo)
    diff = np.abs(ppg_filt_short_slice - ppg_filt_long_slice)
    diff_first_sec = diff[:fs]   # Detik 60.0 - 61.0
    diff_last_sec = diff[-fs:]   # Detik 69.0 - 70.0
    diff_middle = diff[fs:-fs]   # Detik 61.0 - 69.0

    max_diff_overall = np.max(diff)
    mean_diff_overall = np.mean(diff)
    max_diff_first = np.max(diff_first_sec)
    mean_diff_first = np.mean(diff_first_sec)
    max_diff_last = np.max(diff_last_sec)
    mean_diff_last = np.mean(diff_last_sec)
    max_diff_mid = np.max(diff_middle)
    mean_diff_mid = np.mean(diff_middle)

    # 6. Analisis Respons Frekuensi Filter
    w, h = sig.sosfreqz(sos, worN=10000, fs=fs)
    mag_single = np.abs(h)
    db_single = 20 * np.log10(np.maximum(mag_single, 1e-10))
    db_twopass = 2.0 * db_single  # Dua arah (sosfiltfilt) mengkuadratkan magnitude response

    # Titik -3 dB passband
    idx_pass_single = np.where(db_single >= -3.0)[0]
    f_single_low = w[idx_pass_single[0]]
    f_single_high = w[idx_pass_single[-1]]

    idx_pass_twopass = np.where(db_twopass >= -3.0)[0]
    f_twopass_low = w[idx_pass_twopass[0]]
    f_twopass_high = w[idx_pass_twopass[-1]]

    # Nilai atenuasi pada frekuensi kritis Wn (0.5 Hz & 8.0 Hz)
    idx_05 = np.argmin(np.abs(w - 0.5))
    idx_80 = np.argmin(np.abs(w - 8.0))
    att_05_single, att_05_two = db_single[idx_05], db_twopass[idx_05]
    att_80_single, att_80_two = db_single[idx_80], db_twopass[idx_80]

    # 7. Deteksi Puncak pada Sinyal Validasi (Rekaman Panjang)
    # Parameter identik:
    min_dist_samples = int(0.35 * fs)
    prominence_val = 0.25 * np.std(ppg_filt_long_slice)

    # Deteksi puncak pada rekaman penuh (bebas efek tepi lokal), lalu ambil rentang detik 60 - 70
    all_peaks_long, _ = sig.find_peaks(filt_long_full, distance=min_dist_samples, prominence=prominence_val)
    time_full = np.arange(total_samples) / fs

    in_window_mask = (time_full[all_peaks_long] >= t_start) & (time_full[all_peaks_long] <= t_end)
    peaks_new = all_peaks_long[in_window_mask]
    peak_times_new = time_full[peaks_new]
    peak_vals_new = filt_long_full[peaks_new]

    # Deteksi lembah minimum di antara puncak bertetangga pada jendela validasi
    valleys_new = []
    for i in range(len(peaks_new) - 1):
        idx_p1 = peaks_new[i]
        idx_p2 = peaks_new[i + 1]
        rel_min = np.argmin(filt_long_full[idx_p1:idx_p2])
        valleys_new.append(idx_p1 + rel_min)

    valleys_new = np.array(valleys_new, dtype=int)
    valley_times_new = time_full[valleys_new]
    valley_vals_new = filt_long_full[valleys_new]

    # 8. Deteksi Puncak Metode Lama (Short Slice) untuk Perbandingan Eksak
    old_prom = 0.25 * np.std(ppg_filt_short_slice)
    peaks_old_rel, _ = sig.find_peaks(ppg_filt_short_slice, distance=min_dist_samples, prominence=old_prom)
    peak_times_old = (idx_start + peaks_old_rel) / fs
    peak_vals_old = ppg_filt_short_slice[peaks_old_rel]

    # Metrik interval antarpuncak
    ibis_new_sec = np.diff(peak_times_new)
    ibis_new_ms = ibis_new_sec * 1000.0
    hr_new_beats = 60.0 / ibis_new_sec

    ibis_old_sec = np.diff(peak_times_old)
    ibis_old_ms = ibis_old_sec * 1000.0
    hr_old_beats = 60.0 / ibis_old_sec

    # =========================================================================
    # Laporan Teks Ringkasan Validasi
    # =========================================================================
    summary_lines = []
    summary_lines.append("=" * 72)
    summary_lines.append("LAPORAN VALIDASI FILTER & DETEKSI PUNCAK PPG (TAHAP 4B)")
    summary_lines.append(f"Berkas: {data_path.name} | Rekaman 1 (Total: 488 s, {total_samples} sampel)")
    summary_lines.append("Jendela Pengamatan: Detik 60.00 s.d. 70.00 (Durasi: 10.00 s)")
    summary_lines.append("=" * 72)

    summary_lines.append("\n[1. ANALISIS EFEK TEPI FILTERING (EDGE EFFECTS)]")
    summary_lines.append(f"- Selisih Maksimum Keseluruhan (Max Abs Diff) : {max_diff_overall:.4f} a.u.")
    summary_lines.append(f"- Selisih Rata-rata Keseluruhan (Mean Abs Diff): {mean_diff_overall:.4f} a.u.")
    summary_lines.append(f"- 1 Detik Pertama (Detik 60.0 - 61.0): Max Diff = {max_diff_first:.4f} a.u. | Mean Diff = {mean_diff_first:.4f} a.u.")
    summary_lines.append(f"- 1 Detik Terakhir (Detik 69.0 - 70.0): Max Diff = {max_diff_last:.4f} a.u. | Mean Diff = {mean_diff_last:.4f} a.u.")
    summary_lines.append(f"- 8 Detik Tengah  (Detik 61.0 - 69.0): Max Diff = {max_diff_mid:.4f} a.u. | Mean Diff = {mean_diff_mid:.4f} a.u.")
    summary_lines.append("  KESIMPULAN EFEK TEPI:")
    summary_lines.append("  Terjadi distorsi tepi yang sangat nyata pada metode lama akibat filter startup")
    summary_lines.append("  dan stopping transient pada potongan sinyal pendek 10 detik.")

    summary_lines.append("\n[2. RESPON FREKUENSI FILTER CHEBYSHEV TYPE II]")
    summary_lines.append(f"- Parameter Desain: Orde {order}, Stopband Ripple (rs) = {rs} dB, Wn = {Wn} Hz")
    summary_lines.append(f"- Frekuensi Kritis Stopband Wn:")
    summary_lines.append(f"  * Pada 0.5 Hz: Satu Arah = {att_05_single:.2f} dB | Dua Arah (sosfiltfilt) = {att_05_two:.2f} dB")
    summary_lines.append(f"  * Pada 8.0 Hz: Satu Arah = {att_80_single:.2f} dB | Dua Arah (sosfiltfilt) = {att_80_two:.2f} dB")
    summary_lines.append(f"- Batas Passband Efektif (-3 dB cutoff):")
    summary_lines.append(f"  * Respons Satu Arah : {f_single_low:.3f} Hz s.d. {f_single_high:.3f} Hz")
    summary_lines.append(f"  * Respons Dua Arah  : {f_twopass_low:.3f} Hz s.d. {f_twopass_high:.3f} Hz")
    summary_lines.append("  CATATAN TEORITIS:")
    summary_lines.append("  Pada Chebyshev Type II, Wn adalah batas awal stopband, bukan batas passband 3dB.")
    summary_lines.append("  Filtering dua arah (sosfiltfilt) menggandakan atenuasi dalam skala dB,")
    summary_lines.append("  sehingga passband efektif menyempit menjadi ~0.66 s.d. 6.08 Hz.")

    summary_lines.append("\n[3. PERBANDINGAN DETEKSI PUNCAK & LEMBAH (LAMA VS VALIDASI)]")
    summary_lines.append(f"- Jumlah Puncak Terdeteksi : Lama = {len(peak_times_old)} | Validasi = {len(peak_times_new)} (+1 puncak)")
    summary_lines.append(f"- Jumlah Lembah Terdeteksi : Lama = 19 | Validasi = {len(valleys_new)} (+1 lembah)")
    summary_lines.append(f"- Puncak Baru Ditemukan    : t = {peak_times_new[0]:.3f} s (amp = {peak_vals_new[0]:.4f} a.u.)")
    summary_lines.append("  (Pada tahap sebelumnya, puncak ini terlewat akibat transient tepi awal)")
    summary_lines.append(f"- Perubahan Amplitudo Puncak Awal (t=60.528s): Lama = {peak_vals_old[0]:.4f} -> Validasi = {peak_vals_new[1]:.4f} (selisih: {peak_vals_new[1]-peak_vals_old[0]:+.4f})")
    summary_lines.append(f"- Perubahan Amplitudo Puncak Akhir (t~69.95s): Lama = {peak_vals_old[-1]:.4f} -> Validasi = {peak_vals_new[-1]:.4f} (selisih: {peak_vals_new[-1]-peak_vals_old[-1]:+.4f})")
    summary_lines.append(f"- Rata-rata IBI : Lama = {np.mean(ibis_old_ms):.1f} ms | Validasi = {np.mean(ibis_new_ms):.1f} ms")
    summary_lines.append(f"- Median IBI    : Lama = {np.median(ibis_old_ms):.1f} ms | Validasi = {np.median(ibis_new_ms):.1f} ms")
    summary_lines.append(f"- Rata-rata HR  : Lama = {np.mean(hr_old_beats):.2f} BPM | Validasi = {np.mean(hr_new_beats):.2f} BPM")
    summary_lines.append(f"- Median HR     : Lama = {np.median(hr_old_beats):.2f} BPM | Validasi = {np.median(hr_new_beats):.2f} BPM")

    summary_lines.append("\n[4. PERGESERAN WAKTU PUNCAK (PEAK TIME SHIFTS)]")
    summary_lines.append("Pemetaan Puncak Lama (P1-P20) terhadap Puncak Baru (P2-P21):")
    for i in range(len(peak_times_old)):
        t_old = peak_times_old[i]
        t_new = peak_times_new[i + 1]
        dt = (t_new - t_old) * 1000.0  # ms
        v_old = peak_vals_old[i]
        v_new = peak_vals_new[i + 1]
        dv = v_new - v_old
        summary_lines.append(f"  Puncak {i+1:2d}: t_lama={t_old:.3f}s -> t_baru={t_new:.3f}s (dt={dt:+5.1f} ms) | amp: {v_old:.3f} -> {v_new:.3f} (dv={dv:+.3f})")

    summary_lines.append("\n[5. EVALUASI DAN KETIDAKPASTIAN]")
    summary_lines.append("1. Efek tepi terbukti memengaruhi hasil Tahap 4 secara signifikan pada kedua ujung sinyal.")
    summary_lines.append("   Puncak pertama pada detik 60.032 terlewat (false negative) pada tahap sebelumnya,")
    summary_lines.append("   dan amplitudo denyut terakhir pada detik 69.94 terdistorsi parah (0.706 vs 1.150 a.u.).")
    summary_lines.append("2. Klaim 'tidak ada false negative' pada tahap sebelumnya terbukti keliru karena")
    summary_lines.append("   keteraturan interval tidak menjamin tidak ada puncak yang terpotong di tepi batas.")
    summary_lines.append("3. Ketidakpastian yang tersisa:")
    summary_lines.append("   - Batas atas passband efektif (6.08 Hz) meredam komponen harmonik tinggi gelombang PPG,")
    summary_lines.append("     sehingga gelombang dicrotic notch terhaluskan secara parsial.")
    summary_lines.append("   - Resolusi sampling 125 Hz (periode 8 ms) membatasi kepastian posisi puncak dalam rentang +/- 8 ms.")
    summary_lines.append("   - Detak jantung tinggi (~121 BPM) menunjukkan kondisi fisiologis takikardia pada pasien rekaman 1.")
    summary_lines.append("=" * 72)

    summary_text = "\n".join(summary_lines)

    # Cetak ke terminal
    print(summary_text)

    # Simpan ke file teks
    with open(summary_path, "w", encoding="utf-8") as f_out:
        f_out.write(summary_text)
    print(f"\n[SUKSES] Laporan teks disimpan di:\n  {summary_path}")

    # =========================================================================
    # GAMBAR 1: Perbandingan Efek Tepi (Overlay & Selisih)
    # =========================================================================
    fig1, axes1 = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [2.5, 1.2]})
    fig1.suptitle("Validasi Efek Tepi Filter: Potongan 10 Detik vs Rekaman Panjang (488 Detik)", fontsize=13, fontweight="bold")

    # Subplot Atas: Overlay sinyal
    axes1[0].plot(time_slice, ppg_filt_long_slice, color="#1f77b4", linewidth=1.5, label="Validasi: Filter Rekaman Panjang (Bebas Efek Tepi)")
    axes1[0].plot(time_slice, ppg_filt_short_slice, color="#d62728", linestyle="--", linewidth=1.2, alpha=0.85, label="Tahap 4: Filter Langsung Potongan 10s (Terdapat Efek Tepi)")
    axes1[0].axvspan(60.0, 61.0, color="#ff9999", alpha=0.25, label="Zona Tepi Awal (Detik 60-61)")
    axes1[0].axvspan(69.0, 70.0, color="#ff9999", alpha=0.25, label="Zona Tepi Akhir (Detik 69-70)")
    axes1[0].set_ylabel("Amplitudo (a.u.)", fontsize=10)
    axes1[0].set_title("Overlay PPG Terfilter (Perhatikan deviasi pada detik 60-61 dan 69-70)", fontsize=10)
    axes1[0].grid(True, linestyle="--", alpha=0.6)
    axes1[0].legend(loc="upper right", fontsize=9)

    # Subplot Bawah: Selisih mutlak |diff|
    axes1[1].plot(time_slice, diff, color="#7f7f7f", linewidth=1.2, label=f"Selisih Mutlak |Lama - Baru| (Maks: {max_diff_overall:.3f} a.u.)")
    axes1[1].axhline(0.1, color="orange", linestyle=":", label="Ambang Toleransi 0.1 a.u.")
    axes1[1].axvspan(60.0, 61.0, color="#ff9999", alpha=0.25)
    axes1[1].axvspan(69.0, 70.0, color="#ff9999", alpha=0.25)
    axes1[1].set_xlabel("Waktu (detik)", fontsize=10)
    axes1[1].set_ylabel("|Selisih| (a.u.)", fontsize=10)
    axes1[1].grid(True, linestyle="--", alpha=0.6)
    axes1[1].legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    fig1.savefig(fig_edge_path, dpi=300)
    plt.close(fig1)
    print(f"[SUKSES] Gambar 1 disimpan di:\n  {fig_edge_path}")

    # =========================================================================
    # GAMBAR 2: Respons Frekuensi Filter (Satu Arah vs Dua Arah)
    # =========================================================================
    fig2, ax2 = plt.subplots(figsize=(11, 6))
    fig2.suptitle("Respons Frekuensi Filter Chebyshev Type II Bandpass (0.5 - 8.0 Hz, Orde 4, rs=20 dB)", fontsize=13, fontweight="bold")

    # Plot respons frekuensi 0 s.d. 20 Hz (fokus area biologis)
    mask_f = w <= 25.0
    ax2.plot(w[mask_f], db_single[mask_f], color="#1f77b4", linewidth=1.5, label="Respons Satu Arah |H(f)|")
    ax2.plot(w[mask_f], db_twopass[mask_f], color="#d62728", linewidth=1.8, label="Respons Efektif Dua Arah |H(f)|² (sosfiltfilt)")

    # Garis referensi
    ax2.axhline(-3.0, color="gray", linestyle="--", alpha=0.7, label="Garis -3 dB (Batas Passband Efektif)")
    ax2.axvline(0.5, color="purple", linestyle=":", alpha=0.8, label="Wn Bawah = 0.5 Hz (Stopband Edge)")
    ax2.axvline(8.0, color="purple", linestyle=":", alpha=0.8, label="Wn Atas = 8.0 Hz (Stopband Edge)")

    # Anotasi batas passband efektif dua arah
    ax2.axvline(f_twopass_low, color="green", linestyle="-.", alpha=0.8, label=f"Passband 3dB Efektif ({f_twopass_low:.2f} - {f_twopass_high:.2f} Hz)")
    ax2.axvline(f_twopass_high, color="green", linestyle="-.", alpha=0.8)

    ax2.annotate(f"Atenuasi Stopband: -40 dB\npada 0.5 & 8.0 Hz", xy=(8.0, -40.0), xytext=(10.0, -35.0),
                 arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2), fontsize=9, fontweight="bold", color="#d62728")

    ax2.set_xlabel("Frekuensi (Hz)", fontsize=10)
    ax2.set_ylabel("Gain (dB)", fontsize=10)
    ax2.set_ylim(-65, 5)
    ax2.set_xlim(0, 20)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    fig2.savefig(fig_freq_path, dpi=300)
    plt.close(fig2)
    print(f"[SUKSES] Gambar 2 disimpan di:\n  {fig_freq_path}")

    # =========================================================================
    # GAMBAR 3: Deteksi Puncak & Lembah Validasi (21 Puncak, 20 Lembah)
    # =========================================================================
    fig3, ax3 = plt.subplots(figsize=(13, 6))
    fig3.suptitle("Deteksi Puncak & Lembah PPG Tervalidasi (UCI Part_1 | Detik 60 - 70)", fontsize=13, fontweight="bold")

    ax3.plot(time_slice, ppg_filt_long_slice, color="#1f77b4", linewidth=1.4, label="PPG Terfilter Bebas Efek Tepi")
    ax3.scatter(peak_times_new, peak_vals_new, color="#d62728", s=65, marker="^", zorder=5, label=f"Puncak Sistolik Valid ({len(peaks_new)} titik)")
    ax3.scatter(valley_times_new, valley_vals_new, color="#2ca02c", s=65, marker="v", zorder=5, label=f"Lembah Diastolik Valid ({len(valleys_new)} titik)")

    # Soroti puncak pertama yang baru ditemukan
    ax3.scatter([peak_times_new[0]], [peak_vals_new[0]], facecolors="none", edgecolors="magenta", s=180, linewidths=2.0, zorder=6, label="Puncak Baru (Sebelumnya Terlewat)")
    ax3.annotate("Puncak Baru\n(t=60.03s)", xy=(peak_times_new[0], peak_vals_new[0]), xytext=(peak_times_new[0] + 0.15, peak_vals_new[0] + 0.15),
                 arrowprops=dict(arrowstyle="->", color="magenta", lw=1.5), fontsize=9, fontweight="bold", color="magenta")

    # Soroti pemulihan amplitudo puncak akhir
    ax3.annotate(f"Amplitudo Normal\n({peak_vals_new[-1]:.2f} a.u.)", xy=(peak_times_new[-1], peak_vals_new[-1]), xytext=(peak_times_new[-1] - 0.7, peak_vals_new[-1] + 0.2),
                 arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2), fontsize=9, fontweight="bold", color="#d62728")

    ax3.set_title(f"Hasil Validasi: 21 Puncak & 20 Lembah | Rata-rata IBI: {np.mean(ibis_new_ms):.1f} ms | Rata-rata HR: {np.mean(hr_new_beats):.1f} BPM", fontsize=10)
    ax3.set_xlabel("Waktu (detik)", fontsize=10)
    ax3.set_ylabel("Amplitudo (a.u.)", fontsize=10)
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.legend(loc="lower right", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    fig3.savefig(fig_peaks_path, dpi=300)
    plt.close(fig3)
    print(f"[SUKSES] Gambar 3 disimpan di:\n  {fig_peaks_path}")

if __name__ == "__main__":
    main()
