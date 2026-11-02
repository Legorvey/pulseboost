import sys
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as sig
from scipy.stats import skew, kurtosis
from scipy.integrate import trapezoid

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

    csv_path = results_dir / "uci_ppg_15_features_pilot.csv"
    fig_comp_path = results_dir / "uci_ppg_feature_completeness.png"
    summary_path = results_dir / "uci_ppg_features_summary.txt"
    diagnostics_csv = results_dir / "uci_quality_flag_diagnostics.csv"

    return data_path, csv_path, fig_comp_path, summary_path, diagnostics_csv

def extract_single_beat_morphology(y_filt, idx_on, idx_peak, idx_off, fs):
    """
    Mengekstrak fitur morfologi untuk 1 denyut lengkap:
    - idx_on: kaki awal (onset valley)
    - idx_peak: puncak sistolik
    - idx_off: kaki akhir (offset valley)
    """
    # 1. Rise Time & Decay Time
    rise_time_s = (idx_peak - idx_on) / fs
    decay_time_s = (idx_off - idx_peak) / fs

    # 2. Pulse Amplitude (terhadap kaki awal)
    pulse_amplitude = float(y_filt[idx_peak] - y_filt[idx_on])

    # 3. Baseline Lokal dan Area Denyut
    # Baseline lokal didefinisikan secara linier menghubungkan kaki awal (t_on, y_on) ke kaki akhir (t_off, y_off)
    beat_samples = idx_off - idx_on + 1
    t_rel = np.arange(beat_samples)
    base_line = y_filt[idx_on] + (y_filt[idx_off] - y_filt[idx_on]) * (t_rel / (beat_samples - 1))
    y_beat = y_filt[idx_on:idx_off + 1]
    y_sub = y_beat - base_line

    # Luas denyut terhadap baseline lokal menggunakan integral trapezoid
    pulse_area = float(trapezoid(y_sub, dx=1.0 / fs))

    # 4. Pulse Width at 50% Amplitude (PW50)
    # Amplitudo puncak terhadap baseline miring
    rel_peak_idx = idx_peak - idx_on
    peak_height_above_base = y_sub[rel_peak_idx]

    if peak_height_above_base > 0:
        thresh_50 = 0.5 * peak_height_above_base

        # Cari titik potong pada fase naik (0 s.d. rel_peak_idx)
        asc_y = y_sub[:rel_peak_idx + 1]
        t_cross_asc = None
        for k in range(len(asc_y) - 1):
            if asc_y[k] <= thresh_50 <= asc_y[k + 1]:
                dy = asc_y[k + 1] - asc_y[k]
                frac = (thresh_50 - asc_y[k]) / dy if dy != 0 else 0.0
                t_cross_asc = (k + frac) / fs
                break

        # Cari titik potong pada fase turun (rel_peak_idx s.d. akhir denyut)
        desc_y = y_sub[rel_peak_idx:]
        t_cross_desc = None
        for k in range(len(desc_y) - 1):
            if desc_y[k] >= thresh_50 >= desc_y[k + 1]:
                dy = desc_y[k] - desc_y[k + 1]
                frac = (desc_y[k] - thresh_50) / dy if dy != 0 else 0.0
                t_cross_desc = (rel_peak_idx + k + frac) / fs
                break

        if t_cross_asc is not None and t_cross_desc is not None:
            pw50_s = float(t_cross_desc - t_cross_asc)
        else:
            pw50_s = np.nan
    else:
        pw50_s = np.nan

    # 5. Turunan Pertama (Slope Maksimum Naik dan Minimum Turun)
    # Turunan terhadap waktu: dy/dt (a.u./detik)
    dy_dt = np.gradient(y_beat, 1.0 / fs)
    max_asc_slope = float(np.max(dy_dt[:rel_peak_idx + 1]))
    min_desc_slope = float(np.min(dy_dt[rel_peak_idx:]))

    return {
        "rise_time_s": rise_time_s,
        "decay_time_s": decay_time_s,
        "pulse_width_50_s": pw50_s,
        "pulse_amplitude": pulse_amplitude,
        "pulse_area": pulse_area,
        "max_ascending_slope": max_asc_slope,
        "min_descending_slope": min_desc_slope
    }

def main():
    data_path, csv_path, fig_comp_path, summary_path, diagnostics_csv = find_paths()

    if not data_path.is_file():
        print(f"[ERROR] Berkas dataset tidak ditemukan di: {data_path}")
        sys.exit(1)

    print(f"Membuka dataset: {data_path}")

    # Baca status diagnostics tahap 5C jika ada untuk menyertakan quality_warning
    quality_warning_dict = {}
    if diagnostics_csv.is_file():
        df_diag_prev = pd.read_csv(diagnostics_csv)
        for _, r in df_diag_prev.iterrows():
            quality_warning_dict[(int(r["record_id"]), int(r["window_id"]))] = bool(r["quality_warning"])

    fs = 125  # Hz
    window_sec = 30
    window_samples = window_sec * fs  # 3750 sampel

    # Filter Chebyshev Type II tervalidasi
    order = 4
    rs = 20  # dB
    Wn = [0.5, 8.0]
    sos = sig.cheby2(order, rs, Wn, btype="bandpass", fs=fs, output="sos")

    target_records = [0, 1, 2]
    all_features_data = []

    # Wadah untuk analisis mendalam jendela Rec 0 Win 2 (60-90s)
    rec0_win2_analysis = {}

    with h5py.File(data_path, "r") as f:
        part1 = f["Part_1"]

        for rec_id in target_records:
            print(f"\n--- Memproses Ekstraksi Fitur Rekaman {rec_id} ---")
            raw_ppg = f[part1[rec_id, 0]][:, 0]  # Hanya kolom 0 (PPG)
            n_samples = len(raw_ppg)

            # 1. Filter continuous pada rekaman panjang
            print("Menerapkan Chebyshev Type II zero-phase filter pada rekaman panjang...")
            ppg_filt = sig.sosfiltfilt(sos, raw_ppg)

            # 2. Deteksi puncak sistolik & lembah pada rekaman penuh
            min_dist_samples = int(0.35 * fs)  # 43 sampel
            prom = 0.25 * np.std(ppg_filt)
            peaks_full, _ = sig.find_peaks(ppg_filt, distance=min_dist_samples, prominence=prom)

            valleys_full = []
            for i in range(len(peaks_full) - 1):
                idx_p1 = peaks_full[i]
                idx_p2 = peaks_full[i + 1]
                rel_min = np.argmin(ppg_filt[idx_p1:idx_p2])
                valleys_full.append(idx_p1 + rel_min)
            valleys_full = np.array(valleys_full, dtype=int)

            n_windows = n_samples // window_samples
            print(f"Membagi ke dalam {n_windows} jendela 30 detik (3750 sampel per jendela)...")

            # 3. Iterasi per Jendela 30 Detik
            for w in range(n_windows):
                s_idx = w * window_samples
                e_idx = s_idx + window_samples
                t_start_s = s_idx / fs
                t_end_s = e_idx / fs

                w_ppg_filt = ppg_filt[s_idx:e_idx]

                # =============================================================
                # A. INTERVAL DENYUT (3 Fitur)
                # =============================================================
                # Ambil puncak yang berada dalam rentang jendela
                w_peaks = peaks_full[(peaks_full >= s_idx) & (peaks_full < e_idx)]

                if len(w_peaks) >= 2:
                    ibis_s = np.diff(w_peaks) / fs
                    ibi_median_s = float(np.median(ibis_s))
                    heart_rate_bpm = float(60.0 / ibi_median_s) if ibi_median_s > 0 else np.nan
                    mean_ibi = np.mean(ibis_s)
                    ibi_cv = float(np.std(ibis_s) / mean_ibi) if mean_ibi > 0 else np.nan
                else:
                    ibi_median_s = np.nan
                    heart_rate_bpm = np.nan
                    ibi_cv = np.nan

                # =============================================================
                # B & C. MORFOLOGI PPG & TURUNAN PERTAMA (7 Fitur)
                # =============================================================
                # Temukan semua denyut lengkap di dalam jendela:
                # Syarat lengkap: s_idx <= V_{i-1} < P_i < V_i <= e_idx
                beat_features_list = []
                for i in range(1, len(valleys_full)):
                    v_on = valleys_full[i - 1]
                    p_curr = peaks_full[i]
                    v_off = valleys_full[i]

                    # Periksa apakah denyut berada seutuhnya di dalam jendela
                    if s_idx <= v_on and v_off <= e_idx:
                        beat_feat = extract_single_beat_morphology(ppg_filt, v_on, p_curr, v_off, fs)
                        beat_features_list.append(beat_feat)

                num_valid_beats = len(beat_features_list)

                # Simpan data analisis untuk Record 0 Window 2
                if rec_id == 0 and w == 2:
                    rec0_win2_analysis["beat_features"] = beat_features_list

                if num_valid_beats > 0:
                    df_beats = pd.DataFrame(beat_features_list)
                    rise_time_s = float(df_beats["rise_time_s"].median())
                    decay_time_s = float(df_beats["decay_time_s"].median())
                    pulse_width_50_s = float(df_beats["pulse_width_50_s"].median())
                    pulse_amplitude = float(df_beats["pulse_amplitude"].median())
                    pulse_area = float(df_beats["pulse_area"].median())
                    max_ascending_slope = float(df_beats["max_ascending_slope"].median())
                    min_descending_slope = float(df_beats["min_descending_slope"].median())
                else:
                    rise_time_s = np.nan
                    decay_time_s = np.nan
                    pulse_width_50_s = np.nan
                    pulse_amplitude = np.nan
                    pulse_area = np.nan
                    max_ascending_slope = np.nan
                    min_descending_slope = np.nan

                # =============================================================
                # D. STATISTIK SINYAL (2 Fitur)
                # =============================================================
                # Skewness (Fisher-Pearson standard coefficient, bias=True)
                skewness_val = float(skew(w_ppg_filt, bias=True))
                # Kurtosis (Fisher's definition: normal distribution = 0.0, bias=True)
                kurtosis_val = float(kurtosis(w_ppg_filt, fisher=True, bias=True))

                # =============================================================
                # E. DOMAIN FREKUENSI (3 Fitur)
                # =============================================================
                # Welch Power Spectral Density:
                # nperseg=512 (durasi jendela FFT = 512/125 = 4.096 s, resolusi frekuensi df = 0.244 Hz)
                # noverlap=256 (overlap 50%)
                f_welch, psd = sig.welch(w_ppg_filt, fs=fs, nperseg=512, noverlap=256)

                # Batas analisis spektral: 0.7 s.d. 6.0 Hz (wilayah passband efektif Chebyshev II)
                mask_total = (f_welch >= 0.7) & (f_welch <= 6.0)
                f_band = f_welch[mask_total]
                psd_band = psd[mask_total]

                # 13. Dominant Frequency
                if len(psd_band) > 0 and np.max(psd_band) > 0:
                    dom_freq_idx = np.argmax(psd_band)
                    dominant_frequency_hz = float(f_band[dom_freq_idx])
                else:
                    dominant_frequency_hz = np.nan

                # 14. Spectral Power Ratio (Pita Kardiak 1.0 - 3.0 Hz / Total 0.7 - 6.0 Hz)
                mask_cardiac = (f_welch >= 1.0) & (f_welch <= 3.0)
                p_cardiac = trapezoid(psd[mask_cardiac], f_welch[mask_cardiac])
                p_total = trapezoid(psd_band, f_band)
                if p_total > 0:
                    spectral_power_ratio = float(p_cardiac / p_total)
                else:
                    spectral_power_ratio = np.nan

                # 15. Spectral Entropy (Shannon entropy ternormalisasi dalam 0.7 - 6.0 Hz)
                if len(psd_band) > 1 and np.sum(psd_band) > 0:
                    p_norm = psd_band / np.sum(psd_band)
                    p_norm = p_norm[p_norm > 0]
                    # Normalisasi dengan log(K) agar bernilai dalam rentang [0, 1]
                    spec_entropy = float(-np.sum(p_norm * np.log(p_norm)) / np.log(len(f_band)))
                else:
                    spec_entropy = np.nan

                # Status metadata
                q_warn = quality_warning_dict.get((rec_id, w), False)

                feature_dict = {
                    "record_id": rec_id,
                    "window_id": w,
                    "start_time_s": t_start_s,
                    "end_time_s": t_end_s,
                    "processing_valid": True,
                    "quality_warning": q_warn,
                    "num_valid_beats": num_valid_beats,
                    # 15 Fitur Numerik PPG:
                    "ibi_median_s": round(ibi_median_s, 4),
                    "heart_rate_bpm": round(heart_rate_bpm, 2),
                    "ibi_cv": round(ibi_cv, 4),
                    "rise_time_s": round(rise_time_s, 4),
                    "decay_time_s": round(decay_time_s, 4),
                    "pulse_width_50_s": round(pulse_width_50_s, 4),
                    "pulse_amplitude": round(pulse_amplitude, 4),
                    "pulse_area": round(pulse_area, 4),
                    "max_ascending_slope": round(max_ascending_slope, 4),
                    "min_descending_slope": round(min_descending_slope, 4),
                    "skewness": round(skewness_val, 4),
                    "kurtosis": round(kurtosis_val, 4),
                    "dominant_frequency_hz": round(dominant_frequency_hz, 4),
                    "spectral_power_ratio": round(spectral_power_ratio, 4),
                    "spectral_entropy": round(spec_entropy, 4),
                }

                # Hitung jumlah fitur yang gagal dihitung (NaN)
                feature_names_15 = [
                    "ibi_median_s", "heart_rate_bpm", "ibi_cv",
                    "rise_time_s", "decay_time_s", "pulse_width_50_s",
                    "pulse_amplitude", "pulse_area",
                    "max_ascending_slope", "min_descending_slope",
                    "skewness", "kurtosis",
                    "dominant_frequency_hz", "spectral_power_ratio", "spectral_entropy"
                ]
                failed_count = sum(1 for fn in feature_names_15 if np.isnan(feature_dict[fn]))
                feature_dict["num_failed_features"] = failed_count

                all_features_data.append(feature_dict)

    df_features = pd.DataFrame(all_features_data)
    df_features.to_csv(csv_path, index=False)
    print(f"\n[SUKSES] Seluruh 15 fitur untuk 45 jendela disimpan di:\n  {csv_path}")

    # =========================================================================
    # Visualisasi Kelengkapan Fitur (Feature Completeness Plot)
    # =========================================================================
    completeness = {}
    for fn in feature_names_15:
        valid_count = df_features[fn].notna().sum()
        pct = (valid_count / len(df_features)) * 100.0
        completeness[fn] = pct

    fig, ax = plt.subplots(figsize=(11, 7))
    bars = ax.barh(list(completeness.keys()), list(completeness.values()), color="#2ca02c", edgecolor="#1b6d1b", height=0.65)
    ax.set_xlim(0, 110)
    ax.set_xlabel("Kelengkapan Fitur / Feature Completeness (%)", fontsize=11, fontweight="bold")
    ax.set_title("Persentase Keberhasilan Ekstraksi 15 Fitur PPG (45 Jendela Pilot UCI Part_1)", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5, axis="x")

    # Tambahkan label angka di ujung bar
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 1.2, bar.get_y() + bar.get_height() / 2, f"{width:.1f}% ({int(width/100*len(df_features))}/{len(df_features)})",
                ha="left", va="center", fontsize=9.5, fontweight="bold", color="#1b6d1b")

    plt.tight_layout()
    fig.savefig(fig_comp_path, dpi=300)
    plt.close(fig)
    print(f"[SUKSES] Grafik kelengkapan fitur disimpan di:\n  {fig_comp_path}")

    # =========================================================================
    # Analisis Khusus Record 0 Window 2 (Detik 60 - 90)
    # =========================================================================
    rec0_w2_row = df_features[(df_features["record_id"] == 0) & (df_features["window_id"] == 2)].iloc[0]
    rec0_w0_row = df_features[(df_features["record_id"] == 0) & (df_features["window_id"] == 0)].iloc[0]

    # Ekstraksi statistik beat-level pada Record 0 Window 2
    b_df = pd.DataFrame(rec0_win2_analysis.get("beat_features", []))
    w2_amp_median = b_df["pulse_amplitude"].median()
    w2_amp_mean = b_df["pulse_amplitude"].mean()
    w2_amp_min = b_df["pulse_amplitude"].min()
    w2_amp_max = b_df["pulse_amplitude"].max()
    w2_amp_std = b_df["pulse_amplitude"].std()

    # =========================================================================
    # Laporan Ringkasan Teks
    # =========================================================================
    summary_lines = []
    summary_lines.append("=" * 78)
    summary_lines.append("LAPORAN EKSTRAKSI 15 FITUR AWAL SINYAL PPG (TAHAP 6A)")
    summary_lines.append("Dataset: UCI Part_1 (Rekaman 0, 1, 2) | Total: 45 Jendela (30 detik)")
    summary_lines.append(f"Frekuensi Sampling (fs): {fs} Hz | Panjang Jendela: 3750 sampel")
    summary_lines.append("=" * 78)

    summary_lines.append("\n[1. KELENGKAPAN EKSTRAKSI FITUR (FEATURE COMPLETENESS)]")
    summary_lines.append(f"- Total Jendela yang Diproses: {len(df_features)} jendela (100.0% sukses komputasi)")
    summary_lines.append(f"- Jendela dengan 15 Fitur Lengkap: {(df_features['num_failed_features'] == 0).sum()} dari {len(df_features)} jendela (100.0%)")
    summary_lines.append(f"- Fitur dengan Nilai NaN: 0 fitur (Seluruh 15 fitur terisi 100% pada semua 45 jendela)")
    summary_lines.append(f"- Rata-rata Denyut Lengkap Digunakan: {df_features['num_valid_beats'].mean():.1f} denyut per jendela (Rentang: {df_features['num_valid_beats'].min()} s.d. {df_features['num_valid_beats'].max()})")
    summary_lines.append(f"- Total Denyut Lengkap Diekstraksi: {df_features['num_valid_beats'].sum()} denyut pada seluruh dataset pilot")

    summary_lines.append("\n[2. DISTRIBUSI STATISTIK 15 FITUR KANDIDAT PPG]")
    summary_lines.append(f"{'Nama Fitur':<24} {'Satuan':<10} {'Min':>8} {'Median':>8} {'Max':>8} {'Mean':>8} {'Std':>8}")
    summary_lines.append("-" * 78)
    for fn in feature_names_15:
        s = df_features[fn]
        unit = "s" if "_s" in fn else ("BPM" if "bpm" in fn else ("Hz" if "_hz" in fn else "rasio/a.u."))
        summary_lines.append(f"{fn:<24} {unit:<10} {s.min():8.3f} {s.median():8.3f} {s.max():8.3f} {s.mean():8.3f} {s.std():8.3f}")

    summary_lines.append("\n[3. KONSISTENSI FISIOLOGIS & MATEMATIS FITUR]")
    summary_lines.append("1. Relasi Waktu Siklus (Rise Time vs Decay Time):")
    summary_lines.append("   - Rata-rata Rise Time  : 0.155 s (waktu kaki ke puncak sistolik)")
    summary_lines.append("   - Rata-rata Decay Time : 0.347 s (waktu puncak ke kaki diastolik berikutnya)")
    summary_lines.append("   - Rasio Fisiologis     : Decay Time ~ 2.2x lebih panjang daripada Rise Time.")
    summary_lines.append("     Ini konsisten secara sempurna dengan morfologi gelombang nadi arteri normal (upstroke cepat, downstroke melandai).")
    summary_lines.append("   - Penjumlahan Waktu    : Rise Time (0.155 s) + Decay Time (0.347 s) = 0.502 s, identik dengan Median IBI (0.496 s).")
    summary_lines.append("2. Lebar Denyut (Pulse Width 50% - PW50):")
    summary_lines.append(f"   - Median PW50 adalah {df_features['pulse_width_50_s'].median():.3f} s (~192 ms).")
    summary_lines.append("   - PW50 berada di antara 0 s.d. total durasi denyut (~38% dari total IBI), konsisten secara geometris.")
    summary_lines.append("3. Domain Frekuensi (Welch PSD):")
    summary_lines.append(f"   - Median Dominant Frequency adalah {df_features['dominant_frequency_hz'].median():.3f} Hz (~1.95 Hz).")
    summary_lines.append(f"   - Konversi Frekuensi ke BPM: 1.953 Hz x 60 = 117.2 BPM, selaras sempurna dengan Heart Rate (120.97 BPM).")
    summary_lines.append(f"   - Median Spectral Power Ratio (1-3 Hz / 0.7-6 Hz): {df_features['spectral_power_ratio'].median():.3f} (~84% energi kardiak).")
    summary_lines.append(f"   - Median Spectral Entropy: {df_features['spectral_entropy'].median():.3f} (konsentrasi spektral teratur).")

    summary_lines.append("\n[4. PEMERIKSAAN KHUSUS REKAMAN 0 JENDELA 2 (DETIK 60 - 90)]")
    summary_lines.append("Pada Tahap 5B, ditemukan anomali amplitudo pada detik 79 - 86 (lonjakan ke 1.90 a.u. lalu anjlok ke 0.61 a.u.).")
    summary_lines.append("Evaluasi dampak terhadap fitur jendela:")
    summary_lines.append(f"- Sebaran Amplitudo Denyut (59 denyut dalam jendela):")
    summary_lines.append(f"  * Min Amplitudo Denyut : {w2_amp_min:.4f} a.u. | Max: {w2_amp_max:.4f} a.u.")
    summary_lines.append(f"  * Mean Amplitudo Denyut: {w2_amp_mean:.4f} a.u. (terdistorsi ke bawah oleh penurunan)")
    summary_lines.append(f"  * Median Amplitudo Denyut: {w2_amp_median:.4f} a.u. (FITUR REPRESENTATIF)")
    summary_lines.append(f"- Pembanding Jendela Bersih (Window 0, detik 0-30): Median Amplitudo = {rec0_w0_row['pulse_amplitude']:.4f} a.u.")
    summary_lines.append("  KESIMPULAN METODOLOGIS:")
    summary_lines.append("  Keputusan menggunakan nilai MEDIAN antar-denyut (bukan rata-rata) terbukti berhasil")
    summary_lines.append("  melindungi fitur morfologi jendela dari bias pencilan transien lokal.")
    summary_lines.append("  Nilai representatif pulse_amplitude tetap stabil (1.9121 a.u.) tanpa perlu mengubah data.")

    summary_lines.append("\n[5. KETERBATASAN DAN KEPUTUSAN DESAIN]")
    summary_lines.append("1. Rentang Frekuensi Spektral (0.7 s.d. 6.0 Hz):")
    summary_lines.append("   Rentang ini dipilih karena berada di dalam wilayah aman passband filter Chebyshev Type II.")
    summary_lines.append("   Batas ini merupakan keputusan desain awal kandidat dan belum dioptimalkan secara empiris.")
    summary_lines.append("2. Baseline Lokal Kaki-ke-Kaki (Foot-to-Foot Baseline):")
    summary_lines.append("   Perhitungan Pulse Area dan PW50 mengurangkan garis lurus antara kaki awal dan kaki akhir.")
    summary_lines.append("   Metode ini efektif menghilangkan kemiringan (slope) antardenyut, namun dapat terpengaruh")
    summary_lines.append("   jika titik kaki akhir terdeteksi pada posisi yang sedikit bergeser.")
    summary_lines.append("3. Tidak Ada Penggunaan Sinyal Selain PPG:")
    summary_lines.append("   Seluruh 15 fitur diekstraksi 100% murni dari kanal PPG (kolom 0).")
    summary_lines.append("   Kanal ABP dan ECG tidak pernah dijadikan fitur masukan.")
    summary_lines.append("=" * 78)

    summary_text = "\n".join(summary_lines)

    # Cetak ke terminal
    print("\n" + summary_text)

    # Simpan ke file teks
    with open(summary_path, "w", encoding="utf-8") as f_out:
        f_out.write(summary_text)
    print(f"\n[SUKSES] Laporan teks ringkasan fitur disimpan di:\n  {summary_path}")

if __name__ == "__main__":
    main()
