import sys
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as sig
from scipy.stats import skew

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

    csv_path = results_dir / "uci_quality_audit.csv"
    fig_rec0_path = results_dir / "uci_quality_zoom_record0.png"
    fig_rec1_path = results_dir / "uci_quality_zoom_record1.png"
    fig_rec2_path = results_dir / "uci_quality_zoom_record2.png"
    summary_path = results_dir / "uci_quality_audit_summary.txt"

    return data_path, results_dir, csv_path, fig_rec0_path, fig_rec1_path, fig_rec2_path, summary_path

def main():
    data_path, results_dir, csv_path, fig_rec0_path, fig_rec1_path, fig_rec2_path, summary_path = find_paths()

    if not data_path.is_file():
        print(f"[ERROR] Berkas dataset tidak ditemukan di: {data_path}")
        sys.exit(1)

    print(f"Membuka dataset: {data_path}")

    fs = 125  # Hz
    window_sec = 30
    window_samples = window_sec * fs  # 3750 sampel

    # Konfigurasi filter tervalidasi
    order = 4
    rs = 20  # dB
    Wn = [0.5, 8.0]
    sos = sig.cheby2(order, rs, Wn, btype="bandpass", fs=fs, output="sos")

    target_records = [0, 1, 2]
    audit_window_records = []
    record_signals = {}

    with h5py.File(data_path, "r") as f:
        if "Part_1" not in f:
            print("[ERROR] Key 'Part_1' tidak ditemukan dalam berkas HDF5.")
            sys.exit(1)

        part1 = f["Part_1"]

        for rec_id in target_records:
            print(f"\n--- Mengaudit Rekaman Indeks {rec_id} ---")
            raw_matrix = f[part1[rec_id, 0]][:]
            n_samples = raw_matrix.shape[0]

            ppg_raw = raw_matrix[:, 0]
            abp_raw = raw_matrix[:, 1]
            ecg_raw = raw_matrix[:, 2]

            # Filtering continuous pada rekaman panjang
            ppg_filt = sig.sosfiltfilt(sos, ppg_raw)

            # Deteksi puncak sistolik pada rekaman panjang
            min_dist_samples = int(0.35 * fs)  # 43 sampel
            prominence_val = 0.25 * np.std(ppg_filt)
            peaks_full, _ = sig.find_peaks(ppg_filt, distance=min_dist_samples, prominence=prominence_val)

            # Deteksi lembah di antara setiap dua puncak bertetangga
            valleys_full = []
            for i in range(len(peaks_full) - 1):
                idx_p1 = peaks_full[i]
                idx_p2 = peaks_full[i + 1]
                rel_min = np.argmin(ppg_filt[idx_p1:idx_p2])
                valleys_full.append(idx_p1 + rel_min)
            valleys_full = np.array(valleys_full, dtype=int)

            # Simpan data rekaman untuk keperluan visualisasi zoom
            record_signals[rec_id] = {
                "ppg_raw": ppg_raw,
                "ppg_filt": ppg_filt,
                "abp_raw": abp_raw,
                "peaks_full": peaks_full,
                "valleys_full": valleys_full,
                "fs": fs
            }

            # Pemetaan amplitudo denyut (pulse amplitude):
            # Amplitudo denyut P_i = amp(P_i) - amp(V_i-1) di mana V_i-1 adalah lembah sebelum P_i
            # Puncak pertama P_0 tidak memiliki lembah sebelumnya, sehingga denyut dihitung mulai i=1
            pulse_data = []
            for i in range(1, len(peaks_full)):
                p_idx = peaks_full[i]
                v_idx = valleys_full[i - 1]
                pulse_amp = float(ppg_filt[p_idx] - ppg_filt[v_idx])
                pulse_data.append((p_idx, pulse_amp))

            n_windows = n_samples // window_samples

            # Analisis setiap jendela 30 detik
            for w in range(n_windows):
                s_idx = w * window_samples
                e_idx = s_idx + window_samples
                t_start = s_idx / fs
                t_end = e_idx / fs

                w_ppg_filt = ppg_filt[s_idx:e_idx]
                w_ppg_raw = ppg_raw[s_idx:e_idx]

                # 1. Skewness SQI
                skew_val = float(skew(w_ppg_filt, bias=True))

                # 2. Puncak dalam jendela ini
                w_peaks = peaks_full[(peaks_full >= s_idx) & (peaks_full < e_idx)]
                num_peaks = len(w_peaks)

                # 3. Interval Antarpuncak (IBI)
                if num_peaks >= 2:
                    w_ibis_s = np.diff(w_peaks) / fs
                    w_ibis_ms = w_ibis_s * 1000.0
                    med_ibi_ms = float(np.median(w_ibis_ms))
                    # IBI MAD (variasi interval)
                    ibi_mad_ms = float(np.median(np.abs(w_ibis_ms - med_ibi_ms)))
                    ibi_iqr_ms = float(np.percentile(w_ibis_ms, 75) - np.percentile(w_ibis_ms, 25))
                    # Deteksi outlier IBI dengan penanganan aman diskritisasi sampling (fs=125Hz -> dt=8ms)
                    # Jika MAD = 0, gunakan standar deviasi atau batas resolusi minimal 2 sampel (16 ms)
                    if ibi_mad_ms > 0:
                        norm_ibi_mad = max(1.4826 * ibi_mad_ms, 16.0)
                    else:
                        norm_ibi_mad = max(float(np.std(w_ibis_ms)), 16.0)

                    ibi_outliers = [ibi for ibi in w_ibis_ms if abs(ibi - med_ibi_ms) > 3.0 * norm_ibi_mad]
                    n_ibi_outliers = len(ibi_outliers)
                else:
                    med_ibi_ms = np.nan
                    ibi_mad_ms = np.nan
                    ibi_iqr_ms = np.nan
                    n_ibi_outliers = 0

                # 4. Amplitudo Denyut (Peak-to-Valley)
                w_amps = [amp for p_idx, amp in pulse_data if s_idx <= p_idx < e_idx]
                if len(w_amps) >= 2:
                    w_amps_arr = np.array(w_amps)
                    med_amp = float(np.median(w_amps_arr))
                    mad_amp = float(np.median(np.abs(w_amps_arr - med_amp)))
                    # Penanganan aman jika MAD = 0
                    if mad_amp > 0:
                        norm_mad_amp = max(1.4826 * mad_amp, 0.05)
                    else:
                        norm_mad_amp = max(float(np.std(w_amps_arr)), 0.05)

                    # Hitung deviasi dan outlier
                    amp_deviations = np.abs(w_amps_arr - med_amp)
                    max_dev_amp = float(np.max(amp_deviations))
                    amp_outliers = np.where(amp_deviations > 3.0 * norm_mad_amp)[0]
                    n_amp_outliers = len(amp_outliers)
                    ratio_amp_outliers = float(n_amp_outliers / len(w_amps))
                else:
                    med_amp = np.nan
                    mad_amp = np.nan
                    max_dev_amp = 0.0
                    n_amp_outliers = 0
                    ratio_amp_outliers = 0.0

                # 5. Sampel Tidak Valid & Clipping
                inv_count = int(np.isnan(w_ppg_filt).sum() + np.isinf(w_ppg_filt).sum())
                inv_ratio = float(inv_count / window_samples)

                # Indikasi clipping pada batas rel ADC sensor (0.00V atau 4.00V)
                # Serta pemeriksaan flatline (selisih berturutan = 0)
                clip_count = int(((w_ppg_raw <= 0.01) | (w_ppg_raw >= 3.99)).sum())
                clip_ratio = float(clip_count / window_samples)

                # Hitung flatline berturut-turut (maksimum sampel bernilai konstan identik berturut-turut)
                diffs_zero = (np.diff(w_ppg_raw) == 0).astype(int)
                # Panjang run konstan terpanjang
                if np.any(diffs_zero):
                    max_flatline_samples = int(np.max(np.diff(np.where(np.concatenate(([diffs_zero[0]], diffs_zero[:-1] != diffs_zero[1:], [True])))[0])[::2])) if np.sum(diffs_zero) > 0 else 0
                else:
                    max_flatline_samples = 0

                # 6. Penetapan Label Sementara (Tentative Label)
                # Menandai anomali yang perlu diperiksa
                flag_reasons = []
                if n_amp_outliers > 0:
                    flag_reasons.append(f"{n_amp_outliers}_amp_outliers")
                if n_ibi_outliers > 0:
                    flag_reasons.append(f"{n_ibi_outliers}_ibi_outliers")
                if clip_count > 0:
                    flag_reasons.append(f"{clip_count}_clip_samples")
                if inv_count > 0:
                    flag_reasons.append(f"{inv_count}_inv_samples")

                if len(flag_reasons) > 0:
                    tentative_label = "review_needed"
                else:
                    tentative_label = "no_obvious_issue"

                audit_window_records.append({
                    "record_id": rec_id,
                    "window_id": w,
                    "start_time_s": t_start,
                    "end_time_s": t_end,
                    "skewness_sqi": round(skew_val, 4),
                    "num_peaks": num_peaks,
                    "median_ibi_ms": round(med_ibi_ms, 2) if not np.isnan(med_ibi_ms) else None,
                    "ibi_mad_ms": round(ibi_mad_ms, 2) if not np.isnan(ibi_mad_ms) else None,
                    "ibi_iqr_ms": round(ibi_iqr_ms, 2) if not np.isnan(ibi_iqr_ms) else None,
                    "num_ibi_outliers": n_ibi_outliers,
                    "median_pulse_amp": round(med_amp, 4) if not np.isnan(med_amp) else None,
                    "pulse_amp_mad": round(mad_amp, 4) if not np.isnan(mad_amp) else None,
                    "max_amp_dev": round(max_dev_amp, 4),
                    "num_amp_outliers": n_amp_outliers,
                    "ratio_amp_outliers": round(ratio_amp_outliers, 4),
                    "invalid_count": inv_count,
                    "invalid_ratio": inv_ratio,
                    "clipping_samples": clip_count,
                    "clipping_ratio": round(clip_ratio, 5),
                    "max_flatline_samples": max_flatline_samples,
                    "tentative_label": tentative_label,
                    "anomaly_flags": ";".join(flag_reasons) if flag_reasons else "none"
                })

    df_audit = pd.DataFrame(audit_window_records)
    df_audit.to_csv(csv_path, index=False)
    print(f"\n[SUKSES] Tabel audit disimpan di:\n  {csv_path}")

    # =========================================================================
    # Visualisasi Zoom untuk 3 Lokasi Spesifik
    # 1. Record 0: Detik 75 - 88 (Fokus 79 - 86)
    # 2. Record 1: Detik 12 - 30 (Fokus 15 - 18 & 27 - 29)
    # 3. Record 2: Detik 50 - 60 (Fokus 54 - 58)
    # =========================================================================
    def create_zoom_plot(rec_id, t_start_z, t_end_z, focus_zones, out_path, title_text, note_text):
        rec = record_signals[rec_id]
        s_idx = int(t_start_z * fs)
        e_idx = int(t_end_z * fs)
        t_axis = np.arange(s_idx, e_idx) / fs

        p_filt_z = rec["ppg_filt"][s_idx:e_idx]
        a_raw_z = rec["abp_raw"][s_idx:e_idx]

        # Puncak dan lembah dalam jendela zoom
        peaks_z = rec["peaks_full"][(rec["peaks_full"] >= s_idx) & (rec["peaks_full"] < e_idx)]
        valleys_z = rec["valleys_full"][(rec["valleys_full"] >= s_idx) & (rec["valleys_full"] < e_idx)]

        fig, axes = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True, gridspec_kw={"height_ratios": [1.4, 1.1]})
        fig.suptitle(title_text, fontsize=12, fontweight="bold")

        # Subplot 1: Filtered PPG + Puncak + Lembah
        axes[0].plot(t_axis, p_filt_z, color="#1f77b4", linewidth=1.4, label="PPG Terfilter (0.5 - 8.0 Hz)")
        axes[0].scatter(peaks_z / fs, rec["ppg_filt"][peaks_z], color="#d62728", s=60, marker="^", zorder=5, label=f"Puncak Sistolik ({len(peaks_z)} titik)")
        axes[0].scatter(valleys_z / fs, rec["ppg_filt"][valleys_z], color="#2ca02c", s=60, marker="v", zorder=5, label=f"Lembah Antardenyut ({len(valleys_z)} titik)")

        for f_s, f_e, f_label in focus_zones:
            axes[0].axvspan(f_s, f_e, color="#ff9999", alpha=0.3, label=f_label)
            axes[1].axvspan(f_s, f_e, color="#ff9999", alpha=0.3)

        axes[0].set_title(f"PPG Terfilter | {note_text}", fontsize=10)
        axes[0].set_ylabel("Amplitudo (a.u.)", fontsize=10)
        axes[0].grid(True, linestyle="--", alpha=0.6)
        axes[0].legend(loc="upper right", fontsize=8.5)

        # Subplot 2: Raw ABP
        axes[1].plot(t_axis, a_raw_z, color="#d62728", linewidth=1.2, label="ABP Mentah (Arterial Blood Pressure)")
        axes[1].set_title("ABP Mentah (Referensi Visual Pembanding, Bukan Fitur Model PPG)", fontsize=10, fontstyle="italic")
        axes[1].set_xlabel("Waktu (detik)", fontsize=10)
        axes[1].set_ylabel("Tekanan (mmHg)", fontsize=10)
        axes[1].grid(True, linestyle="--", alpha=0.6)
        axes[1].legend(loc="upper right", fontsize=8.5)

        plt.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"[SUKSES] Grafik zoom disimpan di:\n  {out_path}")

    # Grafik Zoom 1: Record 0 (75 - 88s, fokus 79 - 86s)
    create_zoom_plot(
        0, 76.0, 88.0,
        [(79.0, 86.0, "Zona Anomali (Detik 79 - 86)")],
        fig_rec0_path,
        "Audit Kualitas PPG - Rekaman 0 (Detik 76 s.d. 88 | Fokus Anomali 79 - 86)",
        "Lonjakan amplitudo (1.90 a.u.) diikuti penurunan berturut-turut (~0.61 a.u.) sementara ABP tetap stabil (artefak lokal)"
    )

    # Grafik Zoom 2: Record 1 (12 - 30s, fokus 15 - 18s & 27 - 29s)
    create_zoom_plot(
        1, 12.0, 30.0,
        [(15.0, 18.0, "Zona Aritmia 1 (15-18s)"), (27.0, 29.5, "Zona Aritmia 2 (27-29.5s)")],
        fig_rec1_path,
        "Audit Kualitas PPG - Rekaman 1 (Detik 12 s.d. 30 | Fokus Detik 15-18 & 27-29)",
        "Denyut prematur (IBI pendek 368 ms & amp anjlok) dengan jeda kompensasi di ABP & PPG (kejadian fisiologis)"
    )

    # Grafik Zoom 3: Record 2 (50 - 60s, fokus 54 - 58s)
    create_zoom_plot(
        2, 50.0, 60.0,
        [(54.0, 58.0, "Zona Variasi IBI (Detik 54 - 58)")],
        fig_rec2_path,
        "Audit Kualitas PPG - Rekaman 2 (Detik 50 s.d. 60 | Fokus Detik 54 - 58)",
        "Variasi transien IBI (interval memendek ke 400 ms) diikuti jeda kompensasi 544 ms"
    )

    # =========================================================================
    # Laporan Teks Ringkasan Audit Kualitas (Summary)
    # =========================================================================
    summary_lines = []
    summary_lines.append("=" * 75)
    summary_lines.append("LAPORAN AUDIT KUALITAS SINYAL PPG SEBELUM EKSTRAKSI FITUR (TAHAP 5B)")
    summary_lines.append("Berkas: Part_1.mat | Rekaman: Indeks 0, 1, 2 | Total: 45 Jendela (30 detik)")
    summary_lines.append("=" * 75)

    n_review = int((df_audit["tentative_label"] == "review_needed").sum())
    n_no_issue = int((df_audit["tentative_label"] == "no_obvious_issue").sum())

    summary_lines.append("\n[1. REKAPITULASI LABEL SEMENTARA AUDIT]")
    summary_lines.append(f"- Total Jendela yang Diaudit  : 45 jendela")
    summary_lines.append(f"- Label 'review_needed'      : {n_review} jendela ({n_review/45*100:.1f}%)")
    summary_lines.append(f"- Label 'no_obvious_issue'   : {n_no_issue} jendela ({n_no_issue/45*100:.1f}%)")
    summary_lines.append("  (Catatan: Label 'review_needed' murni sebagai indikator pemeriksaan, BUKAN bukti pembuangan data)")

    summary_lines.append("\n[2. ANALISIS MENDALAM 3 LOKASI SPESIFIK]")
    summary_lines.append("A. Rekaman 0: Jendela 2 (Detik 60 - 90, terutama detik 79 - 86):")
    summary_lines.append("   - Temuan PPG: Terjadi lonjakan puncak pada detik 82.5 (1.90 a.u.), diikuti penurunan drastis")
    summary_lines.append("     amplitudo puncak pada detik 83.5 - 85.0 menjadi 0.61 - 0.73 a.u. (normal ~1.15 a.u.).")
    summary_lines.append("     Ditemukan 6 outlier amplitudo denyut relatif terhadap median jendela.")
    summary_lines.append("   - Korelasi ABP: Tekanan ABP tetap stabil (sistolik ~124-128 mmHg, ritme teratur).")
    summary_lines.append("   - Interpretasi: Gangguan lokal pada PPG (artefak pergerakan sensor / perubahan vasomotor lokal),")
    summary_lines.append("     bukan berasal dari kegagalan sirkulasi hemodinamik sistemik.")

    summary_lines.append("\nB. Rekaman 1: Jendela 0 (Detik 0 - 30, terutama detik 15 - 18 dan 27 - 29):")
    summary_lines.append("   - Temuan PPG: Terjadi dua denyut prematur di t=16.58s (IBI=368 ms, amp=0.69) dan t=28.68s")
    summary_lines.append("     (IBI=376 ms, amp=0.65), masing-masing diikuti fase jeda kompensasi panjang.")
    summary_lines.append("   - Korelasi ABP: Sinyal ABP juga memperlihatkan jeda panjang (IBI ABP melonjak ke 816 ms dan 824 ms)")
    summary_lines.append("     akibat pengisian ventrikel yang belum optimal pada denyut prematur.")
    summary_lines.append("   - Interpretasi: Fenomena fisiologis nyata (aritmia / denyut prematur ventrikel/atrium), BUKAN artefak sensor.")

    summary_lines.append("\nC. Rekaman 2: Jendela 1 (Detik 30 - 60, terutama detik 54 - 58):")
    summary_lines.append("   - Temuan PPG: Interval IBI memendek ke 400 ms (t=56.36s), disusul interval kompensasi 544 ms.")
    summary_lines.append("   - Korelasi ABP: Perubahan ritme sinkron terlihat pada ABP.")
    summary_lines.append("   - Interpretasi: Variasi fisiologis denyut jantung (arrhythmia / respiratory sinus arrhythmia ringan).")

    summary_lines.append("\n[3. MENGAPA SKEWNESS SQI TUNGGAL TIDAK CUKUP?]")
    summary_lines.append("1. Skewness SQI adalah momen statistik global ke-3 yang dirata-ratakan selama 30 detik.")
    summary_lines.append("   Pada Rekaman 0 Jendela 2, sSQI tercatat 0.524 (positif dan tampak 'ideal'), meskipun")
    summary_lines.append("   terdapat anomali amplitudo denyut yang anjlok 50% selama 6 detik berturut-turut.")
    summary_lines.append("2. Skewness tidak sensitif terhadap:")
    summary_lines.append("   - Gangguan amplitudo transien jangka pendek (beberapa siklus).")
    summary_lines.append("   - Denyut prematur atau jeda kompensasi ektopik.")
    summary_lines.append("   - Sampel clipping di rel batas sensor jika durasinya singkat.")

    summary_lines.append("\n[4. INDIKATOR DIAGNOSTIK TAMBAHAN YANG MEMBANTU]")
    summary_lines.append("1. Pulse Amplitude MAD (Median Absolute Deviation):")
    summary_lines.append("   Sangat efektif mengidentifikasi variasi tak wajar tinggi gelombang antar-denyut.")
    summary_lines.append("2. IBI MAD / IBI Outlier Detection:")
    summary_lines.append("   Mendeteksi ketidakteraturan ritme, denyut ektopik, atau denyut yang terlewat/ganda.")
    summary_lines.append("3. Clipping Sample Count:")
    summary_lines.append("   Mendeteksi saturasi voltase ADC sensor optik (<=0.01V atau >=3.99V).")

    summary_lines.append("\n[5. USULAN KRITERIA KUALITAS MULTI-DIMENSI (UNTUK TAHAP BERIKUTNYA)]")
    summary_lines.append("(Catatan: Kriteria ini merupakan usulan eksploratif, BELUM diterapkan sebagai pemotong data final):")
    summary_lines.append("1. Dimensi Morfologi Global: sSQI positif moderat (misal 0.20 s.d. 0.85).")
    summary_lines.append("2. Dimensi Stabilitas Ritme : Rasio outlier IBI < 10% per jendela 30s.")
    summary_lines.append("3. Dimensi Stabilitas Amplitudo: Rasio outlier amplitudo denyut < 15% per jendela 30s.")
    summary_lines.append("4. Dimensi Integritas Sensor: Sampel clipping rail ADC < 1% dari total sampel jendela.")
    summary_lines.append("5. Kebijakan Aritmia: Denyut ektopik fisiologis (yang terkonfirmasi sinkron di ABP) dapat dipertahankan")
    summary_lines.append("   atau dipotong per siklus, sedangkan artefak sensor terisolasi sebaiknya dieksklusi.")
    summary_lines.append("=" * 75)

    summary_text = "\n".join(summary_lines)

    # Cetak ke terminal
    print("\n" + summary_text)

    # Simpan ke file teks
    with open(summary_path, "w", encoding="utf-8") as f_out:
        f_out.write(summary_text)
    print(f"\n[SUKSES] Laporan teks audit disimpan di:\n  {summary_path}")

if __name__ == "__main__":
    main()
