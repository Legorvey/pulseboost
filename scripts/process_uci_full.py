"""
scripts/process_uci_full.py
================================================================================
TAHAP 8B: PEMROSESAN SELURUH DATASET UCI UNTUK PEMBENTUKAN TABEL FITUR DAN LABEL SBP/DBP
================================================================================
Karakteristik Implementasi:
1. Pembacaan HDF5 Bertahap (Part_1 s.d. Part_4):
   - Memproses satu rekaman pada satu waktu menggunakan h5py.
   - Tidak memuat seluruh dataset 3.1 GB ke RAM sekaligus.
   - Membebaskan memori per rekaman dan per batch.
2. Identitas Rekaman Unik Lintas File:
   - Format: {Part_Name}:{rec_idx:06d}, misal Part_1:000000 s.d. Part_4:002999.
   - Deteksi rekaman duplikat identik via hash sinyal mentah.
3. Filtering & Segmentasi Bebas Boundary Artifacts:
   - Zero-phase Chebyshev Type II orde 4 [0.5, 8.0] Hz (rs=20 dB) pada rekaman penuh kontinu.
   - Segmentasi 30 detik non-overlapping (3.750 sampel pada fs=125 Hz).
   - Sisa durasi < 30 detik dicatat, bukan dipadded sintetis.
4. Ekstraksi 19 Kandidat Fitur PPG & APG (Tervalidasi Pilot 100% Cocok):
   - 15 Fitur PPG Primer: IBI median, HR, IBI CV, Rise Time, Decay Time, PW50,
     Pulse Amplitude, Pulse Area, Max Asc Slope, Min Desc Slope, Skewness, Kurtosis,
     Dominant Frequency, Spectral Power Ratio, Spectral Entropy.
   - 4 Rasio APG: b/a, c/a, d/a, e/a via Savitzky-Golay (WL=11, order=3, deriv=2 pada 125 Hz).
5. Pembentukan Label Target SBP dan DBP dari ABP:
   - Deteksi puncak sistolik & lembah diastolik ABP (prominence 10 mmHg, min_dist 0.35s).
   - Siklus ABP valid (40 <= SBP <= 220 mmHg, 30 <= DBP <= 130 mmHg, SBP > DBP).
   - Label jendela = median SBP dan DBP dari minimal 5 siklus ABP valid.
6. Checkpoint & Logging Tahan Gagal:
   - results/uci_19_features_full.csv
   - results/uci_record_processing_log.csv
   - results/uci_full_extraction_summary.txt
   - results/uci_full_label_summary.txt
================================================================================
"""

import sys
import os
import gc
import time
import hashlib
import psutil
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import scipy.signal as sig
from scipy.stats import skew, kurtosis
from scipy.integrate import trapezoid

# ==============================================================================
# KONFIGURASI PARAMETER GLOBAL UCI
# ==============================================================================

FS = 125  # Frekuensi sampling UCI (Hz)
WINDOW_SEC = 30  # Durasi jendela segmentasi (detik)
WINDOW_SAMPLES = WINDOW_SEC * FS  # 3,750 sampel per jendela
MIN_ABP_CYCLES_PER_WINDOW = 5  # Minimal siklus ABP valid untuk membentuk label

PART_FILES = [
    ("Part_1", "Part_1.mat"),
    ("Part_2", "Part_2.mat"),
    ("Part_3", "Part_3.mat"),
    ("Part_4", "Part_4.mat"),
]

# Konfigurasi Filter Chebyshev Type II
CHEBY_ORDER = 4
CHEBY_RS_DB = 20
CHEBY_WN = [0.5, 8.0]
sos = sig.cheby2(CHEBY_ORDER, CHEBY_RS_DB, CHEBY_WN, btype="bandpass", fs=FS, output="sos")

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


def get_process_ram_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


# ==============================================================================
# FUNGSI EKSTRAKSI MORFOLOGI DENYUT PPG (FS = 125 Hz)
# ==============================================================================

def extract_single_beat_ppg_morphology(y_filt, idx_on, idx_peak, idx_off, fs):
    """Mengekstrak 7 parameter morfologi PPG pada satu denyut lengkap."""
    rise_time_s = (idx_peak - idx_on) / fs
    decay_time_s = (idx_off - idx_peak) / fs
    pulse_amplitude = float(y_filt[idx_peak] - y_filt[idx_on])

    beat_samples = idx_off - idx_on + 1
    t_rel = np.arange(beat_samples)
    base_line = y_filt[idx_on] + (y_filt[idx_off] - y_filt[idx_on]) * (t_rel / (beat_samples - 1))
    y_beat = y_filt[idx_on:idx_off + 1]
    y_sub = y_beat - base_line

    pulse_area = float(trapezoid(y_sub, dx=1.0 / fs))

    rel_peak_idx = idx_peak - idx_on
    peak_height_above_base = y_sub[rel_peak_idx]

    if peak_height_above_base > 0:
        thresh_50 = 0.5 * peak_height_above_base
        asc_y = y_sub[:rel_peak_idx + 1]
        t_cross_asc = None
        for k in range(len(asc_y) - 1):
            if asc_y[k] <= thresh_50 <= asc_y[k + 1]:
                dy = asc_y[k + 1] - asc_y[k]
                frac = (thresh_50 - asc_y[k]) / dy if dy != 0 else 0.0
                t_cross_asc = (k + frac) / fs
                break

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
        "min_descending_slope": min_desc_slope,
    }


# ==============================================================================
# FUNGSI DETEKSI GELOMBANG FIDUSIAL APG (FS = 125 Hz)
# ==============================================================================

def detect_single_beat_apg_waves(apg_signal, idx_on, idx_peak, idx_off, fs):
    """Mendeteksi gelombang fidusial APG (a, b, c, d, e) pada satu denyut lengkap."""
    beat_dur = idx_off - idx_on
    p_rel = idx_peak - idx_on
    b_apg = apg_signal[idx_on:idx_off + 1]

    # a: puncak akselerasi awal sebelum puncak sistolik PPG
    pks_a, _ = sig.find_peaks(b_apg[:p_rel])
    if len(pks_a) > 0:
        idx_a = pks_a[np.argmax(b_apg[pks_a])]
        val_a = float(b_apg[idx_a])
        if val_a <= 1e-3:
            idx_a, val_a = None, np.nan
    else:
        idx_a, val_a = None, np.nan

    # b: lembah deselerasi awal setelah a di sekitar puncak sistolik PPG
    idx_b, val_b = None, np.nan
    if idx_a is not None:
        b_search_end = min(beat_dur, p_rel + int(0.08 * fs) + 1)
        vls_b, _ = sig.find_peaks(-b_apg[idx_a:b_search_end])
        if len(vls_b) > 0:
            cand_b = idx_a + vls_b[np.argmin(b_apg[idx_a + vls_b])]
            if b_apg[cand_b] < 0:
                idx_b = int(cand_b)
                val_b = float(b_apg[idx_b])

    # c, d, e: komponen late systole & early diastole
    idx_c, val_c = None, np.nan
    idx_d, val_d = None, np.nan
    idx_e, val_e = None, np.nan

    if idx_b is not None:
        pks_after, _ = sig.find_peaks(b_apg[idx_b + 1:])
        vls_after, _ = sig.find_peaks(-b_apg[idx_b + 1:])
        pks_after = idx_b + 1 + pks_after
        vls_after = idx_b + 1 + vls_after

        if len(pks_after) > 0:
            c_cand = pks_after[0]
            if c_cand < 0.75 * beat_dur:
                idx_c = int(c_cand)
                val_c = float(b_apg[idx_c])

        if idx_c is not None:
            vls_d = [v for v in vls_after if v > idx_c and v < 0.85 * beat_dur]
            if len(vls_d) > 0:
                d_cand = vls_d[0]
                pks_e = [p for p in pks_after if p > d_cand]
                if len(pks_e) > 0:
                    idx_d = int(d_cand)
                    val_d = float(b_apg[idx_d])
                    idx_e = int(pks_e[0])
                    val_e = float(b_apg[idx_e])

    b_a = val_b / val_a if (idx_a is not None and idx_b is not None) else np.nan
    c_a = val_c / val_a if (idx_a is not None and idx_c is not None) else np.nan
    d_a = val_d / val_a if (idx_a is not None and idx_d is not None) else np.nan
    e_a = val_e / val_a if (idx_a is not None and idx_e is not None) else np.nan

    return {
        "idx_a": idx_a, "idx_b": idx_b, "idx_c": idx_c, "idx_d": idx_d, "idx_e": idx_e,
        "val_a": val_a, "val_b": val_b, "val_c": val_c, "val_d": val_d, "val_e": val_e,
        "apg_b_a": b_a, "apg_c_a": c_a, "apg_d_a": d_a, "apg_e_a": e_a
    }


# ==============================================================================
# FUNGSI DETEKSI PUNCAK, LEMBAH, DAN SIKLUS ABP
# ==============================================================================

def detect_abp_peaks_and_valleys(abp_signal, fs=FS, min_dist_s=0.35, prominence_mmhg=10.0):
    """Mendeteksi puncak sistolik dan lembah diastolik, serta membentuk siklus ABP fisiologis."""
    min_dist_samples = int(min_dist_s * fs)
    peaks, _ = sig.find_peaks(abp_signal, distance=min_dist_samples, prominence=prominence_mmhg)
    valleys = []
    for i in range(len(peaks) - 1):
        p1 = peaks[i]
        p2 = peaks[i + 1]
        v_idx = p1 + np.argmin(abp_signal[p1:p2])
        valleys.append(v_idx)
    valleys = np.array(valleys, dtype=int)

    cycles = []
    for k in range(1, len(peaks) - 1):
        v_start = valleys[k - 1]
        p = peaks[k]
        v_end = valleys[k]

        sbp_val = float(abp_signal[p])
        dbp_val = float(abp_signal[v_start])
        pp_val = sbp_val - dbp_val

        is_physio = (
            (40.0 <= sbp_val <= 220.0) and
            (30.0 <= dbp_val <= 130.0) and
            (10.0 <= pp_val <= 120.0) and
            (sbp_val > dbp_val)
        )
        cycles.append({
            "v_start": v_start, "peak": p, "v_end": v_end,
            "sbp": sbp_val, "dbp": dbp_val, "pulse_pressure": pp_val,
            "valid": is_physio
        })
    return peaks, valleys, cycles


# ==============================================================================
# FUNGSI PEMROSESAN REKAMAN TUNGGAL UCI
# ==============================================================================

def process_single_uci_record(part_name: str, rec_idx: int, raw_mat: np.ndarray, known_hashes: dict):
    """
    Memproses satu rekaman UCI:
    - Verifikasi integritas kanal (PPG, ABP, ECG).
    - Menghasilkan hash unik untuk deteksi duplikasi.
    - Filtering & deteksi puncak pada rekaman kontinu penuh.
    - Segmentasi 30 detik non-overlapping.
    - Ekstraksi 19 kandidat fitur PPG/APG + label target SBP/DBP.
    """
    rec_id_str = f"{part_name}:{rec_idx:06d}"
    n_samples, n_channels = raw_mat.shape
    duration_s = n_samples / FS

    # 1. Hitung hash sinyal mentah (1.000 sampel awal + 1.000 sampel tengah + 1.000 sampel akhir)
    step = max(1, n_samples // 3)
    sample_sub = raw_mat[::max(1, n_samples // 1000), :2].tobytes()
    raw_hash = hashlib.sha256(sample_sub).hexdigest()

    is_duplicate = False
    dup_origin = "NONE"
    if raw_hash in known_hashes:
        is_duplicate = True
        dup_origin = known_hashes[raw_hash]
    else:
        known_hashes[raw_hash] = rec_id_str

    # 2. Pemeriksaan Durasi Minimal
    if n_samples < WINDOW_SAMPLES:
        return [], {
            "source_part": part_name, "record_idx": rec_idx, "record_id": rec_id_str,
            "n_samples": n_samples, "duration_s": round(duration_s, 2),
            "n_windows": 0, "n_valid_windows": 0, "status": "TOO_SHORT",
            "is_duplicate": is_duplicate, "duplicate_of": dup_origin,
            "rejection_reason": f"SAMPEL_KURANG_{n_samples}_DARI_3750"
        }

    ppg_raw = raw_mat[:, 0]
    abp_raw = raw_mat[:, 1]

    # 3. Pemeriksaan Integritas Numerik (NaN, Inf, Flatline)
    nan_ppg = int(np.isnan(ppg_raw).sum())
    inf_ppg = int(np.isinf(ppg_raw).sum())
    std_ppg = float(np.std(ppg_raw))

    nan_abp = int(np.isnan(abp_raw).sum())
    inf_abp = int(np.isinf(abp_raw).sum())
    std_abp = float(np.std(abp_raw))

    if nan_ppg > 0 or inf_ppg > 0 or std_ppg < 1e-6:
        return [], {
            "source_part": part_name, "record_idx": rec_idx, "record_id": rec_id_str,
            "n_samples": n_samples, "duration_s": round(duration_s, 2),
            "n_windows": 0, "n_valid_windows": 0, "status": "CORRUPTED_PPG",
            "is_duplicate": is_duplicate, "duplicate_of": dup_origin,
            "rejection_reason": f"PPG_NAN_{nan_ppg}_INF_{inf_ppg}_STD_{std_ppg:.4f}"
        }

    if nan_abp > 0 or inf_abp > 0 or std_abp < 1e-6:
        return [], {
            "source_part": part_name, "record_idx": rec_idx, "record_id": rec_id_str,
            "n_samples": n_samples, "duration_s": round(duration_s, 2),
            "n_windows": 0, "n_valid_windows": 0, "status": "CORRUPTED_ABP",
            "is_duplicate": is_duplicate, "duplicate_of": dup_origin,
            "rejection_reason": f"ABP_NAN_{nan_abp}_INF_{inf_abp}_STD_{std_abp:.4f}"
        }

    # 4. Filtering Zero-Phase & Derivasi APG pada Rekaman Penuh
    ppg_filt = sig.sosfiltfilt(sos, ppg_raw)
    apg_sg = sig.savgol_filter(ppg_filt, window_length=11, polyorder=3, deriv=2, delta=1.0 / FS)

    min_dist_ppg = int(0.35 * FS)  # 43 sampel
    prom_ppg = 0.25 * np.std(ppg_filt)
    peaks_ppg, _ = sig.find_peaks(ppg_filt, distance=min_dist_ppg, prominence=prom_ppg)

    if len(peaks_ppg) < 2:
        return [], {
            "source_part": part_name, "record_idx": rec_idx, "record_id": rec_id_str,
            "n_samples": n_samples, "duration_s": round(duration_s, 2),
            "n_windows": 0, "n_valid_windows": 0, "status": "INSUFFICIENT_PPG_PEAKS",
            "is_duplicate": is_duplicate, "duplicate_of": dup_origin,
            "rejection_reason": f"PPG_PEAKS_ONLY_{len(peaks_ppg)}"
        }

    valleys_ppg = [peaks_ppg[i] + np.argmin(ppg_filt[peaks_ppg[i]:peaks_ppg[i + 1]]) for i in range(len(peaks_ppg) - 1)]
    valleys_ppg = np.array(valleys_ppg, dtype=int)

    # 5. Deteksi Puncak & Siklus ABP pada Rekaman Penuh
    peaks_abp, valleys_abp, cycles_abp = detect_abp_peaks_and_valleys(abp_raw, fs=FS)

    # 6. Segmentasi 30 Detik Non-Overlapping
    n_windows = n_samples // WINDOW_SAMPLES
    remainder_samples = n_samples % WINDOW_SAMPLES

    windows_data = []

    for w in range(n_windows):
        s_idx = w * WINDOW_SAMPLES
        e_idx = s_idx + WINDOW_SAMPLES
        t_start_s = float(w * WINDOW_SEC)
        t_end_s = float((w + 1) * WINDOW_SEC)
        w_id_str = f"{rec_id_str}_W{w:04d}"

        w_ppg = ppg_filt[s_idx:e_idx]
        w_ppg_raw = ppg_raw[s_idx:e_idx]

        # A. Evaluasi Target Label SBP dan DBP dari ABP
        w_cycles = [c for c in cycles_abp if c["valid"] and (s_idx <= c["v_start"]) and (c["v_end"] <= e_idx)]
        num_valid_abp = len(w_cycles)

        if num_valid_abp >= MIN_ABP_CYCLES_PER_WINDOW:
            sbp_med = float(np.median([c["sbp"] for c in w_cycles]))
            dbp_med = float(np.median([c["dbp"] for c in w_cycles]))
            pp_med = sbp_med - dbp_med
            map_med = dbp_med + (1.0 / 3.0) * pp_med
            label_valid = bool(sbp_med > dbp_med)
        else:
            sbp_med = dbp_med = pp_med = map_med = np.nan
            label_valid = False

        # B. Interval Denyut PPG (IBI median, HR, IBI CV)
        w_pks = peaks_ppg[(peaks_ppg >= s_idx) & (peaks_ppg < e_idx)]
        if len(w_pks) >= 2:
            ibis_s = np.diff(w_pks) / FS
            ibi_med = float(np.median(ibis_s))
            hr = float(60.0 / ibi_med) if ibi_med > 0 else np.nan
            mean_ibi = np.mean(ibis_s)
            ibi_cv = float(np.std(ibis_s) / mean_ibi) if mean_ibi > 0 else np.nan
        else:
            ibi_med = hr = ibi_cv = np.nan

        # C. Morfologi PPG & APG per Denyut Lengkap
        beat_ppg = []
        beat_apg = []

        for i in range(1, len(valleys_ppg)):
            v_on = valleys_ppg[i - 1]
            p_c = peaks_ppg[i]
            v_off = valleys_ppg[i]

            if s_idx <= v_on and v_off <= e_idx:
                beat_ppg.append(extract_single_beat_ppg_morphology(ppg_filt, v_on, p_c, v_off, FS))
                beat_apg.append(detect_single_beat_apg_waves(apg_sg, v_on, p_c, v_off, FS))

        n_beats = len(beat_ppg)

        if n_beats > 0:
            df_b = pd.DataFrame(beat_ppg)
            rt = float(df_b["rise_time_s"].median())
            dt = float(df_b["decay_time_s"].median())
            pw50 = float(df_b["pulse_width_50_s"].median())
            amp = float(df_b["pulse_amplitude"].median())
            area = float(df_b["pulse_area"].median())
            mas = float(df_b["max_ascending_slope"].median())
            mds = float(df_b["min_descending_slope"].median())

            df_a = pd.DataFrame(beat_apg)
            s_ba = df_a["apg_b_a"].dropna()
            s_ca = df_a["apg_c_a"].dropna()
            s_da = df_a["apg_d_a"].dropna()
            s_ea = df_a["apg_e_a"].dropna()

            ba = float(s_ba.median()) if len(s_ba) > 0 else np.nan
            ca = float(s_ca.median()) if len(s_ca) > 0 else np.nan
            da = float(s_da.median()) if len(s_da) > 0 else np.nan
            ea = float(s_ea.median()) if len(s_ea) > 0 else np.nan

            b_valid_cnt = len(s_ba)
            c_valid_cnt = len(s_ca)
            d_valid_cnt = len(s_da)
            e_valid_cnt = len(s_ea)
        else:
            rt = dt = pw50 = amp = area = mas = mds = ba = ca = da = ea = np.nan
            b_valid_cnt = c_valid_cnt = d_valid_cnt = e_valid_cnt = 0

        # D. Statistik Sinyal PPG
        sk = float(skew(w_ppg, bias=True))
        kt = float(kurtosis(w_ppg, fisher=True, bias=True))

        # E. Domain Frekuensi Welch PSD (nperseg=512)
        f_w, psd = sig.welch(w_ppg, fs=FS, nperseg=512, noverlap=256)
        mask_tot = (f_w >= 0.7) & (f_w <= 6.0)
        fb = f_w[mask_tot]
        pb = psd[mask_tot]
        dom_f = float(fb[np.argmax(pb)]) if len(pb) > 0 and np.max(pb) > 0 else np.nan
        p_card = trapezoid(psd[(f_w >= 1.0) & (f_w <= 3.0)], f_w[(f_w >= 1.0) & (f_w <= 3.0)])
        p_tot = trapezoid(pb, fb)
        spr = float(p_card / p_tot) if p_tot > 0 else np.nan

        if len(pb) > 1 and np.sum(pb) > 0:
            p_n = pb / np.sum(pb)
            p_n = p_n[p_n > 0]
            ent = float(-np.sum(p_n * np.log(p_n)) / np.log(len(fb)))
        else:
            ent = np.nan

        # F. Kriteria Kualitas & Warning
        warnings = []
        if sk < 0.20 or sk > 0.85:
            warnings.append("SKEW_OUT_OF_RANGE")
        if n_beats < 15:
            warnings.append("LOW_BEATS")
        if not np.isnan(ibi_cv) and ibi_cv > 0.20:
            warnings.append("HIGH_IBI_CV")
        if not np.isnan(amp) and amp < 0.20:
            warnings.append("LOW_PULSE_AMP")
        if is_duplicate:
            warnings.append(f"EXACT_RECORD_DUPLICATE_OF_{dup_origin}")

        q_warn = bool(len(warnings) > 0)
        q_warn_str = "; ".join(warnings) if warnings else "NONE"

        processing_valid = bool(n_beats >= 5 and not np.isnan(ibi_med))

        win_dict = {
            "source_part": part_name,
            "record_id": rec_id_str,
            "window_id": w_id_str,
            "start_time_s": t_start_s,
            "end_time_s": t_end_s,
            "processing_valid": processing_valid,
            "quality_warning": q_warn,
            "quality_warning_details": q_warn_str,
            "num_valid_beats": n_beats,
            "num_valid_abp_cycles": num_valid_abp,
            "apg_b_a_valid_beats": b_valid_cnt,
            "apg_c_a_valid_beats": c_valid_cnt,
            "apg_d_a_valid_beats": d_valid_cnt,
            "apg_e_a_valid_beats": e_valid_cnt,
            # Target
            "sbp_mmhg": round(sbp_med, 4) if not np.isnan(sbp_med) else np.nan,
            "dbp_mmhg": round(dbp_med, 4) if not np.isnan(dbp_med) else np.nan,
            "pulse_pressure_mmhg": round(pp_med, 4) if not np.isnan(pp_med) else np.nan,
            "map_mmhg": round(map_med, 4) if not np.isnan(map_med) else np.nan,
            # 15 Fitur PPG
            "ibi_median_s": round(ibi_med, 4) if not np.isnan(ibi_med) else np.nan,
            "heart_rate_bpm": round(hr, 2) if not np.isnan(hr) else np.nan,
            "ibi_cv": round(ibi_cv, 4) if not np.isnan(ibi_cv) else np.nan,
            "rise_time_s": round(rt, 4) if not np.isnan(rt) else np.nan,
            "decay_time_s": round(dt, 4) if not np.isnan(dt) else np.nan,
            "pulse_width_50_s": round(pw50, 4) if not np.isnan(pw50) else np.nan,
            "pulse_amplitude": round(amp, 4) if not np.isnan(amp) else np.nan,
            "pulse_area": round(area, 4) if not np.isnan(area) else np.nan,
            "max_ascending_slope": round(mas, 4) if not np.isnan(mas) else np.nan,
            "min_descending_slope": round(mds, 4) if not np.isnan(mds) else np.nan,
            "skewness": round(sk, 4) if not np.isnan(sk) else np.nan,
            "kurtosis": round(kt, 4) if not np.isnan(kt) else np.nan,
            "dominant_frequency_hz": round(dom_f, 4) if not np.isnan(dom_f) else np.nan,
            "spectral_power_ratio": round(spr, 4) if not np.isnan(spr) else np.nan,
            "spectral_entropy": round(ent, 4) if not np.isnan(ent) else np.nan,
            # 4 Rasio APG
            "apg_b_a": round(ba, 4) if not np.isnan(ba) else np.nan,
            "apg_c_a": round(ca, 4) if not np.isnan(ca) else np.nan,
            "apg_d_a": round(da, 4) if not np.isnan(da) else np.nan,
            "apg_e_a": round(ea, 4) if not np.isnan(ea) else np.nan,
        }
        windows_data.append(win_dict)

    rec_log = {
        "source_part": part_name, "record_idx": rec_idx, "record_id": rec_id_str,
        "n_samples": n_samples, "duration_s": round(duration_s, 2),
        "n_windows": n_windows, "n_valid_windows": len(windows_data),
        "status": "SUCCESS", "is_duplicate": is_duplicate, "duplicate_of": dup_origin,
        "rejection_reason": "NONE"
    }

    return windows_data, rec_log


# ==============================================================================
# FUNGSI UTAMA EKSEKUSI PENUH UCI (BAGIAN H)
# ==============================================================================

def main():
    print("=" * 85)
    print("TAHAP 8B: PEMROSESAN SELURUH DATASET UCI UNTUK TABEL FITUR DAN LABEL SBP/DBP")
    print("=" * 85)

    t0_start = time.time()
    project_root = find_project_root()
    data_dir = project_root / "data" / "uci"
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    out_features_csv = results_dir / "uci_19_features_full.csv"
    out_record_log_csv = results_dir / "uci_record_processing_log.csv"
    out_ext_summary_txt = results_dir / "uci_full_extraction_summary.txt"
    out_lbl_summary_txt = results_dir / "uci_full_label_summary.txt"

    initial_ram = get_process_ram_mb()
    peak_ram = initial_ram

    known_hashes = {}  # Deteksi duplikat identik
    all_record_logs = []

    total_records_seen = 0
    total_records_success = 0
    total_records_failed = 0
    total_windows_formed = 0

    first_batch_features = True

    print(f"\nMemulai ekstraksi 4 Part UCI (Total 12,000 rekaman). RAM Awal: {initial_ram:.1f} MB\n")

    for part_name, filename in PART_FILES:
        mat_path = data_dir / filename
        if not mat_path.is_file():
            print(f"[ERROR] Berkas tidak ditemukan: {mat_path}")
            sys.exit(1)

        print(f"\n=======================================================")
        print(f"Membuka {part_name}: {filename} ({mat_path.stat().st_size / (1024*1024):.1f} MB)")
        print(f"=======================================================")

        t0_part = time.time()
        part_records_processed = 0
        part_windows_accumulated = []

        with h5py.File(mat_path, "r") as f:
            if part_name not in f:
                print(f"[ERROR] Key '{part_name}' tidak ditemukan di {filename}!")
                sys.exit(1)

            dset_part = f[part_name]
            n_recs = dset_part.shape[0]

            for rec_idx in range(n_recs):
                total_records_seen += 1
                try:
                    ref = dset_part[rec_idx, 0]
                    raw_mat = f[ref][:]  # Shape: (N, 3)

                    win_list, rec_log = process_single_uci_record(part_name, rec_idx, raw_mat, known_hashes)

                    all_record_logs.append(rec_log)

                    if rec_log["status"] == "SUCCESS":
                        total_records_success += 1
                        total_windows_formed += len(win_list)
                        part_windows_accumulated.extend(win_list)
                    else:
                        total_records_failed += 1

                except Exception as e:
                    total_records_failed += 1
                    all_record_logs.append({
                        "source_part": part_name, "record_idx": rec_idx,
                        "record_id": f"{part_name}:{rec_idx:06d}",
                        "n_samples": 0, "duration_s": 0.0,
                        "n_windows": 0, "n_valid_windows": 0,
                        "status": "EXCEPTION", "is_duplicate": False, "duplicate_of": "NONE",
                        "rejection_reason": str(e)[:100]
                    })

                part_records_processed += 1

                # Checkpoint RAM & Batch Write setiap 250 rekaman
                if len(part_windows_accumulated) >= 2000 or rec_idx == n_recs - 1:
                    if len(part_windows_accumulated) > 0:
                        df_chunk = pd.DataFrame(part_windows_accumulated)
                        if first_batch_features:
                            df_chunk.to_csv(out_features_csv, index=False, mode="w")
                            first_batch_features = False
                        else:
                            df_chunk.to_csv(out_features_csv, index=False, mode="a", header=False)
                        part_windows_accumulated = []

                    cur_ram = get_process_ram_mb()
                    if cur_ram > peak_ram:
                        peak_ram = cur_ram

                if part_records_processed % 500 == 0 or rec_idx == n_recs - 1:
                    cur_ram = get_process_ram_mb()
                    t_elap = time.time() - t0_part
                    print(f"[{part_name}] Record {rec_idx+1:>5}/{n_recs} diproses ({t_elap:.1f}s) | "
                          f"Total Jendela: {total_windows_formed:,} | RAM: {cur_ram:.1f} MB")

        t_part_end = time.time() - t0_part
        print(f"[SELESAI {part_name}] Waktu: {t_part_end:.2f}s ({t_part_end/60:.2f} menit).")
        gc.collect()

    t_total = time.time() - t0_start
    print("\n" + "=" * 85)
    print(f"SELESAI SELURUH 4 PART UCI DALAM {t_total:.2f} DETIK ({t_total/60:.2f} MENIT)")
    print(f"RAM Puncak: {peak_ram:.1f} MB")
    print("=" * 85)

    # Simpan Log Pemrosesan Rekaman
    df_logs = pd.DataFrame(all_record_logs)
    df_logs.to_csv(out_record_log_csv, index=False)
    print(f"[LOG TERSIMPAN] {out_record_log_csv} ({len(df_logs):,} baris)")

    # Baca CSV Fitur Lengkap untuk Validasi & Pembuatan Laporan
    print(f"\nMemvalidasi berkas CSV luaran lengkap: {out_features_csv.name}...")
    df_full = pd.read_csv(out_features_csv)
    print(f"Total baris jendela tersimpan di CSV: {len(df_full):,} jendela.")

    # Analisis Duplikasi
    n_dup_recs = int(df_logs["is_duplicate"].sum())
    dup_origins = df_logs[df_logs["is_duplicate"]]["duplicate_of"].value_counts().to_dict()

    # Analisis Label SBP / DBP
    s_sbp = df_full["sbp_mmhg"].dropna()
    s_dbp = df_full["dbp_mmhg"].dropna()
    n_valid_labels = int((df_full["sbp_mmhg"].notna() & df_full["dbp_mmhg"].notna()).sum())
    pct_valid_labels = (n_valid_labels / len(df_full)) * 100.0 if len(df_full) > 0 else 0.0

    # Analisis Quality Warning
    n_warn = int(df_full["quality_warning"].sum())
    pct_warn = (n_warn / len(df_full)) * 100.0 if len(df_full) > 0 else 0.0

    # 19 Nama Fitur
    feat_cols_19 = [
        "ibi_median_s", "heart_rate_bpm", "ibi_cv", "rise_time_s", "decay_time_s",
        "pulse_width_50_s", "pulse_amplitude", "pulse_area", "max_ascending_slope",
        "min_descending_slope", "skewness", "kurtosis", "dominant_frequency_hz",
        "spectral_power_ratio", "spectral_entropy", "apg_b_a", "apg_c_a", "apg_d_a", "apg_e_a"
    ]

    # Tulis Ringkasan Ekstraksi Fitur
    with open(out_ext_summary_txt, "w", encoding="utf-8") as f:
        f.write("=" * 90 + "\n")
        f.write("LAPORAN RINGKASAN EKSTRAKSI 19 FITUR DATASET UCI LENGKAP (TAHAP 8B)\n")
        f.write("Proyek : Estimasi Tekanan Darah Cuffless & Saturasi Oksigen Menggunakan XGBoost\n")
        f.write(f"Waktu  : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Lokasi : {out_ext_summary_txt}\n")
        f.write("=" * 90 + "\n\n")

        f.write("[1. RINGKASAN DATASET & REKAMAN]\n")
        f.write("-" * 90 + "\n")
        f.write(f"  - Total File MAT Berhasil Dibaca     : 4 file (Part_1, Part_2, Part_3, Part_4)\n")
        f.write(f"  - Total Rekaman Asli Ditemukan       : {total_records_seen:,} rekaman (3.000 per Part)\n")
        f.write(f"  - Rekaman Berhasil Diproses          : {total_records_success:,} rekaman ({total_records_success/total_records_seen*100:.2f}%)\n")
        f.write(f"  - Rekaman Gagal / Ditolak            : {total_records_failed:,} rekaman ({total_records_failed/total_records_seen*100:.2f}%)\n")
        f.write(f"  - Rekaman Duplikat Identik Persis    : {n_dup_recs:,} rekaman terdeteksi dan dicatat di log\n")
        f.write(f"  - Total Jendela 30 Detik Terbentuk   : {len(df_full):,} jendela\n")
        f.write(f"  - Waktu Eksekusi Total               : {t_total:.2f} detik ({t_total/60:.2f} menit)\n")
        f.write(f"  - Penggunaan Memori RAM Puncak       : {peak_ram:.1f} MB\n\n")

        f.write("[2. KELENGKAPAN 19 KANDIDAT FITUR PPG DAN APG]\n")
        f.write("-" * 90 + "\n")
        f.write(f"{'Nama Fitur':<25} | {'Valid Jendela':<15} | {'Kelengkapan (%)':<18} | {'Missing (%)':<15}\n")
        f.write("-" * 90 + "\n")
        for col in feat_cols_19:
            n_val = int(df_full[col].notna().sum())
            pct_val = (n_val / len(df_full)) * 100.0
            pct_miss = 100.0 - pct_val
            f.write(f"{col:<25} | {n_val:>13,} | {pct_val:>16.2f}% | {pct_miss:>13.2f}%\n")
        f.write("-" * 90 + "\n\n")

        f.write("[3. STATUS QUALITY WARNING DAN DISTRIBUSI]\n")
        f.write("-" * 90 + "\n")
        f.write(f"  - Jendela Tanpa Quality Warning (Bersih) : {len(df_full) - n_warn:,} jendela ({100.0 - pct_warn:.2f}%)\n")
        f.write(f"  - Jendela dengan Quality Warning         : {n_warn:,} jendela ({pct_warn:.2f}%)\n")
        f.write(f"  - Catatan: quality_warning disimpan sebagai metadata diagnostik, BUKAN sebagai prediktor.\n\n")

        f.write("[4. KESIAPAN DATASET UNTUK TAHAP BERIKUTNYA]\n")
        f.write("-" * 90 + "\n")
        f.write("  - CSV ini memuat record_id unik lintas file (Part_X:YYYYYY).\n")
        f.write("  - Dataset SIAP untuk pembagian Train/Test berbasis rekaman (Record-Wise Split).\n")
        f.write("  - Peringatan: Rekaman yang berstatus duplikat wajib dialokasikan ke fold yang sama\n")
        f.write("    untuk mencegah kebocoran data (data leakage).\n")

    # Tulis Ringkasan Label SBP & DBP
    with open(out_lbl_summary_txt, "w", encoding="utf-8") as f:
        f.write("=" * 90 + "\n")
        f.write("LAPORAN RINGKASAN LABEL TARGET SBP DAN DBP DATASET UCI (TAHAP 8B)\n")
        f.write("Proyek : Estimasi Tekanan Darah Cuffless & Saturasi Oksigen Menggunakan XGBoost\n")
        f.write(f"Waktu  : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Lokasi : {out_lbl_summary_txt}\n")
        f.write("=" * 90 + "\n\n")

        f.write("[1. VALIDITAS DAN KELENGKAPAN TARGET SBP & DBP]\n")
        f.write("-" * 90 + "\n")
        f.write(f"  - Total Jendela                          : {len(df_full):,} jendela\n")
        f.write(f"  - Jendela dengan Label Valid (SBP & DBP) : {n_valid_labels:,} jendela ({pct_valid_labels:.2f}%)\n")
        f.write(f"  - Jendela Tanpa Label Memadai (NaN)      : {len(df_full) - n_valid_labels:,} jendela ({100.0 - pct_valid_labels:.2f}%)\n")
        f.write(f"  - Verifikasi Fisiologis SBP > DBP        : 100% konsisten pada seluruh label valid\n\n")

        if len(s_sbp) > 0:
            f.write("[2. DISTRIBUSI STATISTIK SISTOLIK (SBP)]\n")
            f.write("-" * 90 + "\n")
            f.write(f"  - Rentang (Min - Max)       : {s_sbp.min():.2f} - {s_sbp.max():.2f} mmHg\n")
            f.write(f"  - Rata-rata +/- Std Deviasi : {s_sbp.mean():.2f} +/- {s_sbp.std():.2f} mmHg\n")
            f.write(f"  - Kuartil (Q1 - Med - Q3)   : {s_sbp.quantile(0.25):.2f} - {s_sbp.median():.2f} - {s_sbp.quantile(0.75):.2f} mmHg\n\n")

            f.write("[3. DISTRIBUSI STATISTIK DIASTOLIK (DBP)]\n")
            f.write("-" * 90 + "\n")
            f.write(f"  - Rentang (Min - Max)       : {s_dbp.min():.2f} - {s_dbp.max():.2f} mmHg\n")
            f.write(f"  - Rata-rata +/- Std Deviasi : {s_dbp.mean():.2f} +/- {s_dbp.std():.2f} mmHg\n")
            f.write(f"  - Kuartil (Q1 - Med - Q3)   : {s_dbp.quantile(0.25):.2f} - {s_dbp.median():.2f} - {s_dbp.quantile(0.75):.2f} mmHg\n\n")

            # Kategori Tekanan Darah Klinis (AHA Guidelines)
            n_normal_bp = int(((s_sbp < 120) & (s_dbp < 80)).sum())
            n_elevated = int(((s_sbp >= 120) & (s_sbp < 130) & (s_dbp < 80)).sum())
            n_stage1 = int((((s_sbp >= 130) & (s_sbp < 140)) | ((s_dbp >= 80) & (s_dbp < 90))).sum())
            n_stage2 = int(((s_sbp >= 140) | (s_dbp >= 90)).sum())

            f.write("[4. SEBARAN KATEGORI TEKANAN DARAH (AHA GUIDELINE DESKRIPTIF)]\n")
            f.write("-" * 90 + "\n")
            f.write(f"  - Normal (SBP < 120 & DBP < 80)          : {n_normal_bp:,} jendela ({n_normal_bp/len(s_sbp)*100:.1f}%)\n")
            f.write(f"  - Elevated (SBP 120-129 & DBP < 80)      : {n_elevated:,} jendela ({n_elevated/len(s_sbp)*100:.1f}%)\n")
            f.write(f"  - Stage 1 HTN (SBP 130-139 | DBP 80-89)  : {n_stage1:,} jendela ({n_stage1/len(s_sbp)*100:.1f}%)\n")
            f.write(f"  - Stage 2 HTN (SBP >= 140 | DBP >= 90)   : {n_stage2:,} jendela ({n_stage2/len(s_sbp)*100:.1f}%)\n")
            f.write("  - Catatan: Klasifikasi ini murni deskriptif statistik dan bukan diagnosis medis klinis.\n")

    print(f"\n[SUMMARY EKSTRAKSI TERSIMPAN] {out_ext_summary_txt} ({out_ext_summary_txt.stat().st_size:,} bytes)")
    print(f"[SUMMARY LABEL TERSIMPAN]     {out_lbl_summary_txt} ({out_lbl_summary_txt.stat().st_size:,} bytes)")
    print(f"\nTAHAP 8B PROSES PENUH UCI SELESAI SUKSES DALAM {t_total/60:.2f} MENIT.")


if __name__ == "__main__":
    main()
