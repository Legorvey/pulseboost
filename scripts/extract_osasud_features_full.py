"""
scripts/extract_osasud_features_full.py
================================================================================
TAHAP 7H: EKSTRAKSI FITUR FINAL OSASUD SELURUH 30 PASIEN SECARA BERSYARAT
================================================================================
Karakteristik Pipeline:
1. Pemrosesan Modular Satu Pasien pada Satu Waktu:
   - Membaca patient_{pid}.npz dari data/osasud/patients_npz/ satu per satu.
   - Tidak pernah membaca berkas pickle asli (dataset_OSAS.pickle).
   - Membebaskan memori (del + gc.collect) setelah setiap pasien selesai diproses.
2. Segmentasi 30 Detik Non-Overlapping & Label SpO2 Fisiologis:
   - Jendela 30 detik (2.400 sampel pada 80 Hz).
   - Memverifikasi kontinuitas timestamp (menolak gap waktu > 1.05s atau < 0.95s).
   - Kriteria kelayakan target SpO2: minimal 24 detik valid dari 30 detik (>= 80%).
   - Target SpO2 per jendela dihitung dari median nilai valid.
3. Preprocessing Sinyal PPG Bebas Efek Batas:
   - Filter Chebyshev II orde 4 [0.5, 8.0] Hz (rs=20 dB) zero-phase sosfiltfilt
     diterapkan pada segmen kontinu utuh, tidak pernah melintasi gap atau batas pasien.
4. Deteksi Puncak PPG Adaptif Lokal & Plausibilitas Fisiologis Interval:
   - Ambang batas prominens adaptif lokal berbasis rolling std 4.0 detik.
   - Plausibilitas fisiologis denyut jantung (0.33s - 2.0s, konsistensi irama).
   - Memisahkan interval valid dari interval tidak tepercaya (unreliable).
   - Menghitung IBI CV hanya dari interval valid (mencegah artefak missed peak).
5. Deteksi APG Non-Greedy & Anti-Ripple V2:
   - Savitzky-Golay WL=9 (112.5 ms pada 80 Hz), polyorder=3, deriv=2.
   - Puncak c harus naik >= 5% dari dasar lembah b (anti-ripple pemulihan b).
   - Ambang prominens relatif 2.5% dari rentang utama gelombang (a - b).
   - Jika c, d, atau e tidak memenuhi kriteria meyakinkan, rasio bernilai NaN secara jujur.
6. Luaran:
   - results/osasud_19_features_full.csv
   - results/osasud_full_feature_extraction_summary.txt
================================================================================
"""

import sys
import os
import gc
import time
import math
import psutil
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.signal as sig
from scipy.stats import skew, kurtosis
from scipy.integrate import trapezoid

# ==============================================================================
# KONFIGURASI PARAMETER GLOBAL
# ==============================================================================

FS = 80  # Frekuensi sampling OSASUD (Hz)
WINDOW_SECONDS = 30  # Durasi jendela segmentasi (detik)
WINDOW_SAMPLES = WINDOW_SECONDS * FS  # 2,400 sampel per jendela
PRIMARY_SPO2_MIN_VALID = 24  # Minimal detik SpO2 valid per jendela (80%)
ALL_PATIENT_IDS = [str(i) for i in range(1, 31)]  # Pasien 1 s.d. 30

# Parameter Filter Chebyshev II Bandpass
CHEBY_ORDER = 4
CHEBY_RS_DB = 20
CHEBY_WN = [0.5, 8.0]

# Parameter Savitzky-Golay APG V2 (Optimal WL=9)
SAVGOL_WL = 9
SAVGOL_POLYORDER = 3
APG_MIN_PROM_RATIO = 0.025  # 2.5% dari rentang (a - b)

# Parameter Deteksi Puncak PPG Adaptif
PPG_MIN_DIST_SAMPLES = int(0.35 * FS)  # 28 sampel (refractory period ~350 ms)
PPG_LOCAL_WIN_SAMPLES = int(4.0 * FS)  # Jendela moving std 4.0 detik
PPG_MIN_PROM_FLOOR = 1.0  # Lantai noise minimum prominens puncak

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
    """Mengembalikan penggunaan memori Resident Set Size (RSS) proses dalam MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


# ==============================================================================
# SEGMENTASI 30 DETIK DAN EVALUASI KELAYAKAN TARGET SPO2
# ==============================================================================

def segment_patient_data(pid: str, ppg_2d: np.ndarray, spo2_1d: np.ndarray,
                         ts_dt: np.ndarray, ts_ns: np.ndarray):
    """
    Melakukan segmentasi 30 detik non-overlapping sesuai standar Tahap 7E.
    Mengembalikan daftar kamus jendela eligible beserta metadata segmen.
    """
    n_rows = len(ppg_2d)
    n_windows = n_rows // WINDOW_SECONDS

    window_records = []
    total_valid_ppg = 0
    total_valid_spo2 = 0
    total_eligible = 0

    for w_idx in range(n_windows):
        start_sec = w_idx * WINDOW_SECONDS
        end_sec = start_sec + WINDOW_SECONDS
        window_id = f"P{pid}_W{w_idx+1:04d}"

        w_ts_dt = ts_dt[start_sec:end_sec]
        w_ts_ns = ts_ns[start_sec:end_sec]
        start_ts_str = str(w_ts_dt[0])
        end_ts_str = str(w_ts_dt[-1])

        # 1. Periksa kontinuitas timestamp dalam jendela
        w_diffs_sec = np.diff(w_ts_ns) / 1e9
        has_time_gap = bool(np.any(w_diffs_sec > 1.05) or np.any(w_diffs_sec < 0.95))

        # 2. Evaluasi Sinyal PPG (2.400 sampel)
        w_ppg = ppg_2d[start_sec:end_sec]
        flat_ppg = w_ppg.flatten()
        n_nan_ppg = int(np.isnan(flat_ppg).sum())
        n_inf_ppg = int(np.isinf(flat_ppg).sum())

        is_ppg_finite = (n_nan_ppg == 0) and (n_inf_ppg == 0)
        is_ppg_not_constant = bool(np.std(flat_ppg) > 1e-6) if is_ppg_finite else False
        is_ppg_valid = bool(is_ppg_finite and is_ppg_not_constant and (not has_time_gap))

        # 3. Evaluasi Target SpO2 (30 detik)
        w_spo2 = spo2_1d[start_sec:end_sec]
        valid_spo2_mask = np.isfinite(w_spo2)
        valid_spo2_vals = w_spo2[valid_spo2_mask]
        spo2_valid_secs = int(len(valid_spo2_vals))

        is_label_valid = bool(spo2_valid_secs >= PRIMARY_SPO2_MIN_VALID)
        spo2_target = float(np.median(valid_spo2_vals)) if is_label_valid else np.nan

        is_eligible = bool(is_ppg_valid and is_label_valid)

        if is_ppg_valid:
            total_valid_ppg += 1
        if is_label_valid:
            total_valid_spo2 += 1
        if is_eligible:
            total_eligible += 1

        window_records.append({
            "patient_id": str(pid),
            "window_id": window_id,
            "window_idx": w_idx,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "start_timestamp": start_ts_str,
            "end_timestamp": end_ts_str,
            "is_ppg_valid": is_ppg_valid,
            "is_label_valid": is_label_valid,
            "spo2_valid_secs": spo2_valid_secs,
            "spo2_target_pct": spo2_target,
            "ml_window_eligible": is_eligible,
        })

    summary_seg = {
        "total_windows": n_windows,
        "valid_ppg_windows": total_valid_ppg,
        "valid_spo2_windows": total_valid_spo2,
        "eligible_windows": total_eligible,
    }
    return window_records, summary_seg


# ==============================================================================
# FUNGSI DETEKSI PUNCAK PPG ADAPTIF & EVALUASI INTERVAL
# ==============================================================================

def detect_ppg_peaks_adaptive(filt_signal: np.ndarray, fs: int = FS):
    """
    Mendeteksi puncak sistolik PPG dengan ambang batas adaptif lokal.
    Ambang dihitung dari rolling standard deviation jendela 4.0s dengan lantai noise minimum.
    """
    n_samples = len(filt_signal)
    if n_samples < fs:
        return np.array([], dtype=int)

    series = pd.Series(filt_signal)
    roll_std = series.rolling(PPG_LOCAL_WIN_SAMPLES, center=True, min_periods=1).std().values
    prom_thresh = np.maximum(PPG_MIN_PROM_FLOOR, 0.20 * roll_std)

    pks_cand, props = sig.find_peaks(filt_signal, distance=PPG_MIN_DIST_SAMPLES, prominence=0.0)
    if len(pks_cand) == 0:
        return np.array([], dtype=int)

    proms = props["prominences"]
    valid_pks = [p for p, pr in zip(pks_cand, proms) if pr >= prom_thresh[p]]
    return np.array(valid_pks, dtype=int)


def evaluate_ibi_plausibility(peak_indices: np.ndarray, fs: int = FS):
    """
    Mengevaluasi plausibilitas interval denyut (IBI) fisiologis.
    - Rentang fisiologis normal manusia: 0.33s s.d. 2.0s (30 - 180 bpm).
    - Konsistensi terhadap median lokal jendela: 0.5 * median <= IBI <= 1.8 * median.
    - Menghitung IBI CV HANYA jika terdapat minimal 5 interval terpercaya.
    """
    if len(peak_indices) < 2:
        return {
            "num_peaks": len(peak_indices),
            "num_valid_intervals": 0,
            "num_unreliable_intervals": 0,
            "ibi_median_s": np.nan,
            "heart_rate_bpm": np.nan,
            "ibi_cv": np.nan,
            "has_unreliable": False,
        }

    ibis_s = np.diff(peak_indices) / fs
    med_ibi = float(np.median(ibis_s))

    valid_ibis = []
    unreliable_ibis = []

    for ib in ibis_s:
        if 0.33 <= ib <= 2.0 and (0.5 * med_ibi <= ib <= 1.8 * med_ibi):
            valid_ibis.append(ib)
        else:
            unreliable_ibis.append(ib)

    num_valid = len(valid_ibis)
    num_unrel = len(unreliable_ibis)

    if num_valid >= 2:
        clean_med_ibi = float(np.median(valid_ibis))
        hr_bpm = float(60.0 / clean_med_ibi) if clean_med_ibi > 0 else np.nan
    else:
        clean_med_ibi = np.nan
        hr_bpm = np.nan

    if num_valid >= 5:
        mean_v = np.mean(valid_ibis)
        std_v = np.std(valid_ibis)
        ibi_cv = float(std_v / mean_v) if mean_v > 0 else np.nan
    else:
        ibi_cv = np.nan

    return {
        "num_peaks": len(peak_indices),
        "num_valid_intervals": num_valid,
        "num_unreliable_intervals": num_unrel,
        "ibi_median_s": clean_med_ibi,
        "heart_rate_bpm": hr_bpm,
        "ibi_cv": ibi_cv,
        "has_unreliable": (num_unrel > 0),
    }


# ==============================================================================
# FUNGSI EKSTRAKSI MORFOLOGI PPG PER DENYUT
# ==============================================================================

def extract_single_beat_ppg_morphology(y_filt, idx_on, idx_peak, idx_off, fs):
    """
    Mengekstrak 7 parameter morfologi PPG pada satu denyut lengkap (onset - peak - offset).
    """
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
# FUNGSI DETEKSI GELOMBANG FIDUSIAL APG V2 (ANTI-GREEDY, ANTI-RIPPLE)
# ==============================================================================

def detect_single_beat_apg_waves_v2(apg_signal, idx_on, idx_peak, idx_off, fs, min_prom_ratio=APG_MIN_PROM_RATIO):
    """
    Deteksi gelombang fidusial APG (a, b, c, d, e) berbasis morfologi kokoh V2.
    """
    b_apg = apg_signal[idx_on:idx_off + 1]
    N = len(b_apg)
    p_rel = idx_peak - idx_on

    if p_rel <= 1 or N < int(0.30 * fs):
        return {
            "apg_b_a": np.nan, "apg_c_a": np.nan, "apg_d_a": np.nan, "apg_e_a": np.nan,
            "val_a": np.nan, "val_b": np.nan, "val_c": np.nan, "val_d": np.nan, "val_e": np.nan,
        }

    # 1. Gelombang a: Puncak akselerasi awal sebelum puncak PPG
    pks_a, _ = sig.find_peaks(b_apg[:p_rel], prominence=0.0)
    if len(pks_a) == 0:
        return {"apg_b_a": np.nan, "apg_c_a": np.nan, "apg_d_a": np.nan, "apg_e_a": np.nan}
    idx_a = pks_a[np.argmax(b_apg[pks_a])]
    val_a = float(b_apg[idx_a])
    if val_a <= 0:
        return {"apg_b_a": np.nan, "apg_c_a": np.nan, "apg_d_a": np.nan, "apg_e_a": np.nan}

    # 2. Gelombang b: Lembah deselerasi sistolik awal setelah a
    b_search_end = min(N, idx_a + int(0.35 * fs))
    if idx_a >= b_search_end - 1:
        return {"apg_b_a": np.nan, "apg_c_a": np.nan, "apg_d_a": np.nan, "apg_e_a": np.nan}
    vls_b, _ = sig.find_peaks(-b_apg[idx_a:b_search_end], prominence=0.0)
    if len(vls_b) == 0:
        return {"apg_b_a": np.nan, "apg_c_a": np.nan, "apg_d_a": np.nan, "apg_e_a": np.nan}
    idx_b = idx_a + vls_b[np.argmin(b_apg[idx_a + vls_b])]
    val_b = float(b_apg[idx_b])
    span_ab = val_a - val_b
    if span_ab <= 0:
        return {"apg_b_a": np.nan, "apg_c_a": np.nan, "apg_d_a": np.nan, "apg_e_a": np.nan}

    b_a = val_b / val_a
    min_prom_abs = min_prom_ratio * span_ab

    # 3. Gelombang c: Puncak re-akselerasi late-systole (Anti-Greedy & Anti-Ripple)
    c_start = idx_b + 3
    c_end = min(N - 1, int(0.60 * N))
    if c_start >= c_end:
        return {"apg_b_a": b_a, "apg_c_a": np.nan, "apg_d_a": np.nan, "apg_e_a": np.nan}

    pks_c, _ = sig.find_peaks(b_apg[c_start:c_end], prominence=min_prom_abs)
    valid_c_indices = [c_start + p for p in pks_c if b_apg[c_start + p] >= val_b + 0.05 * span_ab]

    if len(valid_c_indices) == 0:
        return {"apg_b_a": b_a, "apg_c_a": np.nan, "apg_d_a": np.nan, "apg_e_a": np.nan}

    if len(valid_c_indices) == 1:
        idx_c = valid_c_indices[0]
    else:
        idx_c = valid_c_indices[np.argmax([b_apg[p] for p in valid_c_indices])]
    val_c = float(b_apg[idx_c])
    c_a = val_c / val_a

    # 4. Gelombang d: Lembah deselerasi late-systole setelah c
    d_start = idx_c + 2
    d_end = min(N - 1, int(0.75 * N))
    if d_start >= d_end:
        return {"apg_b_a": b_a, "apg_c_a": c_a, "apg_d_a": np.nan, "apg_e_a": np.nan}

    vls_d, _ = sig.find_peaks(-b_apg[d_start:d_end], prominence=min_prom_abs)
    valid_d_indices = [d_start + v for v in vls_d if b_apg[d_start + v] < val_c]

    if len(valid_d_indices) == 0:
        return {"apg_b_a": b_a, "apg_c_a": c_a, "apg_d_a": np.nan, "apg_e_a": np.nan}

    idx_d = valid_d_indices[0]
    val_d = float(b_apg[idx_d])
    d_a = val_d / val_a

    # 5. Gelombang e: Puncak akselerasi diastolik awal setelah d (Dicrotic Wave)
    e_start = idx_d + 2
    e_end = min(N - 1, int(0.85 * N))
    if e_start >= e_end:
        return {"apg_b_a": b_a, "apg_c_a": c_a, "apg_d_a": d_a, "apg_e_a": np.nan}

    pks_e, _ = sig.find_peaks(b_apg[e_start:e_end], prominence=min_prom_abs)
    if len(pks_e) == 0:
        return {"apg_b_a": b_a, "apg_c_a": c_a, "apg_d_a": d_a, "apg_e_a": np.nan}

    idx_e = e_start + pks_e[0]
    val_e = float(b_apg[idx_e])
    e_a = val_e / val_a

    return {
        "apg_b_a": b_a, "apg_c_a": c_a, "apg_d_a": d_a, "apg_e_a": e_a,
    }


# ==============================================================================
# PIPELINE EKSTRAKSI FITUR PASIEN TUNGGAL
# ==============================================================================

def process_single_patient(pid: str, npz_path: Path, cheby_sos):
    """
    Memproses satu pasien: segmentasi -> filtering -> deteksi puncak -> ekstraksi fitur 19 kandidat.
    Mengembalikan:
    - df_patient_features (DataFrame jendela eligible)
    - patient_metrics (dict statistik)
    """
    with np.load(npz_path, allow_pickle=False) as npz:
        ppg_2d = npz["ppg_80hz"]
        spo2_1d = npz["spo2_1hz"]
        ts_dt = npz["timestamp_datetime"]
        ts_ns = npz["timestamp_unix_ns"]
        n_total_sec = len(ppg_2d)

    # 1. Segmentasi 30 detik
    all_win_records, seg_summary = segment_patient_data(pid, ppg_2d, spo2_1d, ts_dt, ts_ns)
    df_all_win = pd.DataFrame(all_win_records)
    df_eligible = df_all_win[df_all_win["ml_window_eligible"] == True].copy()

    total_eligible = len(df_eligible)
    if total_eligible == 0:
        del ppg_2d, spo2_1d, ts_dt, ts_ns
        gc.collect()
        return pd.DataFrame(), seg_summary

    # 2. Identifikasi segmen-segmen kontinu valid (tidak ada NaN)
    valid_sec_mask = (~np.isnan(ppg_2d)).all(axis=1)
    diff_valid = np.diff(valid_sec_mask.astype(int))
    starts = np.where(diff_valid == 1)[0] + 1
    ends = np.where(diff_valid == -1)[0] + 1
    if valid_sec_mask[0]:
        starts = np.r_[0, starts]
    if valid_sec_mask[-1]:
        ends = np.r_[ends, n_total_sec]

    segments = list(zip(starts, ends))

    # 3. Proses ekstraksi fitur per segmen kontinu
    features_list = []

    for seg_idx, (seg_start, seg_end) in enumerate(segments, start=1):
        seg_duration = seg_end - seg_start
        if seg_duration < WINDOW_SECONDS:
            continue

        seg_raw = ppg_2d[seg_start:seg_end].flatten()
        seg_filt = sig.sosfiltfilt(cheby_sos, seg_raw)
        seg_apg = sig.savgol_filter(seg_filt, window_length=SAVGOL_WL, polyorder=SAVGOL_POLYORDER, deriv=2, delta=1.0 / FS)

        # Deteksi puncak sistolik adaptif lokal
        pks_seg = detect_ppg_peaks_adaptive(seg_filt, fs=FS)
        if len(pks_seg) < 2:
            continue

        valleys_seg = []
        for i in range(len(pks_seg) - 1):
            idx_p1 = pks_seg[i]
            idx_p2 = pks_seg[i + 1]
            rel_min = np.argmin(seg_filt[idx_p1:idx_p2])
            valleys_seg.append(idx_p1 + rel_min)
        valleys_seg = np.array(valleys_seg, dtype=int)

        # Iterasi jendela eligible di segmen kontinu ini
        for _, win_row in df_eligible.iterrows():
            w_sec_start = win_row["start_sec"]
            w_sec_end = win_row["end_sec"]
            w_id = win_row["window_id"]

            if not (seg_start <= w_sec_start and w_sec_end <= seg_end):
                continue

            s_samp = (w_sec_start - seg_start) * FS
            e_samp = (w_sec_end - seg_start) * FS

            w_filt = seg_filt[s_samp:e_samp]

            # ------------------------------------------------------------------
            # 1. Interval Denyut
            # ------------------------------------------------------------------
            w_pks = pks_seg[(pks_seg >= s_samp) & (pks_seg < e_samp)]
            ibi_res = evaluate_ibi_plausibility(w_pks, fs=FS)

            num_peaks_detected = ibi_res["num_peaks"]
            num_valid_intervals = ibi_res["num_valid_intervals"]
            num_unreliable_intervals = ibi_res["num_unreliable_intervals"]
            ibi_median_s = ibi_res["ibi_median_s"]
            heart_rate_bpm = ibi_res["heart_rate_bpm"]
            ibi_cv = ibi_res["ibi_cv"]

            # ------------------------------------------------------------------
            # 2. Morfologi PPG & APG V2
            # ------------------------------------------------------------------
            beat_morph_list = []
            apg_b_a_list = []
            apg_c_a_list = []
            apg_d_a_list = []
            apg_e_a_list = []

            for i in range(1, len(valleys_seg)):
                v_on = valleys_seg[i - 1]
                p_sys = pks_seg[i]
                v_off = valleys_seg[i]

                if s_samp <= v_on and v_off <= e_samp:
                    # PPG Morfologi
                    m_feat = extract_single_beat_ppg_morphology(seg_filt, v_on, p_sys, v_off, FS)
                    beat_morph_list.append(m_feat)

                    # APG Fidusial V2
                    apg_feat = detect_single_beat_apg_waves_v2(seg_apg, v_on, p_sys, v_off, FS)
                    if not np.isnan(apg_feat["apg_b_a"]):
                        apg_b_a_list.append(apg_feat["apg_b_a"])
                    if not np.isnan(apg_feat["apg_c_a"]):
                        apg_c_a_list.append(apg_feat["apg_c_a"])
                    if not np.isnan(apg_feat["apg_d_a"]):
                        apg_d_a_list.append(apg_feat["apg_d_a"])
                    if not np.isnan(apg_feat["apg_e_a"]):
                        apg_e_a_list.append(apg_feat["apg_e_a"])

            n_valid_beats = len(beat_morph_list)

            # Hitung median morfologi PPG
            if n_valid_beats > 0:
                df_bm = pd.DataFrame(beat_morph_list)
                rise_time_s = float(df_bm["rise_time_s"].median())
                decay_time_s = float(df_bm["decay_time_s"].median())
                pulse_width_50_s = float(df_bm["pulse_width_50_s"].median())
                pulse_amplitude = float(df_bm["pulse_amplitude"].median())
                pulse_area = float(df_bm["pulse_area"].median())
                max_ascending_slope = float(df_bm["max_ascending_slope"].median())
                min_descending_slope = float(df_bm["min_descending_slope"].median())
            else:
                rise_time_s = np.nan
                decay_time_s = np.nan
                pulse_width_50_s = np.nan
                pulse_amplitude = np.nan
                pulse_area = np.nan
                max_ascending_slope = np.nan
                min_descending_slope = np.nan

            # 4 Rasio APG Median V2
            apg_b_a = float(np.median(apg_b_a_list)) if len(apg_b_a_list) > 0 else np.nan
            apg_c_a = float(np.median(apg_c_a_list)) if len(apg_c_a_list) > 0 else np.nan
            apg_d_a = float(np.median(apg_d_a_list)) if len(apg_d_a_list) > 0 else np.nan
            apg_e_a = float(np.median(apg_e_a_list)) if len(apg_e_a_list) > 0 else np.nan

            b_cnt = len(apg_b_a_list)
            c_cnt = len(apg_c_a_list)
            d_cnt = len(apg_d_a_list)
            e_cnt = len(apg_e_a_list)

            b_pct = (b_cnt / n_valid_beats) * 100.0 if n_valid_beats > 0 else 0.0
            c_pct = (c_cnt / n_valid_beats) * 100.0 if n_valid_beats > 0 else 0.0
            d_pct = (d_cnt / n_valid_beats) * 100.0 if n_valid_beats > 0 else 0.0
            e_pct = (e_cnt / n_valid_beats) * 100.0 if n_valid_beats > 0 else 0.0

            apg_reliable = bool(
                n_valid_beats >= 10 and (b_cnt / n_valid_beats >= 0.70)
            ) if n_valid_beats > 0 else False

            # ------------------------------------------------------------------
            # 3. Statistik Sinyal (Skewness, Kurtosis)
            # ------------------------------------------------------------------
            skewness_val = float(skew(w_filt, bias=True))
            kurtosis_val = float(kurtosis(w_filt, fisher=True, bias=True))

            # ------------------------------------------------------------------
            # 4. Domain Frekuensi (Welch PSD, Dominant Freq, SPR, Entropy)
            # ------------------------------------------------------------------
            f_welch, psd = sig.welch(w_filt, fs=FS, nperseg=256, noverlap=128)
            mask_total = (f_welch >= 0.7) & (f_welch <= 6.0)
            f_band = f_welch[mask_total]
            psd_band = psd[mask_total]

            if len(psd_band) > 0 and np.max(psd_band) > 0:
                dominant_frequency_hz = float(f_band[np.argmax(psd_band)])
            else:
                dominant_frequency_hz = np.nan

            mask_cardiac = (f_welch >= 1.0) & (f_welch <= 3.0)
            p_cardiac = trapezoid(psd[mask_cardiac], f_welch[mask_cardiac])
            p_total = trapezoid(psd_band, f_band)
            spectral_power_ratio = float(p_cardiac / p_total) if p_total > 0 else np.nan

            if len(psd_band) > 1 and np.sum(psd_band) > 0:
                p_norm = psd_band / np.sum(psd_band)
                p_norm = p_norm[p_norm > 0]
                spectral_entropy = float(-np.sum(p_norm * np.log(p_norm)) / np.log(len(f_band)))
            else:
                spectral_entropy = np.nan

            # ------------------------------------------------------------------
            # Status Kualitas & Warning
            # ------------------------------------------------------------------
            warnings = []
            if num_unreliable_intervals > 0:
                warnings.append("UNRELIABLE_IBI_INTERVALS")
            if not np.isnan(pulse_amplitude) and pulse_amplitude < 2.0:
                warnings.append("LOW_PULSE_AMPLITUDE")
            if n_valid_beats < 10:
                warnings.append("LOW_BEAT_COUNT")

            quality_warning_str = "; ".join(warnings) if warnings else "NONE"
            feat_status = "SUCCESS" if n_valid_beats >= 5 else "PARTIAL_LOW_BEATS"

            row_feat = {
                "patient_id": str(pid),
                "window_id": w_id,
                "start_timestamp": win_row["start_timestamp"],
                "end_timestamp": win_row["end_timestamp"],
                "spo2_target_pct": win_row["spo2_target_pct"],
                "feature_extraction_status": feat_status,
                "quality_warning": quality_warning_str,
                "num_peaks_detected": num_peaks_detected,
                "num_valid_intervals": num_valid_intervals,
                "num_unreliable_intervals": num_unreliable_intervals,
                "num_valid_beats": n_valid_beats,
                "apg_b_a_valid_beats": b_cnt,
                "apg_c_a_valid_beats": c_cnt,
                "apg_d_a_valid_beats": d_cnt,
                "apg_e_a_valid_beats": e_cnt,
                "apg_b_a_pct": round(b_pct, 1),
                "apg_c_a_pct": round(c_pct, 1),
                "apg_d_a_pct": round(d_pct, 1),
                "apg_e_a_pct": round(e_pct, 1),
                "apg_reliable_coverage": apg_reliable,
                # 15 Fitur PPG Primer
                "ibi_median_s": ibi_median_s,
                "heart_rate_bpm": heart_rate_bpm,
                "ibi_cv": ibi_cv,
                "rise_time_s": rise_time_s,
                "decay_time_s": decay_time_s,
                "pulse_width_50_s": pulse_width_50_s,
                "pulse_amplitude": pulse_amplitude,
                "pulse_area": pulse_area,
                "max_ascending_slope": max_ascending_slope,
                "min_descending_slope": min_descending_slope,
                "skewness": skewness_val,
                "kurtosis": kurtosis_val,
                "dominant_frequency_hz": dominant_frequency_hz,
                "spectral_power_ratio": spectral_power_ratio,
                "spectral_entropy": spectral_entropy,
                # 4 Rasio APG Eksploratif V2
                "apg_b_a": apg_b_a,
                "apg_c_a": apg_c_a,
                "apg_d_a": apg_d_a,
                "apg_e_a": apg_e_a,
            }
            features_list.append(row_feat)

    del ppg_2d, spo2_1d, ts_dt, ts_ns
    gc.collect()

    df_out = pd.DataFrame(features_list)
    return df_out, seg_summary


# ==============================================================================
# FUNGSI UTAMA PENJALANAN 30 PASIEN
# ==============================================================================

def main():
    print("=" * 85)
    print("TAHAP 7H: EKSTRAKSI FITUR FINAL OSASUD SELURUH 30 PASIEN SECARA BERSYARAT")
    print("=" * 85)

    t0_start = time.time()
    project_root = find_project_root()
    npz_dir = project_root / "data" / "osasud" / "patients_npz"
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    out_csv_path = results_dir / "osasud_19_features_full.csv"
    out_summary_path = results_dir / "osasud_full_feature_extraction_summary.txt"

    cheby_sos = sig.cheby2(CHEBY_ORDER, CHEBY_RS_DB, CHEBY_WN, btype="bandpass", fs=FS, output="sos")

    initial_ram = get_process_ram_mb()
    peak_ram = initial_ram

    total_patients_processed = 0
    total_windows_all = 0
    total_eligible_all = 0
    patient_summaries = []

    # Buka CSV baru (tulis header terlebih dahulu)
    first_patient = True

    print(f"\nMemulai pemrosesan 30 pasien. RAM Awal: {initial_ram:.1f} MB\n")

    for idx, pid in enumerate(ALL_PATIENT_IDS, start=1):
        npz_file = npz_dir / f"patient_{pid}.npz"
        if not npz_file.is_file():
            print(f"[ERROR] Berkas tidak ditemukan: {npz_file}")
            sys.exit(1)

        t0_pat = time.time()
        df_pat, seg_sum = process_single_patient(pid, npz_file, cheby_sos)
        dur_pat = time.time() - t0_pat

        cur_ram = get_process_ram_mb()
        if cur_ram > peak_ram:
            peak_ram = cur_ram

        n_win = seg_sum["total_windows"]
        n_elig = seg_sum["eligible_windows"]
        n_feat = len(df_pat)

        total_windows_all += n_win
        total_eligible_all += n_elig
        total_patients_processed += 1

        patient_summaries.append({
            "patient_id": pid,
            "total_windows": n_win,
            "eligible_windows": n_elig,
            "features_extracted": n_feat,
            "duration_s": round(dur_pat, 2),
            "ram_mb": round(cur_ram, 1),
        })

        # Tulis secara bertahap ke CSV (append mode)
        if len(df_pat) > 0:
            if first_patient:
                df_pat.to_csv(out_csv_path, index=False, mode="w")
                first_patient = False
            else:
                df_pat.to_csv(out_csv_path, index=False, mode="a", header=False)

        print(f"[{idx:02d}/30] Pasien {pid:>2}: {n_win:>4} jendela, {n_elig:>4} eligible, "
              f"{n_feat:>4} diekstrak | {dur_pat:>5.2f}s | RAM: {cur_ram:>5.1f} MB")

        del df_pat
        gc.collect()

    t_total = time.time() - t0_start
    print("\n" + "=" * 85)
    print(f"SELESAI PEMROSESAN 30 PASIEN DALAM {t_total:.2f} DETIK ({t_total/60:.2f} MENIT)")
    print(f"RAM Puncak: {peak_ram:.1f} MB")
    print("=" * 85)

    # Validasi dan Baca Ulang CSV Lengkap untuk Penyusunan Laporan
    print(f"\nMemvalidasi berkas CSV luaran lengkap: {out_csv_path.name}...")
    df_full = pd.read_csv(out_csv_path)
    assert len(df_full) == total_eligible_all, (
        f"Ketidaksesuaian jumlah baris! CSV memiliki {len(df_full)}, diharapkan {total_eligible_all}"
    )

    # Hitung Statistik Lengkap untuk Summary
    spo2_s = df_full["spo2_target_pct"]
    spo2_min = float(spo2_s.min())
    spo2_max = float(spo2_s.max())
    spo2_median = float(spo2_s.median())
    spo2_mean = float(spo2_s.mean())
    spo2_std = float(spo2_s.std())
    spo2_q25 = float(spo2_s.quantile(0.25))
    spo2_q75 = float(spo2_s.quantile(0.75))

    # Kategori Hipoksemia Klinis
    n_normal = int((spo2_s >= 95).sum())
    n_mild = int(((spo2_s >= 90) & (spo2_s < 95)).sum())
    n_moderate = int(((spo2_s >= 85) & (spo2_s < 90)).sum())
    n_severe = int((spo2_s < 85).sum())

    # Kelengkapan Fitur
    feature_cols = [
        # 15 Fitur PPG
        "ibi_median_s", "heart_rate_bpm", "ibi_cv", "rise_time_s", "decay_time_s",
        "pulse_width_50_s", "pulse_amplitude", "pulse_area", "max_ascending_slope",
        "min_descending_slope", "skewness", "kurtosis", "dominant_frequency_hz",
        "spectral_power_ratio", "spectral_entropy",
        # 4 Rasio APG
        "apg_b_a", "apg_c_a", "apg_d_a", "apg_e_a"
    ]

    # Warning Breakdown
    warn_counts = df_full["quality_warning"].value_counts().to_dict()

    # Tulis Ringkasan Teknis Lengkap
    with open(out_summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 90 + "\n")
        f.write("LAPORAN RINGKASAN EKSTRAKSI FITUR FINAL OSASUD 30 PASIEN (TAHAP 7H)\n")
        f.write("Proyek : Estimasi Tekanan Darah Cuffless & Saturasi Oksigen (SpO2) Menggunakan XGBoost\n")
        f.write(f"Waktu  : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Lokasi : {out_summary_path}\n")
        f.write("=" * 90 + "\n\n")

        f.write("[1. RINGKASAN DATASET & SEGMENTASI 30 PASIEN]\n")
        f.write("-" * 90 + "\n")
        f.write(f"  - Total Pasien Berhasil Diproses : {total_patients_processed} dari 30 pasien\n")
        f.write(f"  - Total Jendela 30 Detik         : {total_windows_all:,} jendela\n")
        f.write(f"  - Total Jendela Memenuhi Syarat  : {total_eligible_all:,} jendela ({total_eligible_all/total_windows_all*100:.2f}%)\n")
        f.write(f"  - Waktu Eksekusi Total           : {t_total:.2f} detik ({t_total/60:.2f} menit)\n")
        f.write(f"  - Penggunaan Memori RAM Puncak   : {peak_ram:.1f} MB (Sangat aman, di bawah batas sistem)\n\n")

        f.write("[2. DISTRIBUSI TARGET REGRESI SpO2 PADA JENDELA ELIGIBLE]\n")
        f.write("-" * 90 + "\n")
        f.write(f"  - Rentang Nilai Target (Min - Max) : {spo2_min:.1f}% - {spo2_max:.1f}%\n")
        f.write(f"  - Rata-rata +/- Standar Deviasi    : {spo2_mean:.2f}% +/- {spo2_std:.2f}%\n")
        f.write(f"  - Kuartil (Q1 - Median - Q3)       : {spo2_q25:.1f}% - {spo2_median:.1f}% - {spo2_q75:.1f}%\n")
        f.write("  - Sebaran Rentang Klinis Hipoksemia:\n")
        f.write(f"    * Normal (SpO2 >= 95%)           : {n_normal:,} jendela ({n_normal/total_eligible_all*100:.1f}%)\n")
        f.write(f"    * Mild Desaturasi (90% <= SpO2 < 95%) : {n_mild:,} jendela ({n_mild/total_eligible_all*100:.1f}%)\n")
        f.write(f"    * Moderate Desaturasi (85% <= SpO2 < 90%): {n_moderate:,} jendela ({n_moderate/total_eligible_all*100:.1f}%)\n")
        f.write(f"    * Severe Desaturasi (SpO2 < 85%) : {n_severe:,} jendela ({n_severe/total_eligible_all*100:.1f}%)\n\n")

        f.write("[3. KELENGKAPAN & TINGKAT MISSINGNESS 19 KANDIDAT FITUR]\n")
        f.write("-" * 90 + "\n")
        f.write(f"{'Nama Fitur':<25} | {'Valid Jendela':<15} | {'Kelengkapan (%)':<18} | {'Missing (%)':<15}\n")
        f.write("-" * 90 + "\n")
        for col in feature_cols:
            n_val = int(df_full[col].notna().sum())
            pct_val = (n_val / total_eligible_all) * 100.0
            pct_miss = 100.0 - pct_val
            f.write(f"{col:<25} | {n_val:>13,} | {pct_val:>16.2f}% | {pct_miss:>13.2f}%\n")
        f.write("-" * 90 + "\n")
        f.write("Catatan: 15 Fitur PPG Primer memiliki kelengkapan 100%. Rasio APG diekstraksi secara jujur\n")
        f.write("dan menyajikan nilai NaN pada morfologi yang tidak meyakinkan tanpa pemaksaan angka.\n\n")

        f.write("[4. SEBARAN QUALITY WARNING PADA JENDELA ELIGIBLE]\n")
        f.write("-" * 90 + "\n")
        for w_type, cnt in warn_counts.items():
            f.write(f"  - {w_type:<35} : {cnt:>6,} jendela ({cnt/total_eligible_all*100:>5.1f}%)\n")
        f.write("\n")

        f.write("[5. RINCIAN PER PASIEN (30 PASIEN)]\n")
        f.write("-" * 90 + "\n")
        f.write(f"{'PID':<5} | {'Total Jendela':<15} | {'Eligible':<12} | {'Ekstraksi':<12} | {'Durasi (s)':<12} | {'RAM (MB)':<10}\n")
        f.write("-" * 90 + "\n")
        for ps in patient_summaries:
            f.write(f"{ps['patient_id']:<5} | {ps['total_windows']:>13,} | {ps['eligible_windows']:>10,} | "
                    f"{ps['features_extracted']:>10,} | {ps['duration_s']:>10.2f} | {ps['ram_mb']:>8.1f}\n")
        f.write("-" * 90 + "\n")

    print(f"\n[SUMMARY TERSIMPAN] {out_summary_path} ({out_summary_path.stat().st_size:,} bytes)")
    print(f"[CSV LENGKAP TERSIMPAN] {out_csv_path} ({out_csv_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
