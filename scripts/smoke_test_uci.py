import sys
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import scipy.signal as sig
from scipy.stats import skew, kurtosis
from scipy.integrate import trapezoid

# Konfigurasi UCI 125 Hz
FS = 125
WINDOW_SEC = 30
WINDOW_SAMPLES = WINDOW_SEC * FS  # 3750

# Filter Chebyshev II
CHEBY_ORDER = 4
CHEBY_RS_DB = 20
CHEBY_WN = [0.5, 8.0]
sos = sig.cheby2(CHEBY_ORDER, CHEBY_RS_DB, CHEBY_WN, btype="bandpass", fs=FS, output="sos")

def extract_single_beat_ppg_morphology(y_filt, idx_on, idx_peak, idx_off, fs):
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

def detect_single_beat_apg_waves(apg_signal, idx_on, idx_peak, idx_off, fs):
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

    # c, d, e
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

def detect_abp_peaks_and_valleys(abp_signal, fs=FS, min_dist_s=0.35, prominence_mmhg=10.0):
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

        # Validasi fisiologis dasar
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

# Jalankan smoke test perbandingan pada 3 record pertama Part_1
print("Menjalankan Smoke Test Perbandingan pada Record 0, 1, 2 dari Part_1...")
data_path = Path("data/uci/Part_1.mat")
pilot_csv_path = Path("results/uci_ml_ready_pilot.csv")
df_pilot = pd.read_csv(pilot_csv_path)

rows_smoke = []
with h5py.File(data_path, "r") as f:
    part1 = f["Part_1"]
    for rec_id in [0, 1, 2]:
        raw_mat = f[part1[rec_id, 0]][:]
        ppg_raw = raw_mat[:, 0]
        abp_raw = raw_mat[:, 1]
        n_samples = len(ppg_raw)

        ppg_filt = sig.sosfiltfilt(sos, ppg_raw)
        apg_sg = sig.savgol_filter(ppg_filt, window_length=11, polyorder=3, deriv=2, delta=1.0 / FS)

        min_dist = int(0.35 * FS)
        prom = 0.25 * np.std(ppg_filt)
        peaks_ppg, _ = sig.find_peaks(ppg_filt, distance=min_dist, prominence=prom)
        valleys_ppg = [peaks_ppg[i] + np.argmin(ppg_filt[peaks_ppg[i]:peaks_ppg[i + 1]]) for i in range(len(peaks_ppg) - 1)]
        valleys_ppg = np.array(valleys_ppg, dtype=int)

        # ABP
        peaks_abp, valleys_abp, cycles_abp = detect_abp_peaks_and_valleys(abp_raw, fs=FS)

        n_windows = n_samples // WINDOW_SAMPLES
        for w in range(n_windows):
            s_idx = w * WINDOW_SAMPLES
            e_idx = s_idx + WINDOW_SAMPLES
            t_start_s = float(w * WINDOW_SEC)
            t_end_s = float((w + 1) * WINDOW_SEC)

            w_ppg = ppg_filt[s_idx:e_idx]

            # 1. IBI
            w_pks = peaks_ppg[(peaks_ppg >= s_idx) & (peaks_ppg < e_idx)]
            if len(w_pks) >= 2:
                ibis_s = np.diff(w_pks) / FS
                ibi_med = float(np.median(ibis_s))
                hr = float(60.0 / ibi_med) if ibi_med > 0 else np.nan
                ibi_cv = float(np.std(ibis_s) / np.mean(ibis_s)) if np.mean(ibis_s) > 0 else np.nan
            else:
                ibi_med = hr = ibi_cv = np.nan

            # 2. PPG & APG Beats
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
                ba = float(df_a["apg_b_a"].dropna().median()) if len(df_a["apg_b_a"].dropna()) > 0 else np.nan
                ca = float(df_a["apg_c_a"].dropna().median()) if len(df_a["apg_c_a"].dropna()) > 0 else np.nan
                da = float(df_a["apg_d_a"].dropna().median()) if len(df_a["apg_d_a"].dropna()) > 0 else np.nan
                ea = float(df_a["apg_e_a"].dropna().median()) if len(df_a["apg_e_a"].dropna()) > 0 else np.nan
            else:
                rt = dt = pw50 = amp = area = mas = mds = ba = ca = da = ea = np.nan

            sk = float(skew(w_ppg, bias=True))
            kt = float(kurtosis(w_ppg, fisher=True, bias=True))

            f_w, psd = sig.welch(w_ppg, fs=FS, nperseg=512, noverlap=256)
            mask_tot = (f_w >= 0.7) & (f_w <= 6.0)
            fb = f_w[mask_tot]
            pb = psd[mask_tot]
            dom_f = float(fb[np.argmax(pb)]) if len(pb) > 0 and np.max(pb) > 0 else np.nan
            p_card = trapezoid(psd[(f_w >= 1.0) & (f_w <= 3.0)], f_w[(f_w >= 1.0) & (f_w <= 3.0)])
            p_tot = trapezoid(pb, fb)
            spr = float(p_card / p_tot) if p_tot > 0 else np.nan
            p_n = pb / np.sum(pb) if np.sum(pb) > 0 else np.array([])
            p_n = p_n[p_n > 0]
            ent = float(-np.sum(p_n * np.log(p_n)) / np.log(len(fb))) if len(p_n) > 1 else np.nan

            # ABP cycles
            w_cycles = [c for c in cycles_abp if c["valid"] and (s_idx <= c["v_start"]) and (c["v_end"] <= e_idx)]
            if len(w_cycles) >= 5:
                sbp_med = float(np.median([c["sbp"] for c in w_cycles]))
                dbp_med = float(np.median([c["dbp"] for c in w_cycles]))
            else:
                sbp_med = dbp_med = np.nan

            rows_smoke.append({
                "record_id": rec_id, "window_id": w,
                "sbp_mmhg": round(sbp_med, 4), "dbp_mmhg": round(dbp_med, 4),
                "ibi_median_s": round(ibi_med, 4), "heart_rate_bpm": round(hr, 2), "ibi_cv": round(ibi_cv, 4),
                "rise_time_s": round(rt, 4), "decay_time_s": round(dt, 4), "pulse_width_50_s": round(pw50, 4),
                "pulse_amplitude": round(amp, 4), "pulse_area": round(area, 4),
                "max_ascending_slope": round(mas, 4), "min_descending_slope": round(mds, 4),
                "skewness": round(sk, 4), "kurtosis": round(kt, 4),
                "dominant_frequency_hz": round(dom_f, 4), "spectral_power_ratio": round(spr, 4), "spectral_entropy": round(ent, 4),
                "apg_b_a": round(ba, 4) if not np.isnan(ba) else np.nan,
                "apg_c_a": round(ca, 4) if not np.isnan(ca) else np.nan,
                "apg_d_a": round(da, 4) if not np.isnan(da) else np.nan,
                "apg_e_a": round(ea, 4) if not np.isnan(ea) else np.nan,
            })

df_smoke = pd.DataFrame(rows_smoke)
print(f"Smoke test menghasilkan {len(df_smoke)} baris.")

# Bandingkan dengan df_pilot
match_cols = ["sbp_mmhg", "dbp_mmhg", "ibi_median_s", "heart_rate_bpm", "ibi_cv",
              "rise_time_s", "decay_time_s", "pulse_width_50_s", "pulse_amplitude", "pulse_area",
              "max_ascending_slope", "min_descending_slope", "skewness", "kurtosis",
              "dominant_frequency_hz", "spectral_power_ratio", "spectral_entropy",
              "apg_b_a", "apg_c_a", "apg_d_a", "apg_e_a"]

all_match = True
for col in match_cols:
    s_smoke = df_smoke[col]
    s_pilot = df_pilot[col]
    # Check NaN match
    nan_match = (s_smoke.isna() == s_pilot.isna()).all()
    # Check values match where notna
    mask = s_smoke.notna() & s_pilot.notna()
    val_diff = np.max(np.abs(s_smoke[mask] - s_pilot[mask])) if mask.sum() > 0 else 0.0
    is_ok = nan_match and (val_diff < 1e-3)
    if not is_ok:
        all_match = False
        print(f"MISMATCH di {col}: max_diff={val_diff}, nan_match={nan_match}")
    else:
        print(f"MATCH: {col:<22} (diff={val_diff:.6f})")

if all_match:
    print("\n[SMOKE TEST LULUS 100%] Seluruh fitur dan label persis identik dengan pilot lama!")
else:
    print("\n[SMOKE TEST GAGAL] Ditemukan ketidaksesuaian.")
