import sys, time
from pathlib import Path
import h5py
import numpy as np
import scipy.signal as sig
from smoke_test_uci import (
    FS, WINDOW_SEC, WINDOW_SAMPLES, sos,
    extract_single_beat_ppg_morphology, detect_single_beat_apg_waves,
    detect_abp_peaks_and_valleys
)
from scipy.stats import skew, kurtosis
from scipy.integrate import trapezoid

t0 = time.time()
n_rec = 50
total_windows = 0
with h5py.File("data/uci/Part_1.mat", "r") as f:
    part1 = f["Part_1"]
    for rec_id in range(n_rec):
        raw = f[part1[rec_id, 0]][:]
        n_samples = len(raw)
        if n_samples < WINDOW_SAMPLES:
            continue
        ppg_raw = raw[:, 0]
        abp_raw = raw[:, 1]
        
        # Cek integritas
        if np.isnan(ppg_raw).any() or np.isinf(ppg_raw).any() or np.std(ppg_raw) == 0:
            continue
            
        ppg_filt = sig.sosfiltfilt(sos, ppg_raw)
        apg_sg = sig.savgol_filter(ppg_filt, 11, 3, deriv=2, delta=1.0 / FS)
        
        min_dist = int(0.35 * FS)
        prom = 0.25 * np.std(ppg_filt)
        peaks_ppg, _ = sig.find_peaks(ppg_filt, distance=min_dist, prominence=prom)
        if len(peaks_ppg) < 2:
            continue
        valleys_ppg = [peaks_ppg[i] + np.argmin(ppg_filt[peaks_ppg[i]:peaks_ppg[i+1]]) for i in range(len(peaks_ppg)-1)]
        valleys_ppg = np.array(valleys_ppg, dtype=int)
        
        peaks_abp, valleys_abp, cycles_abp = detect_abp_peaks_and_valleys(abp_raw, fs=FS)
        
        n_win = n_samples // WINDOW_SAMPLES
        for w in range(n_win):
            s_idx = w * WINDOW_SAMPLES
            e_idx = s_idx + WINDOW_SAMPLES
            total_windows += 1
            w_ppg = ppg_filt[s_idx:e_idx]
            
            # IBI
            w_pks = peaks_ppg[(peaks_ppg >= s_idx) & (peaks_ppg < e_idx)]
            if len(w_pks) >= 2:
                ibis_s = np.diff(w_pks) / FS
                ibi_med = float(np.median(ibis_s))
                hr = float(60.0 / ibi_med)
                ibi_cv = float(np.std(ibis_s) / np.mean(ibis_s))
                
            # Morfologi beats
            beat_ppg = []
            beat_apg = []
            for i in range(1, len(valleys_ppg)):
                v_on = valleys_ppg[i-1]
                p_c = peaks_ppg[i]
                v_off = valleys_ppg[i]
                if s_idx <= v_on and v_off <= e_idx:
                    beat_ppg.append(extract_single_beat_ppg_morphology(ppg_filt, v_on, p_c, v_off, FS))
                    beat_apg.append(detect_single_beat_apg_waves(apg_sg, v_on, p_c, v_off, FS))
                    
            if len(beat_ppg) > 0:
                df_b = pd.DataFrame(beat_ppg)
                df_a = pd.DataFrame(beat_apg)
                # median features
                
            sk = float(skew(w_ppg, bias=True))
            kt = float(kurtosis(w_ppg, fisher=True, bias=True))
            f_w, psd = sig.welch(w_ppg, fs=FS, nperseg=512, noverlap=256)

t1 = time.time()
dur = t1 - t0
print(f"50 record lengkap ({total_windows} jendela) selesai dalam {dur:.2f}s ({dur/50:.3f} s/record)")
est_total = (dur / 50) * 12000
print(f"Estimasi total 12.000 record: {est_total:.1f} detik ({est_total/60:.1f} menit)")
