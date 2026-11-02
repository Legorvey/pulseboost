import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent))

import h5py
import numpy as np
import scipy.signal as sig
from smoke_test_uci import (
    FS, WINDOW_SEC, WINDOW_SAMPLES, sos,
    extract_single_beat_ppg_morphology, detect_single_beat_apg_waves,
    detect_abp_peaks_and_valleys
)

print("Uji kecil rekaman dari Part_2, Part_3, dan Part_4...")

parts = [
    ("Part_2", Path("data/uci/Part_2.mat")),
    ("Part_3", Path("data/uci/Part_3.mat")),
    ("Part_4", Path("data/uci/Part_4.mat")),
]

for part_name, fpath in parts:
    print(f"\n--- Menguji {part_name} ({fpath.name}) ---")
    with h5py.File(fpath, "r") as f:
        dset = f[part_name]
        print(f"Total record di {part_name}: {dset.shape[0]}")
        # Ambil record indeks 0
        ref = dset[0, 0]
        raw = f[ref][:]
        print(f"Record 0 shape: {raw.shape}")
        ppg_raw = raw[:, 0]
        abp_raw = raw[:, 1]
        
        # Filter & deteksi
        ppg_filt = sig.sosfiltfilt(sos, ppg_raw)
        apg_sg = sig.savgol_filter(ppg_filt, window_length=11, polyorder=3, deriv=2, delta=1.0/FS)
        
        peaks_ppg, _ = sig.find_peaks(ppg_filt, distance=int(0.35*FS), prominence=0.25*np.std(ppg_filt))
        peaks_abp, valleys_abp, cycles_abp = detect_abp_peaks_and_valleys(abp_raw, fs=FS)
        
        print(f"   PPG peaks terdeteksi: {len(peaks_ppg)}")
        print(f"   ABP cycles valid: {len([c for c in cycles_abp if c['valid']])}")
        
        # Ekstrak 1 jendela 30s pertama
        if len(raw) >= WINDOW_SAMPLES:
            w_cycles = [c for c in cycles_abp if c["valid"] and c["v_start"] >= 0 and c["v_end"] <= WINDOW_SAMPLES]
            if len(w_cycles) >= 5:
                sbp_med = np.median([c["sbp"] for c in w_cycles])
                dbp_med = np.median([c["dbp"] for c in w_cycles])
                print(f"   Jendela 0 SBP={sbp_med:.1f} mmHg, DBP={dbp_med:.1f} mmHg (SBP > DBP: {sbp_med > dbp_med})")
            else:
                print(f"   Jendela 0 siklus valid < 5 ({len(w_cycles)})")

print("\n[UJI KECIL PART_2, 3, 4 SUKSES] Semua file MAT dapat diakses dan diekstraksi normal!")
