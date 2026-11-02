import pandas as pd
import numpy as np

print("=== REKONSILIASI KUALITAS APG (BAGIAN C) ===")

# 1. UCI
df_uci = pd.read_csv('results/uci_19_features_full.csv')
print(f"\n--- DATASET UCI (Total: {len(df_uci):,} jendela) ---")
for r in ['b', 'c', 'd', 'e']:
    col_beats = f"apg_{r}_a_valid_beats"
    col_feat = f"apg_{r}_a"
    
    # Syarat: beats >= 10 DAN (beats / num_valid_beats) >= 0.20
    cond_beats_10 = df_uci[col_beats] >= 10
    cond_ratio_20 = (df_uci[col_beats] / df_uci['num_valid_beats']) >= 0.20
    cond_rep = cond_beats_10 & cond_ratio_20
    
    n_rep = int(cond_rep.sum())
    n_not_rep = len(df_uci) - n_rep
    
    # Berapa banyak yang sebelumnya non-NaN tetapi tidak representatif (akan diubah jadi NaN)
    n_was_not_nan = int(df_uci[col_feat].notna().sum())
    n_to_nan = int((df_uci[col_feat].notna() & (~cond_rep)).sum())
    
    print(f"Rasio {r}/a:")
    print(f"  - Semula Not-NaN           : {n_was_not_nan:>6,} ({n_was_not_nan/len(df_uci)*100:>5.2f}%)")
    print(f"  - Memenuhi Kriteria (>=10 & >=20%): {n_rep:>6,} ({n_rep/len(df_uci)*100:>5.2f}%)")
    print(f"  - Tidak Memenuhi Kriteria  : {n_not_rep:>6,} ({n_not_rep/len(df_uci)*100:>5.2f}%)")
    print(f"  - Diubah Menjadi NaN       : {n_to_nan:>6,} jendela")

# 2. OSASUD
df_osa = pd.read_csv('results/osasud_19_features_full.csv')
print(f"\n--- DATASET OSASUD (Total: {len(df_osa):,} jendela) ---")
for r in ['b', 'c', 'd', 'e']:
    col_beats = f"apg_{r}_a_valid_beats"
    col_feat = f"apg_{r}_a"
    
    cond_beats_10 = df_osa[col_beats] >= 10
    cond_ratio_20 = (df_osa[col_beats] / df_osa['num_valid_beats']) >= 0.20
    cond_rep = cond_beats_10 & cond_ratio_20
    
    n_rep = int(cond_rep.sum())
    n_not_rep = len(df_osa) - n_rep
    n_was_not_nan = int(df_osa[col_feat].notna().sum())
    n_to_nan = int((df_osa[col_feat].notna() & (~cond_rep)).sum())
    
    print(f"Rasio {r}/a:")
    print(f"  - Semula Not-NaN           : {n_was_not_nan:>6,} ({n_was_not_nan/len(df_osa)*100:>5.2f}%)")
    print(f"  - Memenuhi Kriteria (>=10 & >=20%): {n_rep:>6,} ({n_rep/len(df_osa)*100:>5.2f}%)")
    print(f"  - Tidak Memenuhi Kriteria  : {n_not_rep:>6,} ({n_not_rep/len(df_osa)*100:>5.2f}%)")
    print(f"  - Diubah Menjadi NaN       : {n_to_nan:>6,} jendela")
