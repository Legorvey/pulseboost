import pandas as pd
import numpy as np

df_uci = pd.read_csv('results/uci_19_features_full.csv')
valid_bp = df_uci[df_uci['sbp_mmhg'].notna() & df_uci['dbp_mmhg'].notna()].copy()
print(f'Total jendela dengan label valid: {len(valid_bp):,}')

sbp = valid_bp['sbp_mmhg'].values
dbp = valid_bp['dbp_mmhg'].values

# Aturan AHA Hierarkis Saling Eksklusif (Prioritas dari Severity Tertinggi):
# 1. Stage 2 HTN: SBP >= 140 OR DBP >= 90
# 2. Stage 1 HTN: SBP in [130, 140) OR DBP in [80, 90) (jika bukan Stage 2)
# 3. Elevated: SBP in [120, 130) AND DBP < 80 (jika bukan Stage 1 atau 2)
# 4. Normal: SBP < 120 AND DBP < 80

is_stage2 = (sbp >= 140.0) | (dbp >= 90.0)
is_stage1 = (~is_stage2) & (((sbp >= 130.0) & (sbp < 140.0)) | ((dbp >= 80.0) & (dbp < 90.0)))
is_elevated = (~is_stage2) & (~is_stage1) & ((sbp >= 120.0) & (sbp < 130.0) & (dbp < 80.0))
is_normal = (sbp < 120.0) & (dbp < 80.0)

n_s2 = int(is_stage2.sum())
n_s1 = int(is_stage1.sum())
n_elev = int(is_elevated.sum())
n_norm = int(is_normal.sum())
total_cat = n_s2 + n_s1 + n_elev + n_norm

print(f'Stage 2 HTN : {n_s2:,} ({n_s2/len(valid_bp)*100:.2f}%)')
print(f'Stage 1 HTN : {n_s1:,} ({n_s1/len(valid_bp)*100:.2f}%)')
print(f'Elevated    : {n_elev:,} ({n_elev/len(valid_bp)*100:.2f}%)')
print(f'Normal      : {n_norm:,} ({n_norm/len(valid_bp)*100:.2f}%)')
print(f'Total Kategori: {total_cat:,} vs Label Valid: {len(valid_bp):,}')
assert total_cat == len(valid_bp), 'Total kategori tidak cocok dengan jumlah label valid!'
print('[VERIFIKASI SUKSES] Keempat kategori 100% saling eksklusif dan jumlahnya tepat sama dengan label valid.')
