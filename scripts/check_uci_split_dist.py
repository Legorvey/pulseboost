import pandas as pd
import numpy as np

# Jalankan simulasi split UCI dan cek sebaran SBP/DBP
from test_split_sim import df_uci, meta_groups, train_groups, val_groups, test_groups

rec_to_split = {}
for g in train_groups:
    for r in meta_groups[g]:
        rec_to_split[r] = 'train'
for g in val_groups:
    for r in meta_groups[g]:
        rec_to_split[r] = 'val'
for g in test_groups:
    for r in meta_groups[g]:
        rec_to_split[r] = 'test'

df_uci['split'] = df_uci['record_id'].map(rec_to_split)

for s_name in ['train', 'val', 'test']:
    sub = df_uci[df_uci['split'] == s_name]
    sbp = sub['sbp_mmhg'].dropna()
    dbp = sub['dbp_mmhg'].dropna()
    print(f"\n--- UCI Split: {s_name.upper()} ({len(sub):,} jendela) ---")
    print(f"SBP: min={sbp.min():.1f}, med={sbp.median():.1f}, mean={sbp.mean():.2f}, std={sbp.std():.2f}")
    print(f"DBP: min={dbp.min():.1f}, med={dbp.median():.1f}, mean={dbp.mean():.2f}, std={dbp.std():.2f}")
