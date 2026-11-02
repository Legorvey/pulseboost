import pandas as pd
import numpy as np

# 1. UCI Split Simulation
df_uci = pd.read_csv('results/uci_19_features_full.csv')
df_log = pd.read_csv('results/uci_record_processing_log.csv')

# Bangun pemetaan duplikat transitif (Connected Components)
rec_to_group = {}
group_members = {}

# Daftar rekaman sukses
succ_recs = sorted(df_log[df_log['status'] == 'SUCCESS']['record_id'].unique().tolist())

# Union-Find
parent = {r: r for r in succ_recs}
def find(i):
    if parent[i] == i:
        return i
    parent[i] = find(parent[i])
    return parent[i]

def union(i, j):
    root_i = find(i)
    root_j = find(j)
    if root_i != root_j:
        parent[root_i] = root_j

dups = df_log[df_log['is_duplicate']]
for _, r in dups.iterrows():
    r_id = r['record_id']
    orig = r['duplicate_of']
    if r_id in parent and orig in parent:
        union(r_id, orig)

# Buat meta groups
meta_groups = {}
for r in succ_recs:
    p = find(r)
    meta_groups.setdefault(p, []).append(r)

print(f"Total rekaman sukses: {len(succ_recs):,}")
print(f"Total meta-groups: {len(meta_groups):,}")

# Hitung jumlah jendela per meta-group
win_counts = df_uci['record_id'].value_counts().to_dict()
group_win_counts = {p: sum(win_counts.get(r, 0) for r in recs) for p, recs in meta_groups.items()}

# Shuffle deterministik meta-groups dengan seed 42
rng = np.random.RandomState(42)
all_p_keys = sorted(list(meta_groups.keys()))
rng.shuffle(all_p_keys)

# Target proporsi jendela: 70% Train, 15% Val, 15% Test
total_wins = len(df_uci)
target_train = 0.70 * total_wins
target_val = 0.15 * total_wins

train_groups, val_groups, test_groups = [], [], []
cur_train, cur_val, cur_test = 0, 0, 0

for g in all_p_keys:
    w_cnt = group_win_counts[g]
    if cur_train + w_cnt <= target_train or (cur_train < target_train and cur_val >= target_val):
        train_groups.append(g)
        cur_train += w_cnt
    elif cur_val + w_cnt <= target_val or (cur_val < target_val and cur_test >= target_val):
        val_groups.append(g)
        cur_val += w_cnt
    else:
        test_groups.append(g)
        cur_test += w_cnt

print(f"UCI Windows: Train={cur_train:,} ({cur_train/total_wins*100:.2f}%), "
      f"Val={cur_val:,} ({cur_val/total_wins*100:.2f}%), "
      f"Test={cur_test:,} ({cur_test/total_wins*100:.2f}%)")
print(f"UCI Groups : Train={len(train_groups)}, Val={len(val_groups)}, Test={len(test_groups)}")

# 2. OSASUD Split Simulation
df_osa = pd.read_csv('results/osasud_19_features_full.csv')
pids = sorted(df_osa['patient_id'].unique().tolist())
osa_win_counts = df_osa['patient_id'].value_counts().to_dict()
total_osa_wins = len(df_osa)

rng_osa = np.random.RandomState(42)
shuffled_pids = list(pids)
rng_osa.shuffle(shuffled_pids)

# 30 pasien: 21 Train (70%), 5 Val (16.7%), 4 Test (13.3%)
# Atau greedy allocation mendekati 70/15/15
target_osa_train = 0.70 * total_osa_wins
target_osa_val = 0.15 * total_osa_wins

train_pids, val_pids, test_pids = [], [], []
c_tr, c_val, c_te = 0, 0, 0

for pid in shuffled_pids:
    w_cnt = osa_win_counts[pid]
    if c_tr + w_cnt <= target_osa_train or (c_tr < target_osa_train and c_val >= target_osa_val):
        train_pids.append(pid)
        c_tr += w_cnt
    elif c_val + w_cnt <= target_osa_val or (c_val < target_osa_val and c_te >= target_osa_val):
        val_pids.append(pid)
        c_val += w_cnt
    else:
        test_pids.append(pid)
        c_te += w_cnt

print(f"\nOSASUD Windows: Train={c_tr:,} ({c_tr/total_osa_wins*100:.2f}%), "
      f"Val={c_val:,} ({c_val/total_osa_wins*100:.2f}%), "
      f"Test={c_te:,} ({c_te/total_osa_wins*100:.2f}%)")
print(f"OSASUD Patients: Train={len(train_pids)} {train_pids}, Val={len(val_pids)} {val_pids}, Test={len(test_pids)} {test_pids}")
