import pandas as pd

df_log = pd.read_csv('results/uci_record_processing_log.csv')
dups = df_log[df_log['is_duplicate']]
print('Daftar 14 rekaman duplikat:')
for _, r in dups.iterrows():
    print(f"{r['record_id']} -> duplikat dari {r['duplicate_of']}")

origins = dups['duplicate_of'].unique()
print('\nUnique origins:', origins)
for orig in origins:
    sub = df_log[df_log['record_id'] == orig]
    print(f"Origin {orig}: is_duplicate={sub['is_duplicate'].values[0]}")
