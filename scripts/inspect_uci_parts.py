import h5py
from pathlib import Path

uci_dir = Path('data/uci')
for p in ['Part_1.mat', 'Part_2.mat', 'Part_3.mat', 'Part_4.mat']:
    file_path = uci_dir / p
    if not file_path.is_file():
        print(f'{p}: File tidak ditemukan!')
        continue
    size_mb = file_path.stat().st_size / (1024*1024)
    with h5py.File(file_path, 'r') as f:
        keys = list(f.keys())
        print(f'{p:<12}: Size={size_mb:.1f} MB, Keys={keys}')
        for k in keys:
            if not k.startswith('#'):
                dset = f[k]
                print(f'   Key "{k}": shape={dset.shape}, dtype={dset.dtype}')
