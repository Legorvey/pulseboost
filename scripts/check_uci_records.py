import h5py
import numpy as np
from pathlib import Path

file_path = Path('data/uci/Part_1.mat')
with h5py.File(file_path, 'r') as f:
    part1 = f['Part_1']
    print(f'Part_1 shape: {part1.shape}')
    for i in range(3):
        ref = part1[i, 0]
        data = np.array(f[ref])
        print(f'Record {i}: shape={data.shape}, dtype={data.dtype}')
        # Cek apakah shape (3, N) atau (N, 3)
        if data.shape[0] == 3:
            ppg = data[0, :]
            abp = data[1, :]
            ecg = data[2, :]
        elif data.shape[1] == 3:
            ppg = data[:, 0]
            abp = data[:, 1]
            ecg = data[:, 2]
        else:
            print('Shape tidak biasa:', data.shape)
            continue
        print(f'   PPG min={ppg.min():.2f}, max={ppg.max():.2f}, std={ppg.std():.2f}')
        print(f'   ABP min={abp.min():.2f}, max={abp.max():.2f}, std={abp.std():.2f}')
        print(f'   ECG min={ecg.min():.2f}, max={ecg.max():.2f}, std={ecg.std():.2f}')
