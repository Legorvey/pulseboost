import sys
from pathlib import Path
import h5py
import numpy as np

def find_dataset_path():
    """Mencari lokasi file Part_1.mat secara fleksibel."""
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir.parent / "data" / "uci" / "Part_1.mat",
        Path.cwd() / "data" / "uci" / "Part_1.mat",
        Path.cwd() / "KTI_XGBoost" / "data" / "uci" / "Part_1.mat",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None

def inspect_uci():
    mat_path = find_dataset_path()
    if not mat_path:
        print("[ERROR] File Part_1.mat tidak ditemukan di direktori data/uci/!")
        sys.exit(1)
        
    print(f"=== Memeriksa File: {mat_path.name} ===")
    print(f"Path: {mat_path}")
    print(f"Ukuran file: {mat_path.stat().st_size / (1024 * 1024):.2f} MB\n")

    # Buka dengan mode read-only ('r') agar file asli tidak berubah
    with h5py.File(mat_path, "r") as f:
        top_keys = list(f.keys())
        print(f"Top-level keys di file HDF5: {top_keys}")
        
        for k in top_keys:
            item = f[k]
            print(f"- Key '{k}':")
            print(f"    Tipe Objek: {type(item).__name__}")
            if hasattr(item, "shape"):
                print(f"    Shape     : {item.shape}")
            if hasattr(item, "dtype"):
                print(f"    Dtype     : {item.dtype}")

        # Identifikasi dataset utama (biasanya 'p' pada UCI cuffless BP dataset)
        main_key = None
        for cand in ["p", "Part_1", "data"]:
            if cand in f:
                main_key = cand
                break
        
        if not main_key:
            # Gunakan key pertama yang bukan metadata/refs jika ada
            non_refs = [k for k in top_keys if not k.startswith("#")]
            if non_refs:
                main_key = non_refs[0]
            else:
                main_key = top_keys[0]

        print(f"\nMemilih key utama: '{main_key}'")
        main_obj = f[main_key]

        # Periksa apakah objek ini menyimpan object reference (cell array MATLAB)
        sample_record = None
        is_cell_reference = False

        if isinstance(main_obj, h5py.Dataset):
            print(f"Dataset '{main_key}' shape: {main_obj.shape}, dtype: {main_obj.dtype}")
            
            # Cek apakah dtype merupakan object reference HDF5 (h5py.ref_dtype atau object)
            if h5py.check_dtype(ref=main_obj.dtype) is not None or str(main_obj.dtype).startswith("|O"):
                is_cell_reference = True
                print("Tipe data adalah HDF5 Object Reference (MATLAB Cell Array).")
                print(f"Total sel/rekaman di '{main_key}': {main_obj.size}")
                
                # Baca HANYA 1 referensi pertama
                # Bentuk shape bisa (1, N) atau (N, 1) atau (N,)
                first_ref = None
                if main_obj.ndim == 2:
                    first_ref = main_obj[0, 0] if main_obj.shape[0] > 0 and main_obj.shape[1] > 0 else None
                elif main_obj.ndim == 1:
                    first_ref = main_obj[0]
                
                print(f"Referensi rekaman pertama: {first_ref}")
                if first_ref:
                    deref_obj = f[first_ref]
                    print(f"Tipe objek hasil dereferensi: {type(deref_obj).__name__}")
                    if isinstance(deref_obj, h5py.Dataset):
                        print(f"Shape rekaman pertama (di disk): {deref_obj.shape}")
                        print(f"Dtype rekaman pertama         : {deref_obj.dtype}")
                        # Baca hanya potongan kecil untuk cek nilai tanpa membebani RAM
                        sample_record = deref_obj[:]
            else:
                # Dataset langsung (bukan cell array)
                print("Dataset bukan berupa referensi objek.")
                sample_record = main_obj[:10]  # ambil sampel kecil

        elif isinstance(main_obj, h5py.Group):
            print(f"'{main_key}' adalah Grup HDF5 dengan keys: {list(main_obj.keys())}")
        
        # Analisis orientasi kanal sinyal
        if sample_record is not None:
            print("\n=== Analisis Orientasi Array dan Kanal Sinyal ===")
            print(f"Shape array rekaman pertama: {sample_record.shape}")
            print(f"Dimensi (ndim)             : {sample_record.ndim}")
            print(f"Dtype array                : {sample_record.dtype}")
            print(f"Ukuran memori array (MB)   : {sample_record.nbytes / (1024 * 1024):.2f} MB")
            
            # UCI dataset dokumentasi:
            # 3 sinyal:
            # - Kanal 1: PPG (Photoplethysmograph)
            # - Kanal 2: ABP (Arterial Blood Pressure) [mmHg]
            # - Kanal 3: ECG (Electrocardiogram)
            # Frekuensi sampling: 125 Hz
            
            if sample_record.ndim == 2:
                rows, cols = sample_record.shape
                print(f"Jumlah baris: {rows}, Jumlah kolom: {cols}")
                
                # Evaluasi orientasi
                if rows == 3:
                    print("-> Kemungkinan Orientasi A: shape (3, N)")
                    print("   Axis 0 (3 baris) = Kanal Sinyal")
                    print("   Axis 1 (N kolom) = Sampel Waktu (Time steps)")
                elif cols == 3:
                    print("-> Kemungkinan Orientasi B: shape (N, 3)")
                    print("   Axis 0 (N baris) = Sampel Waktu (Time steps)")
                    print("   Axis 1 (3 kolom) = Kanal Sinyal")
                else:
                    print(f"-> CATATAN: Tidak ada dimensi yang persis bernilai 3! (rows={rows}, cols={cols})")

                # Cek ringkasan nilai statistik awal untuk masing-masing kemungkinan kanal
                # Agar kita bisa memverifikasi kanal ABP yang biasanya berada pada rentang tekanan darah manusia (~40 - 200 mmHg)
                print("\nInspeksi nilai statistik kanal (min, max, mean) untuk menentukan orientasi:")
                if rows == 3:
                    for ch_idx in range(3):
                        sig = sample_record[ch_idx, :]
                        print(f"   Baris index {ch_idx}: min={sig.min():.2f}, max={sig.max():.2f}, mean={sig.mean():.2f}, std={sig.std():.2f}")
                elif cols == 3:
                    for ch_idx in range(3):
                        sig = sample_record[:, ch_idx]
                        print(f"   Kolom index {ch_idx}: min={sig.min():.2f}, max={sig.max():.2f}, mean={sig.mean():.2f}, std={sig.std():.2f}")
                else:
                    print("   Dimensi bukan (3, N) maupun (N, 3).")

if __name__ == "__main__":
    inspect_uci()
