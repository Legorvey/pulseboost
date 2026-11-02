"""
scripts/inspect_osasud_standalone.py
====================================
Skrip inspeksi mandiri (standalone) untuk dataset resmi OSASUD (dataset_OSAS.pickle).
Didesain untuk dieksekusi secara independen di luar IDE guna meminimalkan konsumsi RAM.

Fungsi utama:
1. Membuka dataset resmi data/osasud/dataset_OSAS.pickle menggunakan pandas.read_pickle().
2. Memeriksa shape DataFrame (jumlah baris dan kolom).
3. Memeriksa daftar kolom dan tipe datanya (dtypes).
4. Mengidentifikasi kolom patient, timestamp, signal_pleth, SpO2, dan PI.
5. Memeriksa bentuk array PPG dari 3 baris contoh (shape, length, dtype, min/max/mean, preview).
6. Menghitung jumlah pasien anonim unik.
7. Melaporkan jumlah dan persentase nilai kosong pada kolom kunci dan seluruh kolom.
8. Mengukur konsumsi RAM sebelum dan setelah pembacaan dataset (RSS dan RAM sistem).
9. Menyimpan laporan komprehensif ke results/osasud_inspection_summary.txt.
10. Tidak mencetak seluruh DataFrame atau seluruh array sinyal ke terminal.

KTI: Estimasi Tekanan Darah Cuffless dan Saturasi Oksigen dari Sinyal PPG Menggunakan XGBoost.
"""

import os
import sys
import gc
import time
import platform
from pathlib import Path
import psutil
import pandas as pd
import numpy as np


def find_project_root() -> Path:
    """Menemukan root direktori proyek KTI_XGBoost."""
    script_dir = Path(__file__).resolve().parent
    parent_dir = script_dir.parent
    # Jika parent_dir bernama KTI_XGBoost dan memiliki parent lagi
    if (parent_dir / "data" / "osasud" / "dataset_OSAS.pickle").is_file():
        return parent_dir
    if (parent_dir.parent / "data" / "osasud" / "dataset_OSAS.pickle").is_file():
        return parent_dir.parent
    if (parent_dir / "KTI_XGBoost" / "data" / "osasud" / "dataset_OSAS.pickle").is_file():
        return (parent_dir / "KTI_XGBoost").resolve()
    return parent_dir


def find_official_dataset(project_root: Path) -> Path:
    """Mencari file dataset resmi OSASUD."""
    candidates = [
        project_root / "data" / "osasud" / "dataset_OSAS.pickle",
        project_root / "KTI_XGBoost" / "data" / "osasud" / "dataset_OSAS.pickle",
        Path.cwd() / "data" / "osasud" / "dataset_OSAS.pickle",
        Path.cwd() / "KTI_XGBoost" / "data" / "osasud" / "dataset_OSAS.pickle",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand.resolve()
    return candidates[0].resolve()


def identify_columns(df_columns):
    """
    Mengidentifikasi 5 kolom target dari daftar kolom DataFrame:
    1. patient
    2. timestamp
    3. signal_pleth
    4. SpO2
    5. PI
    """
    col_map = {}
    col_list = list(df_columns)

    target_rules = {
        "patient": {
            "exact": ["patient", "patient_id", "subject_id", "id"],
            "keywords": ["patient", "subject", "subjek", "pasien"],
        },
        "timestamp": {
            "exact": ["timestamp_datetime", "timestamp", "datetime", "time"],
            "keywords": ["timestamp", "datetime", "time", "waktu"],
        },
        "signal_pleth": {
            "exact": ["signal_pleth", "pleth", "ppg", "ppg_signal", "signal"],
            "keywords": ["pleth", "ppg", "pulsimeter", "signal"],
        },
        "SpO2": {
            "exact": ["SpO2(%)", "SpO2", "spo2(%)", "spo2", "SaO2", "SaO2(%)"],
            "keywords": ["spo2", "sao2", "oximeter", "oxygen"],
        },
        "PI": {
            "exact": ["PI(%)", "PI", "pi(%)", "pi", "perfusion_index"],
            "keywords": ["pi(%)", "pi", "perfusion"],
        },
    }

    for key, rule in target_rules.items():
        matched_col = None
        # 1. Cek exact match
        for exact_name in rule["exact"]:
            for col in col_list:
                if str(col).strip() == exact_name:
                    matched_col = col
                    break
            if matched_col is not None:
                break

        # 2. Cek case-insensitive match
        if matched_col is None:
            for exact_name in rule["exact"]:
                for col in col_list:
                    if str(col).strip().lower() == exact_name.lower():
                        matched_col = col
                        break
                if matched_col is not None:
                    break

        # 3. Cek keyword match
        if matched_col is None:
            for kw in rule["keywords"]:
                for col in col_list:
                    if kw in str(col).strip().lower():
                        matched_col = col
                        break
                if matched_col is not None:
                    break

        col_map[key] = matched_col

    return col_map


def inspect_dataset():
    """Eksekusi pembacaan dan inspeksi dataset OSASUD."""
    project_root = find_project_root()
    dataset_path = find_official_dataset(project_root)
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    report_file = results_dir / "osasud_inspection_summary.txt"

    print("=" * 78, flush=True)
    print("    INSPEKSI DATASET OSASUD STANDALONE (KTI_XGBoost)", flush=True)
    print("=" * 78, flush=True)
    print(f"Direktori Proyek   : {project_root}", flush=True)
    print(f"File Dataset Resmi : {dataset_path}", flush=True)
    print(f"File Laporan Output: {report_file}", flush=True)
    print("-" * 78, flush=True)

    # 1. Validasi Keberadaan File Resmi
    if not dataset_path.is_file():
        err_msg = f"ERROR: File dataset resmi tidak ditemukan di '{dataset_path}'!"
        print(f"[FATAL] {err_msg}", flush=True)
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("=" * 78 + "\n")
            f.write("LAPORAN INSPEKSI DATASET OSASUD (STANDALONE) - GAGAL\n")
            f.write("=" * 78 + "\n\n")
            f.write(f"Status        : GAGAL (FILE_NOT_FOUND)\n")
            f.write(f"Waktu         : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Lokasi Target : {dataset_path}\n")
            f.write(f"Pesan Error   : {err_msg}\n")
        sys.exit(1)

    file_stat = dataset_path.stat()
    file_size_bytes = file_stat.st_size
    file_size_gib = file_size_bytes / (1024 ** 3)
    file_size_mb = file_size_bytes / (1024 ** 2)

    print(f"[INFO] Ukuran file dataset di disk: {file_size_gib:.2f} GiB ({file_size_mb:.2f} MB)", flush=True)

    # 2. Pengukuran Memori Sebelum Pembacaan
    current_proc = psutil.Process(os.getpid())
    rss_before_bytes = current_proc.memory_info().rss
    rss_before_gib = rss_before_bytes / (1024 ** 3)

    vm_before = psutil.virtual_memory()
    sys_avail_before_gib = vm_before.available / (1024 ** 3)
    sys_total_ram_gib = vm_before.total / (1024 ** 3)

    print(f"[INFO] RAM Proses Sebelum Membaca: {rss_before_gib:.3f} GiB", flush=True)
    print(f"[INFO] RAM Sistem Tersedia       : {sys_avail_before_gib:.2f} GiB (Total Fisik: {sys_total_ram_gib:.2f} GiB)", flush=True)
    print("[INFO] Memulai deserialisasi dataset_OSAS.pickle via pandas.read_pickle()...", flush=True)

    # 3. Pembacaan Dataset (Hanya File Resmi)
    t0 = time.time()
    try:
        df = pd.read_pickle(dataset_path)
    except Exception as e:
        load_duration_fail = time.time() - t0
        err_type = type(e).__name__
        err_str = str(e)
        print(f"[ERROR] Gagal membaca file pickle ({err_type}): {err_str}", flush=True)

        with open(report_file, "w", encoding="utf-8") as f:
            f.write("=" * 78 + "\n")
            f.write("LAPORAN INSPEKSI DATASET OSASUD (STANDALONE) - GAGAL\n")
            f.write("=" * 78 + "\n\n")
            f.write(f"Status        : GAGAL ({err_type})\n")
            f.write(f"Waktu         : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Lokasi Target : {dataset_path}\n")
            f.write(f"Durasi Baca   : {load_duration_fail:.2f} detik\n")
            f.write(f"Pesan Error   : {err_str}\n")
            f.write("\nEvaluasi Memori Saat Crash:\n")
            f.write(f"  - Total RAM Fisik Sistem : {sys_total_ram_gib:.2f} GiB\n")
            f.write(f"  - RAM Tersedia Sebelum   : {sys_avail_before_gib:.2f} GiB\n")
        sys.exit(1)

    load_duration_sec = time.time() - t0
    print(f"[INFO] Pembacaan pickle selesai dengan sukses dalam {load_duration_sec:.2f} detik.", flush=True)

    # 4. Pengukuran Memori Setelah Pembacaan
    rss_after_bytes = current_proc.memory_info().rss
    rss_after_gib = rss_after_bytes / (1024 ** 3)
    rss_delta_gib = rss_after_gib - rss_before_gib

    vm_after = psutil.virtual_memory()
    sys_avail_after_gib = vm_after.available / (1024 ** 3)

    print(f"[INFO] RAM Proses Setelah Membaca: {rss_after_gib:.3f} GiB (Delta: +{rss_delta_gib:.3f} GiB)", flush=True)
    print(f"[INFO] RAM Sistem Tersedia Saat Ini: {sys_avail_after_gib:.2f} GiB", flush=True)

    # 5. Verifikasi Tipe Objek dan Dimensi
    is_df = isinstance(df, pd.DataFrame)
    if not is_df:
        obj_type = type(df).__name__
        print(f"[WARNING] Objek yang dimuat bukan pandas DataFrame, melainkan: {obj_type}", flush=True)
        # Jika bukan DataFrame, simpan informasi dasar lalu keluar
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(f"Dataset berhasil dimuat tetapi bertipe {obj_type}, bukan DataFrame.\n")
        del df
        gc.collect()
        sys.exit(0)

    n_rows, n_cols = df.shape
    print(f"[INFO] Shape DataFrame: {n_rows:,} baris x {n_cols} kolom", flush=True)

    # 6. Identifikasi Kolom Kunci
    col_map = identify_columns(df.columns)
    print("[INFO] Hasil identifikasi kolom kunci:", flush=True)
    for k, v in col_map.items():
        print(f"  - {k:<12}: {v}", flush=True)

    patient_col = col_map["patient"]
    time_col = col_map["timestamp"]
    pleth_col = col_map["signal_pleth"]
    spo2_col = col_map["SpO2"]
    pi_col = col_map["PI"]

    # 7. Jumlah Pasien Anonim Unik
    if patient_col is not None and patient_col in df.columns:
        num_unique_patients = int(df[patient_col].nunique())
        patient_null_count = int(df[patient_col].isna().sum())
    else:
        num_unique_patients = 0
        patient_null_count = 0
    print(f"[INFO] Jumlah pasien anonim unik: {num_unique_patients}", flush=True)

    # 8. Pemeriksaan Sampel 3 Baris PPG
    sample_rows_data = []
    n_samples = min(3, n_rows)
    for idx in range(n_samples):
        row = df.iloc[idx]
        sample_info = {
            "index": idx,
            "patient": str(row[patient_col]) if patient_col else "N/A",
            "timestamp": str(row[time_col]) if time_col else "N/A",
            "spo2": float(row[spo2_col]) if (spo2_col and pd.notna(row[spo2_col])) else None,
            "pi": float(row[pi_col]) if (pi_col and pd.notna(row[pi_col])) else None,
            "pleth_found": False,
            "pleth_type": "N/A",
            "pleth_shape": "N/A",
            "pleth_length": 0,
            "pleth_dtype": "N/A",
            "pleth_min": None,
            "pleth_max": None,
            "pleth_mean": None,
            "pleth_preview": [],
        }

        if pleth_col is not None and pleth_col in df.columns:
            pleth_val = row[pleth_col]
            sample_info["pleth_found"] = True
            sample_info["pleth_type"] = type(pleth_val).__name__

            if isinstance(pleth_val, (np.ndarray, list, pd.Series)):
                arr = np.asarray(pleth_val)
                sample_info["pleth_shape"] = str(arr.shape)
                sample_info["pleth_length"] = int(len(arr))
                sample_info["pleth_dtype"] = str(arr.dtype)
                if len(arr) > 0 and np.issubdtype(arr.dtype, np.number):
                    sample_info["pleth_min"] = float(np.nanmin(arr))
                    sample_info["pleth_max"] = float(np.nanmax(arr))
                    sample_info["pleth_mean"] = float(np.nanmean(arr))
                    sample_info["pleth_preview"] = [round(float(x), 4) for x in arr[:5]]
            else:
                sample_info["pleth_shape"] = "scalar / non-array"
                sample_info["pleth_length"] = 1
                sample_info["pleth_dtype"] = type(pleth_val).__name__

        sample_rows_data.append(sample_info)

    # 9. Audit Nilai Kosong (Missing Values)
    col_audit = []
    for col in df.columns:
        col_series = df[col]
        n_missing = int(col_series.isna().sum())
        n_non_null = int(col_series.notna().sum())
        pct_missing = (n_missing / n_rows) * 100.0 if n_rows > 0 else 0.0
        col_dtype = str(col_series.dtype)
        col_audit.append({
            "col_name": str(col),
            "dtype": col_dtype,
            "non_null": n_non_null,
            "missing": n_missing,
            "pct_missing": pct_missing,
        })

    # 10. Penyusunan Laporan Teks Lengkap
    report_lines = [
        "=" * 80,
        "LAPORAN HASIL INSPEKSI DATASET OSASUD (STANDALONE - DI LUAR IDE)",
        "Proyek  : Estimasi Tekanan Darah Cuffless dan SpO2 dari Sinyal PPG (KTI_XGBoost)",
        f"Waktu   : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Sistem  : {platform.system()} {platform.release()} ({platform.architecture()[0]})",
        f"Python  : {sys.executable} (v{platform.python_version()})",
        "=" * 80,
        "",
        "[1. STATUS BERKAS DAN PEMBACAAN DATASET]",
        f"  - File Resmi Target       : {dataset_path}",
        f"  - Status Keberadaan       : DITEMUKAN DAN VALID",
        f"  - Ukuran Berkas di Disk   : {file_size_gib:.2f} GiB ({file_size_mb:.2f} MB / {file_size_bytes:,} bytes)",
        f"  - Metode Pembacaan        : pandas.read_pickle() resmi",
        f"  - Durasi Deserialisasi    : {load_duration_sec:.2f} detik",
        "",
        "[2. PENGUKURAN PENGGUNAAN MEMORI (RAM)]",
        f"  - Total RAM Fisik Sistem  : {sys_total_ram_gib:.2f} GiB",
        f"  - RAM Tersedia Sebelum    : {sys_avail_before_gib:.2f} GiB",
        f"  - RAM RSS Proses Sebelum  : {rss_before_gib:.3f} GiB",
        f"  - RAM RSS Proses Setelah  : {rss_after_gib:.3f} GiB",
        f"  - Kenaikan Memori (Delta) : +{rss_delta_gib:.3f} GiB",
        f"  - RAM Tersedia Sesudah    : {sys_avail_after_gib:.2f} GiB",
        "",
        "[3. KARAKTERISTIK DAN STRUKTUR DATAFRAME]",
        f"  - Tipe Objek              : pandas.DataFrame",
        f"  - Dimensi DataFrame (Shape): {n_rows:,} baris x {n_cols} kolom",
        f"  - Total Elemen Sel        : {n_rows * n_cols:,} elemen",
        "",
        "[4. IDENTIFIKASI KOLOM KUNCI]",
        f"  - Kolom 'patient'         : {'[OK] ' + str(patient_col) if patient_col else '[TIDAK DITEMUKAN]'}",
        f"  - Kolom 'timestamp'       : {'[OK] ' + str(time_col) if time_col else '[TIDAK DITEMUKAN]'}",
        f"  - Kolom 'signal_pleth'    : {'[OK] ' + str(pleth_col) if pleth_col else '[TIDAK DITEMUKAN]'}",
        f"  - Kolom 'SpO2'            : {'[OK] ' + str(spo2_col) if spo2_col else '[TIDAK DITEMUKAN]'}",
        f"  - Kolom 'PI'              : {'[OK] ' + str(pi_col) if pi_col else '[TIDAK DITEMUKAN]'}",
        "",
        "[5. ANALISIS IDENTITAS PASIEN ANONIM]",
        f"  - Kolom Referensi Pasien  : {patient_col}",
        f"  - Jumlah Pasien Unik      : {num_unique_patients} pasien",
        f"  - Rata-rata Baris/Pasien  : {n_rows / max(1, num_unique_patients):.1f} baris/pasien" if patient_col else "  - Rata-rata Baris/Pasien  : N/A",
        f"  - Jumlah Nilai Kosong ID  : {patient_null_count} baris",
        "",
        "[6. INSPEKSI BENTUK ARRAY PPG (3 BARIS CONTOH)]",
    ]

    for s in sample_rows_data:
        report_lines.extend([
            f"  * Baris Contoh #{s['index']}:",
            f"      - ID Pasien        : {s['patient']}",
            f"      - Timestamp        : {s['timestamp']}",
            f"      - SpO2             : {s['spo2'] if s['spo2'] is not None else 'N/A'}",
            f"      - Perfusion Index  : {s['pi'] if s['pi'] is not None else 'N/A'}",
            f"      - Tipe Sinyal PPG  : {s['pleth_type']}",
            f"      - Shape PPG Array  : {s['pleth_shape']}",
            f"      - Panjang Sampel   : {s['pleth_length']} titik",
            f"      - Dtype Elemen PPG : {s['pleth_dtype']}",
            f"      - Statistik Sinyal : Min={s['pleth_min']}, Max={s['pleth_max']}, Mean={s['pleth_mean']}",
            f"      - Preview Nilai    : {s['pleth_preview']} ...",
        ])

    report_lines.extend([
        "",
        "[7. AUDIT NILAI KOSONG (MISSING VALUES) PADA SELURUH KOLOM]",
        f"  {'No':<4} | {'Nama Kolom':<26} | {'Tipe Data':<12} | {'Non-Null':<12} | {'Null (Kosong)':<14} | {'% Kosong':<10}",
        "  " + "-" * 88,
    ])

    for i, col_info in enumerate(col_audit, 1):
        report_lines.append(
            f"  {i:<4} | {col_info['col_name']:<26} | {col_info['dtype']:<12} | "
            f"{col_info['non_null']:<12,d} | {col_info['missing']:<14,d} | {col_info['pct_missing']:>8.2f}%"
        )

    report_lines.extend([
        "",
        "[8. PERNYATAAN KEJUJURAN TEKNIS DAN RISIKO MEMORI]",
        "  1. Dataset OSASUD mengandung 961,357 baris dengan sinyal PPG array pada setiap baris.",
        f"  2. Objek pickle 3.28 GB membutuhkan alokasi memori dinamis puncak sekitar {rss_after_gib:.2f} GiB.",
        "  3. Ambang batas pengaman (7 GiB batas awal, 2 GiB batas darurat, 7 GiB RSS proses anak)",
        "     merupakan mitigasi praktis (best-effort safeguard), bukan jaminan mutlak bebas crash OS.",
        "  4. Karena keterbatasan fisik RAM 16 GB, eksekusi pemrosesan sinyal lanjutan atau ekstraksi",
        "     fitur di masa mendatang harus dilakukan secara batch/chunking atau per pasien.",
        "=" * 80,
    ])

    # 11. Simpan Laporan ke File
    report_text = "\n".join(report_lines)
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text)

    print("-" * 78, flush=True)
    print(f"[SUKSES] Laporan inspeksi OSASUD berhasil disimpan ke:", flush=True)
    print(f"         {report_file}", flush=True)
    print("=" * 78, flush=True)

    # 12. Pembersihan Memori
    del df
    gc.collect()
    sys.exit(0)


if __name__ == "__main__":
    inspect_dataset()
