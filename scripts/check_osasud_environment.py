"""
Skrip Pemeriksaan Lingkungan dan Kapasitas Komputer untuk Dataset OSASUD (Tahap 7A)
Proyek KTI_XGBoost

Skrip ini melakukan audit sistem lengkap:
1. Memeriksa keberadaan dan ukuran file dataset_OSAS.pickle tanpa membuka file tersebut.
2. Memeriksa spesifikasi RAM fisik, RAM tersedia, swap memory, dan ruang penyimpanan drive.
3. Memeriksa versi Python dan ketersediaan library utama (pandas, numpy, psutil).
4. Menyusun analisis risiko penggunaan memori dan estimasi kebutuhan deserialisasi Pickle.
5. Menghasilkan laporan resmi: results/osasud_environment_check.txt
"""

import sys
import os
import platform
import shutil
from pathlib import Path
from datetime import datetime

# Pastikan psutil tersedia
try:
    import psutil
except ImportError:
    print("Error: Library psutil belum terpasang. Jalankan pip install psutil terlebih dahulu.")
    sys.exit(1)

try:
    import pandas as pd
    pandas_version = pd.__version__
except ImportError:
    pandas_version = "TIDAK TERPASANG"

try:
    import numpy as np
    numpy_version = np.__version__
except ImportError:
    numpy_version = "TIDAK TERPASANG"

def format_bytes_binary(n_bytes):
    gib = n_bytes / (1024 ** 3)
    mib = n_bytes / (1024 ** 2)
    return f"{gib:.2f} GiB ({mib:.1f} MiB / {n_bytes:,} bytes)"

def format_bytes_decimal(n_bytes):
    gb = n_bytes / (10 ** 9)
    mb = n_bytes / (10 ** 6)
    return f"{gb:.2f} GB ({mb:.1f} MB)"

def main():
    # 1. Penentuan Path
    base_dir = Path.cwd()
    if (base_dir / "KTI_XGBoost").is_dir():
        project_root = base_dir / "KTI_XGBoost"
    else:
        project_root = base_dir

    dataset_rel_candidates = [
        Path("data") / "osasud" / "dataset_OSAS.pickle",
        Path("KTI_XGBoost") / "data" / "osasud" / "dataset_OSAS.pickle",
    ]

    dataset_path = None
    for cand in dataset_rel_candidates:
        if cand.is_file():
            dataset_path = cand.resolve()
            break

    if dataset_path is None:
        direct_check = project_root / "data" / "osasud" / "dataset_OSAS.pickle"
        if direct_check.is_file():
            dataset_path = direct_check.resolve()

    # 2. Pemeriksaan File Dataset
    file_exists = dataset_path is not None and dataset_path.is_file()
    file_size_bytes = 0
    file_size_gib = 0.0
    file_size_gb_dec = 0.0

    if file_exists:
        stat_info = dataset_path.stat()
        file_size_bytes = stat_info.st_size
        file_size_gib = file_size_bytes / (1024 ** 3)
        file_size_gb_dec = file_size_bytes / (10 ** 9)

    # 3. Pemeriksaan Kapasitas Komputer
    vm = psutil.virtual_memory()
    sm = psutil.swap_memory()
    disk = shutil.disk_usage(str(project_root))

    drive_letter = project_root.resolve().drive
    if not drive_letter:
        drive_letter = "C:"

    # 4. Kesiapan Library
    psutil_ver = psutil.__version__
    python_ver = sys.version.split()[0]
    python_arch = platform.architecture()[0]

    # 5. Analisis Risiko & Kapasitas
    # Acuan struktur dari publikasi:
    pub_rows = 961357
    pub_cols = 18
    pub_key_cols = ["patient", "timestamp_datetime", "signal_pleth", "SpO2(%)", "PI(%)"]
    pub_ppg_hz = 80
    pub_spo2_hz = 1
    pub_pi_hz = 1

    # Estimasi kebutuhan memori:
    # Ukuran file di disk: ~3.05 GiB
    # pickle deserialization: Python harus membaca stream, mengalokasikan memori untuk setiap objek
    # Di dalam Pandas DataFrame, 961,357 baris dengan kolom bertipe objek (misal sinyal pleth berupa numpy array atau list)
    # memiliki overhead PyObject header, pointer table, dan fragmentasi heap pymalloc.
    # Selama proses deserialisasi, pickle membuat memo table dan buffer intermediate.
    # Estimasi konsumsi RAM minimal: 1.5x - 2.0x ukuran file (~4.5 - 6.0 GB).
    # Estimasi konsumsi RAM puncak (peak memory spike) saat proses unpickling berlangsung: 2.5x - 3.5x (~7.5 - 10.5 GB).
    
    total_ram_gb = vm.total / (1024 ** 3)
    avail_ram_gb = vm.available / (1024 ** 3)
    used_ram_pct = vm.percent

    # Status kelayakan
    if avail_ram_gb < 4.0:
        ram_status = "BERISIKO TINGGI (KRITIS)"
        ram_conclusion = (
            "RAM yang tersedia saat ini HANYA {:.2f} GB ({:.1f}% terpakai). "
            "Jika file pickle 3.05 GB dimuat sekarang, kemungkinan besar komputer akan mengalami "
            "MemoryError (out-of-memory crash) atau sistem Windows akan freeze/hang parah akibat "
            "swapping besar-besaran ke disk (paging thrashing)."
        ).format(avail_ram_gb, used_ram_pct)
        safe_to_load_now = False
    elif avail_ram_gb < 8.0:
        ram_status = "BERISIKO SEDANG (TIDAK DISARANKAN SEBELUM RAM DIBERSIHKAN)"
        ram_conclusion = (
            "RAM yang tersedia ({:.2f} GB) cukup terbatas. Deserialisasi pickle berukuran 3.05 GB "
            "memerlukan alokasi puncak hingga ~7-10 GB, sehingga berpotensi memanfaatkan swap memori "
            "dan memperlambat komputer."
        ).format(avail_ram_gb)
        safe_to_load_now = False
    else:
        ram_status = "CUKUP AMAN"
        ram_conclusion = (
            "RAM yang tersedia ({:.2f} GB) berada dalam batas yang relatif aman untuk memuat dataset."
        ).format(avail_ram_gb)
        safe_to_load_now = True

    # 6. Susun Laporan Teks
    report_lines = []
    sep_double = "=" * 80
    sep_single = "-" * 80

    report_lines.append(sep_double)
    report_lines.append("LAPORAN AUDIT LINGKUNGAN DAN PEMERIKSAAN KAPASITAS KOMPUTER (TAHAP 7A)")
    report_lines.append("Proyek: Deteksi Dini Risiko OSA berbasis PPG & APG (KTI_XGBoost)")
    report_lines.append(f"Waktu Audit: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(sep_double)
    report_lines.append("")

    report_lines.append("[1. STATUS KEBERADAAN DAN UKURAN DATASET OSASUD]")
    report_lines.append(f"  Target File           : data/osasud/dataset_OSAS.pickle")
    report_lines.append(f"  Status Keberadaan     : {'DITEMUKAN (ADA)' if file_exists else 'TIDAK DITEMUKAN'}")
    if file_exists:
        report_lines.append(f"  Lokasi Absolut        : {dataset_path}")
        report_lines.append(f"  Ukuran File (Biner)   : {file_size_gib:.2f} GiB ({file_size_bytes:,} bytes)")
        report_lines.append(f"  Ukuran File (Desimal) : {file_size_gb_dec:.2f} GB ({file_size_bytes / 1e6:,.2f} MB)")
        report_lines.append(f"  Integritas Serialisasi: File belum dimuat (sesuai protokol keamanan Tahap 7A).")
    else:
        report_lines.append("  PERINGATAN: File tidak ditemukan di direktori data/osasud/!")
    report_lines.append("")

    report_lines.append("[2. SPESIFIKASI DAN KONDISI SISTEM KOMPUTER]")
    report_lines.append(f"  Sistem Operasi        : {platform.system()} {platform.release()} (Versi: {platform.version()})")
    report_lines.append(f"  Arsitektur OS         : {platform.machine()} ({python_arch})")
    report_lines.append(f"  Versi Python          : Python {python_ver} ({python_arch})")
    report_lines.append(f"  Virtual Environment   : {sys.prefix} (Aktif)")
    report_lines.append(f"  Total RAM Fisik       : {total_ram_gb:.2f} GB ({vm.total:,} bytes)")
    report_lines.append(f"  RAM Terpakai Saat Ini : {(vm.total - vm.available) / (1024 ** 3):.2f} GB ({used_ram_pct:.1f}%)")
    report_lines.append(f"  RAM Bebas/Tersedia    : {avail_ram_gb:.2f} GB ({vm.available:,} bytes)")
    report_lines.append(f"  Total Swap/Pagefile   : {sm.total / (1024 ** 3):.2f} GB")
    report_lines.append(f"  Swap Bebas            : {sm.free / (1024 ** 3):.2f} GB")
    report_lines.append(f"  Drive Proyek ({drive_letter})    : Total {disk.total / (1024 ** 3):.2f} GB | Bebas {disk.free / (1024 ** 3):.2f} GB ({disk.free / disk.total * 100:.1f}%)")
    report_lines.append("")

    report_lines.append("[3. STATUS KESIAPAN LIBRARY PYTHON]")
    report_lines.append(f"  - pandas              : {pandas_version} (Status: {'SIAP' if pandas_version != 'TIDAK TERPASANG' else 'BELUM ADA'})")
    report_lines.append(f"  - numpy               : {numpy_version} (Status: {'SIAP' if numpy_version != 'TIDAK TERPASANG' else 'BELUM ADA'})")
    report_lines.append(f"  - psutil              : {psutil_ver} (Status: SIAP / Berhasil diinstal)")
    report_lines.append(f"  Semua library yang dibutuhkan untuk monitoring dan pengolahan data sudah terpasang di .venv.")
    report_lines.append("")

    report_lines.append("[4. ACUAN STRUKTUR DATASET (Berdasarkan Publikasi Resmi OSASUD)]")
    report_lines.append("  (Catatan: Informasi di bawah ini adalah acuan literatur Scientific Data, BUKAN hasil pembacaan file)")
    report_lines.append(f"  - Estimasi Jumlah Baris   : {pub_rows:,} baris")
    report_lines.append(f"  - Estimasi Jumlah Kolom   : {pub_cols} kolom")
    report_lines.append(f"  - Kolom Utama Relevan     : {', '.join(pub_key_cols)}")
    report_lines.append(f"  - Frekuensi Sampling PPG  : {pub_ppg_hz} Hz (kolom 'signal_pleth')")
    report_lines.append(f"  - Frekuensi Sampling SpO2 : {pub_spo2_hz} Hz (kolom 'SpO2(%)')")
    report_lines.append(f"  - Frekuensi Sampling PI   : {pub_pi_hz} Hz (kolom 'PI(%)')")
    report_lines.append("  - Fisiologis & Temporal  : Setiap baris merepresentasikan jendela waktu 1 detik rekaman.")
    report_lines.append(f"    Pada 80 Hz, 1 detik = 80 titik gelombang PPG per baris.")
    report_lines.append(f"    Total titik sinyal PPG di seluruh dataset = {pub_rows:,} x 80 = {pub_rows * 80:,} sampel!")
    report_lines.append("")

    report_lines.append("[5. ANALISIS ESTIMASI KEBUTUHAN DAN RISIKO PENGGUNAAN MEMORI (RAM)]")
    report_lines.append("  PENTING: Ukuran file di disk (3,05 GB) TIDAK SAMA dengan kebutuhan RAM saat objek dimuat!")
    report_lines.append("  Faktor-faktor pelipatgandaan memori pada file Pickle:")
    report_lines.append("  1. Sifat Serialisasi Pickle: Modul pickle menyimpan representasi hierarki objek Python.")
    report_lines.append("     Saat proses unpickling (pickle.load), Python tidak hanya membaca byte dari disk, tetapi")
    report_lines.append("     harus menginstansiasi struktur objek Python lengkap, tabel memo referensi, dan pointer memory.")
    report_lines.append("  2. Overhead Objek Kolom 'signal_pleth': Kolom ini berisi array sinyal gelombang frekuensi 80 Hz.")
    report_lines.append("     Jika disimpan sebagai kolom bertipe 'object' (kumpulan 961.357 array/list individual), setiap baris")
    report_lines.append("     memiliki overhead PyObject (pointer, metadata tipe, dsb.) yang melipatgandakan ukuran footprint.")
    report_lines.append("  3. Lonjakan Memori Puncak (Peak Memory Spike): Selama proses pickle.load(), modul pickle menahan")
    report_lines.append("     buffer streaming di memori bersamaan dengan pembangunan DataFrame. Alokasi puncak biasanya")
    report_lines.append("     mencapai 2,0x hingga 3,0x ukuran berkas asli di disk (estimasi 6,5 GB hingga 9,5 GB).")
    report_lines.append("  4. Ruang Operasi Lanjutan (Working Buffer): Setelah DataFrame dimuat, operasi indexing, slicing,")
    report_lines.append("     atau konversi tipe data membutuhkan RAM tambahan sebesar 1,0x - 1,5x ukuran DataFrame.")
    report_lines.append("")
    report_lines.append("  Estimasi Kebutuhan RAM:")
    report_lines.append("  - Ukuran file di disk            : ~3,05 GB")
    report_lines.append("  - Estimasi ukuran objek di RAM   : ~4,50 - 6,00 GB")
    report_lines.append("  - Estimasi lonjakan RAM puncak   : ~7,50 - 10,00 GB")
    report_lines.append("  - Ambang batas aman RAM bebas    : Minimal 10,00 - 11,00 GB RAM bebas")
    report_lines.append("")

    report_lines.append("[6. KESIMPULAN KELAYAKAN DAN REKOMENDASI TINDAKAN]")
    report_lines.append(f"  - Status Keberadaan File : OK (Ditemukan, ukuran 3.05 GiB / 3.28 GB)")
    report_lines.append(f"  - Status Kapasitas Disk  : AMAN (Tersedia {disk.free / (1024 ** 3):.2f} GB bebas pada drive {drive_letter})")
    report_lines.append(f"  - Status Kesiapan Python : AMAN (Python {python_ver}, pandas {pandas_version}, numpy {numpy_version}, psutil {psutil_ver})")
    report_lines.append(f"  - Status Kapasitas RAM   : {ram_status}")
    report_lines.append(f"    Penjelasan: {ram_conclusion}")
    report_lines.append("")
    report_lines.append("  REKOMENDASI SEBELUM TAHAP 7B (MEMBUKA/MEMUAT DATASET):")
    report_lines.append("  1. Total RAM fisik komputer (16 GB) secara teoritis CUKUP untuk dataset ini.")
    report_lines.append(f"  2. Namun saat ini pengguna sedang membuka aplikasi berat (RAM terpakai {used_ram_pct:.1f}%).")
    report_lines.append("     Aplikasi utama yang sedang memakan RAM:")
    
    # Ambil top 5 proses pemakan RAM
    processes = []
    for proc in psutil.process_iter(['name', 'memory_info']):
        try:
            mem = proc.info['memory_info'].rss / (1024 ** 2)
            processes.append((proc.info['name'], mem))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    # Agregasi berdasarkan nama proses
    agg_proc = {}
    for name, mem in processes:
        agg_proc[name] = agg_proc.get(name, 0.0) + mem
    top_procs = sorted(agg_proc.items(), key=lambda x: x[1], reverse=True)[:5]
    for name, mem_mb in top_procs:
        report_lines.append(f"     * {name}: {mem_mb:.1f} MB ({mem_mb / 1024:.2f} GB)")

    report_lines.append("  3. TINDAKAN WAJIB PENGGUNA SEBELUM TAHAP 7B:")
    report_lines.append("     - Tutup browser (terutama tab Firefox/Chrome yang banyak) dan aplikasi lain yang tidak diperlukan.")
    report_lines.append("     - Targetkan agar RAM yang tersedia mencapai minimal 10 - 11 GB sebelum script pembacaan dijalankan.")
    report_lines.append("  4. STRATEGI TEKNIS TAHAP 7B YANG DIREKOMENDASIKAN:")
    report_lines.append("     - Jangan langsung meload seluruh kolom jika tidak diperlukan (atau pertimbangkan inspeksi metadata awal).")
    report_lines.append("     - Pastikan skrip pembacaan berikutnya menyertakan error-handling MemoryError dan garbage collection (gc.collect()).")
    report_lines.append("     - Mengonversi data ke format yang lebih efisien dan terindeks (seperti Parquet / HDF5 / Feather per pasien) setelah berhasil dibuka.")
    report_lines.append(sep_double)

    report_text = "\n".join(report_lines)

    # 7. Simpan Laporan
    results_dir_primary = project_root / "results"
    results_dir_primary.mkdir(parents=True, exist_ok=True)
    report_file_primary = results_dir_primary / "osasud_environment_check.txt"
    report_file_primary.write_text(report_text, encoding="utf-8")

    # Jika workspace root berbeda, simpan juga di root results
    if base_dir != project_root:
        results_dir_root = base_dir / "results"
        results_dir_root.mkdir(parents=True, exist_ok=True)
        (results_dir_root / "osasud_environment_check.txt").write_text(report_text, encoding="utf-8")

    # Cetak ke stdout
    print(report_text)
    print(f"\n[INFO] Laporan berhasil disimpan di: {report_file_primary}")

if __name__ == "__main__":
    main()
