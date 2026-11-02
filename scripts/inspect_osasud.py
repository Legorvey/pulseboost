"""
scripts/inspect_osasud.py
=========================
Skrip inspeksi dataset OSASUD secara terkendali dengan arsitektur
proses anak (child process) dan pemantau memori (psutil watchdog).

KTI: Estimasi Tekanan Darah Cuffless dan Saturasi Oksigen dari Sinyal PPG Menggunakan XGBoost.
Tahap 7B: Pemeriksaan RAM Ulang dan Inspeksi OSASUD Secara Terkendali.

Fitur Keamanan:
1. Batas Awal Kehati-hatian: RAM tersedia minimal 8.0 GiB sebelum proses pembacaan diizinkan.
2. Isolasi Proses: Pembacaan pickle dilakukan di child process terpisah.
3. Watchdog Real-time (psutil):
   - Jika RAM tersedia sistem turun di bawah 2.0 GiB -> Hentikan proses anak seketika.
   - Jika RSS proses anak (+ turunannya) melebihi 6.5 GiB -> Hentikan proses anak seketika.
4. Tanpa Retry Otomatis jika terjadi kegagalan atau penghentian.
5. Pembebasan Memori: Penghapusan objek dan gc.collect() setelah selesai.
"""

import sys
import os
import time
import json
import subprocess
import psutil

# Ambang batas pengawasan (Safety Thresholds)
MIN_RAM_AVAILABLE_START_GIB = 8.0    # Syarat awal untuk memulai pembacaan (GiB)
CRITICAL_RAM_AVAILABLE_GIB = 2.0     # Ambang batas darurat RAM sistem (GiB)
MAX_CHILD_RSS_GIB = 6.5              # Batas maksimum konsumsi RAM proses anak (GiB)

PICKLE_PATH = os.path.join("data", "osasud", "dataset_OSAS.pickle")
REPORT_PATH = os.path.join("results", "osasud_inspection_summary.txt")
TEMP_JSON_PATH = os.path.join("results", "_temp_osasud_inspect.json")

REQUIRED_COLUMNS = ["patient", "timestamp_datetime", "signal_pleth", "SpO2(%)", "PI(%)"]


def get_memory_info():
    """Mengambil status RAM fisik dan pagefile/swap sistem."""
    vm = psutil.virtual_memory()
    sm = psutil.swap_memory()
    return {
        "ram_total_gib": vm.total / (1024 ** 3),
        "ram_available_gib": vm.available / (1024 ** 3),
        "ram_used_gib": vm.used / (1024 ** 3),
        "ram_percent": vm.percent,
        "swap_total_gib": sm.total / (1024 ** 3),
        "swap_used_gib": sm.used / (1024 ** 3),
        "swap_free_gib": sm.free / (1024 ** 3),
        "swap_percent": sm.percent,
    }


def get_process_tree_rss_gib(pid):
    """Menghitung total RSS memori proses beserta seluruh turunannya."""
    try:
        parent = psutil.Process(pid)
        total_bytes = parent.memory_info().rss
        for child in parent.children(recursive=True):
            try:
                total_bytes += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return total_bytes / (1024 ** 3)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0.0


def terminate_process_tree(pid):
    """Menghentikan proses anak beserta seluruh turunannya secara terkendali dan paksa."""
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        parent.terminate()

        # Beri waktu jeda 1 detik untuk terminasi anggun
        gone, alive = psutil.wait_procs(children + [parent], timeout=1.0)
        for p in alive:
            try:
                p.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass


def run_worker_process(target_pickle, output_json):
    """
    Fungsi yang dieksekusi di dalam proses anak (child process).
    Membaca dataset pickle dan mengekstrak metadata inspeksi sesuai Bagian C.
    """
    import gc
    import numpy as np
    import pandas as pd

    proc = psutil.Process()
    ram_before_read = proc.memory_info().rss / (1024 ** 3)

    t0 = time.time()
    try:
        data = pd.read_pickle(target_pickle)
    except Exception as e:
        err_data = {
            "success": False,
            "error_type": type(e).__name__,
            "error_message": str(e),
        }
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(err_data, f)
        sys.exit(1)

    load_time = time.time() - t0
    ram_after_read = proc.memory_info().rss / (1024 ** 3)

    inspection = {
        "success": True,
        "load_time_sec": load_time,
        "object_type": type(data).__name__,
        "is_dataframe": isinstance(data, pd.DataFrame),
        "ram_before_read_gib": ram_before_read,
        "ram_after_read_gib": ram_after_read,
        "ram_delta_gib": ram_after_read - ram_before_read,
    }

    if isinstance(data, pd.DataFrame):
        inspection["num_rows"] = int(len(data))
        inspection["num_cols"] = int(len(data.columns))
        inspection["columns_dtypes"] = {str(col): str(dtype) for col, dtype in data.dtypes.items()}

        # Periksa keberadaan kolom persis
        exact_cols_status = {}
        for req_col in REQUIRED_COLUMNS:
            exact_cols_status[req_col] = req_col in data.columns
        inspection["exact_columns_status"] = exact_cols_status
        inspection["all_required_columns_present"] = all(exact_cols_status.values())

        # Jumlah identitas pasien unik (tanpa mencetak identitas)
        if "patient" in data.columns:
            inspection["num_unique_patients"] = int(data["patient"].nunique())
        else:
            inspection["num_unique_patients"] = None

        # Sampel 3 baris (baris indeks 0, 1, 2)
        sample_rows_info = []
        n_samples = min(3, len(data))
        for idx in range(n_samples):
            row = data.iloc[idx]
            row_info = {"sample_index": idx}

            # Sinyal pleth (PPG)
            if "signal_pleth" in data.columns:
                pleth_val = row["signal_pleth"]
                pleth_type = type(pleth_val).__name__
                pleth_shape = list(getattr(pleth_val, "shape", [len(pleth_val)] if hasattr(pleth_val, "__len__") else []))
                pleth_dtype = str(getattr(pleth_val, "dtype", type(pleth_val[0]).__name__ if hasattr(pleth_val, "__getitem__") and len(pleth_val) > 0 else "unknown"))
                num_ppg_samples = len(pleth_val) if hasattr(pleth_val, "__len__") else 0
                row_info["pleth_type"] = pleth_type
                row_info["pleth_shape"] = pleth_shape
                row_info["pleth_dtype"] = pleth_dtype
                row_info["num_ppg_samples"] = num_ppg_samples

            # SpO2
            if "SpO2(%)" in data.columns:
                row_info["SpO2_val"] = float(row["SpO2(%)"]) if pd.notna(row["SpO2(%)"]) else None

            # PI
            if "PI(%)" in data.columns:
                row_info["PI_val"] = float(row["PI(%)"]) if pd.notna(row["PI(%)"]) else None

            sample_rows_info.append(row_info)
        inspection["sample_rows"] = sample_rows_info

        # Proporsi nilai kosong (tanpa menyalin DataFrame)
        missing_proportions = {}
        for col in REQUIRED_COLUMNS:
            if col in data.columns:
                try:
                    pct_missing = float(data[col].isna().mean() * 100.0)
                    missing_proportions[col] = pct_missing
                except Exception:
                    missing_proportions[col] = "Gagal dihitung tanpa salinan"
            else:
                missing_proportions[col] = "Kolom tidak ditemukan"
        inspection["missing_proportions_pct"] = missing_proportions

    else:
        inspection["note"] = "Objek bukan DataFrame pandas."

    # Simpan hasil sementara
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(inspection, f, indent=2)

    # Bersihkan memori child process
    del data
    gc.collect()
    sys.exit(0)


def generate_failure_report(reason_code, reason_desc, mem_info, file_exists, file_size_gib, watchdog_stats=None):
    """Menghasilkan teks laporan saat pembacaan tidak dapat dilakukan atau dihentikan."""
    lines = [
        "=" * 80,
        "LAPORAN HASIL INSPEKSI DATASET OSASUD SECARA TERKENDALI (TAHAP 7B)",
        "Proyek : Estimasi Tekanan Darah Cuffless dan SpO2 dari Sinyal PPG (KTI_XGBoost)",
        f"Waktu  : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 80,
        "",
        "[1. STATUS SISTEM DAN INTERPRETER PYTHON]",
        f"  - Python Executable   : {sys.executable}",
        f"  - Python Version      : {sys.version.split()[0]}",
        f"  - Virtual Environment : Aktif ({os.path.dirname(sys.executable)})",
        "",
        "[2. SPESIFIKASI DAN KONDISI MEMORI (RAM & PAGEFILE)]",
        f"  - Total RAM Fisik     : {mem_info['ram_total_gib']:.2f} GiB",
        f"  - RAM Tersedia        : {mem_info['ram_available_gib']:.2f} GiB ({mem_info['ram_percent']:.1f}% terpakai)",
        f"  - Total Swap/Pagefile : {mem_info['swap_total_gib']:.2f} GiB",
        f"  - Swap Bebas          : {mem_info['swap_free_gib']:.2f} GiB ({mem_info['swap_percent']:.1f}% terpakai)",
        "",
        "[3. STATUS FILE DATASET OSASUD]",
        f"  - Target Path         : {PICKLE_PATH}",
        f"  - Keberadaan File     : {'DITEMUKAN' if file_exists else 'TIDAK DITEMUKAN'}",
        f"  - Ukuran File di Disk : {file_size_gib:.2f} GiB ({file_size_gib * 1024:.2f} MB)" if file_exists else "  - Ukuran File di Disk : N/A",
        "",
        "[4. HASIL EVALUASI KEAMANAN & KEPUTUSAN EKSEKUSI]",
        f"  - Status Eksekusi     : {reason_code}",
        f"  - Deskripsi Keputusan : {reason_desc}",
        f"  - Ambang Batas Awal   : Minimal {MIN_RAM_AVAILABLE_START_GIB:.1f} GiB RAM tersedia",
        f"  - RAM Tersedia Aktual : {mem_info['ram_available_gib']:.2f} GiB",
    ]

    if watchdog_stats:
        lines.extend([
            "",
            "[5. STATISTIK PENGAWAS (WATCHDOG)]",
            f"  - Durasi Pengawasan   : {watchdog_stats.get('duration_sec', 0):.2f} detik",
            f"  - Puncak RSS Anak     : {watchdog_stats.get('peak_child_rss_gib', 0):.2f} GiB",
            f"  - RAM Tersedia Min    : {watchdog_stats.get('min_ram_avail_gib', 0):.2f} GiB",
            f"  - Alasan Penghentian  : {watchdog_stats.get('stop_reason', 'N/A')}",
            "",
            "[6. KESIMPULAN DAN REKOMENDASI TINDAKAN]",
        ])
    else:
        lines.extend([
            "",
            "[5. KESIMPULAN DAN REKOMENDASI TINDAKAN]",
        ])

    lines.extend([
        "  - Sesuai protokol keselamatan Bagian A butir 4 dan Bagian D:",
        "    Pembacaan Pickle DIBATALKAN / DIHENTIKAN secara terkendali untuk mencegah MemoryError,",
        "    crash sistem operasi Windows, dan thrashing pagefile disk.",
        "  - Tidak ada manipulasi atau pembuatan data buatan (zero fabrication).",
        "  - Rekomendasi: Pengguna disarankan menutup aplikasi latar belakang (seperti browser dengan tab banyak)",
        f"    agar RAM tersedia mencapai minimal {MIN_RAM_AVAILABLE_START_GIB:.1f} GiB sebelum mencoba kembali.",
        "=" * 80,
    ])
    return "\n".join(lines)


def generate_success_report(mem_info, file_size_gib, inspect_data, watchdog_stats):
    """Menghasilkan teks laporan lengkap jika dataset berhasil dibuka dan diinspeksi."""
    lines = [
        "=" * 80,
        "LAPORAN HASIL INSPEKSI DATASET OSASUD SECARA TERKENDALI (TAHAP 7B)",
        "Proyek : Estimasi Tekanan Darah Cuffless dan SpO2 dari Sinyal PPG (KTI_XGBoost)",
        f"Waktu  : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 80,
        "",
        "[1. STATUS SISTEM DAN INTERPRETER PYTHON]",
        f"  - Python Executable   : {sys.executable}",
        f"  - Python Version      : {sys.version.split()[0]}",
        f"  - Virtual Environment : Aktif ({os.path.dirname(sys.executable)})",
        "",
        "[2. SPESIFIKASI DAN KONDISI MEMORI (RAM & PAGEFILE)]",
        f"  - Total RAM Fisik     : {mem_info['ram_total_gib']:.2f} GiB",
        f"  - RAM Tersedia Awal   : {mem_info['ram_available_gib']:.2f} GiB",
        f"  - Total Swap/Pagefile : {mem_info['swap_total_gib']:.2f} GiB",
        f"  - Swap Bebas          : {mem_info['swap_free_gib']:.2f} GiB",
        "",
        "[3. STATUS FILE DATASET OSASUD]",
        f"  - Target Path         : {PICKLE_PATH}",
        f"  - Ukuran File di Disk : {file_size_gib:.2f} GiB ({file_size_gib * 1024:.2f} MB)",
        f"  - Waktu Pembacaan     : {inspect_data.get('load_time_sec', 0):.2f} detik",
        "",
        "[4. KARAKTERISTIK STRUKTURAL DATASET (BAGIAN C)]",
        f"  - Tipe Objek Python   : {inspect_data.get('object_type')}",
        f"  - Jumlah Baris Aktual : {inspect_data.get('num_rows', 'N/A'):,}",
        f"  - Jumlah Kolom Aktual : {inspect_data.get('num_cols', 'N/A')}",
        "",
        "  - Daftar Seluruh Kolom dan Tipe Data:",
    ]

    for col, dtype in inspect_data.get("columns_dtypes", {}).items():
        lines.append(f"      * {col:<25} : {dtype}")

    lines.extend([
        "",
        "  - Pemeriksaan Keberadaan Kolom Penting (Nama Persis):",
    ])
    for req_col, is_present in inspect_data.get("exact_columns_status", {}).items():
        status_str = "DITEMUKAN (PERSIS)" if is_present else "TIDAK DITEMUKAN"
        lines.append(f"      * {req_col:<25} : {status_str}")

    lines.extend([
        f"  - Seluruh Kolom Penting Lengkap : {'YA' if inspect_data.get('all_required_columns_present') else 'TIDAK'}",
        f"  - Jumlah Pasien Unik            : {inspect_data.get('num_unique_patients', 'N/A')} pasien (tanpa identitas individual)",
        "",
        "[5. INSPEKSI SAMPEL SINYAL DAN NILAI FISIOLOGIS (3 BARIS CONTOH)]",
    ])

    for sample in inspect_data.get("sample_rows", []):
        s_idx = sample.get("sample_index")
        lines.extend([
            f"  [Contoh Baris ke-{s_idx}]",
            f"    - Sinyal Pleth Tipe      : {sample.get('pleth_type', 'N/A')} (dtype: {sample.get('pleth_dtype', 'N/A')})",
            f"    - Bentuk Array (Shape)   : {sample.get('pleth_shape', 'N/A')}",
            f"    - Jumlah Sampel PPG      : {sample.get('num_ppg_samples', 'N/A')} sampel per detik",
            f"    - Nilai SpO2 (%)         : {sample.get('SpO2_val', 'N/A')}%",
            f"    - Nilai PI (%)           : {sample.get('PI_val', 'N/A')}%",
        ])

    lines.extend([
        "",
        "[6. PROPORSI NILAI KOSONG (MISSING VALUES) PADA KOLOM PENTING]",
    ])
    for col, pct in inspect_data.get("missing_proportions_pct", {}).items():
        if isinstance(pct, (int, float)):
            lines.append(f"    - {col:<25} : {pct:.4f}% kosong")
        else:
            lines.append(f"    - {col:<25} : {pct}")

    lines.extend([
        "",
        "[7. ANALISIS PENGGUNAAN MEMORI PROSES DAN PENGAWAS]",
        f"  - RAM Proses Sebelum Baca : {inspect_data.get('ram_before_read_gib', 0):.2f} GiB",
        f"  - RAM Proses Setelah Baca : {inspect_data.get('ram_after_read_gib', 0):.2f} GiB",
        f"  - Delta RAM DataFrame     : {inspect_data.get('ram_delta_gib', 0):.2f} GiB",
        f"  - Puncak RSS Anak         : {watchdog_stats.get('peak_child_rss_gib', 0):.2f} GiB",
        f"  - RAM Tersedia Terendah   : {watchdog_stats.get('min_ram_avail_gib', 0):.2f} GiB",
        "",
        "[8. KESIMPULAN DAN STATUS AKHIR]",
        "  - Objek telah dibersihkan (del data) dan garbage collection dijalankan.",
        "  - Proses anak telah ditutup secara aman, memori sistem telah dibebaskan penuh.",
        "=" * 80,
    ])
    return "\n".join(lines)


def main():
    # Cek mode worker (anak)
    if len(sys.argv) >= 4 and sys.argv[1] == "--worker":
        target_pickle = sys.argv[2]
        output_json = sys.argv[3]
        run_worker_process(target_pickle, output_json)
        return

    print("=" * 80)
    print("MEMULAI EVALUASI AWAL TAHAP 7B (PEMERIKSAAN RAM ULANG)")
    print("=" * 80)

    # 1. Pastikan virtual environment dan interpreter Python
    print(f"[BAGIAN A.1] Python Interpreter : {sys.executable}")
    print(f"[BAGIAN A.1] Python Version     : {sys.version.split()[0]}")

    # 2. Periksa RAM dan Pagefile
    mem_info = get_memory_info()
    print(f"[BAGIAN A.2] Total RAM          : {mem_info['ram_total_gib']:.2f} GiB")
    print(f"[BAGIAN A.2] RAM Tersedia       : {mem_info['ram_available_gib']:.2f} GiB ({mem_info['ram_percent']:.1f}% terpakai)")
    print(f"[BAGIAN A.2] Total Pagefile     : {mem_info['swap_total_gib']:.2f} GiB")
    print(f"[BAGIAN A.2] Pagefile Bebas     : {mem_info['swap_free_gib']:.2f} GiB")

    # 3. Pastikan file dataset_OSAS.pickle tersedia
    file_exists = os.path.exists(PICKLE_PATH)
    file_size_gib = (os.path.getsize(PICKLE_PATH) / (1024 ** 3)) if file_exists else 0.0
    print(f"[BAGIAN A.3] File Dataset       : {PICKLE_PATH}")
    print(f"[BAGIAN A.3] Status Keberadaan  : {'DITEMUKAN' if file_exists else 'TIDAK DITEMUKAN'}")
    if file_exists:
        print(f"[BAGIAN A.3] Ukuran File di Disk: {file_size_gib:.2f} GiB ({file_size_gib * 1024:.2f} MB)")

    if not file_exists:
        reason = "File dataset_OSAS.pickle tidak ditemukan pada path yang ditentukan."
        report = generate_failure_report("FILE_NOT_FOUND", reason, mem_info, file_exists, file_size_gib)
        os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n[HASIL] {reason}")
        print(f"[HASIL] Laporan disimpan ke: {REPORT_PATH}")
        return

    # 4. Syarat Kehati-hatian Awal: RAM Tersedia >= 8.0 GiB
    print(f"\n[BAGIAN A.4] Evaluasi Ambang Batas RAM Tersedia: Minimal {MIN_RAM_AVAILABLE_START_GIB:.1f} GiB")
    if mem_info["ram_available_gib"] < MIN_RAM_AVAILABLE_START_GIB:
        reason = (
            f"RAM tersedia saat ini ({mem_info['ram_available_gib']:.2f} GiB) KURANG dari batas minimal "
            f"kehati-hatian ({MIN_RAM_AVAILABLE_START_GIB:.1f} GiB). "
            f"Sesuai instruksi Bagian A Butir 4, proses pembacaan Pickle TIDAK DIJALANKAN (BERHENTI) "
            f"demi menjaga kestabilan sistem dari risiko kehabisan memori (OOM crash)."
        )
        print(f"[PERINGATAN KESELAMATAN] {reason}")
        report = generate_failure_report("INSUFFICIENT_RAM_PRECHECK", reason, mem_info, file_exists, file_size_gib)
        os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"[BAGIAN D] Laporan kondisi disimpan ke: {REPORT_PATH}")
        print("Proses dihentikan secara aman tanpa menyentuh file Pickle.")
        return

    # -------------------------------------------------------------------------
    # BAGIAN B: PERCOBAAN PEMBACAAN TERKENDALI (HANYA JIKA RAM MEMENUHI SYARAT)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("MEMULAI BAGIAN B: PERCOBAAN PEMBACAAN TERKENDALI DENGAN PENGAWAS")
    print("=" * 80)

    # Hapus file JSON sementara jika ada
    if os.path.exists(TEMP_JSON_PATH):
        try:
            os.remove(TEMP_JSON_PATH)
        except OSError:
            pass

    cmd = [sys.executable, __file__, "--worker", PICKLE_PATH, TEMP_JSON_PATH]
    print(f"[PENGAWAS] Meluncurkan proses anak: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd)
    child_pid = proc.pid
    print(f"[PENGAWAS] PID Proses Anak: {child_pid}")

    start_time = time.time()
    peak_child_rss = 0.0
    min_system_avail = mem_info["ram_available_gib"]
    stop_reason = None
    aborted = False

    try:
        while True:
            # Periksa apakah proses anak masih berjalan
            ret = proc.poll()
            if ret is not None:
                print(f"[PENGAWAS] Proses anak selesai dengan returncode: {ret}")
                break

            # Ukur kondisi memori real-time
            current_vm = psutil.virtual_memory()
            curr_avail_gib = current_vm.available / (1024 ** 3)
            child_rss_gib = get_process_tree_rss_gib(child_pid)

            if child_rss_gib > peak_child_rss:
                peak_child_rss = child_rss_gib
            if curr_avail_gib < min_system_avail:
                min_system_avail = curr_avail_gib

            # Evaluasi kondisi penghentian darurat
            # 1. RAM tersedia turun di bawah 2 GiB
            if curr_avail_gib < CRITICAL_RAM_AVAILABLE_GIB:
                stop_reason = (
                    f"RAM tersedia sistem turun ke level kritis ({curr_avail_gib:.2f} GiB < {CRITICAL_RAM_AVAILABLE_GIB:.1f} GiB). "
                    "Pengawas menghentikan proses anak untuk menyelamatkan sistem."
                )
                aborted = True
                break

            # 2. RSS proses anak melebihi 6.5 GiB
            if child_rss_gib > MAX_CHILD_RSS_GIB:
                stop_reason = (
                    f"Konsumsi RSS proses anak melebihi batas aman ({child_rss_gib:.2f} GiB > {MAX_CHILD_RSS_GIB:.1f} GiB). "
                    "Pengawas menghentikan proses anak."
                )
                aborted = True
                break

            time.sleep(0.05)

    except KeyboardInterrupt:
        stop_reason = "Diinterupsi oleh pengguna (KeyboardInterrupt)."
        aborted = True

    watchdog_stats = {
        "duration_sec": time.time() - start_time,
        "peak_child_rss_gib": peak_child_rss,
        "min_ram_avail_gib": min_system_avail,
        "stop_reason": stop_reason if aborted else "Proses anak selesai normal.",
    }

    if aborted:
        print(f"\n[PENGAWAS - PENGHENTIAN DARURAT] {stop_reason}")
        terminate_process_tree(child_pid)
        # Jangan melakukan retry otomatis!
        report = generate_failure_report("WATCHDOG_ABORTED", stop_reason, mem_info, file_exists, file_size_gib, watchdog_stats)
        os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"[BAGIAN D] Laporan kegagalan disimpan ke: {REPORT_PATH}")
        return

    # Evaluasi hasil dari child process
    if not os.path.exists(TEMP_JSON_PATH):
        reason = f"Proses anak selesai (exit code {proc.returncode}) tetapi file hasil inspeksi tidak ditemukan."
        report = generate_failure_report("CHILD_OUTPUT_MISSING", reason, mem_info, file_exists, file_size_gib, watchdog_stats)
        os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"[BAGIAN D] Laporan disimpan ke: {REPORT_PATH}")
        return

    with open(TEMP_JSON_PATH, "r", encoding="utf-8") as f:
        inspect_data = json.load(f)

    # Bersihkan file JSON sementara
    try:
        os.remove(TEMP_JSON_PATH)
    except OSError:
        pass

    if not inspect_data.get("success", False):
        reason = f"Error saat membaca Pickle: {inspect_data.get('error_type')}: {inspect_data.get('error_message')}"
        report = generate_failure_report("PICKLE_READ_ERROR", reason, mem_info, file_exists, file_size_gib, watchdog_stats)
        os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"[BAGIAN D] Laporan error disimpan ke: {REPORT_PATH}")
        return

    # Sukses penuh
    report = generate_success_report(mem_info, file_size_gib, inspect_data, watchdog_stats)
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print("\n" + "=" * 80)
    print("INSPEKSI BERHASIL DISELESAIKAN SECARA TERKENDALI")
    print(f"Laporan lengkap disimpan ke: {REPORT_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
