"""
scripts/run_osasud_safe.py
==========================
Skrip pengawas memori (watchdog supervisor) untuk menjalankan inspeksi OSASUD
secara terkendali dan aman di luar IDE.

Fitur & Protokol Keamanan:
1. Pra-pemeriksaan RAM:
   - Jika RAM sistem tersedia < 7.0 GiB, proses DIBATALKAN seketika dan dilaporkan.
2. Isolasi Proses Anak (Subprocess):
   - Menjalankan scripts/inspect_osasud_standalone.py sebagai child process.
3. Pemantauan Real-Time (psutil watchdog):
   - Jika RAM sistem tersedia turun < 2.0 GiB saat proses berlangsung:
     Hentikan proses anak beserta seluruh turunannya seketika.
   - Jika RSS proses anak (+ turunan) melebihi > 7.0 GiB:
     Hentikan proses anak beserta seluruh turunannya seketika.
4. Batas Waktu Wajar (Timeout):
   - Batas maksimal 600 detik (10 menit) untuk menghindari proses menggantung.
5. Tanpa Retry Otomatis:
   - Tidak melakukan percobaan ulang jika terjadi pembatalan atau kegagalan.
6. Pencatatan Status & Kejujuran Risiko:
   - Mencatat status keberhasilan, pembatalan, atau kegagalan ke terminal dan
     results/osasud_inspection_summary.txt.
   - Ambang pengaman adalah mitigasi praktis (best-effort safeguard), bukan
     jaminan mutlak bebas crash OS.

KTI: Estimasi Tekanan Darah Cuffless dan Saturasi Oksigen dari Sinyal PPG Menggunakan XGBoost.
"""

import os
import sys
import time
import threading
import subprocess
import platform
from pathlib import Path
import psutil

# Ambang Batas Keamanan Memori (GiB)
MIN_RAM_AVAILABLE_START_GIB = 7.0   # Batas minimal RAM tersedia untuk memulai eksekusi
CRITICAL_SYSTEM_RAM_GIB = 2.0       # Batas darurat RAM sistem saat proses berjalan
MAX_CHILD_RSS_GIB = 7.0             # Batas konsumsi RSS proses anak
TIMEOUT_SECONDS = 600.0             # Batas waktu maksimal eksekusi (10 menit)
MONITOR_INTERVAL_SECONDS = 0.25     # Interval polling watchdog


def find_project_root() -> Path:
    """Menemukan direktori root proyek KTI_XGBoost."""
    script_dir = Path(__file__).resolve().parent
    parent_dir = script_dir.parent
    if (parent_dir / "data" / "osasud" / "dataset_OSAS.pickle").is_file():
        return parent_dir
    if (parent_dir.parent / "data" / "osasud" / "dataset_OSAS.pickle").is_file():
        return parent_dir.parent
    if (parent_dir / "KTI_XGBoost" / "data" / "osasud" / "dataset_OSAS.pickle").is_file():
        return (parent_dir / "KTI_XGBoost").resolve()
    return parent_dir


def find_python_interpreter(project_root: Path) -> Path:
    """Menemukan interpreter Python virtual environment yang valid."""
    candidates = [
        project_root / ".venv" / "Scripts" / "python.exe",
        project_root.parent / ".venv" / "Scripts" / "python.exe",
        project_root / "KTI_XGBoost" / ".venv" / "Scripts" / "python.exe",
        Path(sys.executable),
    ]
    for cand in candidates:
        if cand.is_file():
            return cand.resolve()
    return Path(sys.executable).resolve()


def get_process_tree_rss_gib(pid: int) -> float:
    """Menghitung total penggunaan RSS memori untuk proses dan seluruh turunannya."""
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


def terminate_process_tree(pid: int):
    """Menghentikan proses anak dan seluruh turunannya secara anggun lalu paksa."""
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        try:
            parent.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        gone, alive = psutil.wait_procs(children + [parent], timeout=2.0)
        for p in alive:
            try:
                p.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass


def save_cancellation_report(report_path: Path, reason: str, vm: psutil.virtual_memory, dataset_path: Path):
    """Menyimpan laporan pembatalan jika RAM awal tidak mencukupi."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    ram_total_gib = vm.total / (1024 ** 3)
    ram_avail_gib = vm.available / (1024 ** 3)
    sm = psutil.swap_memory()
    swap_total_gib = sm.total / (1024 ** 3)
    swap_free_gib = sm.free / (1024 ** 3)

    file_size_str = "N/A"
    if dataset_path.is_file():
        file_size_gib = dataset_path.stat().st_size / (1024 ** 3)
        file_size_str = f"{file_size_gib:.2f} GiB"

    lines = [
        "=" * 80,
        "LAPORAN PENGAWAS MEMORI DATASET OSASUD - STATUS: DIBATALKAN",
        "Proyek  : Estimasi Tekanan Darah Cuffless dan SpO2 dari Sinyal PPG (KTI_XGBoost)",
        f"Waktu   : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Python  : {sys.executable} (v{platform.python_version()})",
        "=" * 80,
        "",
        "[1. STATUS EVALUASI PRA-PEMERIKSAAN RAM]",
        "  - Status Eksekusi         : DIBATALKAN (CANCELLED_INSUFFICIENT_RAM)",
        f"  - Alasan Pembatalan       : {reason}",
        f"  - Ambang Minimal Memulai  : {MIN_RAM_AVAILABLE_START_GIB:.2f} GiB RAM tersedia",
        f"  - RAM Tersedia Aktual     : {ram_avail_gib:.2f} GiB (Terpakai: {vm.percent:.1f}%)",
        f"  - Total RAM Fisik Sistem  : {ram_total_gib:.2f} GiB",
        f"  - Total Pagefile / Swap   : {swap_total_gib:.2f} GiB (Bebas: {swap_free_gib:.2f} GiB)",
        "",
        "[2. INFORMASI DATASET TARGET]",
        f"  - File Target             : {dataset_path}",
        f"  - Ukuran Berkas di Disk   : {file_size_str}",
        "",
        "[3. PENJELASAN RISIKO KETERBATASAN MEMORI & KEJUJURAN TEKNIS]",
        "  1. Dataset resmi OSASUD (dataset_OSAS.pickle) memiliki ukuran ~3.28 GB di disk.",
        "  2. Proses deserialisasi pickle membutuhkan alokasi memori dinamis puncak hingga 7-10 GiB.",
        f"  3. Dengan RAM tersedia saat ini ({ram_avail_gib:.2f} GiB < {MIN_RAM_AVAILABLE_START_GIB:.2f} GiB), sistem berisiko",
        "     tinggi mengalami OS freeze/hang parah, thrashing pagefile disk, atau MemoryError.",
        "  4. Ambang batas ini merupakan pengaman praktis (best-effort safeguard), bukan jaminan",
        "     mutlak bebas crash OS jika kondisi memori terlalu rendah.",
        "  5. Demi stabilitas sistem, pembacaan TIDAK DILANJUTKAN dan TIDAK ADA RETRY OTOMATIS.",
        "",
        "[4. REKOMENDASI TINDAKAN BAGI PENGGUNA]",
        "  1. Tutup aplikasi yang menggunakan RAM besar (IDE seperti VS Code / PyCharm, browser dengan",
        "     banyak tab, aplikasi background lainnya).",
        "  2. Pastikan RAM yang tersedia mencapai minimal 7.0 GiB.",
        "  3. Jalankan kembali launcher mandiri melalui: run_osasud_inspection.cmd",
        "=" * 80,
    ]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_abort_report(report_path: Path, reason: str, duration: float, peak_child_rss: float, min_sys_avail: float):
    """Menyimpan laporan penghentian darurat jika watchdog menghentikan proses anak."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    vm = psutil.virtual_memory()

    lines = [
        "=" * 80,
        "LAPORAN PENGAWAS MEMORI DATASET OSASUD - STATUS: DIHENTIKAN PENGAWAS",
        "Proyek  : Estimasi Tekanan Darah Cuffless dan SpO2 dari Sinyal PPG (KTI_XGBoost)",
        f"Waktu   : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 80,
        "",
        "[1. STATUS PENGHENTIAN DARURAT (EMERGENCY ABORT)]",
        "  - Status Eksekusi         : DIHENTIKAN_PENGAWAS (WATCHDOG_TERMINATED)",
        f"  - Alasan Penghentian      : {reason}",
        f"  - Durasi Berjalan         : {duration:.2f} detik",
        f"  - Puncak RSS Anak         : {peak_child_rss:.2f} GiB (Batas Aman: {MAX_CHILD_RSS_GIB:.2f} GiB)",
        f"  - RAM Tersedia Terendah   : {min_sys_avail:.2f} GiB (Ambang Kritis: {CRITICAL_SYSTEM_RAM_GIB:.2f} GiB)",
        f"  - RAM Sistem Akhir        : {vm.available / (1024 ** 3):.2f} GiB tersedia dari {vm.total / (1024 ** 3):.2f} GiB",
        "",
        "[2. PERNYATAAN KEJUJURAN TEKNIS & KESIMPULAN]",
        "  1. Mekanisme pengawas (psutil watchdog) berhasil memotong proses anak sebelum terjadi crash",
        "     total atau OS freeze.",
        "  2. Ambang batas pengaman merupakan mitigasi praktis (best-effort safeguard), bukan jaminan",
        "     mutlak bebas crash OS.",
        "  3. Sesuai protokol keselamatan, TIDAK ADA PERCOBAAN ULANG OTOMATIS (no auto-retry).",
        "  4. Keterbatasan fisik RAM 16 GB mengharuskan penutupan seluruh program lain secara menyeluruh",
        "     sebelum mencoba kembali, atau mempertimbangkan metode pembacaan streaming/chunking.",
        "=" * 80,
    ]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def append_watchdog_success_log(report_path: Path, duration: float, peak_child_rss: float, min_sys_avail: float):
    """Menambahkan ringkasan log pengawasan memori ke laporan akhir jika proses sukses."""
    lines = [
        "",
        "=" * 80,
        "[9. LOG MONITORING PENGAWAS MEMORI (WATCHDOG MONITORING LOG)]",
        "=" * 80,
        "  - Status Pengawasan       : SUKSES (NORMAL TERMINATION)",
        f"  - Durasi Eksekusi         : {duration:.2f} detik (Batas Timeout: {TIMEOUT_SECONDS:.0f} detik)",
        f"  - Puncak Konsumsi RSS Anak: {peak_child_rss:.2f} GiB (Batas Aman: {MAX_CHILD_RSS_GIB:.2f} GiB)",
        f"  - RAM Sistem Tersedia Min : {min_sys_avail:.2f} GiB (Ambang Kritis: {CRITICAL_SYSTEM_RAM_GIB:.2f} GiB)",
        "  - Pelanggaran Ambang      : NIHIL (Seluruh kriteria keselamatan dipatuhi)",
        "  - Status Retry            : Tidak diperlukan percobaan ulang (berhasil pada run pertama)",
        "  - Validasi Keamanan       : Pengaman praktis berhasil menjaga integritas sistem operasi",
        "                              sepanjang deserialisasi dan inspeksi dataset OSASUD.",
        "=" * 80,
    ]
    try:
        with open(report_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except Exception as e:
        print(f"[WARNING] Gagal menambahkan log watchdog ke {report_path}: {e}", flush=True)


def main():
    project_root = find_project_root()
    python_exe = find_python_interpreter(project_root)
    inspect_script = project_root / "scripts" / "inspect_osasud_standalone.py"
    if not inspect_script.is_file():
        # Coba alternatif path
        alt_script = project_root / "KTI_XGBoost" / "scripts" / "inspect_osasud_standalone.py"
        if alt_script.is_file():
            inspect_script = alt_script

    dataset_path = project_root / "data" / "osasud" / "dataset_OSAS.pickle"
    if not dataset_path.is_file():
        alt_dataset = project_root / "KTI_XGBoost" / "data" / "osasud" / "dataset_OSAS.pickle"
        if alt_dataset.is_file():
            dataset_path = alt_dataset

    report_path = project_root / "results" / "osasud_inspection_summary.txt"

    print("=" * 80, flush=True)
    print("    PENGAWAS MEMORI AMAN OSASUD (SAFE SUPERVISOR WATCHDOG)", flush=True)
    print("    Proyek KTI_XGBoost: Estimasi Tekanan Darah Cuffless & SpO2", flush=True)
    print("=" * 80, flush=True)
    print(f"Direktori Proyek    : {project_root}", flush=True)
    print(f"Interpreter Python  : {python_exe}", flush=True)
    print(f"Skrip Target Anak   : {inspect_script}", flush=True)
    print(f"File Dataset Resmi  : {dataset_path}", flush=True)
    print(f"File Laporan Akhir  : {report_path}", flush=True)
    print("-" * 80, flush=True)

    # 1. Pra-pemeriksaan RAM Tersedia Sebelum Membuka Dataset
    vm = psutil.virtual_memory()
    ram_total_gib = vm.total / (1024 ** 3)
    ram_avail_gib = vm.available / (1024 ** 3)

    print(f"[STATUS SISTEM] Total RAM Fisik : {ram_total_gib:.2f} GiB", flush=True)
    print(f"[STATUS SISTEM] RAM Tersedia    : {ram_avail_gib:.2f} GiB ({vm.percent:.1f}% terpakai)", flush=True)
    print(f"[PENGAMAN] Ambang Mulai Minimal : {MIN_RAM_AVAILABLE_START_GIB:.2f} GiB RAM Tersedia", flush=True)
    print(f"[PENGAMAN] Ambang Darurat Sistem: {CRITICAL_SYSTEM_RAM_GIB:.2f} GiB RAM Tersedia", flush=True)
    print(f"[PENGAMAN] Batas Maks RSS Anak  : {MAX_CHILD_RSS_GIB:.2f} GiB RSS", flush=True)
    print(f"[PENGAMAN] Batas Waktu (Timeout): {TIMEOUT_SECONDS:.0f} detik", flush=True)
    print("-" * 80, flush=True)

    # Cek Syarat Awal (Butir 4: Jika RAM < 7 GiB, batalkan dan laporkan)
    if ram_avail_gib < MIN_RAM_AVAILABLE_START_GIB:
        cancel_msg = (
            f"RAM tersedia saat ini ({ram_avail_gib:.2f} GiB) KURANG dari batas minimal "
            f"kehati-hatian ({MIN_RAM_AVAILABLE_START_GIB:.2f} GiB). "
            "Demi mencegah out-of-memory crash atau Windows freeze, pembacaan DIBATALKAN."
        )
        print(f"[PEMBATALAN KEAMANAN] {cancel_msg}", flush=True)
        print("Mencatat laporan pembatalan ke results/osasud_inspection_summary.txt...", flush=True)
        save_cancellation_report(report_path, cancel_msg, vm, dataset_path)
        print(f"[INFO] Laporan pembatalan tersimpan di: {report_path}", flush=True)
        print("TIDAK ADA PERCOBAAN ULANG OTOMATIS.", flush=True)
        print("Saran: Tutup IDE dan peramban web, lalu jalankan kembali melalui launcher .cmd.", flush=True)
        sys.exit(2)

    # Validasi Skrip Anak
    if not inspect_script.is_file():
        print(f"[FATAL ERROR] Skrip inspeksi tidak ditemukan: {inspect_script}", flush=True)
        sys.exit(1)

    print("[INFO] Memori memenuhi syarat awal (>= 7.0 GiB).", flush=True)
    print("[INFO] Menjalankan proses anak secara terisolasi...", flush=True)
    print("-" * 80, flush=True)

    # 2. Menjalankan Skrip Inspeksi Sebagai Proses Anak (Subprocess)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [str(python_exe), str(inspect_script)]
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            bufsize=1,
        )
    except Exception as e:
        print(f"[FATAL ERROR] Gagal meluncurkan child process: {e}", flush=True)
        sys.exit(1)

    # Thread pembaca output proses anak secara real-time
    captured_output = []

    def stream_stdout(pipe):
        try:
            for line in iter(pipe.readline, ""):
                captured_output.append(line)
                sys.stdout.write(line)
                sys.stdout.flush()
        except Exception:
            pass
        finally:
            try:
                pipe.close()
            except Exception:
                pass

    reader_thread = threading.Thread(target=stream_stdout, args=(proc.stdout,), daemon=True)
    reader_thread.start()

    # 3. Watchdog Monitoring Loop
    start_time = time.time()
    peak_child_rss_gib = 0.0
    min_system_avail_gib = ram_avail_gib
    stop_reason = None

    while True:
        # Cek apakah proses anak sudah selesai
        ret_code = proc.poll()
        if ret_code is not None:
            break

        elapsed = time.time() - start_time

        # A. Cek Timeout
        if elapsed > TIMEOUT_SECONDS:
            stop_reason = f"TIMEOUT: Waktu eksekusi melebihi batas {TIMEOUT_SECONDS:.0f} detik"
            print(f"\n[WATCHDOG INTERVENTION] {stop_reason}! Menghentikan proses anak...", flush=True)
            terminate_process_tree(proc.pid)
            break

        # B. Cek RAM Sistem Tersedia (< 2.0 GiB)
        vm_cur = psutil.virtual_memory()
        cur_sys_avail = vm_cur.available / (1024 ** 3)
        if cur_sys_avail < min_system_avail_gib:
            min_system_avail_gib = cur_sys_avail

        if cur_sys_avail < CRITICAL_SYSTEM_RAM_GIB:
            stop_reason = (
                f"RAM_SISTEM_KRITIS: RAM sistem tersedia turun menjadi {cur_sys_avail:.2f} GiB "
                f"(di bawah ambang darurat {CRITICAL_SYSTEM_RAM_GIB:.2f} GiB)"
            )
            print(f"\n[WATCHDOG INTERVENTION] {stop_reason}! Menghentikan proses anak seketika...", flush=True)
            terminate_process_tree(proc.pid)
            break

        # C. Cek RSS Proses Anak (> 7.0 GiB)
        child_rss = get_process_tree_rss_gib(proc.pid)
        if child_rss > peak_child_rss_gib:
            peak_child_rss_gib = child_rss

        if child_rss > MAX_CHILD_RSS_GIB:
            stop_reason = (
                f"KONSUMSI_RSS_ANAK_BERLEBIH: RSS proses anak mencapai {child_rss:.2f} GiB "
                f"(melebihi batas aman {MAX_CHILD_RSS_GIB:.2f} GiB)"
            )
            print(f"\n[WATCHDOG INTERVENTION] {stop_reason}! Menghentikan proses anak seketika...", flush=True)
            terminate_process_tree(proc.pid)
            break

        time.sleep(MONITOR_INTERVAL_SECONDS)

    total_duration = time.time() - start_time
    reader_thread.join(timeout=2.0)

    print("-" * 80, flush=True)
    print("                 HASIL EVALUASI PENGAWAS MEMORI", flush=True)
    print("-" * 80, flush=True)
    print(f"Total Durasi Pengawasan       : {total_duration:.2f} detik", flush=True)
    print(f"Puncak Konsumsi RSS Anak      : {peak_child_rss_gib:.2f} GiB", flush=True)
    print(f"RAM Sistem Minimum Tersedia   : {min_system_avail_gib:.2f} GiB", flush=True)

    # 4. Evaluasi Akhir & Pelaporan
    if stop_reason is not None:
        print(f"[STATUS] DIHENTIKAN OLEH PENGAWAS KEAMANAN", flush=True)
        print(f"[ALASAN] {stop_reason}", flush=True)
        save_abort_report(report_path, stop_reason, total_duration, peak_child_rss_gib, min_system_avail_gib)
        print(f"[INFO] Laporan penghentian darurat disimpan ke: {report_path}", flush=True)
        print("TIDAK ADA PERCOBAAN ULANG OTOMATIS.", flush=True)
        sys.exit(3)

    # Proses selesai sendiri tanpa intervensi watchdog
    final_exit_code = proc.returncode
    if final_exit_code == 0:
        print("[STATUS] PROSES SELESAI DENGAN SUKSES (KODE 0)", flush=True)
        append_watchdog_success_log(report_path, total_duration, peak_child_rss_gib, min_system_avail_gib)
        print(f"[INFO] Laporan inspeksi lengkap tersimpan di: {report_path}", flush=True)
        print("=" * 80, flush=True)
        sys.exit(0)
    else:
        print(f"[STATUS] PROSES GAGAL / ERROR DENGAN KODE: {final_exit_code}", flush=True)
        # Jika gagal di dalam proses anak, catat ke laporan
        err_report_lines = [
            "",
            "=" * 80,
            f"[LOG ERROR PENGAWAS] Proses anak keluar dengan error code: {final_exit_code}",
            f"Durasi: {total_duration:.2f}s | Puncak RSS: {peak_child_rss_gib:.2f} GiB",
            "=" * 80,
        ]
        try:
            with open(report_path, "a", encoding="utf-8") as f:
                f.write("\n".join(err_report_lines) + "\n")
        except Exception:
            pass
        sys.exit(final_exit_code if final_exit_code is not None else 1)


if __name__ == "__main__":
    main()
