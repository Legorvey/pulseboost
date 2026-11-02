@echo off
setlocal enabledelayedexpansion

title OSASUD Safe Dataset Inspector - KTI_XGBoost

echo ===============================================================================
echo        OSASUD SAFE DATASET INSPECTOR (STANDALONE LAUNCHER)
echo        Proyek KTI_XGBoost - Estimasi Tekanan Darah dan SpO2
echo ===============================================================================
echo.

:: 1. Tentukan Root Folder Proyek
set "CURRENT_DIR=%~dp0"
cd /d "%CURRENT_DIR%"

:: Deteksi letak folder proyek
set "PROJECT_ROOT="
if exist "%CURRENT_DIR%scripts\run_osasud_safe.py" (
    set "PROJECT_ROOT=%CURRENT_DIR%"
) else if exist "%CURRENT_DIR%KTI_XGBoost\scripts\run_osasud_safe.py" (
    set "PROJECT_ROOT=%CURRENT_DIR%KTI_XGBoost\"
) else if exist "%CURRENT_DIR%..\scripts\run_osasud_safe.py" (
    set "PROJECT_ROOT=%CURRENT_DIR%..\"
)

if "%PROJECT_ROOT%"=="" (
    echo [ERROR] Tidak dapat menemukan skrip scripts\run_osasud_safe.py!
    echo Pastikan launcher ini berada di direktori proyek Metode Penelitian KTI.
    echo.
    pause
    exit /b 1
)

:: 2. Deteksi Interpreter Python Virtual Environment
set "PYTHON_EXE="
if exist "%PROJECT_ROOT%.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%PROJECT_ROOT%.venv\Scripts\python.exe"
) else if exist "%CURRENT_DIR%.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%CURRENT_DIR%.venv\Scripts\python.exe"
) else if exist "%PROJECT_ROOT%KTI_XGBoost\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%PROJECT_ROOT%KTI_XGBoost\.venv\Scripts\python.exe"
) else if exist "%CURRENT_DIR%..\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%CURRENT_DIR%..\.venv\Scripts\python.exe"
)

if "%PYTHON_EXE%"=="" (
    echo [ERROR] Virtual Environment Python (.venv) tidak ditemukan!
    echo Pastikan environment .venv telah terkonfigurasi.
    echo.
    pause
    exit /b 1
)

echo [INFO] Direktori Proyek   : %PROJECT_ROOT%
echo [INFO] Python Interpreter : %PYTHON_EXE%
echo [INFO] Skrip Pengawas     : %PROJECT_ROOT%scripts\run_osasud_safe.py
echo [INFO] Skrip Inspeksi     : %PROJECT_ROOT%scripts\inspect_osasud_standalone.py
echo [INFO] Dataset Target     : %PROJECT_ROOT%data\osasud\dataset_OSAS.pickle
echo [INFO] Lokasi Laporan     : %PROJECT_ROOT%results\osasud_inspection_summary.txt
echo.
echo Memulai pengawas memori (psutil watchdog)...
echo -------------------------------------------------------------------------------
echo.

:: 3. Eksekusi Safe Runner
"%PYTHON_EXE%" "%PROJECT_ROOT%scripts\run_osasud_safe.py"
set "EXIT_CODE=%ERRORLEVEL%"

echo.
echo -------------------------------------------------------------------------------
if %EXIT_CODE% equ 0 (
    echo [STATUS: SUKSES]
    echo Inspeksi OSASUD selesai dengan aman tanpa kendala memori.
    echo Laporan lengkap telah tersimpan di:
    echo %PROJECT_ROOT%results\osasud_inspection_summary.txt
) else if %EXIT_CODE% equ 2 (
    echo [STATUS: DIBATALKAN DEMI KEAMANAN RAM]
    echo RAM yang tersedia saat ini kurang dari 7.0 GiB.
    echo Silakan tutup aplikasi berat (IDE, browser web, dll) lalu jalankan kembali launcher ini.
    echo Laporan evaluasi tersimpan di:
    echo %PROJECT_ROOT%results\osasud_inspection_summary.txt
) else if %EXIT_CODE% equ 3 (
    echo [STATUS: DIHENTIKAN OLEH PENGAWAS KEAMANAN]
    echo Pengawas memori menghentikan proses anak karena melewati ambang batas keamanan.
    echo Laporan darurat tersimpan di:
    echo %PROJECT_ROOT%results\osasud_inspection_summary.txt
) else (
    echo [STATUS: TERJADI ERROR (KODE: %EXIT_CODE%)]
    echo Periksa rincian pesan error di atas atau buka file laporan:
    echo %PROJECT_ROOT%results\osasud_inspection_summary.txt
)
echo ===============================================================================
echo.
echo Selesai. Tekan tombol apa saja untuk menutup jendela terminal ini...
pause >nul
exit /b %EXIT_CODE%
