================================================================================
PETUNJUK PENGGUNAAN NOTEBOOK KONVERSI DATASET OSASUD DI KAGGLE
Berkas  : osasud_convert_kaggle.ipynb
Proyek  : Estimasi Tekanan Darah Cuffless & Saturasi Oksigen (SpO2) Menggunakan XGBoost
Status  : Khusus Dijalankan di Kaggle Cloud (Bukan Komputer Lokal)
================================================================================

1. LATAR BELAKANG & ALASAN PENGGUNAAN KAGGLE
--------------------------------------------------------------------------------
- Dataset resmi OSASUD (dataset_OSAS.pickle) memiliki ukuran berkas ~3.05 GiB.
- Pada komputer Windows lokal (RAM 16 GB), proses pembacaan pickle memerlukan
  alokasi memori RSS hingga > 6.08 GiB, menyisakan RAM sistem < 1.66 GiB.
  Pengawas memori lokal menghentikan proses demi mencegah OS freeze/crash.
- Kaggle Notebook versi CPU menyediakan RAM fisik ~30 GiB dan ruang disk 20 GiB,
  sehingga sangat aman dan leluasa untuk membaca dataset monolitik OSASUD dan
  mengonversinya menjadi berkas NPZ terpisah per pasien.


2. SPESIFIKASI LINGKUNGAN KAGGLE
--------------------------------------------------------------------------------
- Accelerator   : None (CPU Only, TANPA GPU)
- Language      : Python 3
- Environment   : Kaggle Default (Python 3.10+, pandas, numpy, psutil telah tersedia)
- Persistence   : Files only / standard
- Internet      : Off atau On (semua pustaka sudah built-in di image Kaggle)


3. PANDUAN LANGKAH-DEMI-LANGKAH (STEP-BY-STEP)
--------------------------------------------------------------------------------

LANGKAH 1: UNGGAH DATASET OSASUD KE KAGGLE DATASETS
1. Buka https://www.kaggle.com dan masuk ke akun Kaggle Anda.
2. Di menu navigasi kiri, klik "Datasets" -> klik tombol "+ New Dataset".
3. Beri judul dataset (misalnya: "OSASUD Official Dataset").
4. Unggah berkas asli:
   data/osasud/dataset_OSAS.pickle
   (Tunggu hingga proses upload dan kompresi Kaggle selesai 100%).
5. PENTING: Pada pengaturan visibilitas, pilih "Private" agar data tidak publik.
6. Klik "Create".

LANGKAH 2: UNGGAH NOTEBOOK KE KAGGLE
1. Di menu navigasi Kaggle, klik "Code" -> klik tombol "+ New Notebook".
2. Setelah editor notebook terbuka, klik menu "File" di bilah atas -> pilih "Import Notebook".
3. Pilih opsi "Upload" (atau seret berkas), lalu pilih berkas dari komputer lokal:
   osasud_convert_kaggle.ipynb
4. Notebook akan termuat dengan seluruh 12 sel instruksi dan kode Python.

LANGKAH 3: PENGATURAN HARDWARE (CPU ONLY)
1. Di bilah menu kanan notebook, periksa bagian "Notebook options":
   - Accelerator : None (CPU Only)
   - Persistence : No persistence atau Variables and Files
2. Jangan mengaktifkan GPU (P100 / T4) karena konversi dataset ini bersifat
   CPU/RAM-bound dan GPU Kaggle memiliki batas RAM yang lebih kecil (13-16 GB).

LANGKAH 4: HUBUNGKAN DATASET INPUT
1. Pada panel kanan notebook ("Data"), klik tombol "+ Add Input".
2. Beralih ke tab "Your Work" -> "Datasets".
3. Cari dataset OSASUD pribadi yang telah Anda buat di Langkah 1.
4. Klik tanda tambah (+) di samping dataset tersebut untuk memasangnya.
5. Jalur input akan otomatis terpasang di bawah direktori /kaggle/input/.
   (Notebook dirancang mencari file 'dataset_OSAS.pickle' secara dinamis
    tanpa terpengaruh nama subfolder).

LANGKAH 5: JALANKAN NOTEBOOK (RUN ALL)
1. Klik tombol "Run All" (atau tekan Ctrl + F9 / menu "Run" -> "Run All").
2. Notebook akan mengeksekusi tahapan berikut secara otomatis:
   a. Pre-flight check: Validasi RAM Kaggle (total >= 16 GB, tersedia >= 12 GB).
   b. Pencarian otomatis berkas dataset_OSAS.pickle di /kaggle/input/.
   c. Pemuatan dataset dengan pandas.read_pickle() disertai pelacakan RSS RAM.
   d. Validasi skema (5 kolom target, jumlah pasien, tipe array PPG, nilai hilang).
   e. Ekspor serial per pasien (memory-safe):
      - PPG 80 Hz (float64)
      - SpO2 1 Hz (float64)
      - PI 1 Hz (float64)
      - Timestamps (ISO 8601 <U32 & Unix ns int64)
      - NPZ terkompresi tanpa objek Python / tanpa pickle.
   f. Pembebasan memori DataFrame mentah (gc.collect()).
   g. Verifikasi independen seluruh berkas NPZ dengan allow_pickle=False.
   h. Penyusunan osasud_manifest.json dan osasud_conversion_summary.txt.
   i. Pengemasan seluruh hasil ke osasud_patients_npz.zip dan pengujian CRC testzip.
3. Estimasi waktu berjalan: sekitar 5 - 10 menit.

LANGKAH 6: UNDUH BERKAS HASIL DARI KAGGLE
1. Setelah notebook selesai berjalan hingga sel terakhir:
2. Di panel sebelah kanan, buka bagian "Data" -> "Output" (/kaggle/working).
3. Anda akan melihat berkas:
   - osasud_patients_npz.zip
   - osasud_manifest.json
   - osasud_conversion_summary.txt
4. Klik tombol titik tiga (...) di samping "osasud_patients_npz.zip" -> pilih "Download".
   (Ukuran unduhan diperkirakan sekitar ~500 - 700 MB).


4. PENATAAN HASIL DI REPOSITORI LOKAL WINDOWS
--------------------------------------------------------------------------------
Setelah berkas zip berhasil diunduh ke komputer lokal:
1. Buat direktori tujuan (jika belum ada):
   data/osasud/patients_npz/
2. Ekstrak seluruh berkas .npz yang ada di dalam folder 'npz/' pada arsip ZIP
   ke dalam:
   C:\Codes\UNIVERSITY PROJECTS (UNPAD)\Metode Penelitian KTI\data\osasud\patients_npz\
3. Letakkan berkas manifes 'osasud_manifest.json' di:
   C:\Codes\UNIVERSITY PROJECTS (UNPAD)\Metode Penelitian KTI\data\osasud\osasud_manifest.json
4. Letakkan berkas ringkasan 'osasud_conversion_summary.txt' di:
   C:\Codes\UNIVERSITY PROJECTS (UNPAD)\Metode Penelitian KTI\results\osasud_conversion_summary.txt


5. KEPATUHAN PROTOKOL & ATURAN ILMIAH
--------------------------------------------------------------------------------
- TIDAK ADA DOWNSAMPLING: PPG tetap 80 sampel per detik (80 Hz).
- TIDAK ADA FILTERING: Sinyal mentah disimpan apa adanya (filtering dilakukan
  pada skrip ekstraksi fitur terpisah).
- TIDAK ADA IMPUTASI BUATAN: Jika terdapat baris sinyal cacat/hilang, ditandai
  NaN eksplisit dan dicatat di manifes tanpa mengarang sinyal sintetis.
- TIDAK ADA TIPE DATA OBJECT / PICKLE: Semua array bertipe numerik float64/int64
  dan string fixed-width <U32, kompatibel murni dengan allow_pickle=False.
- TIDAK ADA MODEL TRAINING / FEATURE SELECTION: Notebook murni konversi data.
- LARANGAN FITUR: ECG, apnea label, dan PI TIDAK DIGUNAKAN sebagai fitur model.


6. LANGKAH PENELITIAN SELANJUTNYA
--------------------------------------------------------------------------------
Dengan tersedianya berkas NPZ per pasien:
1. Tahap segmentasi jendela 30 detik dapat dijalankan secara modular per pasien.
2. Ekstraksi fitur APG (derivatif pertama & kedua) dan deteksi puncak sistolik/
   diastolik PPG dapat diproses secara serial dengan pemakaian RAM lokal hanya
   ~50 - 100 MB per pasien.
3. Sistem lokal sepenuhnya bebas dari risiko crash atau kehabisan memori.
================================================================================
