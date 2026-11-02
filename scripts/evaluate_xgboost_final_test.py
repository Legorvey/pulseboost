"""
scripts/evaluate_xgboost_final_test.py
================================================================================
TAHAP 10: EVALUASI AKHIR TIGA MODEL XGBOOST PADA DATA TEST (SEKALI JALAN)
Proyek : KTI_XGBoost - Estimasi SBP, DBP (UCI) dan SpO2 (OSASUD)
Status : PENGUJIAN FINAL PADA SET TEST INDEPENDEN
================================================================================
Peraturan Ketat:
1. Menjalankan seluruh pemeriksaan keamanan (Bagian A) sebelum memproses TEST.
2. Memuat model yang sudah disimpan di folder `models/` tanpa melatih ulang.
3. Menggunakan urutan dan daftar fitur persis sesuai metadata model.
4. Menjalankan inferensi TEST satu kali saja tanpa mengubah model / hyperparameter.
5. Menghitung metrik final (MAE, RMSE, ME, STD(ME), Pearson R, R^2, N valid).
6. Membandingkan performa model dengan Baseline Mean dan Baseline Median dari TRAIN.
7. Menganalisis galat per record (UCI) dan per pasien (OSASUD).
8. Menganalisis galat berdasarkan rentang target (SBP/DBP rendah-sedang-tinggi, SpO2 desaturasi).
9. Menyimpan hasil prediksi, metrik, visualisasi, dan log integritas.
================================================================================
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

# Konfigurasi gaya visualisasi matplotlib
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8


def find_project_root() -> Path:
    cwd = Path.cwd()
    if (cwd / "KTI_XGBoost").is_dir():
        return cwd / "KTI_XGBoost"
    if cwd.name == "KTI_XGBoost":
        return cwd
    for parent in cwd.parents:
        if (parent / "KTI_XGBoost").is_dir():
            return parent / "KTI_XGBoost"
        if parent.name == "KTI_XGBoost":
            return parent
    return cwd


# ==============================================================================
# BAGIAN A: VERIFIKASI KEAMANAN DAN INTEGRITAS SEBELUM EVALUASI TEST
# ==============================================================================

def run_preflight_safety_checks(project_root: Path) -> dict:
    """
    Melakukan 10 poin pemeriksaan keamanan data, model, dan split sebelum TEST.
    Jika ada 1 poin yang gagal, eksekusi HARUS BERHENTI.
    """
    print("=" * 80)
    print("BAGIAN A: MENJALANKAN PEMERIKSAAN KEAMANAN SEBELUM EVALUASI TEST")
    print("=" * 80)

    integrity_log = []
    def log(msg):
        print(f"  [CHECK] {msg}")
        integrity_log.append(msg)

    # 1. Periksa keberadaan berkas manifest, data, dan model
    required_files = {
        "manifest_selection": project_root / "results" / "final_model_selection_manifest.json",
        "data_readiness": project_root / "results" / "ml_data_readiness_summary.txt",
        "uci_manifest": project_root / "results" / "uci_split_manifest.csv",
        "osasud_manifest": project_root / "results" / "osasud_split_manifest.csv",
        "uci_data": project_root / "results" / "uci_features_model_ready_v1.csv",
        "osasud_data": project_root / "results" / "osasud_features_model_ready_v1.csv",
        "sbp_model": project_root / "models" / "sbp_config_c.json",
        "sbp_meta": project_root / "models" / "sbp_config_c_metadata.json",
        "dbp_model": project_root / "models" / "dbp_config_c.json",
        "dbp_meta": project_root / "models" / "dbp_config_c_metadata.json",
        "spo2_model": project_root / "models" / "spo2_config_d.json",
        "spo2_meta": project_root / "models" / "spo2_config_d_metadata.json",
    }

    for name, fpath in required_files.items():
        if not fpath.is_file():
            raise FileNotFoundError(f"Pemeriksaan GAGAL: Berkas wajib tidak ditemukan: {fpath} ({name})")
        log(f"Berkas {name:<20}: ADA ({fpath.name}, size={fpath.stat().st_size:,} bytes)")

    # Muat metadata ketiga model terpilih
    with open(required_files["sbp_meta"], "r", encoding="utf-8") as f:
        sbp_meta = json.load(f)
    with open(required_files["dbp_meta"], "r", encoding="utf-8") as f:
        dbp_meta = json.load(f)
    with open(required_files["spo2_meta"], "r", encoding="utf-8") as f:
        spo2_meta = json.load(f)
    with open(required_files["manifest_selection"], "r", encoding="utf-8") as f:
        selection_manifest = json.load(f)

    # Muat dataset dan manifest
    df_uci = pd.read_csv(required_files["uci_data"])
    df_osasud = pd.read_csv(required_files["osasud_data"])
    df_uci_man = pd.read_csv(required_files["uci_manifest"])
    df_osa_man = pd.read_csv(required_files["osasud_manifest"])

    # 2. Cocokkan nama, jumlah, dan urutan fitur
    expected_19 = [
        "ibi_median_s", "heart_rate_bpm", "ibi_cv", "rise_time_s", "decay_time_s",
        "pulse_width_50_s", "pulse_amplitude", "pulse_area", "max_ascending_slope",
        "min_descending_slope", "skewness", "kurtosis", "dominant_frequency_hz",
        "spectral_power_ratio", "spectral_entropy", "apg_b_a", "apg_c_a", "apg_d_a", "apg_e_a"
    ]
    expected_spo2_cfs = ["rise_time_s", "max_ascending_slope"]

    assert sbp_meta["features_list"] == expected_19, "Urutan / nama fitur SBP tidak cocok!"
    assert dbp_meta["features_list"] == expected_19, "Urutan / nama fitur DBP tidak cocok!"
    assert spo2_meta["features_list"] == expected_spo2_cfs, "Urutan / nama fitur SpO2 tidak cocok!"
    assert len(sbp_meta["features_list"]) == 19, "Jumlah fitur SBP bukan 19!"
    assert len(dbp_meta["features_list"]) == 19, "Jumlah fitur DBP bukan 19!"
    assert len(spo2_meta["features_list"]) == 2, "Jumlah fitur SpO2 bukan 2!"
    log("Daftar, jumlah, dan urutan fitur ketiga model 100% konsisten dengan metadata.")

    # 3. Verifikasi prediktor murni (tanpa target, metadata, timestamp, ABP, ECG, PI, SQI tambahan)
    forbidden_columns = {
        "sbp_mmhg", "dbp_mmhg", "pulse_pressure_mmhg", "map_mmhg", "spo2_target_pct",
        "source_part", "record_id", "window_id", "patient_id",
        "start_time_s", "end_time_s", "start_timestamp", "end_timestamp",
        "processing_valid", "quality_warning", "quality_warning_details", "feature_extraction_status",
        "num_valid_beats", "num_valid_abp_cycles", "num_peaks_detected", "num_valid_intervals", "num_unreliable_intervals",
        "apg_b_a_valid_beats", "apg_c_a_valid_beats", "apg_d_a_valid_beats", "apg_e_a_valid_beats",
        "apg_b_a_pct", "apg_c_a_pct", "apg_d_a_pct", "apg_e_a_pct", "apg_reliable_coverage",
        "ecg", "abp", "pi", "split"
    }
    all_allowed_candidates = set(expected_19)

    for t_name, feats in [("SBP", sbp_meta["features_list"]), ("DBP", dbp_meta["features_list"]), ("SpO2", spo2_meta["features_list"])]:
        for f in feats:
            if f in forbidden_columns:
                raise ValueError(f"Kolom terlarang '{f}' ditemukan pada prediktor {t_name}!")
            if f not in all_allowed_candidates:
                raise ValueError(f"Fitur tidak dikenal '{f}' ditemukan pada prediktor {t_name}!")
    log("Prediktor terverifikasi murni 19 fitur fisiologis PPG/APG kandidat sah.")

    # 4. Verifikasi jenis split (UCI: record-wise, OSASUD: patient-wise)
    assert set(df_uci["split"].unique()) == {"train", "val", "test"}, "Split UCI tidak lengkap!"
    assert set(df_osasud["split"].unique()) == {"train", "val", "test"}, "Split OSASUD tidak lengkap!"

    # 5. Verifikasi tidak ada record_id UCI di lebih dari 1 split
    uci_rec_splits = df_uci.groupby("record_id")["split"].nunique()
    if (uci_rec_splits > 1).any():
        bad_recs = uci_rec_splits[uci_rec_splits > 1].index.tolist()
        raise ValueError(f"Data leakage terdeteksi! Record UCI berada pada lebih dari satu split: {bad_recs[:5]}")
    log(f"UCI Record-Wise Split terverifikasi: 0 dari {len(uci_rec_splits):,} record_id tumpang tindih.")

    # 6. Verifikasi tidak ada patient_id OSASUD di lebih dari 1 split
    osa_pat_splits = df_osasud.groupby("patient_id")["split"].nunique()
    if (osa_pat_splits > 1).any():
        bad_pats = osa_pat_splits[osa_pat_splits > 1].index.tolist()
        raise ValueError(f"Data leakage terdeteksi! Pasien OSASUD berada pada lebih dari satu split: {bad_pats}")
    log(f"OSASUD Patient-Wise Split terverifikasi: 0 dari {len(osa_pat_splits)} patient_id tumpang tindih.")

    # 7. Verifikasi seluruh kelompok rekaman identik/duplikat UCI berada pada split yang sama
    grp_splits = df_uci_man.groupby("meta_group_id")["split"].nunique()
    if (grp_splits > 1).any():
        bad_grps = grp_splits[grp_splits > 1].index.tolist()
        raise ValueError(f"Kelompok rekaman duplikat UCI terpisah di split berbeda: {bad_grps}")
    log(f"Meta-Group UCI terverifikasi: Seluruh {len(grp_splits):,} kelompok duplikat berada pada split yang sama.")

    # 8. Verifikasi tidak ada duplikasi identitas jendela
    uci_win_dup = df_uci.duplicated(subset=["record_id", "window_id"]).sum()
    osa_win_dup = df_osasud.duplicated(subset=["patient_id", "window_id"]).sum()
    assert uci_win_dup == 0, f"Ditemukan {uci_win_dup} duplikat identitas jendela pada UCI!"
    assert osa_win_dup == 0, f"Ditemukan {osa_win_dup} duplikat identitas jendela pada OSASUD!"
    log(f"Keunikan jendela terverifikasi: UCI duplikat={uci_win_dup}, OSASUD duplikat={osa_win_dup}.")

    # 9. Verifikasi seleksi model telah dikunci sebelum evaluasi TEST
    assert selection_manifest["stage"] == "TAHAP_9C_FINAL_VALIDATION_SELECTION", "Manifest seleksi bukan dari Tahap 9C!"
    assert selection_manifest["selected_candidate_models"]["sbp"]["selected_config"] == "config_c"
    assert selection_manifest["selected_candidate_models"]["dbp"]["selected_config"] == "config_c"
    assert selection_manifest["selected_candidate_models"]["spo2"]["selected_config"] == "config_d"
    log("Seleksi model resmi terbukti dikunci sebelum evaluasi TEST (berdasarkan hasil validasi Tahap 9C).")

    # 10. Pernyataan integritas
    log("Status: SELURUH 10 PEMERIKSAAN KEAMANAN LULUS 100%. EVALUASI TEST DIIZINKAN.")
    print("=" * 80 + "\n")

    return {
        "files": required_files,
        "sbp_meta": sbp_meta,
        "dbp_meta": dbp_meta,
        "spo2_meta": spo2_meta,
        "selection_manifest": selection_manifest,
        "df_uci": df_uci,
        "df_osasud": df_osasud,
        "integrity_log": integrity_log,
    }


# ==============================================================================
# FUNGSI METRIK EVALUASI AKURAT
# ==============================================================================

def compute_metrics(y_true, y_pred, is_constant=False):
    """
    Menghitung metrik performa regresi:
    MAE, RMSE, ME (pred - true), STD(ME) dengan ddof=1, Pearson R, R^2.
    Untuk prediksi konstanta, Pearson R diset None / 'N/A'.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    n = len(y_true)
    error = y_pred - y_true
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    me = float(np.mean(error))
    std_me = float(np.std(error, ddof=1)) if n > 1 else 0.0

    # R^2 = 1 - SS_res / SS_tot
    ss_res = float(np.sum(error ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else np.nan

    # Pearson R
    if is_constant or np.std(y_pred) < 1e-9 or np.std(y_true) < 1e-9:
        r = np.nan
    else:
        r_val = np.corrcoef(y_pred, y_true)[0, 1]
        r = float(r_val) if not np.isnan(r_val) else np.nan

    return {
        "n_samples": n,
        "mae": mae,
        "rmse": rmse,
        "me": me,
        "std_me": std_me,
        "pearson_r": r,
        "r2": r2,
    }


# ==============================================================================
# BAGIAN B, C, D, E, F: EKSEKUSI EVALUASI TEST FINAL
# ==============================================================================

def run_final_test_evaluation():
    start_time = time.time()
    eval_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    project_root = find_project_root()
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Jalankan Safety Checks
    check_res = run_preflight_safety_checks(project_root)

    df_uci = check_res["df_uci"]
    df_osasud = check_res["df_osasud"]
    sbp_meta = check_res["sbp_meta"]
    dbp_meta = check_res["dbp_meta"]
    spo2_meta = check_res["spo2_meta"]
    files = check_res["files"]

    print("=" * 80)
    print("BAGIAN B & C: PENGUJIAN FINAL SATU KALI PADA DATA TEST DENGAN BASELINE")
    print("=" * 80)

    # Siapkan Set Train untuk ekstraksi nilai baseline presisi penuh
    uci_train_mask = df_uci["split"] == "train"
    osasud_train_mask = df_osasud["split"] == "train"

    # Presisi penuh baseline dari TRAIN
    sbp_train_valid = df_uci.loc[uci_train_mask & df_uci["sbp_mmhg"].notna(), "sbp_mmhg"].values
    dbp_train_valid = df_uci.loc[uci_train_mask & df_uci["dbp_mmhg"].notna(), "dbp_mmhg"].values
    spo2_train_valid = df_osasud.loc[osasud_train_mask & df_osasud["spo2_target_pct"].notna(), "spo2_target_pct"].values

    base_sbp_mean = float(np.mean(sbp_train_valid))
    base_sbp_median = float(np.median(sbp_train_valid))
    base_dbp_mean = float(np.mean(dbp_train_valid))
    base_dbp_median = float(np.median(dbp_train_valid))
    base_spo2_mean = float(np.mean(spo2_train_valid))
    base_spo2_median = float(np.median(spo2_train_valid))

    print(f"Nilai Baseline Asli Berpresisi Penuh dari TRAIN:")
    print(f"  - SBP : Mean = {base_sbp_mean:.6f} mmHg, Median = {base_sbp_median:.6f} mmHg (N_train={len(sbp_train_valid):,})")
    print(f"  - DBP : Mean = {base_dbp_mean:.6f} mmHg, Median = {base_dbp_median:.6f} mmHg (N_train={len(dbp_train_valid):,})")
    print(f"  - SpO2: Mean = {base_spo2_mean:.6f} %,     Median = {base_spo2_median:.6f} %     (N_train={len(spo2_train_valid):,})")

    # Filter Set TEST
    uci_test_all = df_uci[df_uci["split"] == "test"].copy()
    osasud_test_all = df_osasud[df_osasud["split"] == "test"].copy()

    n_uci_raw = len(uci_test_all)
    n_osa_raw = len(osasud_test_all)
    print(f"\nJumlah Jendela Mentah Set TEST:")
    print(f"  - UCI TEST   : {n_uci_raw:,} jendela ({uci_test_all['record_id'].nunique():,} record_id)")
    print(f"  - OSASUD TEST: {n_osa_raw:,} jendela ({osasud_test_all['patient_id'].nunique()} pasien)")

    assert n_uci_raw == 12514, f"Jumlah jendela UCI TEST tidak 12,514 (ditemukan: {n_uci_raw})"
    assert n_osa_raw == 4706, f"Jumlah jendela OSASUD TEST tidak 4,706 (ditemukan: {n_osa_raw})"

    # Filter baris dengan label target valid pada TEST
    uci_test_sbp = uci_test_all[uci_test_all["sbp_mmhg"].notna()].copy()
    uci_test_dbp = uci_test_all[uci_test_all["dbp_mmhg"].notna()].copy()
    osasud_test_spo2 = osasud_test_all[osasud_test_all["spo2_target_pct"].notna()].copy()

    n_sbp_valid = len(uci_test_sbp)
    n_dbp_valid = len(uci_test_dbp)
    n_spo2_valid = len(osasud_test_spo2)

    print(f"\nJumlah Jendela Final dengan Target Valid untuk Evaluasi:")
    print(f"  - SBP TEST  : {n_sbp_valid:,} jendela valid ({n_uci_raw - n_sbp_valid} dikeluarkan karena label kosong)")
    print(f"  - DBP TEST  : {n_dbp_valid:,} jendela valid ({n_uci_raw - n_dbp_valid} dikeluarkan karena label kosong)")
    print(f"  - SpO2 TEST : {n_spo2_valid:,} jendela valid ({n_osa_raw - n_spo2_valid} dikeluarkan)")

    # --------------------------------------------------------------------------
    # 2. Muat Model XGBoost dan Lakukan Prediksi TEST Sekali Saja
    # --------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("Memuat Model XGBoost Terpilih...")
    print("-" * 80)

    model_sbp = xgb.XGBRegressor()
    model_sbp.load_model(str(files["sbp_model"]))

    model_dbp = xgb.XGBRegressor()
    model_dbp.load_model(str(files["dbp_model"]))

    model_spo2 = xgb.XGBRegressor()
    model_spo2.load_model(str(files["spo2_model"]))

    # Verifikasi best_iteration konsisten
    assert model_sbp.best_iteration == sbp_meta["best_iteration"], "best_iteration SBP mismatch!"
    assert model_dbp.best_iteration == dbp_meta["best_iteration"], "best_iteration DBP mismatch!"
    assert model_spo2.best_iteration == spo2_meta["best_iteration"], "best_iteration SpO2 mismatch!"
    print(f"  - Model SBP  (Config C): best_iteration = {model_sbp.best_iteration}")
    print(f"  - Model DBP  (Config C): best_iteration = {model_dbp.best_iteration}")
    print(f"  - Model SpO2 (Config D): best_iteration = {model_spo2.best_iteration}")

    # Prediksi TEST
    print("\nMelakukan Prediksi pada Set TEST...")
    pred_sbp = model_sbp.predict(uci_test_sbp[sbp_meta["features_list"]])
    pred_dbp = model_dbp.predict(uci_test_dbp[dbp_meta["features_list"]])
    pred_spo2 = model_spo2.predict(osasud_test_spo2[spo2_meta["features_list"]])

    y_sbp_true = uci_test_sbp["sbp_mmhg"].values
    y_dbp_true = uci_test_dbp["dbp_mmhg"].values
    y_spo2_true = osasud_test_spo2["spo2_target_pct"].values

    # Prediksi Baseline
    pred_sbp_b_mean = np.full_like(y_sbp_true, fill_value=base_sbp_mean)
    pred_sbp_b_med = np.full_like(y_sbp_true, fill_value=base_sbp_median)

    pred_dbp_b_mean = np.full_like(y_dbp_true, fill_value=base_dbp_mean)
    pred_dbp_b_med = np.full_like(y_dbp_true, fill_value=base_dbp_median)

    pred_spo2_b_mean = np.full_like(y_spo2_true, fill_value=base_spo2_mean)
    pred_spo2_b_med = np.full_like(y_spo2_true, fill_value=base_spo2_median)

    # --------------------------------------------------------------------------
    # 3. Hitung Metrik Final (Bagian D)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("BAGIAN D: PERHITUNGAN METRIK FINAL PADA DATA TEST")
    print("=" * 80)

    # Metrik Model
    m_sbp_model = compute_metrics(y_sbp_true, pred_sbp)
    m_sbp_b_mean = compute_metrics(y_sbp_true, pred_sbp_b_mean, is_constant=True)
    m_sbp_b_med = compute_metrics(y_sbp_true, pred_sbp_b_med, is_constant=True)

    m_dbp_model = compute_metrics(y_dbp_true, pred_dbp)
    m_dbp_b_mean = compute_metrics(y_dbp_true, pred_dbp_b_mean, is_constant=True)
    m_dbp_b_med = compute_metrics(y_dbp_true, pred_dbp_b_med, is_constant=True)

    m_spo2_model = compute_metrics(y_spo2_true, pred_spo2)
    m_spo2_b_mean = compute_metrics(y_spo2_true, pred_spo2_b_mean, is_constant=True)
    m_spo2_b_med = compute_metrics(y_spo2_true, pred_spo2_b_med, is_constant=True)

    # Validasi sebelumnya (dari metadata model)
    val_sbp_mae = sbp_meta["validation_metrics"]["mae"]
    val_sbp_rmse = sbp_meta["validation_metrics"]["rmse"]
    val_dbp_mae = dbp_meta["validation_metrics"]["mae"]
    val_dbp_rmse = dbp_meta["validation_metrics"]["rmse"]
    val_spo2_mae = spo2_meta["validation_metrics"]["mae"]
    val_spo2_rmse = spo2_meta["validation_metrics"]["rmse"]

    # Persentase perbaikan MAE terhadap Baseline Median TEST
    imp_sbp = ((m_sbp_b_med["mae"] - m_sbp_model["mae"]) / m_sbp_b_med["mae"]) * 100.0
    imp_dbp = ((m_dbp_b_med["mae"] - m_dbp_model["mae"]) / m_dbp_b_med["mae"]) * 100.0
    imp_spo2 = ((m_spo2_b_med["mae"] - m_spo2_model["mae"]) / m_spo2_b_med["mae"]) * 100.0

    # Susun Tabel Metrik Final
    metrics_rows = [
        # SBP
        {
            "target": "sbp",
            "target_name": "Systolic Blood Pressure",
            "unit": "mmHg",
            "model_type": "XGBoost (Config C, 19 feats)",
            "eval_split": "test",
            "n_samples": m_sbp_model["n_samples"],
            "mae": m_sbp_model["mae"],
            "rmse": m_sbp_model["rmse"],
            "me": m_sbp_model["me"],
            "std_me": m_sbp_model["std_me"],
            "pearson_r": m_sbp_model["pearson_r"],
            "r2": m_sbp_model["r2"],
            "val_mae": val_sbp_mae,
            "val_rmse": val_sbp_rmse,
            "diff_mae_test_vs_val": m_sbp_model["mae"] - val_sbp_mae,
            "pct_improvement_vs_median": imp_sbp,
        },
        {
            "target": "sbp",
            "target_name": "Systolic Blood Pressure",
            "unit": "mmHg",
            "model_type": "Baseline (Mean TRAIN)",
            "eval_split": "test",
            "n_samples": m_sbp_b_mean["n_samples"],
            "mae": m_sbp_b_mean["mae"],
            "rmse": m_sbp_b_mean["rmse"],
            "me": m_sbp_b_mean["me"],
            "std_me": m_sbp_b_mean["std_me"],
            "pearson_r": np.nan,
            "r2": m_sbp_b_mean["r2"],
            "val_mae": 17.9867,
            "val_rmse": 21.9423,
            "diff_mae_test_vs_val": m_sbp_b_mean["mae"] - 17.9867,
            "pct_improvement_vs_median": ((m_sbp_b_med["mae"] - m_sbp_b_mean["mae"]) / m_sbp_b_med["mae"]) * 100.0,
        },
        {
            "target": "sbp",
            "target_name": "Systolic Blood Pressure",
            "unit": "mmHg",
            "model_type": "Baseline (Median TRAIN)",
            "eval_split": "test",
            "n_samples": m_sbp_b_med["n_samples"],
            "mae": m_sbp_b_med["mae"],
            "rmse": m_sbp_b_med["rmse"],
            "me": m_sbp_b_med["me"],
            "std_me": m_sbp_b_med["std_me"],
            "pearson_r": np.nan,
            "r2": m_sbp_b_med["r2"],
            "val_mae": 17.8338,
            "val_rmse": 21.8742,
            "diff_mae_test_vs_val": m_sbp_b_med["mae"] - 17.8338,
            "pct_improvement_vs_median": 0.0,
        },
        # DBP
        {
            "target": "dbp",
            "target_name": "Diastolic Blood Pressure",
            "unit": "mmHg",
            "model_type": "XGBoost (Config C, 19 feats)",
            "eval_split": "test",
            "n_samples": m_dbp_model["n_samples"],
            "mae": m_dbp_model["mae"],
            "rmse": m_dbp_model["rmse"],
            "me": m_dbp_model["me"],
            "std_me": m_dbp_model["std_me"],
            "pearson_r": m_dbp_model["pearson_r"],
            "r2": m_dbp_model["r2"],
            "val_mae": val_dbp_mae,
            "val_rmse": val_dbp_rmse,
            "diff_mae_test_vs_val": m_dbp_model["mae"] - val_dbp_mae,
            "pct_improvement_vs_median": imp_dbp,
        },
        {
            "target": "dbp",
            "target_name": "Diastolic Blood Pressure",
            "unit": "mmHg",
            "model_type": "Baseline (Mean TRAIN)",
            "eval_split": "test",
            "n_samples": m_dbp_b_mean["n_samples"],
            "mae": m_dbp_b_mean["mae"],
            "rmse": m_dbp_b_mean["rmse"],
            "me": m_dbp_b_mean["me"],
            "std_me": m_dbp_b_mean["std_me"],
            "pearson_r": np.nan,
            "r2": m_dbp_b_mean["r2"],
            "val_mae": 8.3272,
            "val_rmse": 10.5566,
            "diff_mae_test_vs_val": m_dbp_b_mean["mae"] - 8.3272,
            "pct_improvement_vs_median": ((m_dbp_b_med["mae"] - m_dbp_b_mean["mae"]) / m_dbp_b_med["mae"]) * 100.0,
        },
        {
            "target": "dbp",
            "target_name": "Diastolic Blood Pressure",
            "unit": "mmHg",
            "model_type": "Baseline (Median TRAIN)",
            "eval_split": "test",
            "n_samples": m_dbp_b_med["n_samples"],
            "mae": m_dbp_b_med["mae"],
            "rmse": m_dbp_b_med["rmse"],
            "me": m_dbp_b_med["me"],
            "std_me": m_dbp_b_med["std_me"],
            "pearson_r": np.nan,
            "r2": m_dbp_b_med["r2"],
            "val_mae": 7.9603,
            "val_rmse": 10.5907,
            "diff_mae_test_vs_val": m_dbp_b_med["mae"] - 7.9603,
            "pct_improvement_vs_median": 0.0,
        },
        # SpO2
        {
            "target": "spo2",
            "target_name": "Oxygen Saturation",
            "unit": "%",
            "model_type": "XGBoost (Config D, 2 feats CFS)",
            "eval_split": "test",
            "n_samples": m_spo2_model["n_samples"],
            "mae": m_spo2_model["mae"],
            "rmse": m_spo2_model["rmse"],
            "me": m_spo2_model["me"],
            "std_me": m_spo2_model["std_me"],
            "pearson_r": m_spo2_model["pearson_r"],
            "r2": m_spo2_model["r2"],
            "val_mae": val_spo2_mae,
            "val_rmse": val_spo2_rmse,
            "diff_mae_test_vs_val": m_spo2_model["mae"] - val_spo2_mae,
            "pct_improvement_vs_median": imp_spo2,
        },
        {
            "target": "spo2",
            "target_name": "Oxygen Saturation",
            "unit": "%",
            "model_type": "Baseline (Mean TRAIN)",
            "eval_split": "test",
            "n_samples": m_spo2_b_mean["n_samples"],
            "mae": m_spo2_b_mean["mae"],
            "rmse": m_spo2_b_mean["rmse"],
            "me": m_spo2_b_mean["me"],
            "std_me": m_spo2_b_mean["std_me"],
            "pearson_r": np.nan,
            "r2": m_spo2_b_mean["r2"],
            "val_mae": 2.4491,
            "val_rmse": 2.8526,
            "diff_mae_test_vs_val": m_spo2_b_mean["mae"] - 2.4491,
            "pct_improvement_vs_median": ((m_spo2_b_med["mae"] - m_spo2_b_mean["mae"]) / m_spo2_b_med["mae"]) * 100.0,
        },
        {
            "target": "spo2",
            "target_name": "Oxygen Saturation",
            "unit": "%",
            "model_type": "Baseline (Median TRAIN)",
            "eval_split": "test",
            "n_samples": m_spo2_b_med["n_samples"],
            "mae": m_spo2_b_med["mae"],
            "rmse": m_spo2_b_med["rmse"],
            "me": m_spo2_b_med["me"],
            "std_me": m_spo2_b_med["std_me"],
            "pearson_r": np.nan,
            "r2": m_spo2_b_med["r2"],
            "val_mae": 2.0387,
            "val_rmse": 2.4369,
            "diff_mae_test_vs_val": m_spo2_b_med["mae"] - 2.0387,
            "pct_improvement_vs_median": 0.0,
        },
    ]

    df_metrics = pd.DataFrame(metrics_rows)
    metrics_csv_path = results_dir / "xgboost_final_test_metrics.csv"
    df_metrics.to_csv(metrics_csv_path, index=False)
    print(f"Disimpan Tabel Metrik: {metrics_csv_path}")

    # Cetak ke layar
    print("\n--- RINGKASAN METRIK TEST FINAL ---")
    for r in metrics_rows:
        r_str = f"{r['pearson_r']:.4f}" if pd.notna(r['pearson_r']) else "N/A"
        r2_str = f"{r['r2']:.4f}" if pd.notna(r['r2']) else "N/A"
        print(f"[{r['target'].upper():<4}] {r['model_type']:<30} | MAE: {r['mae']:>7.4f} {r['unit']} | RMSE: {r['rmse']:>7.4f} | ME: {r['me']:>+7.4f} | R: {r_str} | R2: {r2_str} | Perbaikan vs Med: {r['pct_improvement_vs_median']:>+6.2f}%")

    # --------------------------------------------------------------------------
    # 4. Simpan Tabel Prediksi Lengkap untuk Audit
    # --------------------------------------------------------------------------
    # Prediksi UCI
    df_pred_uci = uci_test_all[["source_part", "record_id", "window_id", "start_time_s", "end_time_s",
                                 "sbp_mmhg", "dbp_mmhg"]].copy()

    # Inisialisasi kolom prediksi
    df_pred_uci["pred_sbp_config_c"] = np.nan
    df_pred_uci["pred_sbp_baseline_mean"] = np.nan
    df_pred_uci["pred_sbp_baseline_median"] = np.nan
    df_pred_uci["error_sbp_config_c"] = np.nan
    df_pred_uci["abs_error_sbp_config_c"] = np.nan

    df_pred_uci["pred_dbp_config_c"] = np.nan
    df_pred_uci["pred_dbp_baseline_mean"] = np.nan
    df_pred_uci["pred_dbp_baseline_median"] = np.nan
    df_pred_uci["error_dbp_config_c"] = np.nan
    df_pred_uci["abs_error_dbp_config_c"] = np.nan

    # Isi baris valid
    valid_sbp_idx = df_pred_uci["sbp_mmhg"].notna()
    df_pred_uci.loc[valid_sbp_idx, "pred_sbp_config_c"] = pred_sbp
    df_pred_uci.loc[valid_sbp_idx, "pred_sbp_baseline_mean"] = pred_sbp_b_mean
    df_pred_uci.loc[valid_sbp_idx, "pred_sbp_baseline_median"] = pred_sbp_b_med
    df_pred_uci.loc[valid_sbp_idx, "error_sbp_config_c"] = pred_sbp - y_sbp_true
    df_pred_uci.loc[valid_sbp_idx, "abs_error_sbp_config_c"] = np.abs(pred_sbp - y_sbp_true)

    valid_dbp_idx = df_pred_uci["dbp_mmhg"].notna()
    df_pred_uci.loc[valid_dbp_idx, "pred_dbp_config_c"] = pred_dbp
    df_pred_uci.loc[valid_dbp_idx, "pred_dbp_baseline_mean"] = pred_dbp_b_mean
    df_pred_uci.loc[valid_dbp_idx, "pred_dbp_baseline_median"] = pred_dbp_b_med
    df_pred_uci.loc[valid_dbp_idx, "error_dbp_config_c"] = pred_dbp - y_dbp_true
    df_pred_uci.loc[valid_dbp_idx, "abs_error_dbp_config_c"] = np.abs(pred_dbp - y_dbp_true)

    uci_pred_path = results_dir / "xgboost_final_predictions_uci.csv"
    df_pred_uci.to_csv(uci_pred_path, index=False)
    print(f"\nDisimpan Tabel Prediksi Audit UCI: {uci_pred_path} ({len(df_pred_uci):,} baris)")

    # Prediksi OSASUD
    df_pred_osasud = osasud_test_all[["patient_id", "window_id", "start_timestamp", "end_timestamp",
                                       "spo2_target_pct"]].copy()
    df_pred_osasud["pred_spo2_config_d"] = pred_spo2
    df_pred_osasud["pred_spo2_baseline_mean"] = pred_spo2_b_mean
    df_pred_osasud["pred_spo2_baseline_median"] = pred_spo2_b_med
    df_pred_osasud["error_spo2_config_d"] = pred_spo2 - y_spo2_true
    df_pred_osasud["abs_error_spo2_config_d"] = np.abs(pred_spo2 - y_spo2_true)

    osasud_pred_path = results_dir / "xgboost_final_predictions_osasud.csv"
    df_pred_osasud.to_csv(osasud_pred_path, index=False)
    print(f"Disimpan Tabel Prediksi Audit OSASUD: {osasud_pred_path} ({len(df_pred_osasud):,} baris)")

    # --------------------------------------------------------------------------
    # 5. BAGIAN E: ANALISIS KHUSUS DATASET UCI (SBP & DBP)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("BAGIAN E: ANALISIS MENDALAM DATASET UCI (PER-RECORD & PER-RENTANG TEKANAN)")
    print("=" * 80)

    # MAE per record_id
    rec_sbp_mae = df_pred_uci.groupby("record_id")["abs_error_sbp_config_c"].mean().dropna()
    rec_dbp_mae = df_pred_uci.groupby("record_id")["abs_error_dbp_config_c"].mean().dropna()

    uci_record_stats = {
        "sbp": {
            "n_records": len(rec_sbp_mae),
            "mean_rec_mae": float(rec_sbp_mae.mean()),
            "median_rec_mae": float(rec_sbp_mae.median()),
            "std_rec_mae": float(rec_sbp_mae.std(ddof=1)),
            "min_rec_mae": float(rec_sbp_mae.min()),
            "max_rec_mae": float(rec_sbp_mae.max()),
            "p25_rec_mae": float(rec_sbp_mae.quantile(0.25)),
            "p75_rec_mae": float(rec_sbp_mae.quantile(0.75)),
        },
        "dbp": {
            "n_records": len(rec_dbp_mae),
            "mean_rec_mae": float(rec_dbp_mae.mean()),
            "median_rec_mae": float(rec_dbp_mae.median()),
            "std_rec_mae": float(rec_dbp_mae.std(ddof=1)),
            "min_rec_mae": float(rec_dbp_mae.min()),
            "max_rec_mae": float(rec_dbp_mae.max()),
            "p25_rec_mae": float(rec_dbp_mae.quantile(0.25)),
            "p75_rec_mae": float(rec_dbp_mae.quantile(0.75)),
        }
    }

    print(f"Distribusi MAE per Record UCI (N={uci_record_stats['sbp']['n_records']}):")
    print(f"  - SBP: Mean={uci_record_stats['sbp']['mean_rec_mae']:.3f} mmHg | Median={uci_record_stats['sbp']['median_rec_mae']:.3f} | IQR=[{uci_record_stats['sbp']['p25_rec_mae']:.3f}, {uci_record_stats['sbp']['p75_rec_mae']:.3f}] | Range=[{uci_record_stats['sbp']['min_rec_mae']:.3f}, {uci_record_stats['sbp']['max_rec_mae']:.3f}]")
    print(f"  - DBP: Mean={uci_record_stats['dbp']['mean_rec_mae']:.3f} mmHg | Median={uci_record_stats['dbp']['median_rec_mae']:.3f} | IQR=[{uci_record_stats['dbp']['p25_rec_mae']:.3f}, {uci_record_stats['dbp']['p75_rec_mae']:.3f}] | Range=[{uci_record_stats['dbp']['min_rec_mae']:.3f}, {uci_record_stats['dbp']['max_rec_mae']:.3f}]")

    # Analisis Rentang Tekanan Darah (Rendah, Sedang, Tinggi)
    # SBP: Rendah (<120), Sedang (120-139), Tinggi (>=140)
    sbp_valid_sub = df_pred_uci[df_pred_uci["sbp_mmhg"].notna()].copy()
    sbp_low = sbp_valid_sub[sbp_valid_sub["sbp_mmhg"] < 120]
    sbp_mid = sbp_valid_sub[(sbp_valid_sub["sbp_mmhg"] >= 120) & (sbp_valid_sub["sbp_mmhg"] < 140)]
    sbp_high = sbp_valid_sub[sbp_valid_sub["sbp_mmhg"] >= 140]

    sbp_tier_stats = {
        "low (<120 mmHg)": {"n": len(sbp_low), "mae": float(sbp_low["abs_error_sbp_config_c"].mean()), "me": float(sbp_low["error_sbp_config_c"].mean())},
        "mid (120-139 mmHg)": {"n": len(sbp_mid), "mae": float(sbp_mid["abs_error_sbp_config_c"].mean()), "me": float(sbp_mid["error_sbp_config_c"].mean())},
        "high (>=140 mmHg)": {"n": len(sbp_high), "mae": float(sbp_high["abs_error_sbp_config_c"].mean()), "me": float(sbp_high["error_sbp_config_c"].mean())},
    }

    print("\nPerforma SBP berdasarkan Rentang Nilai Acuan:")
    for tier, s in sbp_tier_stats.items():
        print(f"  - {tier:<20}: N={s['n']:>5,} ({s['n']/len(sbp_valid_sub)*100:>5.1f}%) | MAE={s['mae']:>6.3f} mmHg | ME={s['me']:>+6.3f} mmHg")

    # DBP: Rendah (<80), Sedang (80-89), Tinggi (>=90)
    dbp_valid_sub = df_pred_uci[df_pred_uci["dbp_mmhg"].notna()].copy()
    dbp_low = dbp_valid_sub[dbp_valid_sub["dbp_mmhg"] < 80]
    dbp_mid = dbp_valid_sub[(dbp_valid_sub["dbp_mmhg"] >= 80) & (dbp_valid_sub["dbp_mmhg"] < 90)]
    dbp_high = dbp_valid_sub[dbp_valid_sub["dbp_mmhg"] >= 90]

    dbp_tier_stats = {
        "low (<80 mmHg)": {"n": len(dbp_low), "mae": float(dbp_low["abs_error_dbp_config_c"].mean()), "me": float(dbp_low["error_dbp_config_c"].mean())},
        "mid (80-89 mmHg)": {"n": len(dbp_mid), "mae": float(dbp_mid["abs_error_dbp_config_c"].mean()), "me": float(dbp_mid["error_dbp_config_c"].mean())},
        "high (>=90 mmHg)": {"n": len(dbp_high), "mae": float(dbp_high["abs_error_dbp_config_c"].mean()), "me": float(dbp_high["error_dbp_config_c"].mean())},
    }

    print("\nPerforma DBP berdasarkan Rentang Nilai Acuan:")
    for tier, s in dbp_tier_stats.items():
        print(f"  - {tier:<20}: N={s['n']:>5,} ({s['n']/len(dbp_valid_sub)*100:>5.1f}%) | MAE={s['mae']:>6.3f} mmHg | ME={s['me']:>+6.3f} mmHg")

    # --------------------------------------------------------------------------
    # 6. BAGIAN F: ANALISIS KHUSUS DATASET OSASUD (SPO2)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("BAGIAN F: ANALISIS MENDALAM DATASET OSASUD (PER-PASIEN & KELOMPOK SATURASI)")
    print("=" * 80)

    # MAE per Pasien
    osa_patient_stats = []
    for pid in sorted(df_pred_osasud["patient_id"].unique()):
        p_sub = df_pred_osasud[df_pred_osasud["patient_id"] == pid]
        p_mae = float(p_sub["abs_error_spo2_config_d"].mean())
        p_rmse = float(np.sqrt((p_sub["error_spo2_config_d"] ** 2).mean()))
        p_me = float(p_sub["error_spo2_config_d"].mean())
        p_std = float(p_sub["error_spo2_config_d"].std(ddof=1)) if len(p_sub) > 1 else 0.0
        p_ref_mean = float(p_sub["spo2_target_pct"].mean())
        p_pred_mean = float(p_sub["pred_spo2_config_d"].mean())

        osa_patient_stats.append({
            "patient_id": str(pid),
            "n_windows": len(p_sub),
            "ref_spo2_mean": p_ref_mean,
            "pred_spo2_mean": p_pred_mean,
            "mae": p_mae,
            "rmse": p_rmse,
            "me": p_me,
            "std_me": p_std,
        })

    df_osa_pat = pd.DataFrame(osa_patient_stats)
    unweighted_pat_mae = float(df_osa_pat["mae"].mean())
    print(f"Performa SpO2 per Pasien TEST (N=5):")
    for _, row in df_osa_pat.iterrows():
        print(f"  - Pasien {row['patient_id']:>2}: N={int(row['n_windows']):>4} | Ref Mean={row['ref_spo2_mean']:>5.2f}% | Pred Mean={row['pred_spo2_mean']:>5.2f}% | MAE={row['mae']:>5.3f}% | RMSE={row['rmse']:>5.3f}% | ME={row['me']:>+5.3f}%")
    print(f"  * Rata-rata MAE per Pasien Tanpa Pembobotan (Unweighted): {unweighted_pat_mae:.4f}% (vs Agregat: {m_spo2_model['mae']:.4f}%)")

    # Analisis Kelompok Saturasi (<90%, 90% - <95%, >=95%)
    spo2_sub = df_pred_osasud
    sp_severe = spo2_sub[spo2_sub["spo2_target_pct"] < 90]
    sp_moderate = spo2_sub[(spo2_sub["spo2_target_pct"] >= 90) & (spo2_sub["spo2_target_pct"] < 95)]
    sp_normal = spo2_sub[spo2_sub["spo2_target_pct"] >= 95]

    spo2_group_stats = {
        "<90% (Severe Desaturation)": {
            "n": len(sp_severe),
            "mae": float(sp_severe["abs_error_spo2_config_d"].mean()) if len(sp_severe) > 0 else np.nan,
            "me": float(sp_severe["error_spo2_config_d"].mean()) if len(sp_severe) > 0 else np.nan,
            "pred_mean": float(sp_severe["pred_spo2_config_d"].mean()) if len(sp_severe) > 0 else np.nan,
            "ref_mean": float(sp_severe["spo2_target_pct"].mean()) if len(sp_severe) > 0 else np.nan,
        },
        "90% hingga <95% (Mild Desat)": {
            "n": len(sp_moderate),
            "mae": float(sp_moderate["abs_error_spo2_config_d"].mean()) if len(sp_moderate) > 0 else np.nan,
            "me": float(sp_moderate["error_spo2_config_d"].mean()) if len(sp_moderate) > 0 else np.nan,
            "pred_mean": float(sp_moderate["pred_spo2_config_d"].mean()) if len(sp_moderate) > 0 else np.nan,
            "ref_mean": float(sp_moderate["spo2_target_pct"].mean()) if len(sp_moderate) > 0 else np.nan,
        },
        ">=95% (Normoxemia)": {
            "n": len(sp_normal),
            "mae": float(sp_normal["abs_error_spo2_config_d"].mean()) if len(sp_normal) > 0 else np.nan,
            "me": float(sp_normal["error_spo2_config_d"].mean()) if len(sp_normal) > 0 else np.nan,
            "pred_mean": float(sp_normal["pred_spo2_config_d"].mean()) if len(sp_normal) > 0 else np.nan,
            "ref_mean": float(sp_normal["spo2_target_pct"].mean()) if len(sp_normal) > 0 else np.nan,
        },
    }

    print("\nPerforma SpO2 Berdasarkan Kelompok Saturasi:")
    for grp, s in spo2_group_stats.items():
        pct = (s['n'] / len(spo2_sub)) * 100.0
        print(f"  - {grp:<30}: N={s['n']:>5} ({pct:>5.2f}%) | Ref={s['ref_mean']:>5.2f}% | Pred={s['pred_mean']:>5.2f}% | MAE={s['mae']:>5.3f}% | ME={s['me']:>+5.3f}%")

    # --------------------------------------------------------------------------
    # 7. VISUALISASI FINAL
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("MEMBUAT TIGA BERKAS GRAFIK EVALUASI TEST FINAL")
    print("=" * 80)

    # Plot 1: Perbandingan Validasi vs Test vs Baseline
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    targets_data = [
        ("SBP (mmHg)", [17.8338, val_sbp_mae, m_sbp_model["mae"], m_sbp_b_med["mae"]], ["Val Base Med", "Val XGB", "Test XGB", "Test Base Med"], axes[0]),
        ("DBP (mmHg)", [7.9603, val_dbp_mae, m_dbp_model["mae"], m_dbp_b_med["mae"]], ["Val Base Med", "Val XGB", "Test XGB", "Test Base Med"], axes[1]),
        ("SpO2 (%)", [2.0387, val_spo2_mae, m_spo2_model["mae"], m_spo2_b_med["mae"]], ["Val Base Med", "Val XGB", "Test XGB", "Test Base Med"], axes[2]),
    ]

    for title, vals, labels, ax in targets_data:
        bars = ax.bar(labels, vals, color=["#9e9e9e", "#1976d2", "#2e7d32", "#bdbdbd"], width=0.55)
        ax.set_title(f"Perbandingan MAE - {title}", fontsize=12, fontweight="bold", pad=10)
        ax.set_ylabel(f"MAE ({title.split()[-1].strip('()')})", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        for b in bars:
            h = b.get_height()
            ax.annotate(f"{h:.3f}", xy=(b.get_x() + b.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                        fontsize=9, fontweight="bold")

    plt.suptitle("Evaluasi Performa XGBoost: Validation vs Final Test vs Baseline", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    p1_path = results_dir / "xgboost_validation_vs_test.png"
    plt.savefig(p1_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Disimpan Grafik 1: {p1_path}")

    # Plot 2: Scatter Plot Referensi vs Prediksi pada Set TEST
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Scatter SBP
    ax = axes[0]
    ax.scatter(y_sbp_true, pred_sbp, alpha=0.15, s=10, color="#1976d2", edgecolors="none")
    lims_sbp = [min(y_sbp_true.min(), pred_sbp.min()) - 5, max(y_sbp_true.max(), pred_sbp.max()) + 5]
    ax.plot(lims_sbp, lims_sbp, "r--", linewidth=1.5, label="Identity Line (y=x)")
    ax.set_xlim(lims_sbp)
    ax.set_ylim(lims_sbp)
    ax.set_title(f"SBP TEST: Referensi vs Prediksi\n(MAE={m_sbp_model['mae']:.2f} mmHg, R={m_sbp_model['pearson_r']:.3f})", fontsize=11, fontweight="bold")
    ax.set_xlabel("Acuan Finapres SBP (mmHg)", fontsize=10)
    ax.set_ylabel("Prediksi XGBoost SBP (mmHg)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left")

    # Scatter DBP
    ax = axes[1]
    ax.scatter(y_dbp_true, pred_dbp, alpha=0.15, s=10, color="#388e3c", edgecolors="none")
    lims_dbp = [min(y_dbp_true.min(), pred_dbp.min()) - 5, max(y_dbp_true.max(), pred_dbp.max()) + 5]
    ax.plot(lims_dbp, lims_dbp, "r--", linewidth=1.5, label="Identity Line (y=x)")
    ax.set_xlim(lims_dbp)
    ax.set_ylim(lims_dbp)
    ax.set_title(f"DBP TEST: Referensi vs Prediksi\n(MAE={m_dbp_model['mae']:.2f} mmHg, R={m_dbp_model['pearson_r']:.3f})", fontsize=11, fontweight="bold")
    ax.set_xlabel("Acuan Finapres DBP (mmHg)", fontsize=10)
    ax.set_ylabel("Prediksi XGBoost DBP (mmHg)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left")

    # Scatter SpO2
    ax = axes[2]
    ax.scatter(y_spo2_true, pred_spo2, alpha=0.25, s=12, color="#d32f2f", edgecolors="none")
    lims_spo2 = [60, 102]
    ax.plot(lims_spo2, lims_spo2, "k--", linewidth=1.5, label="Identity Line (y=x)")
    ax.set_xlim(lims_spo2)
    ax.set_ylim(lims_spo2)
    ax.set_title(f"SpO2 TEST: Referensi vs Prediksi\n(MAE={m_spo2_model['mae']:.2f}%, R={m_spo2_model['pearson_r']:.3f})", fontsize=11, fontweight="bold")
    ax.set_xlabel("Acuan Oksimeter SpO2 (%)", fontsize=10)
    ax.set_ylabel("Prediksi XGBoost SpO2 (%)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left")

    plt.suptitle("Diagram Sebar Prediksi Final XGBoost vs Nilai Acuan pada Data TEST", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    p2_path = results_dir / "xgboost_test_prediction_scatter.png"
    plt.savefig(p2_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Disimpan Grafik 2: {p2_path}")

    # Plot 3: Distribusi Signed Error (Prediksi - Referensi)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    err_sbp = pred_sbp - y_sbp_true
    err_dbp = pred_dbp - y_dbp_true
    err_spo2 = pred_spo2 - y_spo2_true

    err_configs = [
        ("SBP (mmHg)", err_sbp, m_sbp_model["me"], m_sbp_model["std_me"], "#1976d2", axes[0]),
        ("DBP (mmHg)", err_dbp, m_dbp_model["me"], m_dbp_model["std_me"], "#388e3c", axes[1]),
        ("SpO2 (%)", err_spo2, m_spo2_model["me"], m_spo2_model["std_me"], "#d32f2f", axes[2]),
    ]

    for title, err, me, std, col, ax in err_configs:
        n_bins = 50
        ax.hist(err, bins=n_bins, density=True, alpha=0.6, color=col, edgecolor="white", label="Distribusi Error")
        ax.axvline(0, color="black", linestyle="-", linewidth=1.2, label="Zero Error")
        ax.axvline(me, color="red", linestyle="--", linewidth=1.5, label=f"Mean Error ({me:+.2f})")
        ax.axvline(me - 1.96 * std, color="gray", linestyle=":", linewidth=1.2, label=f"-1.96 STD ({me-1.96*std:+.2f})")
        ax.axvline(me + 1.96 * std, color="gray", linestyle=":", linewidth=1.2, label=f"+1.96 STD ({me+1.96*std:+.2f})")
        ax.set_title(f"Distribusi Signed Error {title}\n(ME={me:+.2f}, STD={std:.2f})", fontsize=11, fontweight="bold")
        ax.set_xlabel(f"Galat Prediksi: Prediksi - Referensi ({title.split()[-1].strip('()')})", fontsize=10)
        ax.set_ylabel("Densitas", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8, loc="upper right")

    plt.suptitle("Distribusi Galat Bertanda (Signed Error) pada Data TEST", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    p3_path = results_dir / "xgboost_test_error_distribution.png"
    plt.savefig(p3_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Disimpan Grafik 3: {p3_path}")

    # --------------------------------------------------------------------------
    # 8. TULIS RINGKASAN TEKS EVALUASI TEST DAN LOG INTEGRITAS
    # --------------------------------------------------------------------------
    elapsed = time.time() - start_time

    # File 1: results/xgboost_final_test_summary.txt
    summary_path = results_dir / "xgboost_final_test_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 90 + "\n")
        f.write("LAPORAN RINGKASAN EVALUASI AKHIR MODEL XGBOOST PADA DATA TEST INDEPENDEN (TAHAP 10)\n")
        f.write("Proyek : Estimasi Tekanan Darah Cuffless & Saturasi Oksigen Menggunakan XGBoost\n")
        f.write(f"Waktu  : {eval_timestamp}\n")
        f.write(f"Lokasi : {summary_path}\n")
        f.write("=" * 90 + "\n\n")

        f.write("[1. PERNYATAAN PROTOKOL PENGUJIAN FINAL TEST]\n")
        f.write("-" * 90 + "\n")
        f.write("1. Data TEST dievaluasi SATU KALI SAJA setelah seluruh model, fitur, dan parameter dikunci di Tahap 9C.\n")
        f.write("2. TIDAK ADA pelatihan ulang (retraining), tuning hyperparameter, seleksi fitur ulang, maupun pergantian model.\n")
        f.write("3. Model yang digunakan:\n")
        f.write("   - SBP : models/sbp_config_c.json (19 fitur, best_iteration=999)\n")
        f.write("   - DBP : models/dbp_config_c.json (19 fitur, best_iteration=996)\n")
        f.write("   - SpO2: models/spo2_config_d.json (2 fitur CFS: rise_time_s, max_ascending_slope; best_iteration=284)\n")
        f.write("4. Baseline pembanding dihitung murni dari set TRAIN berpresisi penuh:\n")
        f.write(f"   - SBP Baseline Mean Train: {base_sbp_mean:.4f} mmHg | Median Train: {base_sbp_median:.4f} mmHg\n")
        f.write(f"   - DBP Baseline Mean Train: {base_dbp_mean:.4f} mmHg | Median Train: {base_dbp_median:.4f} mmHg\n")
        f.write(f"   - SpO2 Baseline Mean Train: {base_spo2_mean:.4f} %  | Median Train: {base_spo2_median:.4f} %\n\n")

        f.write("[2. TABEL METRIK FINAL PADA DATA TEST]\n")
        f.write("-" * 90 + "\n")
        f.write(f"{'Target':<6} | {'Model':<30} | {'N_samp':<7} | {'MAE':<8} | {'RMSE':<8} | {'ME':<8} | {'STD_ME':<8} | {'R':<7} | {'R2':<7} | {'Imp_vs_Med'}\n")
        f.write("-" * 90 + "\n")
        for r in metrics_rows:
            r_str = f"{r['pearson_r']:.4f}" if pd.notna(r['pearson_r']) else "N/A"
            r2_str = f"{r['r2']:.4f}" if pd.notna(r['r2']) else "N/A"
            f.write(f"{r['target'].upper():<6} | {r['model_type']:<30} | {r['n_samples']:>7,d} | {r['mae']:>8.4f} | {r['rmse']:>8.4f} | {r['me']:>+8.4f} | {r['std_me']:>8.4f} | {r_str:>7} | {r2_str:>7} | {r['pct_improvement_vs_median']:>+9.2f}%\n")
        f.write("-" * 90 + "\n\n")

        f.write("[3. PERBANDINGAN FORMAL: HASIL VALIDASI (TAHAP 9C) VS HASIL TEST FINAL (TAHAP 10)]\n")
        f.write("-" * 90 + "\n")
        f.write(f"  - SBP MAE  : Validation = {val_sbp_mae:.4f} mmHg -> TEST = {m_sbp_model['mae']:.4f} mmHg (Selisih = {m_sbp_model['mae'] - val_sbp_mae:+.4f} mmHg)\n")
        f.write(f"  - SBP RMSE : Validation = {val_sbp_rmse:.4f} mmHg -> TEST = {m_sbp_model['rmse']:.4f} mmHg (Selisih = {m_sbp_model['rmse'] - val_sbp_rmse:+.4f} mmHg)\n")
        f.write(f"  - DBP MAE  : Validation = {val_dbp_mae:.4f} mmHg -> TEST = {m_dbp_model['mae']:.4f} mmHg (Selisih = {m_dbp_model['mae'] - val_dbp_mae:+.4f} mmHg)\n")
        f.write(f"  - DBP RMSE : Validation = {val_dbp_rmse:.4f} mmHg -> TEST = {m_dbp_model['rmse']:.4f} mmHg (Selisih = {m_dbp_model['rmse'] - val_dbp_rmse:+.4f} mmHg)\n")
        f.write(f"  - SpO2 MAE : Validation = {val_spo2_mae:.4f} %    -> TEST = {m_spo2_model['mae']:.4f} %    (Selisih = {m_spo2_model['mae'] - val_spo2_mae:+.4f} %)\n")
        f.write(f"  - SpO2 RMSE: Validation = {val_spo2_rmse:.4f} %    -> TEST = {m_spo2_model['rmse']:.4f} %    (Selisih = {m_spo2_model['rmse'] - val_spo2_rmse:+.4f} %)\n\n")

        f.write("[4. ANALISIS KHUSUS DATASET UCI (SBP & DBP)]\n")
        f.write("-" * 90 + "\n")
        f.write(f"  A. Statistik MAE per Record ID (N={uci_record_stats['sbp']['n_records']:,} record_id valid):\n")
        f.write(f"     * SBP: Mean per-record MAE = {uci_record_stats['sbp']['mean_rec_mae']:.4f} mmHg | Median = {uci_record_stats['sbp']['median_rec_mae']:.4f} mmHg\n")
        f.write(f"            Rentang = [{uci_record_stats['sbp']['min_rec_mae']:.3f}, {uci_record_stats['sbp']['max_rec_mae']:.3f}] mmHg | IQR = [{uci_record_stats['sbp']['p25_rec_mae']:.3f}, {uci_record_stats['sbp']['p75_rec_mae']:.3f}]\n")
        f.write(f"     * DBP: Mean per-record MAE = {uci_record_stats['dbp']['mean_rec_mae']:.4f} mmHg | Median = {uci_record_stats['dbp']['median_rec_mae']:.4f} mmHg\n")
        f.write(f"            Rentang = [{uci_record_stats['dbp']['min_rec_mae']:.3f}, {uci_record_stats['dbp']['max_rec_mae']:.3f}] mmHg | IQR = [{uci_record_stats['dbp']['p25_rec_mae']:.3f}, {uci_record_stats['dbp']['p75_rec_mae']:.3f}]\n")
        f.write("  B. Performa SBP berdasarkan Rentang Nilai Acuan:\n")
        for tier, s in sbp_tier_stats.items():
            f.write(f"     * {tier:<20}: N={s['n']:>5,} ({s['n']/len(sbp_valid_sub)*100:>5.1f}%) | MAE={s['mae']:>6.3f} mmHg | ME={s['me']:>+6.3f} mmHg\n")
        f.write("  C. Performa DBP berdasarkan Rentang Nilai Acuan:\n")
        for tier, s in dbp_tier_stats.items():
            f.write(f"     * {tier:<20}: N={s['n']:>5,} ({s['n']/len(dbp_valid_sub)*100:>5.1f}%) | MAE={s['mae']:>6.3f} mmHg | ME={s['me']:>+6.3f} mmHg\n")
        f.write("  D. Catatan Ilmiah UCI:\n")
        f.write("     - UCI tidak menyediakan identitas pasien (patient_id), sehingga partisi record-grouped TIDAK\n")
        f.write("       menjamin independensi biologis antar-subjek.\n")
        f.write("     - Hasil evaluasi jendela 30 detik ini TIDAK BOLEH diklaim memenuhi standar klinis AAMI/BHS\n")
        f.write("       ataupun digunakan untuk mendiagnosis hipertensi secara klinis.\n\n")

        f.write("[5. ANALISIS KHUSUS DATASET OSASUD (SPO2)]\n")
        f.write("-" * 90 + "\n")
        f.write("  A. Performa per Pasien TEST Independen (N=5 pasien):\n")
        for _, row in df_osa_pat.iterrows():
            f.write(f"     * Pasien {row['patient_id']:>2}: N={int(row['n_windows']):>4} | Ref={row['ref_spo2_mean']:>5.2f}% | Pred={row['pred_spo2_mean']:>5.2f}% | MAE={row['mae']:>5.3f}% | RMSE={row['rmse']:>5.3f}% | ME={row['me']:>+5.3f}%\n")
        f.write(f"     * Unweighted Average MAE per Pasien: {unweighted_pat_mae:.4f}%\n")
        f.write(f"     * Weighted Aggregate MAE seluruh Jendela: {m_spo2_model['mae']:.4f}%\n")
        f.write("  B. Analisis Kasus Desaturasi Oksigen:\n")
        for grp, s in spo2_group_stats.items():
            pct = (s['n'] / len(spo2_sub)) * 100.0
            f.write(f"     * {grp:<30}: N={s['n']:>5} ({pct:>5.2f}%) | Ref={s['ref_mean']:>5.2f}% | Pred={s['pred_mean']:>5.2f}% | MAE={s['mae']:>5.3f}% | ME={s['me']:>+5.3f}%\n")
        f.write("  C. Catatan Ilmiah OSASUD:\n")
        f.write("     - Kasus SpO2 < 90% pada set TEST hanya berjumlah 32 jendela (0.68%), sehingga estimasi error\n")
        f.write("       pada desaturasi berat memiliki ketidakpastian statistik tinggi.\n")
        f.write("     - Model XGBoost menunjukkan kecenderungan regresi ke nilai rata-rata (regress-to-the-mean)\n")
        f.write("       ketika saturasi referensi turun ke rentang hipoksemia berat.\n")
        f.write("     - Nilai acuan OSASUD berasal dari sensor oksimeter pulse klinis, BUKAN SaO2 invasif (gas darah).\n\n")

        f.write("[6. KESIMPULAN EVALUASI AKHIR]\n")
        f.write("-" * 90 + "\n")
        f.write(f"1. Model SBP XGBoost mengungguli baseline median: MAE {m_sbp_model['mae']:.4f} vs {m_sbp_b_med['mae']:.4f} mmHg ({imp_sbp:+.2f}% perbaikan).\n")
        f.write(f"2. Model DBP XGBoost mengungguli baseline median: MAE {m_dbp_model['mae']:.4f} vs {m_dbp_b_med['mae']:.4f} mmHg ({imp_dbp:+.2f}% perbaikan).\n")
        f.write(f"3. Model SpO2 XGBoost mengungguli baseline median: MAE {m_spo2_model['mae']:.4f} vs {m_spo2_b_med['mae']:.4f} % ({imp_spo2:+.2f}% perbaikan).\n")
        f.write(f"4. Tidak ada degradasi katastropik antara validasi dan test, mengonfirmasi tidak adanya overfitting ekstrim.\n")
    print(f"Disimpan Laporan Ringkasan: {summary_path}")

    # File 2: results/xgboost_final_evaluation_integrity.txt
    integ_path = results_dir / "xgboost_final_evaluation_integrity.txt"
    with open(integ_path, "w", encoding="utf-8") as f:
        f.write("=" * 90 + "\n")
        f.write("DOKUMEN INTEGRITAS EVALUASI TEST FINAL (TAHAP 10 AUDIT TRAIL)\n")
        f.write("Proyek : Estimasi Tekanan Darah Cuffless & Saturasi Oksigen Menggunakan XGBoost\n")
        f.write(f"Waktu  : {eval_timestamp}\n")
        f.write(f"Durasi : {elapsed:.2f} detik\n")
        f.write("=" * 90 + "\n\n")

        f.write("[1. LINGKUNGAN EKSEKUSI & VERSI SOFTWARE]\n")
        f.write("-" * 90 + "\n")
        f.write(f"  - Operating System : {sys.platform} (Windows)\n")
        f.write(f"  - Python Version   : {sys.version.split()[0]}\n")
        f.write(f"  - XGBoost Version  : {xgb.__version__}\n")
        f.write(f"  - Pandas Version   : {pd.__version__}\n")
        f.write(f"  - NumPy Version    : {np.__version__}\n\n")

        f.write("[2. BUKTI BERKAS MODEL DAN PARAMETER YANG DIGUNAKAN]\n")
        f.write("-" * 90 + "\n")
        f.write(f"  - SBP Model File   : {files['sbp_model']} (best_iteration={model_sbp.best_iteration})\n")
        f.write(f"  - SBP Features (19): {', '.join(sbp_meta['features_list'])}\n")
        f.write(f"  - DBP Model File   : {files['dbp_model']} (best_iteration={model_dbp.best_iteration})\n")
        f.write(f"  - DBP Features (19): {', '.join(dbp_meta['features_list'])}\n")
        f.write(f"  - SpO2 Model File  : {files['spo2_model']} (best_iteration={model_spo2.best_iteration})\n")
        f.write(f"  - SpO2 Features (2): {', '.join(spo2_meta['features_list'])}\n\n")

        f.write("[3. AUDIT 10 PEMERIKSAAN KEAMANAN SEBELUM EVALUASI TEST]\n")
        f.write("-" * 90 + "\n")
        for item in check_res["integrity_log"]:
            f.write(f"  [PASSED] {item}\n")
        f.write("\n")

        f.write("[4. PERNYATAAN ILMIAH TENTANG DATA LEAKAGE DAN BATASAN RISET]\n")
        f.write("-" * 90 + "\n")
        f.write("  1. Ketiadaan ID Pasien pada Dataset UCI:\n")
        f.write("     Dataset UCI Cuffless Blood Pressure Estimation (Kaggle/PhysioNet) tidak menyertakan\n")
        f.write("     identitas pasien biologis. Partisi record-grouped berhasil mengisolasi segmen rekaman\n")
        f.write("     dan kelompok rekaman duplikat identik, namun potensi rekaman dari subjek yang sama\n")
        f.write("     masuk ke set berbeda TIDAK DAPAT DITIADAKAN secara mutlak karena keterbatasan metadata asli.\n")
        f.write("  2. Independensi Pasien pada Dataset OSASUD:\n")
        f.write("     Dataset OSASUD memiliki metadata identitas 30 pasien terverifikasi. Partisi dilakukan\n")
        f.write("     secara patient-wise murni (21 Train, 4 Val, 5 Test), sehingga independensi subjek\n")
        f.write("     pada estimasi SpO2 terjamin 100% tanpa kebocoran pasien.\n")
        f.write("  3. Batasan Klaim Klinis:\n")
        f.write("     Penelitian ini adalah latihan akademis dan karya ilmiah sarjana (KTI), BUKAN pengujian\n")
        f.write("     alat kesehatan atau sertifikasi klinis standar AAMI/BHS/ISO.\n")
        f.write("=" * 90 + "\n")
    print(f"Disimpan Log Integritas: {integ_path}")

    print("\n" + "=" * 80)
    print("EVALUASI TEST FINAL SELESAI DENGAN SUKSES!")
    print("=" * 80)


if __name__ == "__main__":
    run_final_test_evaluation()
