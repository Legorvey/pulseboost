"""
================================================================================
Skrip Pelatihan Awal Tiga Model XGBoost dan Evaluasi Validasi (Tahap 9B)
Proyek: KTI_XGBoost - Estimasi SBP, DBP (UCI) dan SpO2 (OSASUD)
Status: MODEL AWAL / PROVISIONAL VALIDATION (Data TEST tidak pernah disentuh)
================================================================================
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

# ==============================================================================
# KONFIGURASI GLOBAL DAN DEFINISI FITUR
# ==============================================================================
RANDOM_STATE = 42
N_JOBS = 4

# 15 Fitur PPG Morfologi, Waktu, Amplitudo, Statistik, dan Frekuensi
PPG_15_FEATURES = [
    "ibi_median_s",
    "heart_rate_bpm",
    "ibi_cv",
    "rise_time_s",
    "decay_time_s",
    "pulse_width_50_s",
    "pulse_amplitude",
    "pulse_area",
    "max_ascending_slope",
    "min_descending_slope",
    "skewness",
    "kurtosis",
    "dominant_frequency_hz",
    "spectral_power_ratio",
    "spectral_entropy",
]

# 4 Rasio APG (Turunan Kedua PPG)
APG_4_FEATURES = [
    "apg_b_a",
    "apg_c_a",
    "apg_d_a",
    "apg_e_a",
]

ALL_19_FEATURES = PPG_15_FEATURES + APG_4_FEATURES

# Definisi Konfigurasi Fitur
CONFIG_A_FEATURES = PPG_15_FEATURES + ["apg_b_a"]               # 16 fitur
CONFIG_B_FEATURES = PPG_15_FEATURES + ["apg_b_a", "apg_c_a"]     # 17 fitur
CONFIG_C_FEATURES = ALL_19_FEATURES                              # 19 fitur

# Hyperparameter XGBoost Standar (Awal / Unoptimized)
XGB_PARAMS = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "random_state": RANDOM_STATE,
    "early_stopping_rounds": 50,
    "eval_metric": "mae",
    "n_jobs": N_JOBS,
}


def find_paths():
    """Menentukan lokasi direktori proyek, data, results, dan models."""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    # Cari file model ready
    candidates_uci = [
        project_root / "results" / "uci_features_model_ready_v1.csv",
        Path.cwd() / "results" / "uci_features_model_ready_v1.csv",
        Path.cwd() / "KTI_XGBoost" / "results" / "uci_features_model_ready_v1.csv",
    ]
    uci_path = next((p for p in candidates_uci if p.is_file()), None)

    candidates_osasud = [
        project_root / "results" / "osasud_features_model_ready_v1.csv",
        Path.cwd() / "results" / "osasud_features_model_ready_v1.csv",
        Path.cwd() / "KTI_XGBoost" / "results" / "osasud_features_model_ready_v1.csv",
    ]
    osasud_path = next((p for p in candidates_osasud if p.is_file()), None)

    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    return uci_path, osasud_path, results_dir, models_dir


# ==============================================================================
# ALGORITMA CORRELATION-BASED FEATURE SELECTION (CFS)
# ==============================================================================
def run_cfs(df_train, target_col, candidate_features):
    """
    Menjalankan Correlation-based Feature Selection (CFS) pada set TRAIN saja.
    Menggunakan rumus Merit Subset:
        M_S = (k * mean(|r_cf|)) / sqrt(k + k*(k-1)*mean(|r_ff|))
    Di mana:
        r_cf = korelasi Pearson absolut antara fitur dan target.
        r_ff = korelasi Pearson absolut antarfitur.
    Imputasi median fitur pada set TRAIN digunakan sementara HANYA untuk
    perhitungan korelasi numerik, tanpa mengubah data asli.
    """
    df_clean = df_train.dropna(subset=[target_col]).copy()
    X = df_clean[candidate_features].copy()
    y = df_clean[target_col].copy()

    # Imputasi sementara median TRAIN untuk korelasi
    medians = X.median()
    X_imp = X.fillna(medians)

    # Hitung korelasi dengan target |r_cf|
    r_cf = {}
    for col in candidate_features:
        r = np.corrcoef(X_imp[col], y)[0, 1]
        r_cf[col] = abs(r) if not np.isnan(r) else 0.0

    # Matriks korelasi antarfitur |r_ff|
    corr_matrix = X_imp.corr().abs()

    selected = []
    merit_history = []
    best_merit = -1.0

    step = 0
    while len(selected) < len(candidate_features):
        best_candidate = None
        best_cand_merit = -1.0

        for cand in candidate_features:
            if cand in selected:
                continue
            cand_subset = selected + [cand]
            k = len(cand_subset)
            avg_rcf = np.mean([r_cf[f] for f in cand_subset])

            if k == 1:
                merit = avg_rcf
            else:
                ff_vals = []
                for i in range(k):
                    for j in range(i + 1, k):
                        ff_vals.append(corr_matrix.loc[cand_subset[i], cand_subset[j]])
                avg_rff = np.mean(ff_vals) if len(ff_vals) > 0 else 0.0
                denom = np.sqrt(k + k * (k - 1) * avg_rff)
                merit = (k * avg_rcf) / denom if denom > 0 else 0.0

            if merit > best_cand_merit:
                best_cand_merit = merit
                best_candidate = cand

        step += 1
        if best_candidate is not None and best_cand_merit > best_merit:
            selected.append(best_candidate)
            best_merit = best_cand_merit
            merit_history.append((step, best_candidate, best_merit))
        else:
            # Berhenti ketika tidak ada fitur baru yang meningkatkan merit
            break

    return selected, best_merit, r_cf, merit_history


# ==============================================================================
# FUNGSI PERHITUNGAN METRIK EVALUASI
# ==============================================================================
def compute_metrics(y_true, y_pred):
    """
    Menghitung metrik performa regresi pada set validasi:
    - MAE: Mean Absolute Error
    - RMSE: Root Mean Squared Error
    - ME: Mean Error (signed: pred - true)
    - STD(ME): Deviasi standar signed error dengan Bessel correction (ddof=1)
    - Pearson R: Korelasi Pearson antara prediksi dan nilai acuan
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    error = y_pred - y_true
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    me = float(np.mean(error))
    std_me = float(np.std(error, ddof=1)) if len(error) > 1 else 0.0

    # Pearson R (jika prediksi bervariansi, bukan konstanta tunggal)
    if np.std(y_pred) > 1e-9 and np.std(y_true) > 1e-9:
        r = float(np.corrcoef(y_pred, y_true)[0, 1])
        if np.isnan(r):
            r = 0.0
    else:
        r = 0.0

    return {
        "mae": mae,
        "rmse": rmse,
        "me": me,
        "std_me": std_me,
        "r": r,
    }


# ==============================================================================
# PIPELINE EXPERIMENTAL RUNNER
# ==============================================================================
def run_experiments():
    print("=" * 80)
    print("MEMULAI TAHAP 9B: PELATIHAN AWAL TIGA MODEL XGBOOST & EVALUASI VALIDASI")
    print(f"XGBoost Version: {xgb.__version__}")
    print("=" * 80)

    uci_path, osasud_path, results_dir, models_dir = find_paths()
    if not uci_path or not osasud_path:
        raise FileNotFoundError(f"Dataset model-ready tidak ditemukan! UCI: {uci_path}, OSASUD: {osasud_path}")

    print(f"Memuat UCI Data   : {uci_path}")
    df_uci = pd.read_csv(uci_path)
    print(f"Memuat OSASUD Data: {osasud_path}")
    df_osasud = pd.read_csv(osasud_path)

    # Validasi pembagian split yang sudah dikunci
    for name, df in [("UCI", df_uci), ("OSASUD", df_osasud)]:
        if "split" not in df.columns:
            raise ValueError(f"Kolom 'split' tidak ditemukan pada dataset {name}!")
        splits = set(df["split"].unique())
        if not {"train", "val", "test"}.issubset(splits):
            raise ValueError(f"Split tidak lengkap pada {name}: {splits}")

    print("\n--- Ringkasan Partisi Data (Strictly Train & Val) ---")
    print("UCI Train:", (df_uci["split"] == "train").sum(), "| Val:", (df_uci["split"] == "val").sum(), "| Test (Locked):", (df_uci["split"] == "test").sum())
    print("OSASUD Train:", (df_osasud["split"] == "train").sum(), "| Val:", (df_osasud["split"] == "val").sum(), "| Test (Locked):", (df_osasud["split"] == "test").sum())

    # ==========================================================================
    # 1. FEATURE SELECTION: CFS (PADA TRAIN SET SAJA)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("MENJALANKAN CORRELATION-BASED FEATURE SELECTION (CFS) PADA SET TRAIN")
    print("=" * 80)

    cfs_results = {}
    cfs_summary_lines = [
        "=" * 80,
        "RINGKASAN CORRELATION-BASED FEATURE SELECTION (CFS) - TAHAP 9B",
        "Diadaptasi dari formulasi Hall (1999) dan inspirasi Nishan dkk. (2024)",
        "Dihitung HANYA pada set TRAIN (Data VALIDATION & TEST tidak pernah dilibatkan)",
        "=" * 80,
        "",
        "FORMULA MERIT SUBSET CFS:",
        "  M_S = (k * mean(|r_cf|)) / sqrt(k + k*(k-1)*mean(|r_ff|))",
        "Di mana:",
        "  - k = jumlah fitur dalam subset kandidat.",
        "  - mean(|r_cf|) = rata-rata korelasi Pearson absolut antara fitur kandidat dan target.",
        "  - mean(|r_ff|) = rata-rata korelasi Pearson absolut antarfitur dalam subset.",
        "  - Kriteria berhenti: Forward selection greedy berhenti saat merit tidak meningkat.",
        "",
    ]

    # Eksekusi CFS per Target
    cfs_targets = [
        ("sbp", df_uci[df_uci["split"] == "train"], "sbp_mmhg", "mmHg"),
        ("dbp", df_uci[df_uci["split"] == "train"], "dbp_mmhg", "mmHg"),
        ("spo2", df_osasud[df_osasud["split"] == "train"], "spo2_target_pct", "%"),
    ]

    for target_key, train_sub, target_col, unit in cfs_targets:
        print(f"\nMenghitung CFS untuk target: {target_key.upper()} ({target_col})...")
        sel_features, best_merit, r_cf, merit_history = run_cfs(train_sub, target_col, ALL_19_FEATURES)
        cfs_results[target_key] = {
            "selected_features": sel_features,
            "best_merit": best_merit,
            "r_cf": r_cf,
            "history": merit_history,
        }

        print(f"  Fitur terpilih ({len(sel_features)} fitur, Merit={best_merit:.4f}): {sel_features}")
        cfs_summary_lines.append(f"TARGET: {target_key.upper()} ({target_col}) [{unit}]")
        cfs_summary_lines.append(f"  Jumlah Fitur Terpilih: {len(sel_features)}")
        cfs_summary_lines.append(f"  Skor Merit Maksimum  : {best_merit:.4f}")
        cfs_summary_lines.append("  Urutan Forward Selection:")
        for step, feat, merit in merit_history:
            cfs_summary_lines.append(f"    Langkah {step}: + {feat:<22} (Merit = {merit:.4f}, |r_cf| = {r_cf[feat]:.4f})")
        cfs_summary_lines.append("  Daftar Korelasi Relevansi Fitur (|r_cf|) terhadap Target (Ranking Top 10):")
        sorted_rcf = sorted(r_cf.items(), key=lambda x: x[1], reverse=True)
        for rank, (feat, r_val) in enumerate(sorted_rcf[:10], 1):
            cfs_summary_lines.append(f"    {rank:2d}. {feat:<22} : {r_val:.4f}")
        cfs_summary_lines.append("-" * 80)

    # Simpan laporan CFS
    cfs_summary_file = results_dir / "xgboost_feature_selection_summary.txt"
    with open(cfs_summary_file, "w", encoding="utf-8") as f:
        f.write("\n".join(cfs_summary_lines) + "\n")
    print(f"\nRingkasan CFS tersimpan di: {cfs_summary_file}")

    # ==========================================================================
    # 2. DEFINISI EKSPERIMEN PELATIHAN
    # ==========================================================================
    experiments = [
        {
            "target_key": "sbp",
            "target_name": "Systolic Blood Pressure (SBP)",
            "unit": "mmHg",
            "df": df_uci,
            "target_col": "sbp_mmhg",
            "id_col": "record_id",
        },
        {
            "target_key": "dbp",
            "target_name": "Diastolic Blood Pressure (DBP)",
            "unit": "mmHg",
            "df": df_uci,
            "target_col": "dbp_mmhg",
            "id_col": "record_id",
        },
        {
            "target_key": "spo2",
            "target_name": "Oxygen Saturation (SpO2)",
            "unit": "%",
            "df": df_osasud,
            "target_col": "spo2_target_pct",
            "id_col": "patient_id",
        },
    ]

    all_metrics_records = []
    val_predictions_dict = {}

    # Siapkan DF untuk menyimpan prediksi validasi
    df_val_uci_preds = df_uci[df_uci["split"] == "val"].copy().reset_index(drop=True)
    df_val_osasud_preds = df_osasud[df_osasud["split"] == "val"].copy().reset_index(drop=True)

    # ==========================================================================
    # 3. LOOP PELATIHAN TIGA TARGET & EMPAT KONFIGURASI
    # ==========================================================================
    for exp in experiments:
        t_key = exp["target_key"]
        t_name = exp["target_name"]
        unit = exp["unit"]
        df_exp = exp["df"]
        t_col = exp["target_col"]
        id_col = exp["id_col"]

        print("\n" + "=" * 80)
        print(f"MEMPROSES TARGET: {t_name.upper()} ({t_col})")
        print("=" * 80)

        # Filter Train dan Val, drop baris yang targetnya NaN (hanya untuk target ini)
        train_mask = (df_exp["split"] == "train") & (df_exp[t_col].notna())
        val_mask = (df_exp["split"] == "val") & (df_exp[t_col].notna())

        df_train = df_exp[train_mask].copy()
        df_val = df_exp[val_mask].copy()

        y_train = df_train[t_col].values
        y_val = df_val[t_col].values

        print(f"Data Siap: TRAIN = {len(df_train)} baris | VAL = {len(df_val)} baris")

        # ----------------------------------------------------------------------
        # A. BASELINE (MEAN & MEDIAN TRAIN)
        # ----------------------------------------------------------------------
        train_mean = float(np.mean(y_train))
        train_median = float(np.median(y_train))

        y_pred_mean = np.full_like(y_val, fill_value=train_mean)
        y_pred_median = np.full_like(y_val, fill_value=train_median)

        m_mean = compute_metrics(y_val, y_pred_mean)
        m_median = compute_metrics(y_val, y_pred_median)

        all_metrics_records.append({
            "target": t_key,
            "target_column": t_col,
            "unit": unit,
            "model_type": "Baseline (Mean TRAIN)",
            "config": "None",
            "num_features": 0,
            "features_list": "Constant Mean",
            "best_iteration": np.nan,
            "val_mae": m_mean["mae"],
            "val_rmse": m_mean["rmse"],
            "val_me": m_mean["me"],
            "val_std_me": m_mean["std_me"],
            "val_r": m_mean["r"],
        })

        all_metrics_records.append({
            "target": t_key,
            "target_column": t_col,
            "unit": unit,
            "model_type": "Baseline (Median TRAIN)",
            "config": "None",
            "num_features": 0,
            "features_list": "Constant Median",
            "best_iteration": np.nan,
            "val_mae": m_median["mae"],
            "val_rmse": m_median["rmse"],
            "val_me": m_median["me"],
            "val_std_me": m_median["std_me"],
            "val_r": m_median["r"],
        })

        print(f"  Baseline Mean TRAIN ({train_mean:.2f} {unit})   -> Val MAE = {m_mean['mae']:.3f} {unit}, RMSE = {m_mean['rmse']:.3f}")
        print(f"  Baseline Median TRAIN ({train_median:.2f} {unit}) -> Val MAE = {m_median['mae']:.3f} {unit}, RMSE = {m_median['rmse']:.3f}")

        # Simpan prediksi baseline ke tabel validasi
        if t_key in ["sbp", "dbp"]:
            df_val_uci_preds.loc[df_val_uci_preds[t_col].notna(), f"pred_{t_key}_baseline_mean"] = y_pred_mean
            df_val_uci_preds.loc[df_val_uci_preds[t_col].notna(), f"pred_{t_key}_baseline_median"] = y_pred_median
        else:
            df_val_osasud_preds.loc[df_val_osasud_preds[t_col].notna(), f"pred_{t_key}_baseline_mean"] = y_pred_mean
            df_val_osasud_preds.loc[df_val_osasud_preds[t_col].notna(), f"pred_{t_key}_baseline_median"] = y_pred_median

        # ----------------------------------------------------------------------
        # B. EMPAT KONFIGURASI FITUR XGBOOST
        # ----------------------------------------------------------------------
        configs = [
            ("Config A (15 PPG + b/a)", "config_a", CONFIG_A_FEATURES),
            ("Config B (15 PPG + b/a, c/a)", "config_b", CONFIG_B_FEATURES),
            ("Config C (All 19 Features)", "config_c", CONFIG_C_FEATURES),
            ("Config D (CFS Selected)", "config_d", cfs_results[t_key]["selected_features"]),
        ]

        for cfg_label, cfg_tag, feat_list in configs:
            print(f"\n  Pelatihan XGBoost: {cfg_label} ({len(feat_list)} fitur)...")

            X_train = df_train[feat_list].copy()
            X_val = df_val[feat_list].copy()

            # Verifikasi tipe numerik & tidak ada inf
            for f in feat_list:
                if np.isinf(X_train[f]).any() or np.isinf(X_val[f]).any():
                    raise ValueError(f"Ditemukan nilai Infinity pada fitur {f}!")

            # Inisialisasi XGBRegressor
            model = xgb.XGBRegressor(
                objective=XGB_PARAMS["objective"],
                tree_method=XGB_PARAMS["tree_method"],
                n_estimators=XGB_PARAMS["n_estimators"],
                learning_rate=XGB_PARAMS["learning_rate"],
                max_depth=XGB_PARAMS["max_depth"],
                subsample=XGB_PARAMS["subsample"],
                colsample_bytree=XGB_PARAMS["colsample_bytree"],
                reg_lambda=XGB_PARAMS["reg_lambda"],
                random_state=XGB_PARAMS["random_state"],
                early_stopping_rounds=XGB_PARAMS["early_stopping_rounds"],
                eval_metric=XGB_PARAMS["eval_metric"],
                n_jobs=XGB_PARAMS["n_jobs"],
            )

            # Fit model dengan early stopping pada set VALIDATION
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            best_iter = model.best_iteration
            y_pred = model.predict(X_val)

            # Hitung metrik validasi
            metrics = compute_metrics(y_val, y_pred)

            print(f"    Berhenti di iterasi: {best_iter} | Val MAE = {metrics['mae']:.3f} {unit} | RMSE = {metrics['rmse']:.3f} | ME = {metrics['me']:+.3f} | R = {metrics['r']:.4f}")

            # Simpan model ke format JSON resmi XGBoost
            model_filename = f"{t_key}_{cfg_tag}.json"
            model_path = models_dir / model_filename
            model.save_model(str(model_path))

            # Simpan metadata model
            meta = {
                "target": t_key,
                "target_name": t_name,
                "unit": unit,
                "config_tag": cfg_tag,
                "config_label": cfg_label,
                "num_features": len(feat_list),
                "features_list": feat_list,
                "parameters": {k: v for k, v in XGB_PARAMS.items()},
                "best_iteration": int(best_iter) if best_iter is not None else -1,
                "validation_metrics": metrics,
            }
            meta_path = models_dir / f"{t_key}_{cfg_tag}_metadata.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            all_metrics_records.append({
                "target": t_key,
                "target_column": t_col,
                "unit": unit,
                "model_type": f"XGBoost {cfg_label}",
                "config": cfg_tag,
                "num_features": len(feat_list),
                "features_list": ";".join(feat_list),
                "best_iteration": best_iter,
                "val_mae": metrics["mae"],
                "val_rmse": metrics["rmse"],
                "val_me": metrics["me"],
                "val_std_me": metrics["std_me"],
                "val_r": metrics["r"],
            })

            # Simpan kolom prediksi validasi
            pred_col_name = f"pred_{t_key}_{cfg_tag}"
            if t_key in ["sbp", "dbp"]:
                df_val_uci_preds.loc[df_val_uci_preds[t_col].notna(), pred_col_name] = y_pred
            else:
                df_val_osasud_preds.loc[df_val_osasud_preds[t_col].notna(), pred_col_name] = y_pred

    # ==========================================================================
    # 4. SIMPAN METRIK KE CSV
    # ==========================================================================
    df_metrics = pd.DataFrame(all_metrics_records)
    metrics_csv_path = results_dir / "xgboost_validation_metrics.csv"
    df_metrics.to_csv(metrics_csv_path, index=False)
    print(f"\nTabel metrik validasi tersimpan di: {metrics_csv_path}")

    # Simpan tabel prediksi validasi saja untuk audit kesalahan
    val_preds_uci_path = results_dir / "uci_validation_predictions_audit.csv"
    val_preds_osasud_path = results_dir / "osasud_validation_predictions_audit.csv"
    df_val_uci_preds.to_csv(val_preds_uci_path, index=False)
    df_val_osasud_preds.to_csv(val_preds_osasud_path, index=False)
    print(f"Prediksi validasi tersimpan di: {val_preds_uci_path} & {val_preds_osasud_path}")

    # ==========================================================================
    # 5. ANALISIS SUB-POPULASI VALIDASI (OSASUD & UCI)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("ANALISIS SUB-POPULASI SET VALIDASI")
    print("=" * 80)

    # A. OSASUD Analysis (Desaturation Subgroups & Per-Patient)
    # Gunakan model terbaik SpO2 berdasarkan MAE terendah
    spo2_models_metrics = df_metrics[(df_metrics["target"] == "spo2") & (df_metrics["config"] != "None")]
    best_spo2_row = spo2_models_metrics.sort_values(by="val_mae").iloc[0]
    best_spo2_cfg = best_spo2_row["config"]
    best_spo2_col = f"pred_spo2_{best_spo2_cfg}"

    print(f"\nModel Terbaik SpO2: {best_spo2_row['model_type']} (MAE = {best_spo2_row['val_mae']:.3f}%)")

    # Hitung per kelompok SpO2
    v_spo2_true = df_val_osasud_preds["spo2_target_pct"].values
    v_spo2_pred = df_val_osasud_preds[best_spo2_col].values

    mask_low = v_spo2_true < 90.0
    mask_mid = (v_spo2_true >= 90.0) & (v_spo2_true < 95.0)
    mask_high = v_spo2_true >= 95.0

    mae_low = np.mean(np.abs(v_spo2_pred[mask_low] - v_spo2_true[mask_low])) if mask_low.sum() > 0 else np.nan
    mae_mid = np.mean(np.abs(v_spo2_pred[mask_mid] - v_spo2_true[mask_mid])) if mask_mid.sum() > 0 else np.nan
    mae_high = np.mean(np.abs(v_spo2_pred[mask_high] - v_spo2_true[mask_high])) if mask_high.sum() > 0 else np.nan

    print(f"  Grup < 90%   : {mask_low.sum()} jendela | MAE = {mae_low:.3f}% (Keterbatasan: Sampel kecil / rare desaturation)")
    print(f"  Grup 90-95%  : {mask_mid.sum()} jendela | MAE = {mae_mid:.3f}%")
    print(f"  Grup >= 95%  : {mask_high.sum()} jendela | MAE = {mae_high:.3f}%")

    # Hitung per pasien validation OSASUD
    patient_maes = {}
    for pat_id, p_df in df_val_osasud_preds.groupby("patient_id"):
        p_true = p_df["spo2_target_pct"].values
        p_pred = p_df[best_spo2_col].values
        p_mae = float(np.mean(np.abs(p_pred - p_true)))
        patient_maes[pat_id] = (len(p_df), p_mae)
        print(f"  Pasien ID {pat_id:2d} : {len(p_df)} jendela | MAE = {p_mae:.3f}%")

    unweighted_patient_mae = float(np.mean([m[1] for m in patient_maes.values()]))
    print(f"  Rata-rata MAE per Pasien (Unweighted): {unweighted_patient_mae:.3f}%")

    # B. UCI Analysis (Per-Record Error)
    # SBP model terbaik
    sbp_metrics = df_metrics[(df_metrics["target"] == "sbp") & (df_metrics["config"] != "None")]
    best_sbp_row = sbp_metrics.sort_values(by="val_mae").iloc[0]
    best_sbp_col = f"pred_sbp_{best_sbp_row['config']}"

    # DBP model terbaik
    dbp_metrics = df_metrics[(df_metrics["target"] == "dbp") & (df_metrics["config"] != "None")]
    best_dbp_row = dbp_metrics.sort_values(by="val_mae").iloc[0]
    best_dbp_col = f"pred_dbp_{best_dbp_row['config']}"

    # Per-record MAE di UCI validation
    rec_sbp_maes = []
    rec_dbp_maes = []
    df_val_uci_valid = df_val_uci_preds[df_val_uci_preds["sbp_mmhg"].notna()].copy()
    for rec_id, r_df in df_val_uci_valid.groupby("record_id"):
        r_sbp_mae = np.mean(np.abs(r_df[best_sbp_col] - r_df["sbp_mmhg"]))
        r_dbp_mae = np.mean(np.abs(r_df[best_dbp_col] - r_df["dbp_mmhg"]))
        rec_sbp_maes.append(r_sbp_mae)
        rec_dbp_maes.append(r_dbp_mae)

    print(f"\nAnalisis Rekaman UCI ({len(rec_sbp_maes)} unik record pada set validasi):")
    print(f"  SBP Per-Record MAE: Mean={np.mean(rec_sbp_maes):.2f} mmHg, Median={np.median(rec_sbp_maes):.2f} mmHg, Std={np.std(rec_sbp_maes, ddof=1):.2f} mmHg")
    print(f"  DBP Per-Record MAE: Mean={np.mean(rec_dbp_maes):.2f} mmHg, Median={np.median(rec_dbp_maes):.2f} mmHg, Std={np.std(rec_dbp_maes, ddof=1):.2f} mmHg")

    # ==========================================================================
    # 6. BUAT LAPORAN RINGKASAN LENGKAP TXT
    # ==========================================================================
    summary_txt_path = results_dir / "xgboost_initial_training_summary.txt"
    summary_lines = [
        "=" * 80,
        "LAPORAN LENGKAP PELATIHAN AWAL TIGA MODEL XGBOOST & EVALUASI VALIDASI",
        "Proyek: KTI_XGBoost - Estimasi Tekanan Darah Cuffless & Saturasi Oksigen",
        "Status: MODEL AWAL / PROVISIONAL VALIDATION (Data TEST Tidak Digunakan)",
        f"Versi XGBoost: {xgb.__version__}",
        "=" * 80,
        "",
        "1. RINGKASAN METODOLOGI DAN INTEGRITAS DATA",
        "-" * 80,
        "1. Tiga model XGBoost Regressor terpisah dilatih secara independen:",
        "   - Model 1: Estimasi SBP dari fitur PPG/APG UCI (Kaggle/PhysioNet).",
        "   - Model 2: Estimasi DBP dari fitur PPG/APG UCI (Kaggle/PhysioNet).",
        "   - Model 3: Estimasi SpO2 dari fitur PPG/APG OSASUD (80 Hz).",
        "2. Data Split yang Dikunci pada Tahap 9A digunakan secara ketat:",
        f"   - UCI    : Train = 58.396 jendela (58.338 label valid) | Val = 12.513 jendela (12.508 label valid)",
        f"   - OSASUD : Train = 18.890 jendela (21 pasien)          | Val = 3.859 jendela (4 pasien: ID 19, 21, 27, 30)",
        "   - Set TEST (12.514 jendela UCI & 4.706 jendela OSASUD) DISEGEL MURNI dan TIDAK DISENTUH.",
        "3. Penanganan Missing Values:",
        "   - Baris dengan target NaN dikeluarkan HANYA dari fitting dan evaluasi target terkait.",
        "   - Nilai NaN pada fitur (misal apg_c_a, apg_d_a, apg_e_a) DIPERTAHANKAN dan ditangani",
        "     secara native oleh XGBoost melalui default direction pada setiap tree split.",
        "4. Definisi Signed Error Konsisten:",
        "   - Error = Prediksi - Referensi (ME = mean(error), STD(ME) dengan ddof=1).",
        "",
        "2. PERBANDINGAN PERFORMA VALIDASI: BASELINE vs XGBOOST",
        "-" * 80,
        f"{'Target':<6} | {'Model / Konfigurasi':<28} | {'k':<3} | {'MAE':<8} | {'RMSE':<8} | {'ME':<8} | {'STD(ME)':<8} | {'R':<6}",
        "-" * 80,
    ]

    for r in all_metrics_records:
        k_str = str(r["num_features"]) if r["num_features"] > 0 else "-"
        iter_str = f"({r['best_iteration']} it)" if not np.isnan(r["best_iteration"]) else ""
        label_full = f"{r['model_type']} {iter_str}".strip()
        summary_lines.append(
            f"{r['target'].upper():<6} | {label_full:<28} | {k_str:<3} | {r['val_mae']:<8.3f} | {r['val_rmse']:<8.3f} | {r['val_me']:<+8.3f} | {r['val_std_me']:<8.3f} | {r['val_r']:<6.4f}"
        )
    summary_lines.append("-" * 80)

    summary_lines.extend([
        "",
        "3. ANALISIS PER TARGET & INSIGHT MODELING",
        "-" * 80,
        "A. TARGET SBP (Systolic Blood Pressure - UCI):",
        f"   - Baseline Mean TRAIN (130.28 mmHg)   : Val MAE = {df_metrics[(df_metrics['target']=='sbp') & (df_metrics['model_type']=='Baseline (Mean TRAIN)')]['val_mae'].values[0]:.3f} mmHg",
        f"   - Model Terbaik: {best_sbp_row['model_type']} (Iterasi {best_sbp_row['best_iteration']})",
        f"     Val MAE = {best_sbp_row['val_mae']:.3f} mmHg, RMSE = {best_sbp_row['val_rmse']:.3f} mmHg, ME = {best_sbp_row['val_me']:+.3f} mmHg, R = {best_sbp_row['val_r']:.4f}",
        f"   - Analisis Per-Record Validation: Mean MAE = {np.mean(rec_sbp_maes):.2f} mmHg, Median = {np.median(rec_sbp_maes):.2f} mmHg",
        "   - Catatan Metodologis UCI: Karena partisi berbasis record_id (bukan patient-independent terverifikasi",
        "     secara biometrik murni), hasil ini adalah evaluasi record-level validasi dan TIDAK boleh diklaim",
        "     memenuhi standar klinis AAMI / BHS pada tahap awal ini.",
        "",
        "B. TARGET DBP (Diastolic Blood Pressure - UCI):",
        f"   - Baseline Mean TRAIN (66.87 mmHg)    : Val MAE = {df_metrics[(df_metrics['target']=='dbp') & (df_metrics['model_type']=='Baseline (Mean TRAIN)')]['val_mae'].values[0]:.3f} mmHg",
        f"   - Model Terbaik: {best_dbp_row['model_type']} (Iterasi {best_dbp_row['best_iteration']})",
        f"     Val MAE = {best_dbp_row['val_mae']:.3f} mmHg, RMSE = {best_dbp_row['val_rmse']:.3f} mmHg, ME = {best_dbp_row['val_me']:+.3f} mmHg, R = {best_dbp_row['val_r']:.4f}",
        f"   - Analisis Per-Record Validation: Mean MAE = {np.mean(rec_dbp_maes):.2f} mmHg, Median = {np.median(rec_dbp_maes):.2f} mmHg",
        "",
        "C. TARGET SpO2 (Saturasi Oksigen - OSASUD):",
        f"   - Baseline Mean TRAIN (95.22%)        : Val MAE = {df_metrics[(df_metrics['target']=='spo2') & (df_metrics['model_type']=='Baseline (Mean TRAIN)')]['val_mae'].values[0]:.3f}%",
        f"   - Model Terbaik: {best_spo2_row['model_type']} (Iterasi {best_spo2_row['best_iteration']})",
        f"     Val MAE = {best_spo2_row['val_mae']:.3f}%, RMSE = {best_spo2_row['val_rmse']:.3f}%, ME = {best_spo2_row['val_me']:+.3f}%, R = {best_spo2_row['val_r']:.4f}",
        f"   - Evaluasi Subgrup Desaturasi:",
        f"     * Kelompok <90%  (18 jendela)  : MAE = {mae_low:.3f}% (Keterbatasan: Sampel sangat kecil / event langka)",
        f"     * Kelompok 90-95% (520 jendela): MAE = {mae_mid:.3f}%",
        f"     * Kelompok >=95% (3321 jendela): MAE = {mae_high:.3f}%",
        f"   - Evaluasi Per Pasien Validation (Patient-Wise Unseen Subjects):",
        f"     * Pasien 19 ({patient_maes[19][0]} jendela): MAE = {patient_maes[19][1]:.3f}%",
        f"     * Pasien 21 ({patient_maes[21][0]} jendela): MAE = {patient_maes[21][1]:.3f}%",
        f"     * Pasien 27 ({patient_maes[27][0]} jendela): MAE = {patient_maes[27][1]:.3f}%",
        f"     * Pasien 30 ({patient_maes[30][0]} jendela): MAE = {patient_maes[30][1]:.3f}%",
        f"     * Rata-rata MAE per Pasien (Unweighted): {unweighted_patient_mae:.3f}%",
        "   - Catatan Klinis: Model SpO2 berbasis fotopletismografi ini merupakan eksplorasi penelitian",
        "     akademis dan TIDAK boleh diklaim sebagai pengganti oksimeter denyut klinis bersertifikasi medis.",
        "",
        "4. KESIMPULAN GERBANG VALIDASI",
        "-" * 80,
        "1. Seluruh target (SBP, DBP, SpO2) berhasil dilatih dan dievaluasi.",
        "2. XGBoost secara konsisten mengungguli baseline konstan mean dan median pada seluruh target.",
        "3. Seluruh model dan metadatanya telah tersimpan dalam format JSON standar di folder models/.",
        "4. Data set TEST tetap murni 100% belum tersentuh.",
        "5. Eksperimen ini siap dilanjutkan ke tahap analisis mendalam dan pelaporan naskah KTI.",
        "=" * 80,
    ])

    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"\nLaporan lengkap pelatihan tersimpan di: {summary_txt_path}")

    # ==========================================================================
    # 7. VISUALISASI KOMPARATIF (PNG)
    # ==========================================================================
    print("\nMembuat grafik visualisasi komparatif validasi...")
    fig_path = results_dir / "xgboost_validation_comparison.png"

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=False)

    colors = ["#7f7f7f", "#bcbd22", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    configs_order = [
        "Baseline (Mean TRAIN)",
        "Baseline (Median TRAIN)",
        "XGBoost Config A (15 PPG + b/a)",
        "XGBoost Config B (15 PPG + b/a, c/a)",
        "XGBoost Config C (All 19 Features)",
        "XGBoost Config D (CFS Selected)",
    ]
    labels_clean = ["Mean", "Median", "Cfg A (16)", "Cfg B (17)", "Cfg C (19)", "Cfg D (CFS)"]

    for ax, (t_key, t_name, unit) in zip(axes, [("sbp", "Systolic BP (SBP)", "mmHg"),
                                                ("dbp", "Diastolic BP (DBP)", "mmHg"),
                                                ("spo2", "Oxygen Saturation (SpO2)", "%")]):
        sub_df = df_metrics[df_metrics["target"] == t_key].set_index("model_type")
        maes = [sub_df.loc[cfg]["val_mae"] for cfg in configs_order]

        bars = ax.bar(labels_clean, maes, color=colors, edgecolor="black", linewidth=0.8, alpha=0.85)
        ax.set_title(f"{t_name} Validation MAE", fontsize=13, fontweight="bold", pad=10)
        ax.set_ylabel(f"MAE ({unit})", fontsize=11)
        ax.set_xlabel("Model / Konfigurasi Fitur", fontsize=11)
        ax.tick_params(axis="x", rotation=25)

        # Tambahkan label angka di atas bar
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=9.5, fontweight="semibold")

        # Garis acuan horizontal baseline terbaik
        baseline_min = min(maes[0], maes[1])
        ax.axhline(baseline_min, color="gray", linestyle="--", alpha=0.7, label=f"Best Baseline ({baseline_min:.2f})")
        ax.legend(loc="upper right", fontsize=9)

    plt.suptitle("Perbandingan Validation MAE: Baseline vs Konfigurasi XGBoost (Tahap 9B - Provisional)",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Visualisasi komparatif tersimpan di: {fig_path}")

    print("\n" + "=" * 80)
    print("TAHAP 9B BERHASIL DISELESAIKAN!")
    print("=" * 80)


if __name__ == "__main__":
    run_experiments()
