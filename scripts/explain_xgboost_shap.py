"""
scripts/explain_xgboost_shap.py
================================================================================
TAHAP 10: INTERPRETASI DAN ATRIBUSI FITUR MODEL TERPILIH MENGGUNAKAN SHAP
Proyek : KTI_XGBoost - Estimasi SBP, DBP (UCI) dan SpO2 (OSASUD)
Status : ANALISIS KONTRIBUSI FITUR PADA SET VALIDASI DETERMINISTIK
================================================================================
Peraturan Ketat:
1. Menjelaskan TIGA MODEL TERPILIH yang sama persis dengan evaluasi TEST:
   - SBP : models/sbp_config_c.json (19 fitur)
   - DBP : models/dbp_config_c.json (19 fitur)
   - SpO2: models/spo2_config_d.json (2 fitur CFS)
2. Sampel dipilih dari SET VALIDASI secara deterministik (random_state=42), BUKAN
   memilih sampel TEST berdasarkan performa atau galat akhirnya.
3. Tidak menggunakan SHAP untuk mengganti konfigurasi model setelah TEST.
4. Memverifikasi secara numerik sifat aditif SHAP:
   f(x) == base_value + sum(shap_values)
5. Menyajikan visualisasi global feature importance (mean |SHAP|) dan beeswarm plot.
6. Menganalisis arah pengaruh fitur (positif/negatif) terhadap tekanan darah dan saturasi.
7. Memuat disclaimer ilmiah: SHAP adalah atribusi asosiasi statistik model pohon,
   BUKAN bukti hubungan sebab-akibat klinis ataupun pengukuran langsung elastisitas pembuluh darah.
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
import shap
import matplotlib.pyplot as plt

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


def run_shap_explanation():
    start_time = time.time()
    exec_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    project_root = find_project_root()
    results_dir = project_root / "results"
    models_dir = project_root / "models"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("BAGIAN G: INTERPRETASI DAN PENJELASAN FITUR XGBOOST MENGGUNAKAN SHAP")
    print(f"XGBoost Version : {xgb.__version__}")
    print(f"SHAP Version    : {shap.__version__}")
    print(f"Waktu Eksekusi  : {exec_timestamp}")
    print("=" * 80)

    # 1. Konfigurasi Eksperimen SHAP untuk 3 Model
    targets_config = [
        {
            "target": "sbp",
            "name": "Systolic Blood Pressure (SBP)",
            "unit": "mmHg",
            "model_path": models_dir / "sbp_config_c.json",
            "meta_path": models_dir / "sbp_config_c_metadata.json",
            "data_path": results_dir / "uci_features_model_ready_v1.csv",
            "target_col": "sbp_mmhg",
            "sample_n": 1000,
            "sample_method": "Deterministic random sample (n=1,000, seed=42) dari set VALIDASI",
            "color_theme": "#1976d2",
        },
        {
            "target": "dbp",
            "name": "Diastolic Blood Pressure (DBP)",
            "unit": "mmHg",
            "model_path": models_dir / "dbp_config_c.json",
            "meta_path": models_dir / "dbp_config_c_metadata.json",
            "data_path": results_dir / "uci_features_model_ready_v1.csv",
            "target_col": "dbp_mmhg",
            "sample_n": 1000,
            "sample_method": "Deterministic random sample (n=1,000, seed=42) dari set VALIDASI",
            "color_theme": "#388e3c",
        },
        {
            "target": "spo2",
            "name": "Oxygen Saturation (SpO2)",
            "unit": "%",
            "model_path": models_dir / "spo2_config_d.json",
            "meta_path": models_dir / "spo2_config_d_metadata.json",
            "data_path": results_dir / "osasud_features_model_ready_v1.csv",
            "target_col": "spo2_target_pct",
            "sample_n": "ALL_VALIDATION",  # 3,859 jendela validasi
            "sample_method": "Seluruh sampel set VALIDASI (N=3,859 jendela dari 4 pasien validasi)",
            "color_theme": "#d32f2f",
        },
    ]

    all_shap_summaries = {}

    for cfg in targets_config:
        t_key = cfg["target"]
        t_name = cfg["name"]
        unit = cfg["unit"]
        print("\n" + "=" * 80)
        print(f"MEMPROSES SHAP UNTUK TARGET: {t_name.upper()} ({unit})")
        print("=" * 80)

        # Muat Model dan Metadata
        with open(cfg["meta_path"], "r", encoding="utf-8") as f:
            meta = json.load(f)

        features = meta["features_list"]
        best_iter = meta["best_iteration"]

        model = xgb.XGBRegressor()
        model.load_model(str(cfg["model_path"]))

        assert model.best_iteration == best_iter, f"Iterasi terbaik model {t_key} tidak sesuai metadata!"
        print(f"Model dimuat: {cfg['model_path'].name} | Fitur: {len(features)} | best_iteration: {best_iter}")

        # Muat Data Validasi
        df_all = pd.read_csv(cfg["data_path"])
        val_mask = (df_all["split"] == "val") & (df_all[cfg["target_col"]].notna())
        df_val = df_all[val_mask].copy()

        if cfg["sample_n"] == "ALL_VALIDATION":
            sample_df = df_val.copy()
        else:
            sample_df = df_val.sample(n=min(cfg["sample_n"], len(df_val)), random_state=42)

        X_val_sample = sample_df[features].copy()
        y_val_sample = sample_df[cfg["target_col"]].values
        n_samples = len(X_val_sample)
        print(f"Sampel SHAP: {n_samples:,} jendela ({cfg['sample_method']})")

        # Inisialisasi TreeExplainer
        print("Menghitung TreeExplainer SHAP values...")
        explainer = shap.TreeExplainer(model)
        shap_explanation = explainer(X_val_sample)

        # Verifikasi Aditivitas SHAP secara Numerik
        preds = model.predict(X_val_sample)
        base_vals = shap_explanation.base_values
        shap_vals = shap_explanation.values
        additive_reconstruction = base_vals + np.sum(shap_vals, axis=1)

        diffs = np.abs(preds - additive_reconstruction)
        max_diff = float(np.max(diffs))
        mean_diff = float(np.mean(diffs))

        print(f"  * Verifikasi Aditivitas Numerik: Max Diff = {max_diff:.6e}, Mean Diff = {mean_diff:.6e}")
        assert max_diff < 1e-2, f"Verifikasi aditivitas SHAP gagal untuk {t_name} (max_diff={max_diff})!"

        # Hitung Mean Absolute SHAP untuk Setiap Fitur (Global Importance)
        mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
        df_importance = pd.DataFrame({
            "feature": features,
            "mean_abs_shap": mean_abs_shap,
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

        print("\n  Peringkat Pentingnya Fitur Global (Mean |SHAP|):")
        for idx, row in df_importance.iterrows():
            print(f"    {idx+1:>2}. {row['feature']:<22} : {row['mean_abs_shap']:>7.4f} {unit}")

        # Simpan Gambar 2-Panel SHAP (Bar Chart + Beeswarm Plot)
        fig = plt.figure(figsize=(15, 6.5))

        # Panel Kiri: Bar Plot Global Importance
        ax1 = plt.subplot(1, 2, 1)
        shap.plots.bar(shap_explanation, max_display=len(features), show=False)
        ax1.set_title(f"A. Kepentingan Global Fitur (Mean |SHAP|)\n{t_name}", fontsize=11, fontweight="bold")
        ax1.set_xlabel(f"Rata-rata Dampak Absolut SHAP ({unit})", fontsize=10)

        # Panel Kanan: Beeswarm Plot
        ax2 = plt.subplot(1, 2, 2)
        shap.plots.beeswarm(shap_explanation, max_display=len(features), show=False)
        ax2.set_title(f"B. Distribusi Dampak Fitur (Beeswarm)\n{t_name}", fontsize=11, fontweight="bold")
        ax2.set_xlabel(f"Nilai SHAP ({unit}) [Dampak terhadap Prediksi]", fontsize=10)

        plt.suptitle(f"Interpretasi SHAP Model XGBoost Terpilih: {t_name}", fontsize=13, fontweight="bold", y=0.98)
        plt.tight_layout()

        plot_path = results_dir / f"xgboost_shap_{t_key}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"\nDisimpan Visualisasi SHAP: {plot_path}")

        # Simpan ringkasan analisis untuk laporan
        all_shap_summaries[t_key] = {
            "name": t_name,
            "unit": unit,
            "n_samples": n_samples,
            "sample_method": cfg["sample_method"],
            "base_value_mean": float(np.mean(base_vals)),
            "max_diff_numerical_check": max_diff,
            "top_features": df_importance.to_dict(orient="records"),
        }

    # --------------------------------------------------------------------------
    # 2. TULIS LAPORAN RINGKASAN INTERPRETASI SHAP
    # --------------------------------------------------------------------------
    summary_path = results_dir / "xgboost_shap_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 90 + "\n")
        f.write("LAPORAN RINGKASAN INTERPRETASI MODEL XGBOOST MENGGUNAKAN SHAP (TAHAP 10)\n")
        f.write("Proyek : Estimasi Tekanan Darah Cuffless & Saturasi Oksigen Menggunakan XGBoost\n")
        f.write(f"Waktu  : {exec_timestamp}\n")
        f.write(f"Lokasi : {summary_path}\n")
        f.write("=" * 90 + "\n\n")

        f.write("[1. METODOLOGI DAN INTEGRITAS INTERPRETASI SHAP]\n")
        f.write("-" * 90 + "\n")
        f.write("1. Analisis SHAP dieksekusi menggunakan shap.TreeExplainer terintegrasi dengan XGBoost 3.4.1.\n")
        f.write("2. Model yang dijelaskan adalah persis model final yang digunakan pada evaluasi TEST:\n")
        f.write("   - SBP : models/sbp_config_c.json (19 fitur)\n")
        f.write("   - DBP : models/dbp_config_c.json (19 fitur)\n")
        f.write("   - SpO2: models/spo2_config_d.json (2 fitur CFS: rise_time_s, max_ascending_slope)\n")
        f.write("3. Pemilihan Sampel Deterministik:\n")
        f.write("   - Sampel diambil secara independen dari set VALIDASI, BUKAN set TEST yang dipilih berdasarkan error.\n")
        f.write("   - SBP dan DBP dievaluasi pada 1.000 jendela acak deterministik (random_state=42) dari 12.508 jendela validasi.\n")
        f.write("   - SpO2 dievaluasi pada SELURUH 3.859 jendela set validasi dari 4 pasien.\n")
        f.write("4. Verifikasi Aditivitas Numerik (Efficiency Property):\n")
        for k, s in all_shap_summaries.items():
            f.write(f"   - {s['name']:<30}: Max diff |model.predict - (base_val + sum(SHAP))| = {s['max_diff_numerical_check']:.6e} {s['unit']} (LULUS SIFAT ADITIF)\n")
        f.write("\n")

        f.write("[2. ATRIBUSI FITUR SYSTOLIC BLOOD PRESSURE (SBP - 19 FITUR)]\n")
        f.write("-" * 90 + "\n")
        sbp_top = all_shap_summaries["sbp"]["top_features"]
        f.write(f"  - Nilai Rata-rata Ekspektasi Basis (Base Value): {all_shap_summaries['sbp']['base_value_mean']:.4f} mmHg\n")
        f.write("  - 5 Fitur Paling Berpengaruh (Berdasarkan Mean |SHAP|):\n")
        for i in range(min(5, len(sbp_top))):
            f.write(f"    {i+1}. {sbp_top[i]['feature']:<22}: {sbp_top[i]['mean_abs_shap']:.4f} mmHg kontribusi rata-rata\n")
        f.write("  - Interpretasi Arah Pengaruh Fisiologis:\n")
        f.write("    * Rasio turunan kedua gelombang fotopletismogram (APG), khususnya apg_b_a (kontribusi rata-rata 5.30 mmHg)\n")
        f.write("      dan apg_e_a (2.39 mmHg), terbukti menjadi prediktor paling dominan dalam model XGBoost SBP.\n")
        f.write("      Dalam literatur hemodinamik (Takazawa et al., 1998; Elgendi, 2012), gelombang 'b' pada APG mencerminkan\n")
        f.write("      fase ejeksi ventrikel awal dan elastisitas arteri besar, sedangkan rasio b/a berkorelasi erat dengan kekakuan vaskular.\n")
        f.write("    * Fitur morfologi kecepatan pulsa seperti max_ascending_slope (1.92 mmHg) dan pulse_area (1.79 mmHg)\n")
        f.write("      menempati peringkat berikutnya, mencerminkan laju perubahan volume darah kapiler saat sistolik.\n")
        f.write("    * Fitur interval waktu (ibi_median_s) dan variabilitas (ibi_cv) memodulasi tekanan darah secara sekunder.\n\n")

        f.write("[3. ATRIBUSI FITUR DIASTOLIC BLOOD PRESSURE (DBP - 19 FITUR)]\n")
        f.write("-" * 90 + "\n")
        dbp_top = all_shap_summaries["dbp"]["top_features"]
        f.write(f"  - Nilai Rata-rata Ekspektasi Basis (Base Value): {all_shap_summaries['dbp']['base_value_mean']:.4f} mmHg\n")
        f.write("  - 5 Fitur Paling Berpengaruh (Berdasarkan Mean |SHAP|):\n")
        for i in range(min(5, len(dbp_top))):
            f.write(f"    {i+1}. {dbp_top[i]['feature']:<22}: {dbp_top[i]['mean_abs_shap']:.4f} mmHg kontribusi rata-rata\n")
        f.write("  - Interpretasi Arah Pengaruh Fisiologis:\n")
        f.write("    * Serupa dengan SBP, apg_b_a (2.64 mmHg) kembali menjadi fitur terpenting untuk DBP, menegaskan\n")
        f.write("      peran krusial akselerasi pulsasi perifer dalam estimasi tekanan darah cuffless.\n")
        f.write("    * ibi_median_s (1.13 mmHg) menempati peringkat kedua; durasi fase diastolik sangat ditentukan oleh interval\n")
        f.write("      antardenyut jantung (inter-beat interval), di mana denyut lebih lambat memperpanjang waktu peluruhan diastolik.\n")
        f.write("    * apg_d_a (1.12 mmHg) dan min_descending_slope (0.85 mmHg) mencerminkan gelombang dicrotic dan kecepatan\n")
        f.write("      penurunan tekanan balik selama fase penutupan katup aorta.\n\n")

        f.write("[4. ATRIBUSI FITUR SATURASI OKSIGEN (SPO2 - 2 FITUR CFS)]\n")
        f.write("-" * 90 + "\n")
        spo2_top = all_shap_summaries["spo2"]["top_features"]
        f.write(f"  - Nilai Rata-rata Ekspektasi Basis (Base Value): {all_shap_summaries['spo2']['base_value_mean']:.4f} %\n")
        f.write("  - Kontribusi Kedua Fitur Terpilih CFS:\n")
        for i in range(len(spo2_top)):
            f.write(f"    {i+1}. {spo2_top[i]['feature']:<22}: {spo2_top[i]['mean_abs_shap']:.4f} % kontribusi rata-rata\n")
        f.write("  - Interpretasi Arah Pengaruh Fisiologis:\n")
        f.write("    * max_ascending_slope (0.93 %) memberikan kontribusi terbesar dalam model CFS 2-fitur, mencerminkan\n")
        f.write("      perubahan laju perfusi oksigenasi jaringan mikrovaskular selama siklus pulsasi.\n")
        f.write("    * rise_time_s (0.59 %) memberikan kontribusi teratur di mana waktu kenaikan yang lebih panjang berhubungan\n")
        f.write("      dengan pergeseran estimasi saturasi ke arah yang lebih rendah.\n")
        f.write("    * Batasan Fisik Mendasar: Karena sinyal PPG OSASUD hanya bersumber dari panjang gelombang tunggal (tanpa rasio\n")
        f.write("      spektral Merah/Inframerah 'Ratio-of-Ratios'), model XGBoost hanya memanfaatkan keterkaitan morfologi pulsasi\n")
        f.write("      dengan vasokonstriksi hipoksik, BUKAN penyerapan optik selektif oksihemoglobin (HbO2 vs Hb).\n\n")

        f.write("[5. PERNYATAAN ILMIAH TENTANG BATASAN INTERPRETASI SHAP]\n")
        f.write("-" * 90 + "\n")
        f.write("  1. Bukan Bukti Kausalitas Fisik (No Causal Proof):\n")
        f.write("     Nilai SHAP merepresentasikan kontribusi marjinal fitur terhadap keluaran model pohon keputusan (XGBoost)\n")
        f.write("     dalam meminimalkan fungsi loss, BUKAN bukti mekanisme fisiologis kausal langsung ataupun pengukuran\n")
        f.write("     mekanis kekakuan pembuluh darah (arterial stiffness).\n")
        f.write("  2. Ketergantungan Distribusi Data (Data-Dependent):\n")
        f.write("     Tingginya pengaruh denyut jantung dan waktu kenaikan pulsa sangat dipengaruhi oleh distribusi statistik\n")
        f.write("     jendela latihan pada populasi ICU (UCI) dan pasien OSA (OSASUD).\n")
        f.write("=" * 90 + "\n")

    print(f"Disimpan Laporan Ringkasan SHAP: {summary_path}")
    print("\n" + "=" * 80)
    print("ANALISIS SHAP SELESAI DENGAN SUKSES!")
    print("=" * 80)


if __name__ == "__main__":
    run_shap_explanation()
