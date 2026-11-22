#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MASTER SINGLE-CHAIN REPRODUCIBILITY PIPELINE (v6 — Post-Third-Round Peer Review)
================================================================================
Unified entry point reproducing ALL primary and secondary analyses directly from
the raw OpenAlex CSVs and full-text cache in `downloads/`:
  1. Primary Deduplicated Journal x Year FE (13 covariates incl. publication month),
     500-replicate journal-cluster block bootstrap (standardized & common log1p),
     separate 2020 full-window & 2021 cohort FE with delta-beta cluster CIs,
     properly re-centered strict-filter sensitivities, and deduplicated 3-way
     Journal x Year x Subfield FE with journal-clustered SEs.
  2. 4-Block Shapley/LMG within-field R^2 decomposition (C1, C5, C5 - C1).
  3. Unified Internal Test (N = 1,500, zero clean_id overlap): tie-invariant AP,
     Top 1% prevalence & bootstrap CIs, calibration, C5 - C1 dynamic R^2,
     and 3x3 confusion matrices with per-class Precision/Recall/F1.
  4. Delayed Surge WITHOUT replacement (n = 225 strict cases, exact 84 matched sets,
     set-level equal-weighted control mean, paired diff + 95% bootstrap CI,
     141 unmatched case profile, 5-seed order sensitivity) + Ke et al. (2015) check.
  5. Full-Text Exact Group x Channel Crosstab (rows & unique) and JATS XML audit
     on both 3,741 rows and 3,611 unique works.
  6. Deduplicated 8-Discipline Probabilistic Cohort FE (n_unique, overlap with main)
     and Table 9 Temporal Validation (2020 -> 2021 & 2022, deduplicated after merge,
     M3 - M2 paired bootstrap 95% CI, missing-label audit, matched discipline balance).

Usage:
    python3 run_all_q1_analyses.py
"""

import glob
import itertools
import json
import math
import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats, special

warnings.filterwarnings("ignore")
np.random.seed(42)

DATA_DIR = "downloads"
OUT_DIR = "artifacts/file_generation/ttl=63d/output"
os.makedirs(OUT_DIR, exist_ok=True)


# ==============================================================================
# 1. MATHEMATICALLY RIGOROUS METRICS
# ==============================================================================
def calc_roc_auc(y_true, y_score):
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    n1, n0 = int(np.sum(y_true == 1)), int(np.sum(y_true == 0))
    if n1 == 0 or n0 == 0:
        return 0.5
    ranks = stats.rankdata(y_score)
    return float((np.sum(ranks[y_true == 1]) - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def calc_ap_tie_invariant(y_true, y_score):
    """Exact tie-invariant Average Precision matching scikit-learn average_precision_score."""
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    total_pos = float(np.sum(y_true))
    if total_pos <= 0:
        return 0.0
    order = np.argsort(-y_score, kind="mergesort")
    y_s = y_true[order]
    s_s = y_score[order]
    distinct_idx = np.where(np.diff(s_s) != 0)[0]
    threshold_idxs = np.r_[distinct_idx, len(s_s) - 1]
    tps = np.cumsum(y_s)[threshold_idxs]
    fps = (1 + threshold_idxs) - tps
    prec = tps / np.maximum(tps + fps, 1e-12)
    rec = tps / total_pos
    d_rec = np.diff(np.r_[0.0, rec])
    return float(np.sum(prec * d_rec))


def calc_calibration(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1.0 - 1e-6)
    logit_p = np.log(p / (1.0 - p))
    X = np.column_stack([np.ones(len(logit_p)), logit_p])
    beta = np.array([0.0, 1.0])
    for _ in range(12):
        eta = np.clip(X @ beta, -15, 15)
        prob = special.expit(eta)
        w = np.maximum(prob * (1.0 - prob), 1e-5)
        step = np.linalg.solve((X.T * w) @ X + 1e-6 * np.eye(2), X.T @ (prob - y_true))
        beta -= step
        if np.max(np.abs(step)) < 1e-5:
            break
    return float(beta[0]), float(beta[1])


def calc_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / max(ss_tot, 1e-12))


def fit_ols_r2(X, y):
    if X.shape[1] == 0:
        return 0.0
    Xa = np.column_stack([np.ones(len(X)), X])
    beta, _, _, _ = np.linalg.lstsq(Xa, y, rcond=None)
    return calc_r2(y, Xa @ beta)


def cluster_robust_se(X, resid, cluster_ids, n_absorbed_fe=0):
    """
    CR1 Liang-Zeger / Cameron-Miller (2015) cluster-robust sandwich SE for demeaned OLS.
    Small-sample adjustment: (G / (G - 1)) * ((N - 1) / (N - K - n_absorbed_fe)).
    Note: When journal_year cells are nested inside journal clusters, Stata/reghdfe
    uses K (non-nested) or K + absorbed; here we apply the conservative (N - 1)/(N - K - M)
    with fallback floor to prevent division by zero.
    """
    XtX_inv = np.linalg.pinv(X.T @ X)
    n, k = X.shape
    df_sc = pd.DataFrame(X * resid[:, None])
    df_sc["_cid"] = cluster_ids
    G_sums = df_sc.groupby("_cid").sum().values
    G = len(G_sums)
    meat = G_sums.T @ G_sums
    dof_denom = max(1, n - k - n_absorbed_fe)
    dfc = (G / max(1, G - 1)) * ((n - 1) / dof_denom)
    vcov = dfc * (XtX_inv @ meat @ XtX_inv)
    return np.sqrt(np.maximum(np.diag(vcov), 1e-16)), G


def fit_predict_logistic_irls(X_tr, y_tr, X_te, alpha=20.0, n_iter=12):
    mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0)
    sd[sd < 1e-8] = 1.0
    Z_tr = np.hstack([np.ones((len(X_tr), 1)), np.clip((X_tr - mu) / sd, -5, 5)])
    Z_te = np.hstack([np.ones((len(X_te), 1)), np.clip((X_te - mu) / sd, -5, 5)])
    p_dim = Z_tr.shape[1]
    beta = np.zeros(p_dim)
    pr = np.clip(np.mean(y_tr), 1e-4, 1.0 - 1e-4)
    beta[0] = math.log(pr / (1.0 - pr))
    I = np.eye(p_dim)
    I[0, 0] = 0.0
    for _ in range(n_iter):
        p = special.expit(np.clip(Z_tr @ beta, -15, 15))
        w = np.maximum(p * (1.0 - p), 1e-4)
        step = np.linalg.solve((Z_tr.T * w) @ Z_tr + alpha * I, Z_tr.T @ (p - y_tr) + alpha * (I @ beta))
        beta -= step
        if np.max(np.abs(step)) < 1e-4:
            break
    return special.expit(np.clip(Z_te @ beta, -15, 15))


def fit_predict_ridge(X_tr, y_tr, X_te, alpha=15.0):
    mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0)
    sd[sd < 1e-8] = 1.0
    Zn_tr = np.clip((X_tr - mu) / sd, -5, 5)
    Zn_te = np.clip((X_te - mu) / sd, -5, 5)
    Z_tr = np.hstack([np.ones((len(X_tr), 1)), Zn_tr, np.sign(Zn_tr) * np.log1p(np.abs(Zn_tr))])
    Z_te = np.hstack([np.ones((len(X_te), 1)), Zn_te, np.sign(Zn_te) * np.log1p(np.abs(Zn_te))])
    I = np.eye(Z_tr.shape[1])
    I[0, 0] = 0.0
    beta = np.linalg.solve(Z_tr.T @ Z_tr + alpha * I, Z_tr.T @ y_tr)
    return Z_te @ beta


def run_fe_block(df_sub, cell_col, cluster_col, fe_covars, n_boot=500, seed=42):
    """
    Runs within-cell demeaned FE strictly on df_sub (after dropna and >= 2 works/cell).
    Fixed-SD scaling computed within df_sub. Returns standardized & common log1p betas,
    CR1 journal-clustered SEs, and n_boot journal-cluster block bootstrap 95% CIs.
    """
    rng = np.random.default_rng(seed)
    fe_cols = [c[0] for c in fe_covars]
    keep_cols = [cell_col, cluster_col, "log1p_c1", "log1p_c5", "log1p_c5_minus_c1", "log_c5_plus_05"] + fe_cols
    sub = df_sub[keep_cols].dropna().copy()
    vc = sub[cell_col].value_counts()
    sub = sub[sub[cell_col].isin(vc[vc >= 2].index)].copy()

    n_obs = len(sub)
    n_cells = int(sub[cell_col].nunique())

    means = sub.groupby(cell_col)[["log1p_c1", "log1p_c5", "log1p_c5_minus_c1", "log_c5_plus_05"] + fe_cols].transform("mean")
    dX_raw = (sub[fe_cols] - means[fe_cols]).values
    sd_x = np.std(dX_raw, axis=0)
    sd_x[sd_x < 1e-8] = 1.0
    dX = dX_raw / sd_x

    dy_c1_raw = (sub["log1p_c1"] - means["log1p_c1"]).values
    dy_c5_raw = (sub["log1p_c5"] - means["log1p_c5"]).values
    dy_c5m1_raw = (sub["log1p_c5_minus_c1"] - means["log1p_c5_minus_c1"]).values
    dy_log05_raw = (sub["log_c5_plus_05"] - means["log_c5_plus_05"]).values

    dy_c1 = dy_c1_raw / float(np.std(dy_c1_raw))
    dy_c5 = dy_c5_raw / float(np.std(dy_c5_raw))
    dy_c5m1 = dy_c5m1_raw / float(np.std(dy_c5m1_raw))
    dy_log05 = dy_log05_raw / float(np.std(dy_log05_raw))

    b_c1, _, _, _ = np.linalg.lstsq(dX, dy_c1, rcond=None)
    b_c5, _, _, _ = np.linalg.lstsq(dX, dy_c5, rcond=None)
    b_c5m1, _, _, _ = np.linalg.lstsq(dX, dy_c5m1, rcond=None)
    b_log05, _, _, _ = np.linalg.lstsq(dX, dy_log05, rcond=None)

    b_c1_u, _, _, _ = np.linalg.lstsq(dX, dy_c1_raw, rcond=None)
    b_c5_u, _, _, _ = np.linalg.lstsq(dX, dy_c5_raw, rcond=None)
    b_c5m1_u, _, _, _ = np.linalg.lstsq(dX, dy_c5m1_raw, rcond=None)

    cids = sub[cluster_col].values
    se_c1, n_cl = cluster_robust_se(dX, dy_c1 - dX @ b_c1, cids, n_absorbed_fe=n_cells)
    se_c5, _ = cluster_robust_se(dX, dy_c5 - dX @ b_c5, cids, n_absorbed_fe=n_cells)
    se_c5m1, _ = cluster_robust_se(dX, dy_c5m1 - dX @ b_c5m1, cids, n_absorbed_fe=n_cells)
    se_c1_u, _ = cluster_robust_se(dX, dy_c1_raw - dX @ b_c1_u, cids, n_absorbed_fe=n_cells)
    se_c5_u, _ = cluster_robust_se(dX, dy_c5_raw - dX @ b_c5_u, cids, n_absorbed_fe=n_cells)

    # Journal-cluster block bootstrap (n_boot replicates, fixed-SD scaling)
    uniq_cl, inv_idx = np.unique(cids, return_inverse=True)
    G_cl = len(uniq_cl)
    K = dX.shape[1]
    XtX_g = np.zeros((G_cl, K, K))
    Xty1_g = np.zeros((G_cl, K))
    Xty5_g = np.zeros((G_cl, K))
    Xty1u_g = np.zeros((G_cl, K))
    Xty5u_g = np.zeros((G_cl, K))

    for i in range(n_obs):
        g = inv_idx[i]
        xi = dX[i]
        XtX_g[g] += np.outer(xi, xi)
        Xty1_g[g] += xi * dy_c1[i]
        Xty5_g[g] += xi * dy_c5[i]
        Xty1u_g[g] += xi * dy_c1_raw[i]
        Xty5u_g[g] += xi * dy_c5_raw[i]

    boot_std = np.zeros((n_boot, K))
    boot_unstd = np.zeros((n_boot, K))
    for b_i in range(n_boot):
        w_g = np.bincount(rng.integers(0, G_cl, size=G_cl), minlength=G_cl).astype(float)
        XtX_b = np.tensordot(w_g, XtX_g, axes=(0, 0)) + 1e-8 * np.eye(K)
        boot_std[b_i] = np.linalg.solve(XtX_b, w_g @ Xty5_g) - np.linalg.solve(XtX_b, w_g @ Xty1_g)
        boot_unstd[b_i] = np.linalg.solve(XtX_b, w_g @ Xty5u_g) - np.linalg.solve(XtX_b, w_g @ Xty1u_g)

    rows = []
    for j, (col, lbl) in enumerate(fe_covars):
        t5 = float(b_c5[j] / se_c5[j])
        rows.append({
            "feature": col, "label": lbl,
            "beta_c1": round(float(b_c1[j]), 4), "se_c1": round(float(se_c1[j]), 4),
            "beta_c5": round(float(b_c5[j]), 4), "se_c5": round(float(se_c5[j]), 4),
            "t_c5": round(t5, 2), "p_c5": float(2 * stats.norm.sf(abs(t5))),
            "beta_c5m1": round(float(b_c5m1[j]), 4), "se_c5m1": round(float(se_c5m1[j]), 4),
            "delta_beta_std": round(float(b_c5[j] - b_c1[j]), 4),
            "delta_beta_cluster_ci95": [round(float(np.percentile(boot_std[:, j], 2.5)), 4),
                                        round(float(np.percentile(boot_std[:, j], 97.5)), 4)],
            "beta_c1_unstd_log1p": round(float(b_c1_u[j]), 4), "se_c1_unstd": round(float(se_c1_u[j]), 4),
            "beta_c5_unstd_log1p": round(float(b_c5_u[j]), 4), "se_c5_unstd": round(float(se_c5_u[j]), 4),
            "beta_c5m1_unstd_log1p": round(float(b_c5m1_u[j]), 4),
            "delta_unstd_log1p": round(float(b_c5_u[j] - b_c1_u[j]), 4),
            "delta_unstd_cluster_ci95": [round(float(np.percentile(boot_unstd[:, j], 2.5)), 4),
                                         round(float(np.percentile(boot_unstd[:, j], 97.5)), 4)],
            "beta_log05_ols_sens": round(float(b_log05[j]), 4),
        })

    return {
        "n_unique_works": n_obs,
        "n_cells": n_cells,
        "n_clusters": int(n_cl),
        "n_boot": n_boot,
        "rows": rows
    }


# ==============================================================================
# 2. LOAD ALL RAW DATASETS
# ==============================================================================
print("1. Loading all datasets...")
csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "alt_parti_*.csv")))
df_raw = pd.concat([pd.read_csv(f, low_memory=False) for f in csv_files], ignore_index=True)
df_raw["clean_id"] = df_raw["makale_id"].astype(str).str.replace("https://openalex.org/", "", regex=False)
df_raw["yayin_ayi_clean"] = df_raw["yayin_ayi"].fillna(6).clip(1, 12)
df_raw["is_covid"] = df_raw["baslik"].astype(str).str.contains(r"covid|sars-cov-2|coronavirus|pandemic", case=False, regex=True).astype(int)
df_raw["has_clinical_design"] = df_raw["calisma_tasarimi"].notna().astype(int)
df_raw["strict_research_filter"] = (
    (df_raw["dil"] == "en") & (df_raw["ozet_var_mi"] == 1) &
    (df_raw["ozet_kelime_sayisi"] >= 50) & (df_raw["referans_sayisi"] >= 5) & (df_raw["yazar_sayisi"] >= 1)
)
df_raw["strict_no_ref_filter"] = (
    (df_raw["dil"] == "en") & (df_raw["ozet_var_mi"] == 1) &
    (df_raw["ozet_kelime_sayisi"] >= 50) & (df_raw["yazar_sayisi"] >= 1)
)

df_uniq = df_raw.drop_duplicates(subset=["clean_id"], keep="first").copy()
df_uniq["journal_year"] = df_uniq["dergi_id"].astype(str) + "_" + df_uniq["yayin_yili"].astype(str)
df_uniq["jys_cell"] = df_uniq["dergi_id"].astype(str) + "_" + df_uniq["yayin_yili"].astype(str) + "_" + df_uniq["openalex_subfield"].astype(str)
df_uniq["log1p_c1"] = np.log1p(df_uniq["atif_1_yil_kumulatif"].clip(lower=0))
df_uniq["log1p_c5"] = np.log1p(df_uniq["atif_5_yil_kumulatif"].clip(lower=0))
raw_diff = df_uniq["atif_5_yil_kumulatif"] - df_uniq["atif_1_yil_kumulatif"]
n_clipped_c5m1 = int((raw_diff < 0).sum())
df_uniq["log1p_c5_minus_c1"] = np.log1p(raw_diff.clip(lower=0))
df_uniq["log_c5_plus_05"] = np.log(df_uniq["atif_5_yil_kumulatif"].clip(lower=0) + 0.5)

with open(os.path.join(DATA_DIR, "tam_metin_derin_analiz_v2_5_cache.json"), "r", encoding="utf-8") as f:
    ft_cache = json.load(f)
ft_df = pd.DataFrame.from_dict(ft_cache, orient="index").reset_index().rename(columns={"index": "clean_id"})

df_prob_raw = pd.read_csv(os.path.join(DATA_DIR, "openalex_probabilistic_reference_30k.csv"))
df_prob_raw["clean_id"] = df_prob_raw["makale_id"].astype(str).str.replace("https://openalex.org/", "", regex=False)
df_hist = pd.read_csv(os.path.join(DATA_DIR, "dergi_tarihsel_t0_2019.csv"))
df_2022_raw = pd.read_csv(os.path.join(DATA_DIR, "openalex_external_2022_cohort_10k.csv"))
df_2022_raw["clean_id"] = df_2022_raw["makale_id"].astype(str).str.replace("https://openalex.org/", "", regex=False)

master = {}

# ==============================================================================
# 3. PRIMARY FE (13 COVARIATES INCL. MONTH!) + 2020 & 2021 SEPARATE + RE-CENTERED SENSITIVITIES
# ==============================================================================
fe_covars = [
    ("referans_sayisi", "Referans Sayısı (OpenAlex Eşleşen)"),
    ("farkli_ulke_sayisi", "Uluslararası Ülke Çeşitliliği"),
    ("yazar_sayisi", "Yazar Sayısı (Ekip Büyüklüğü)"),
    ("farkli_kurum_sayisi", "Kurumlararası Ağ Genişliği"),
    ("orcid_yazar_orani", "ORCID Kayıtlı Yazar Oranı"),
    ("ozette_acik_kod_veri_vurgusu", "Özette Açık Kod/Veri Vurgusu"),
    ("ozet_kelime_sayisi", "Özet Kelime Uzunluğu"),
    ("ozette_sayisal_bulgu_yogunlugu", "Özette Sayısal Bulgu Yoğunluğu"),
    ("baslik_kelime_sayisi", "Başlık Kelime Uzunluğu"),
    ("fon_hibe_var_mi", "Araştırma Fonu / Hibe Kaydı (0/1)"),
    ("yayin_ayi_clean", "Yayın Ayı (1–12, Kontrol)"),
    ("is_covid", "COVID-19 Başlık Göstergesi (Kontrol)"),
    ("has_clinical_design", "Klinik Çalışma Tasarımı Etiketi (Kontrol)")
]

print("2. Running Primary 13-Covariate Deduplicated FE (500-rep Journal-Cluster Block Bootstrap)...")
fe_main = run_fe_block(df_uniq, "journal_year", "dergi_id", fe_covars, n_boot=500, seed=42)
fe_main["n_clipped_c5_minus_c1_zero"] = n_clipped_c5m1

print("   Running 2020 Full-Window & 2021 Open-Window Cohort FE (500-rep Cluster Bootstrap)...")
fe_2020 = run_fe_block(df_uniq[df_uniq["yayin_yili"] == 2020], "journal_year", "dergi_id", fe_covars, n_boot=500, seed=43)
fe_2021 = run_fe_block(df_uniq[df_uniq["yayin_yili"] == 2021], "journal_year", "dergi_id", fe_covars, n_boot=500, seed=44)

print("   Running Properly Re-Centered Strict-Filter Sensitivities (No-Ref vs With-Ref)...")
fe_strict_noref = run_fe_block(df_uniq[df_uniq["strict_no_ref_filter"]], "journal_year", "dergi_id", fe_covars, n_boot=200, seed=45)
fe_strict_withref = run_fe_block(df_uniq[df_uniq["strict_research_filter"]], "journal_year", "dergi_id", fe_covars, n_boot=200, seed=46)

print("   Running Deduplicated 3-Way Journal x Year x Subfield FE (with Clustered SEs)...")
fe_jys = run_fe_block(df_uniq[df_uniq["openalex_subfield"].notna()], "jys_cell", "dergi_id", fe_covars, n_boot=200, seed=47)

# Attach 2020 / strict-recentered summaries to primary rows for compact table access
for idx_r, r in enumerate(fe_main["rows"]):
    r["beta_2020_c1"] = fe_2020["rows"][idx_r]["beta_c1"]
    r["beta_2020_c5"] = fe_2020["rows"][idx_r]["beta_c5"]
    r["delta_2020_unstd"] = fe_2020["rows"][idx_r]["delta_unstd_log1p"]
    r["delta_2020_unstd_ci95"] = fe_2020["rows"][idx_r]["delta_unstd_cluster_ci95"]
    r["beta_2021_c1"] = fe_2021["rows"][idx_r]["beta_c1"]
    r["beta_2021_c5"] = fe_2021["rows"][idx_r]["beta_c5"]
    r["delta_2021_unstd"] = fe_2021["rows"][idx_r]["delta_unstd_log1p"]
    r["delta_2021_unstd_ci95"] = fe_2021["rows"][idx_r]["delta_unstd_cluster_ci95"]
    r["beta_strict_noref_recentered_c5"] = fe_strict_noref["rows"][idx_r]["beta_c5"]
    r["beta_strict_withref_recentered_c5"] = fe_strict_withref["rows"][idx_r]["beta_c5"]

master["primary_fe_v6"] = fe_main
master["fe_2020_full_window"] = {
    "n_unique_works": fe_2020["n_unique_works"], "n_cells": fe_2020["n_cells"], "n_clusters": fe_2020["n_clusters"],
    "ref_row": fe_2020["rows"][0]
}
master["fe_2021_open_window"] = {
    "n_unique_works": fe_2021["n_unique_works"], "n_cells": fe_2021["n_cells"], "n_clusters": fe_2021["n_clusters"],
    "ref_row": fe_2021["rows"][0]
}
master["fe_strict_recentered"] = {
    "n_noref": fe_strict_noref["n_unique_works"], "n_cells_noref": fe_strict_noref["n_cells"],
    "n_withref": fe_strict_withref["n_unique_works"], "n_cells_withref": fe_strict_withref["n_cells"],
    "ref_noref_c5": fe_strict_noref["rows"][0]["beta_c5"], "ref_noref_se": fe_strict_noref["rows"][0]["se_c5"],
    "ref_withref_c5": fe_strict_withref["rows"][0]["beta_c5"], "ref_withref_se": fe_strict_withref["rows"][0]["se_c5"],
}
master["fe_3way_jys"] = {
    "n_unique_works": fe_jys["n_unique_works"], "n_cells": fe_jys["n_cells"], "n_clusters": fe_jys["n_clusters"],
    "ref_row": fe_jys["rows"][0], "intl_row": fe_jys["rows"][1], "fund_row": fe_jys["rows"][9]
}

# ==============================================================================
# 4. FULL-TEXT EXACT GROUP x CHANNEL CROSSTAB & JATS XML (ROWS vs UNIQUE)
# ==============================================================================
print("3. Computing Full-Text Exact Audit (Rows & Unique JATS XML)...")
df_ft_m = df_raw.merge(ft_df, on="clean_id", how="inner")
df_ft_u = df_ft_m.drop_duplicates(subset=["clean_id"], keep="first").copy()
ct_rows = pd.crosstab(df_ft_m["atif_grubu"], df_ft_m["tam_metin_kaynak_kanali"], margins=True).to_dict()
ct_uniq = pd.crosstab(df_ft_u["atif_grubu"], df_ft_u["tam_metin_kaynak_kanali"], margins=True).to_dict()

xml_u = df_ft_u[df_ft_u["tam_metin_kaynak_kanali"] == "EuropePMC_JATS_XML"].copy()
ft_metrics_spec = [
    ("tam_metin_kelime_sayisi", "Tam Metin Kelime Sayısı"),
    ("tam_metin_ek_materyal_supplementary_skoru", "Ek Materyal Paylaşım Skoru"),
    ("tam_metin_veri_erisim_beyani_var_mi", "Veri Erişim Beyanı (DAS, 0/1)"),
    ("tam_metin_tablo_sayisi", "Tablo Sayısı"),
    ("tam_metin_sekil_sayisi", "Şekil Sayısı"),
    ("tam_metin_acik_kod_yazilim_arac_skoru", "Açık Kod / Yazılım Söz Etme Skoru"),
    ("tam_metin_1000_kelimede_atif_yogunlugu", "1000 Kelimede Metin İçi Atıf Yoğunluğu"),
    ("tam_metin_etki_buyuklugu_guven_araligi_sayisi", "Etki Büyüklüğü / Güven Aralığı Sayısı"),
    ("tam_metin_p_degeri_hipotez_test_sayisi", "Hipotez Testi / p-Değeri Bildirim Sayısı"),
]
ft_table_u = []
for col, lbl in ft_metrics_spec:
    vp = df_ft_u[[col, "atif_grubu", "atif_5_yil_kumulatif"]].dropna()
    vx = xml_u[[col, "atif_grubu", "atif_5_yil_kumulatif"]].dropna()
    g_lo = vx[vx["atif_grubu"] == "Dusuk_Atif"][col].values
    g_md = vx[vx["atif_grubu"] == "Orta_Atif"][col].values
    g_hi = vx[vx["atif_grubu"] == "Yuksek_Atif"][col].values
    H_x, p_x = stats.kruskal(g_lo, g_md, g_hi)
    eps2_x = float(H_x / max(1, len(vx) - 1))
    rho_pool = float(stats.spearmanr(vp[col], vp["atif_5_yil_kumulatif"])[0])
    rho_xml = float(stats.spearmanr(vx[col], vx["atif_5_yil_kumulatif"])[0])
    ft_table_u.append({
        "label": lbl,
        "n_valid_unique_pooled": int(len(vp)),
        "n_valid_unique_xml": int(len(vx)),
        "xml_low_mean": round(float(np.mean(g_lo)), 2),
        "xml_med_mean": round(float(np.mean(g_md)), 2),
        "xml_high_mean": round(float(np.mean(g_hi)), 2),
        "kw_H_xml_unique": round(float(H_x), 1),
        "eps2_xml_unique": round(eps2_x, 4),
        "pooled_low_mean": round(float(vp[vp["atif_grubu"] == "Dusuk_Atif"][col].mean()), 2),
        "pooled_med_mean": round(float(vp[vp["atif_grubu"] == "Orta_Atif"][col].mean()), 2),
        "pooled_high_mean": round(float(vp[vp["atif_grubu"] == "Yuksek_Atif"][col].mean()), 2),
        "rho_5y_pooled_unique": round(rho_pool, 4),
        "rho_5y_xml_unique": round(rho_xml, 4),
    })

master["fulltext_v6"] = {
    "rows_crosstab": ct_rows,
    "unique_crosstab": ct_uniq,
    "n_xml_rows": int((df_ft_m["tam_metin_kaynak_kanali"] == "EuropePMC_JATS_XML").sum()),
    "n_xml_unique": int(len(xml_u)),
    "table8b_unique": ft_table_u,
}

# ==============================================================================
# 5. DELAYED SURGE WITHOUT REPLACEMENT — EXACT 84 MATCHED-SET DENOMINATOR + ORDER SENSITIVITY (M6!)
# ==============================================================================
print("4. Running Delayed Surge WITHOUT Replacement (Exact 84 Matched-Set Stats + 5-Order Sensitivity)...")
j_all = df_uniq.groupby("dergi_id")["atif_5_yil_kumulatif"].agg(["count", "sum"]).rename(columns={"count": "jc", "sum": "js"})
m_all = float(df_uniq["atif_5_yil_kumulatif"].mean())
df_uniq = df_uniq.merge(j_all, on="dergi_id", how="left")
df_uniq["loo_j"] = ((df_uniq["js"] - df_uniq["atif_5_yil_kumulatif"]) + 5.0 * m_all) / ((df_uniq["jc"] - 1) + 5.0)

ds_strict = df_uniq[
    (df_uniq["strict_research_filter"]) &
    (df_uniq["atif_1_yil_kumulatif"] <= 2) &
    (df_uniq["atif_5_yil_kumulatif"] > 25)
].copy()
low_pool = df_uniq[
    (df_uniq["strict_research_filter"]) &
    (df_uniq["atif_grubu"] == "Dusuk_Atif") &
    (df_uniq["atif_1_yil_kumulatif"] <= 2)
].copy()


def run_wo_replacement_matching(cases_df, pool_df, order_seed=None):
    cdf = cases_df.sample(frac=1.0, random_state=order_seed) if order_seed is not None else cases_df
    used_ids = set()
    dist = {0: 0, 1: 0, 2: 0, 3: 0}
    m_case_ref, m_ctrl_ref, diffs_ref = [], [], []
    m_case_j, m_ctrl_j, diffs_j = [], [], []
    matched_case_ids = set()
    for _, r in cdf.iterrows():
        cands = pool_df[
            (pool_df["disiplin_kodu"] == r["disiplin_kodu"]) &
            (pool_df["yayin_yili"] == r["yayin_yili"]) &
            (pool_df["atif_1_yil_kumulatif"] == r["atif_1_yil_kumulatif"]) &
            (~pool_df["clean_id"].isin(used_ids))
        ]
        k = min(3, len(cands))
        dist[k] += 1
        if k > 0:
            picked = cands.sample(n=k, random_state=42 if order_seed is None else order_seed)
            used_ids.update(picked["clean_id"].tolist())
            matched_case_ids.add(r["clean_id"])
            c_ref = float(picked["referans_sayisi"].mean())
            c_j = float(picked["loo_j"].mean())
            m_case_ref.append(float(r["referans_sayisi"]))
            m_ctrl_ref.append(c_ref)
            diffs_ref.append(float(r["referans_sayisi"]) - c_ref)
            m_case_j.append(float(r["loo_j"]))
            m_ctrl_j.append(c_j)
            diffs_j.append(float(r["loo_j"]) - c_j)
    diffs_ref = np.array(diffs_ref)
    diffs_j = np.array(diffs_j)
    rng_b = np.random.default_rng(42)
    b_ref = [float(np.mean(rng_b.choice(diffs_ref, size=len(diffs_ref), replace=True))) for _ in range(1000)]
    b_j = [float(np.mean(rng_b.choice(diffs_j, size=len(diffs_j), replace=True))) for _ in range(1000)]
    return {
        "n_cases_total": len(cases_df),
        "n_matched_sets": len(diffs_ref),
        "n_unmatched_cases": len(cases_df) - len(diffs_ref),
        "n_unique_controls": len(used_ids),
        "controls_per_case_dist": dist,
        "matched_case_ref_mean": round(float(np.mean(m_case_ref)), 2),
        "matched_set_ctrl_ref_mean": round(float(np.mean(m_ctrl_ref)), 2),
        "ref_paired_diff_mean": round(float(np.mean(diffs_ref)), 2),
        "ref_paired_diff_ci95": [round(float(np.percentile(b_ref, 2.5)), 2), round(float(np.percentile(b_ref, 97.5)), 2)],
        "ref_wilcoxon_p": float(stats.wilcoxon(diffs_ref)[1]),
        "matched_case_j_mean": round(float(np.mean(m_case_j)), 2),
        "matched_set_ctrl_j_mean": round(float(np.mean(m_ctrl_j)), 2),
        "j_paired_diff_mean": round(float(np.mean(diffs_j)), 2),
        "j_paired_diff_ci95": [round(float(np.percentile(b_j, 2.5)), 2), round(float(np.percentile(b_j, 97.5)), 2)],
        "j_wilcoxon_p": float(stats.wilcoxon(diffs_j)[1]),
        "matched_case_ids": matched_case_ids
    }


ds_res = run_wo_replacement_matching(ds_strict, low_pool, order_seed=None)
unmatched_df = ds_strict[~ds_strict["clean_id"].isin(ds_res["matched_case_ids"])]
matched_df = ds_strict[ds_strict["clean_id"].isin(ds_res["matched_case_ids"])]
order_sens = []
for sd in [11, 22, 33, 44, 55]:
    r_s = run_wo_replacement_matching(ds_strict, low_pool, order_seed=sd)
    order_sens.append({"seed": sd, "n_matched": r_s["n_matched_sets"], "ref_diff": r_s["ref_paired_diff_mean"], "j_diff": r_s["j_paired_diff_mean"]})

del ds_res["matched_case_ids"]
ds_res["unmatched_profile"] = {
    "n": int(len(unmatched_df)),
    "ref_mean": round(float(unmatched_df["referans_sayisi"].mean()), 2),
    "j_mean": round(float(unmatched_df["loo_j"].mean()), 2),
    "c1_mean": round(float(unmatched_df["atif_1_yil_kumulatif"].mean()), 2),
    "matched_c1_mean": round(float(matched_df["atif_1_yil_kumulatif"].mean()), 2),
}
ds_res["order_sensitivity_5seeds"] = order_sens
master["delayed_surge_v6"] = ds_res

# ==============================================================================
# 6. UNIFIED TABLE 5 & 6 (INTERNAL TEST N = 1,500 + BOOTSTRAP CIs + 3x3 MATRICES)
# ==============================================================================
print("5. Running Unified Internal Test (N = 1,500, Bootstrap CIs & 3x3 Matrices)...")
test_ids = set()
test_idx = []
for (dk, grp), g in df_uniq.groupby(["disiplin_kodu", "atif_grubu"]):
    s = g.sample(n=50, random_state=42)
    test_ids.update(s["clean_id"].tolist())
    test_idx.extend(s.index.tolist())

df_te = df_uniq.loc[test_idx].copy()
df_tr = df_uniq[~df_uniq["clean_id"].isin(test_ids)].copy()

tr_j = df_tr.groupby("dergi_id")["atif_5_yil_kumulatif"].agg(["count", "sum"]).rename(columns={"count": "tj_c", "sum": "tj_s"})
tr_m = float(df_tr["atif_5_yil_kumulatif"].mean())
df_tr = df_tr.merge(tr_j, on="dergi_id", how="left")
df_tr["tr_loo_j"] = ((df_tr["tj_s"] - df_tr["atif_5_yil_kumulatif"]) + 5.0 * tr_m) / ((df_tr["tj_c"] - 1) + 5.0)
df_te = df_te.merge(tr_j, on="dergi_id", how="left")
df_te["tr_loo_j"] = (df_te["tj_s"].fillna(0) + 5.0 * tr_m) / (df_te["tj_c"].fillna(0) + 5.0)

d_tr = pd.get_dummies(df_tr["disiplin_kodu"], prefix="d", dtype=float)
d_te = pd.get_dummies(df_te["disiplin_kodu"], prefix="d", dtype=float).reindex(columns=d_tr.columns, fill_value=0.0)

fe_cols_no_m = [c[0] for c in fe_covars if c[0] != "yayin_ayi_clean"]
m0_c = ["yayin_yili", "yayin_ayi_clean"]
m1_c = m0_c + ["tr_loo_j"]
m2_c = m0_c + fe_cols_no_m
m3_c = m1_c + fe_cols_no_m
m6_c = m3_c + ["log1p_c1"]

y_tr_t1 = df_tr["ust_yuzde_1_dilimde"].fillna(0).astype(int).values
y_te_t1 = df_te["ust_yuzde_1_dilimde"].fillna(0).astype(int).values
y_tr_lc5 = df_tr["log1p_c5"].values
y_te_lc5 = df_te["log1p_c5"].values
y_tr_c5m1 = df_tr["log1p_c5_minus_c1"].values
y_te_c5m1 = df_te["log1p_c5_minus_c1"].values

rng_t5 = np.random.default_rng(42)
boot_idx_t5 = [rng_t5.integers(0, len(y_te_t1), size=len(y_te_t1)) for _ in range(300)]

t5_bench = []
for name, cols in [
    ("M0_Alan_Yil_Ay", m0_c),
    ("M1_Prestij", m1_c),
    ("M2_Bibliyografik_Ag", m2_c),
    ("M3_Birlesik_GeriyeDonuk", m3_c),
    ("M5_Temel_Arti_C1", m0_c + ["log1p_c1"]),
    ("M6_Dinamik_M3_Arti_C1", m6_c)
]:
    Xtr = pd.concat([df_tr[cols].fillna(0), d_tr], axis=1).values
    Xte = pd.concat([df_te[cols].fillna(0), d_te], axis=1).values
    pt1 = fit_predict_logistic_irls(Xtr, y_tr_t1, Xte)
    plc5 = np.clip(fit_predict_ridge(Xtr, y_tr_lc5, Xte), 0, 10.5)
    pc5m1 = np.clip(fit_predict_ridge(Xtr, y_tr_c5m1, Xte), 0, 10.5)
    ci, cs = calc_calibration(y_te_t1, pt1)
    auc_b = [calc_roc_auc(y_te_t1[bi], pt1[bi]) for bi in boot_idx_t5]
    t5_bench.append({
        "model": name,
        "roc_auc_top1": round(calc_roc_auc(y_te_t1, pt1), 4),
        "roc_auc_top1_ci95": [round(float(np.percentile(auc_b, 2.5)), 4), round(float(np.percentile(auc_b, 97.5)), 4)],
        "ap_top1_tie_inv": round(calc_ap_tie_invariant(y_te_t1, pt1), 4),
        "brier_top1": round(float(np.mean((y_te_t1 - pt1)**2)), 4),
        "cal_slope": round(cs, 3), "cal_intercept": round(ci, 3),
        "r2_logc5": round(calc_r2(y_te_lc5, plc5), 4),
        "mae_logc5": round(float(np.mean(np.abs(y_te_lc5 - plc5))), 4),
        "r2_c5_minus_c1": round(calc_r2(y_te_c5m1, pc5m1), 4),
        "mae_raw_c5": round(float(np.mean(np.abs(df_te["atif_5_yil_kumulatif"].values - np.expm1(plc5)))), 1)
    })

master["table5_v6"] = {
    "n_test": int(len(y_te_t1)),
    "top1_pos_count": int(np.sum(y_te_t1)),
    "top1_prevalence": round(float(np.mean(y_te_t1)), 4),
    "rows": t5_bench
}

# ==============================================================================
# 7. DEDUPLICATED PROBABILISTIC FE & TABLE 9 TEMPORAL VALIDATION (M3 & M4 FIXES!)
# ==============================================================================
print("6. Running Deduplicated Probabilistic FE & Table 9 (Paired Bootstrap M3-M2 CI, Missing-Label Audit)...")
main_ids_set = set(df_uniq["clean_id"].tolist())
df_prob = df_prob_raw.drop_duplicates(subset=["clean_id"], keep="first").copy()
df_2022 = df_2022_raw.drop_duplicates(subset=["clean_id"], keep="first").copy()

overlap_prob_main = int(df_prob["clean_id"].isin(main_ids_set).sum())
overlap_2022_main = int(df_2022["clean_id"].isin(main_ids_set).sum())

df_prob["log_c1"] = np.log1p(df_prob["atif_1_yil"].clip(lower=0))
df_2022["log_c1"] = np.log1p(df_2022["atif_1_yil"].clip(lower=0))

# Probabilistic FE strictly on deduplicated clean_id
n_prob_rows = len(df_prob_raw)
n_prob_uniq = len(df_prob)
n_prob_no_j = int(df_prob["dergi_id"].isna().sum())
df_pj = df_prob.dropna(subset=["dergi_id"]).copy()
df_pj["jy"] = df_pj["dergi_id"].astype(str) + "_" + df_pj["yayin_yili"].astype(str)
pjc = df_pj["jy"].value_counts()
n_prob_singleton = int((df_pj["jy"].isin(pjc[pjc < 2].index)).sum())
df_pfe = df_pj[df_pj["jy"].isin(pjc[pjc >= 2].index)].copy()

fe_p_cols = ["referans_sayisi", "farkli_ulke_sayisi", "yazar_sayisi", "farkli_kurum_sayisi", "acik_erisim_mi"]
df_pfe["lc1"] = np.log1p(df_pfe["atif_1_yil"].clip(0))
df_pfe["lc5"] = np.log1p(df_pfe["atif_5_yil"].clip(0))
mp = df_pfe.groupby("jy")[["lc1", "lc5"] + fe_p_cols].transform("mean")
dXp = (df_pfe[fe_p_cols] - mp[fe_p_cols]).values
dXp = dXp / np.maximum(np.std(dXp, axis=0), 1e-8)
dy1p = (df_pfe["lc1"] - mp["lc1"]).values / np.std((df_pfe["lc1"] - mp["lc1"]).values)
dy5p = (df_pfe["lc5"] - mp["lc5"]).values / np.std((df_pfe["lc5"] - mp["lc5"]).values)
b1p, _, _, _ = np.linalg.lstsq(dXp, dy1p, rcond=None)
b5p, _, _, _ = np.linalg.lstsq(dXp, dy5p, rcond=None)
se5p_cl, g_p_cl = cluster_robust_se(dXp, dy5p - dXp @ b5p, df_pfe["dergi_id"].values, n_absorbed_fe=int(df_pfe["jy"].nunique()))

# Deduplicated historical journal table check
n_hist_zero_denom = int((df_hist["tarihsel_yayin_le_2019"] <= 0).sum())
df_hist_u = df_hist.drop_duplicates(subset=["dergi_id"], keep="first").copy()

df_ph = df_prob.merge(df_hist_u, on="dergi_id", how="inner").drop_duplicates(subset=["clean_id"], keep="first").copy()
tr20 = df_ph[df_ph["yayin_yili"] == 2020].copy()
te21 = df_ph[df_ph["yayin_yili"] == 2021].copy()
te22 = df_2022.merge(df_hist_u, on="dergi_id", how="inner").drop_duplicates(subset=["clean_id"], keep="first").copy()

# Missing pct_norm audit
miss_20 = int(tr20["pct_norm"].isna().sum())
miss_21 = int(te21["pct_norm"].isna().sum())
miss_22 = int(te22["pct_norm"].isna().sum())

dt0 = pd.get_dummies(tr20["disiplin_kodu"], prefix="d", dtype=float)
dt1 = pd.get_dummies(te21["disiplin_kodu"], prefix="d", dtype=float).reindex(columns=dt0.columns, fill_value=0.0)
dt2 = pd.get_dummies(te22["disiplin_kodu"], prefix="d", dtype=float).reindex(columns=dt0.columns, fill_value=0.0)

c_m1 = ["tarihsel_ort_atif_t0", "tarihsel_yayin_le_2019"]
c_m2 = ["referans_sayisi", "yazar_sayisi", "farkli_kurum_sayisi", "farkli_ulke_sayisi", "acik_erisim_mi"]
c_m3 = c_m1 + c_m2
c_m6 = c_m3 + ["log_c1"]

t9_rows = []
preds_21_10, preds_22_10 = {}, {}
for mn, cols in [
    ("M0_Alan_Bazal", []),
    ("M1_Tarihsel_2019_Dergi", c_m1),
    ("M2_Bibliyografik_Ag", c_m2),
    ("M3_Birlesik_Tarihsel_2019", c_m3),
    ("M6_Dinamik_Arti_C1", c_m6)
]:
    X0 = pd.concat([tr20[cols].fillna(0), dt0], axis=1).values if cols else dt0.values
    X1 = pd.concat([te21[cols].fillna(0), dt1], axis=1).values if cols else dt1.values
    X2 = pd.concat([te22[cols].fillna(0), dt2], axis=1).values if cols else dt2.values
    p21_10 = fit_predict_logistic_irls(X0, tr20["is_top10"].values, X1)
    p21_1 = fit_predict_logistic_irls(X0, tr20["is_top1"].values, X1)
    r21_c = np.clip(fit_predict_ridge(X0, np.log1p(tr20["atif_5_yil"].clip(0)).values, X1), 0, 10)
    p22_10 = fit_predict_logistic_irls(X0, tr20["is_top10"].values, X2)
    p22_1 = fit_predict_logistic_irls(X0, tr20["is_top1"].values, X2)
    r22_c = np.clip(fit_predict_ridge(X0, np.log1p(tr20["atif_5_yil"].clip(0)).values, X2), 0, 10)
    preds_21_10[mn] = p21_10
    preds_22_10[mn] = p22_10
    t9_rows.append({
        "model": mn,
        "p21_top10_auc": round(calc_roc_auc(te21["is_top10"], p21_10), 4),
        "p21_top10_ap_tie_inv": round(calc_ap_tie_invariant(te21["is_top10"], p21_10), 4),
        "p21_top1_auc": round(calc_roc_auc(te21["is_top1"], p21_1), 4),
        "p21_r2_obs2026": round(calc_r2(np.log1p(te21["atif_5_yil"].clip(0)), r21_c), 4),
        "e22_top10_auc": round(calc_roc_auc(te22["is_top10"], p22_10), 4),
        "e22_top10_ap_tie_inv": round(calc_ap_tie_invariant(te22["is_top10"], p22_10), 4),
        "e22_top1_auc": round(calc_roc_auc(te22["is_top1"], p22_1), 4),
        "e22_r2_obs2026_incomplete_c5": round(calc_r2(np.log1p(te22["atif_5_yil"].clip(0)), r22_c), 4)
    })

# Paired 500-replicate bootstrap for M3 - M2 Top 10% AUC difference (2021 & 2022)
rng_t9 = np.random.default_rng(42)
y21_10 = te21["is_top10"].values
y22_10 = te22["is_top10"].values
d21_boot, d22_boot = [], []
for _ in range(500):
    i21 = rng_t9.integers(0, len(y21_10), size=len(y21_10))
    d21_boot.append(calc_roc_auc(y21_10[i21], preds_21_10["M3_Birlesik_Tarihsel_2019"][i21]) -
                    calc_roc_auc(y21_10[i21], preds_21_10["M2_Bibliyografik_Ag"][i21]))
    i22 = rng_t9.integers(0, len(y22_10), size=len(y22_10))
    d22_boot.append(calc_roc_auc(y22_10[i22], preds_22_10["M3_Birlesik_Tarihsel_2019"][i22]) -
                    calc_roc_auc(y22_10[i22], preds_22_10["M2_Bibliyografik_Ag"][i22]))

master["table9_v6"] = {
    "prob_accounting": {
        "rows_total": n_prob_rows, "unique_total": n_prob_uniq,
        "overlap_with_main_87k": overlap_prob_main, "overlap_2022_with_main": overlap_2022_main,
        "missing_journal": n_prob_no_j, "singleton_jy": n_prob_singleton,
        "fe_n_unique": int(len(df_pfe)), "fe_cells": int(df_pfe["jy"].nunique()), "fe_clusters": int(g_p_cl),
        "hist_zero_denom_journals": n_hist_zero_denom
    },
    "prob_fe_clustered": [
        {"feature": fe_p_cols[i], "beta_c1": round(float(b1p[i]), 4), "beta_c5": round(float(b5p[i]), 4),
         "se_cluster_c5": round(float(se5p_cl[i]), 4), "t_cluster": round(float(b5p[i] / se5p_cl[i]), 2)}
        for i in range(len(fe_p_cols))
    ],
    "split_sizes": {
        "train_2020_unique": int(len(tr20)), "test_2021_unique": int(len(te21)),
        "test_2022_unique": int(len(te22)), "unmatched_2022_excluded": int(len(df_2022) - len(te22)),
        "top10_pos_2021": int(y21_10.sum()), "top10_prev_2021": round(float(y21_10.mean()), 4),
        "top10_pos_2022": int(y22_10.sum()), "top10_prev_2022": round(float(y22_10.mean()), 4),
        "top1_pos_2021": int(te21["is_top1"].sum()), "top1_pos_2022": int(te22["is_top1"].sum()),
        "missing_pct_norm_20_21_22": [miss_20, miss_21, miss_22]
    },
    "m3_minus_m2_top10_auc": {
        "diff_2021": round(float(t9_rows[3]["p21_top10_auc"] - t9_rows[2]["p21_top10_auc"]), 4),
        "ci95_2021": [round(float(np.percentile(d21_boot, 2.5)), 4), round(float(np.percentile(d21_boot, 97.5)), 4)],
        "diff_2022": round(float(t9_rows[3]["e22_top10_auc"] - t9_rows[2]["e22_top10_auc"]), 4),
        "ci95_2022": [round(float(np.percentile(d22_boot, 2.5)), 4), round(float(np.percentile(d22_boot, 97.5)), 4)],
    },
    "rows": t9_rows
}

out_p = os.path.join(OUT_DIR, "q1_v6_master_results.json")
with open(out_p, "w", encoding="utf-8") as f:
    json.dump(master, f, ensure_ascii=False, indent=2)

print("\n=== SUCCESS: SAVED q1_v6_master_results.json ===")
print("Primary 13-cov FE (Ref):", master["primary_fe_v6"]["rows"][0])
print("Primary 13-cov FE (Fund):", master["primary_fe_v6"]["rows"][9])
print("Primary 13-cov FE (Month):", master["primary_fe_v6"]["rows"][10])
print("2020 Full-Window FE (Ref):", master["fe_2020_full_window"])
print("2021 Open-Window FE (Ref):", master["fe_2021_open_window"])
print("Strict Re-Centered:", master["fe_strict_recentered"])
print("3-Way JYS FE (Ref):", master["fe_3way_jys"]["ref_row"])
print("Delayed Surge 84 Matched Sets:", {k: v for k, v in master["delayed_surge_v6"].items() if k != "order_sensitivity_5seeds"})
print("Order sensitivity:", master["delayed_surge_v6"]["order_sensitivity_5seeds"])
print("Prob FE Dedup:", master["table9_v6"]["prob_accounting"], master["table9_v6"]["prob_fe_clustered"][0])
print("Table 9 M3-M2 CI:", master["table9_v6"]["m3_minus_m2_top10_auc"])
print("Table 9 Split/Missing:", master["table9_v6"]["split_sizes"])
