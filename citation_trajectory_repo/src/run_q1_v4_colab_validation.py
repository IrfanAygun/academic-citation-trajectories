#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q1 External & Probabilistic Validation Suite (v4)
=================================================
Analyzes the 3 newly harvested OpenAlex Colab datasets:
1) openalex_probabilistic_reference_30k.csv (unbiased random `sample` cohort, N=24,000)
   - Empirical natural universe stratum shares (including pi_3 buffer!)
   - Within-Journal-Year FE replication on unbiased probabilistic sample
   - Cross-evaluation: Model trained on 90k enriched sample -> evaluated on unbiased probabilistic sample
2) dergi_tarihsel_t0_2019.csv (true historical pre-publication journal prestige <= 2019)
   - Zero-leakage strict ex-ante t0 benchmark on 2020-2021 articles
3) openalex_external_2022_cohort_10k.csv (unseen 2022 future external cohort)
   - True future-cohort external validation (trained on 2020-2021 -> evaluated on 2022)
"""

import glob
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

def calc_roc_auc(y_true, y_score):
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    n1, n0 = np.sum(y_true == 1), np.sum(y_true == 0)
    if n1 == 0 or n0 == 0:
        return 0.5
    ranks = stats.rankdata(y_score)
    return float((np.sum(ranks[y_true == 1]) - n1 * (n1 + 1) / 2.0) / (n1 * n0))

def calc_ap(y_true, y_score):
    y_true = np.asarray(y_true, dtype=float)
    order = np.argsort(-np.asarray(y_score, dtype=float))
    ys = y_true[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1.0 - ys)
    if tp[-1] <= 0:
        return 0.0
    prec = tp / np.maximum(tp + fp, 1e-12)
    rec = tp / tp[-1]
    return float(np.sum(prec * np.diff(np.concatenate([[0.0], rec]))))

def calc_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / max(ss_tot, 1e-12))

def fit_predict_logistic_irls(X_tr, y_tr, X_te, alpha=15.0, n_iter=10):
    mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0)
    sd[sd < 1e-8] = 1.0
    Z_tr = (X_tr - mu) / sd
    Z_te = (X_te - mu) / sd
    Z_tr_nl = np.hstack([np.ones((len(Z_tr), 1)), Z_tr, np.sign(Z_tr) * np.log1p(np.abs(Z_tr))])
    Z_te_nl = np.hstack([np.ones((len(Z_te), 1)), Z_te, np.sign(Z_te) * np.log1p(np.abs(Z_te))])
    p_dim = Z_tr_nl.shape[1]
    beta = np.zeros(p_dim)
    pos_rate = np.clip(np.mean(y_tr), 1e-4, 1 - 1e-4)
    beta[0] = math.log(pos_rate / (1.0 - pos_rate))
    I = np.eye(p_dim)
    I[0, 0] = 0.0
    for _ in range(n_iter):
        eta = np.clip(Z_tr_nl @ beta, -15, 15)
        prob = special.expit(eta)
        w = np.maximum(prob * (1.0 - prob), 1e-4)
        grad = Z_tr_nl.T @ (prob - y_tr) + alpha * (I @ beta)
        H = (Z_tr_nl.T * w) @ Z_tr_nl + alpha * I
        step = np.linalg.solve(H, grad)
        beta -= step
        if np.max(np.abs(step)) < 1e-4:
            break
    return special.expit(np.clip(Z_te_nl @ beta, -15, 15))

def fit_predict_ridge(X_tr, y_tr, X_te, alpha=15.0):
    mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0)
    sd[sd < 1e-8] = 1.0
    Z_tr = (X_tr - mu) / sd
    Z_te = (X_te - mu) / sd
    Z_tr_nl = np.hstack([np.ones((len(Z_tr), 1)), Z_tr, np.sign(Z_tr) * np.log1p(np.abs(Z_tr)), np.clip(Z_tr, -3, 3) ** 2])
    Z_te_nl = np.hstack([np.ones((len(Z_te), 1)), Z_te, np.sign(Z_te) * np.log1p(np.abs(Z_te)), np.clip(Z_te, -3, 3) ** 2])
    I = np.eye(Z_tr_nl.shape[1])
    I[0, 0] = 0.0
    beta = np.linalg.solve(Z_tr_nl.T @ Z_tr_nl + alpha * I, Z_tr_nl.T @ y_tr)
    return Z_te_nl @ beta

print("1. Loading Colab External Datasets & 90k Main Dataset...")
df_prob = pd.read_csv(os.path.join(DATA_DIR, "openalex_probabilistic_reference_30k.csv"))
df_hist = pd.read_csv(os.path.join(DATA_DIR, "dergi_tarihsel_t0_2019.csv"))
df_2022 = pd.read_csv(os.path.join(DATA_DIR, "openalex_external_2022_cohort_10k.csv"))
with open(os.path.join(DATA_DIR, "deney_a_evren_oranlari.json")) as f:
    pi_emp = json.load(f)

print(f"   Probabilistic sample: {len(df_prob):,} rows across {df_prob['disiplin_kodu'].nunique()} disciplines.")
print(f"   Historical <=2019 journals: {len(df_hist):,} journals.")
print(f"   External 2022 cohort: {len(df_2022):,} rows across {df_2022['disiplin_kodu'].nunique()} disciplines.")

# Load 90k main dataset
csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "alt_parti_*.csv")))
df90 = pd.concat([pd.read_csv(f, low_memory=False) for f in csv_files], ignore_index=True)
df90["clean_id"] = df90["makale_id"].astype(str).str.replace("https://openalex.org/", "", regex=False)

# Standardize column names between 90k and Colab datasets
df90_std = df90[[
    "clean_id", "disiplin_kodu", "yayin_yili", "referans_sayisi", "yazar_sayisi",
    "farkli_kurum_sayisi", "farkli_ulke_sayisi", "fon_hibe_var_mi", "acik_erisim_mi",
    "dergi_id", "atif_1_yil_kumulatif", "atif_5_yil_kumulatif", "atif_toplam_guncel",
    "ust_yuzde_10_dilimde", "ust_yuzde_1_dilimde"
]].rename(columns={
    "atif_1_yil_kumulatif": "atif_1_yil",
    "atif_5_yil_kumulatif": "atif_5_yil",
    "atif_toplam_guncel": "atif_toplam",
    "ust_yuzde_10_dilimde": "is_top10",
    "ust_yuzde_1_dilimde": "is_top1"
}).drop_duplicates(subset=["clean_id"]).copy()

df_prob["clean_id"] = df_prob["makale_id"].astype(str).str.replace("https://openalex.org/", "", regex=False)
df_2022["clean_id"] = df_2022["makale_id"].astype(str).str.replace("https://openalex.org/", "", regex=False)

# Merge strict historical t0 <= 2019 journal prestige into df_prob, df90_std, and df_2022
df_prob = df_prob.merge(df_hist, on="dergi_id", how="left")
df90_std = df90_std.merge(df_hist, on="dergi_id", how="left")
df_2022 = df_2022.merge(df_hist, on="dergi_id", how="left")

# Fill missing historical journal mean with median of observed journals
med_hist = float(df_hist["tarihsel_ort_atif_t0"].median())
for d_sub in [df_prob, df90_std, df_2022]:
    d_sub["hist_j_t0"] = d_sub["tarihsel_ort_atif_t0"].fillna(med_hist)
    d_sub["hist_j_works_t0"] = d_sub["tarihsel_yayin_le_2019"].fillna(0)

v4_res = {}

# ==============================================================================
# ANALYSIS A: PROBABILISTIC SAMPLE (N=24,000) NATURAL UNIVERSE & FE REPLICATION
# ==============================================================================
print("2. Running Analysis A: Probabilistic Sample FE & Natural Universe Proportions...")
c_tot = df_prob["atif_toplam"].fillna(0)
pi_by_disc = []
for dk, g in df_prob.groupby("disiplin_kodu"):
    ct = g["atif_toplam"].fillna(0)
    pi_by_disc.append({
        "disiplin_kodu": dk,
        "n": int(len(g)),
        "pi_0_2": round(float((ct <= 2).mean()), 4),
        "pi_3": round(float((ct == 3).mean()), 4),
        "pi_4_25": round(float(((ct >= 4) & (ct <= 25)).mean()), 4),
        "pi_gt25": round(float((ct > 25).mean()), 4),
        "top10_rate_pct": round(float(g["is_top10"].mean() * 100), 2),
        "top1_rate_pct": round(float(g["is_top1"].mean() * 100), 2),
        "mean_c5": round(float(g["atif_5_yil"].mean()), 2),
        "median_c5": round(float(g["atif_5_yil"].median()), 1)
    })

# Within-Journal-Year FE replication on the UNBIASED Probabilistic Sample!
df_prob_jy = df_prob.dropna(subset=["dergi_id"]).copy()
df_prob_jy["jy"] = df_prob_jy["dergi_id"].astype(str) + "_" + df_prob_jy["yayin_yili"].astype(str)
jy_c = df_prob_jy["jy"].value_counts()
df_prob_fe = df_prob_jy[df_prob_jy["jy"].isin(jy_c[jy_c >= 2].index)].copy()
df_prob_fe["lc1"] = np.log1p(df_prob_fe["atif_1_yil"].clip(lower=0))
df_prob_fe["lc5"] = np.log1p(df_prob_fe["atif_5_yil"].clip(lower=0))

fe_cols_prob = ["referans_sayisi", "farkli_ulke_sayisi", "yazar_sayisi", "farkli_kurum_sayisi", "acik_erisim_mi"]
m_pf = df_prob_fe.groupby("jy")[["lc1", "lc5"] + fe_cols_prob].transform("mean")
dXp = (df_prob_fe[fe_cols_prob] - m_pf[fe_cols_prob]).values
dXp = dXp / np.maximum(np.std(dXp, axis=0), 1e-8)
dy1p = (df_prob_fe["lc1"] - m_pf["lc1"]).values / max(np.std((df_prob_fe["lc1"] - m_pf["lc1"]).values), 1e-8)
dy5p = (df_prob_fe["lc5"] - m_pf["lc5"]).values / max(np.std((df_prob_fe["lc5"] - m_pf["lc5"]).values), 1e-8)

b1p, _, _, _ = np.linalg.lstsq(dXp, dy1p, rcond=None)
b5p, _, _, _ = np.linalg.lstsq(dXp, dy5p, rcond=None)

# SE for probabilistic FE
res5 = dy5p - dXp @ b5p
XtXi = np.linalg.pinv(dXp.T @ dXp)
se5p = np.sqrt(np.diag(XtXi) * np.sum(res5**2) / max(1, len(dXp) - df_prob_fe["jy"].nunique() - len(fe_cols_prob)))

prob_fe_summary = {
    "n_articles_prob_fe": int(len(df_prob_fe)),
    "n_jy_cells_prob": int(df_prob_fe["jy"].nunique()),
    "rows": [
        {
            "feature": col,
            "beta_c1_prob": round(float(b1p[i]), 4),
            "beta_c5_prob": round(float(b5p[i]), 4),
            "se_c5_prob": round(float(se5p[i]), 4),
            "t_c5_prob": round(float(b5p[i] / se5p[i]), 2),
            "p_c5_prob": float(2 * stats.norm.sf(abs(b5p[i] / se5p[i])))
        }
        for i, col in enumerate(fe_cols_prob)
    ]
}

v4_res["analysis_a_probabilistic_sample"] = {
    "empirical_universe_shares": pi_emp,
    "by_discipline": pi_by_disc,
    "within_journal_fe_replication": prob_fe_summary
}

# ==============================================================================
# ANALYSIS B & C: TRUE HISTORICAL t0 (<=2019) & EXTERNAL VALIDATION ON PROBABILISTIC (2020-2021) + FUTURE 2022 COHORT
# ==============================================================================
print("3. Running External Generalization & True Historical t0 (<=2019) Validation...")
# Deduplicate against 90k training set so zero overlap!
prob_ids = set(df_prob["clean_id"])
ext22_ids = set(df_2022["clean_id"])
df90_tr = df90_std[~df90_std["clean_id"].isin(prob_ids | ext22_ids)].copy()

# Restrict to disciplines present in both (check intersection)
common_discs = sorted(list(set(df_prob["disiplin_kodu"]).intersection(set(df90_tr["disiplin_kodu"]))))
df90_tr_c = df90_tr[df90_tr["disiplin_kodu"].isin(common_discs)].copy()
df_prob_c = df_prob[df_prob["disiplin_kodu"].isin(common_discs)].copy()
df_2022_c = df_2022[df_2022["disiplin_kodu"].isin(common_discs)].copy()

d_tr = pd.get_dummies(df90_tr_c["disiplin_kodu"], prefix="d", dtype=float)
d_pr = pd.get_dummies(df_prob_c["disiplin_kodu"], prefix="d", dtype=float).reindex(columns=d_tr.columns, fill_value=0.0)
d_22 = pd.get_dummies(df_2022_c["disiplin_kodu"], prefix="d", dtype=float).reindex(columns=d_tr.columns, fill_value=0.0)

# Strictly Historical Pre-Publication (<=2019) Feature Sets (Zero future leakage!)
hist_m1_cols = ["hist_j_t0", "hist_j_works_t0"]
hist_m2_cols = ["referans_sayisi", "yazar_sayisi", "farkli_kurum_sayisi", "farkli_ulke_sayisi", "fon_hibe_var_mi", "acik_erisim_mi"]
hist_m3_cols = hist_m1_cols + hist_m2_cols
hist_m6_cols = hist_m3_cols + ["atif_1_yil"]

ext_benchmarks = []
for mname, cols in [
    ("M0_Alan_Bazal", []),
    ("M1_Gercek_Tarihsel_t0_2019_Dergi", hist_m1_cols),
    ("M2_Sadece_Bibliyografik_Ag", hist_m2_cols),
    ("M3_Birlesik_Gercek_Tarihsel_t0", hist_m3_cols),
    ("M6_Dinamik_Tarihsel_t0_Arti_C1", hist_m6_cols)
]:
    X_tr = pd.concat([df90_tr_c[cols].fillna(0), d_tr], axis=1).values if cols else d_tr.values
    X_pr = pd.concat([df_prob_c[cols].fillna(0), d_pr], axis=1).values if cols else d_pr.values
    X_22 = pd.concat([df_2022_c[cols].fillna(0), d_22], axis=1).values if cols else d_22.values
    
    # Evaluate Top 10% and Top 1% and log1p(C5) on BOTH:
    # 1) Unbiased Probabilistic Reference Sample (2020-2021)
    # 2) Unseen Future 2022 External Cohort
    p_pr_t10 = fit_predict_logistic_irls(X_tr, df90_tr_c["is_top10"].fillna(0).astype(int).values, X_pr)
    p_pr_t1 = fit_predict_logistic_irls(X_tr, df90_tr_c["is_top1"].fillna(0).astype(int).values, X_pr)
    pred_pr_lc5 = np.clip(fit_predict_ridge(X_tr, np.log1p(df90_tr_c["atif_5_yil"].clip(lower=0)).values, X_pr), 0.0, 10.5)
    
    p_22_t10 = fit_predict_logistic_irls(X_tr, df90_tr_c["is_top10"].fillna(0).astype(int).values, X_22)
    p_22_t1 = fit_predict_logistic_irls(X_tr, df90_tr_c["is_top1"].fillna(0).astype(int).values, X_22)
    pred_22_lc5 = np.clip(fit_predict_ridge(X_tr, np.log1p(df90_tr_c["atif_5_yil"].clip(lower=0)).values, X_22), 0.0, 10.5)
    
    y_pr_t10 = df_prob_c["is_top10"].fillna(0).astype(int).values
    y_pr_t1 = df_prob_c["is_top1"].fillna(0).astype(int).values
    y_pr_lc5 = np.log1p(df_prob_c["atif_5_yil"].clip(lower=0)).values
    
    y_22_t10 = df_2022_c["is_top10"].fillna(0).astype(int).values
    y_22_t1 = df_2022_c["is_top1"].fillna(0).astype(int).values
    y_22_lc5 = np.log1p(df_2022_c["atif_5_yil"].clip(lower=0)).values
    
    ext_benchmarks.append({
        "model": mname,
        "prob_2020_21_n": int(len(df_prob_c)),
        "prob_top10_roc_auc": round(calc_roc_auc(y_pr_t10, p_pr_t10), 4),
        "prob_top10_ap": round(calc_ap(y_pr_t10, p_pr_t10), 4),
        "prob_top1_roc_auc": round(calc_roc_auc(y_pr_t1, p_pr_t1), 4),
        "prob_r2_logc5": round(calc_r2(y_pr_lc5, pred_pr_lc5), 4),
        "ext_2022_n": int(len(df_2022_c)),
        "ext22_top10_roc_auc": round(calc_roc_auc(y_22_t10, p_22_t10), 4),
        "ext22_top10_ap": round(calc_ap(y_22_t10, p_22_t10), 4),
        "ext22_top1_roc_auc": round(calc_roc_auc(y_22_t1, p_22_t1), 4),
        "ext22_r2_logc5": round(calc_r2(y_22_lc5, pred_22_lc5), 4)
    })

v4_res["analysis_b_c_historical_t0_and_external_validation"] = ext_benchmarks

out_v4_path = os.path.join(OUT_DIR, "q1_v4_colab_validation_results.json")
with open(out_v4_path, "w", encoding="utf-8") as f:
    json.dump(v4_res, f, ensure_ascii=False, indent=2)
print(f"SUCCESS: Saved Colab external validation results to {out_v4_path}")
print(json.dumps(v4_res, ensure_ascii=False, indent=2))