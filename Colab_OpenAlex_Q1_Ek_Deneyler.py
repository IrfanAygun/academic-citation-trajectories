#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GOOGLE COLAB — TEK SEFERDE OTOMATİK ÇALIŞAN OPENALEX API Q1 EK DOĞRULAMA PAKETİ
================================================================================
Bu betiği Google Colab'da tek bir hücrede çalıştırmanız yeterlidir.
1) Google Drive'ınızı otomatik bağlar (/content/drive/MyDrive/Q1_Makale_Ek_Deneyler).
2) Deney A (Olasılıklı 30.000 Referans Örneklemi),
   Deney B (2019 Sonu Gerçek Tarihsel Dergi t0 Prestiji),
   Deney C (2022 Gelecek Dış Kohort 10.000 Makale)
   adımlarını sırayla ve otomatik olarak çalıştırıp Drive'a kaydeder.
3) İşlem bitince yalnızca sohbete "Colab deneyleri bitti" yazmanız yeterlidir;
   çıktılar Drive'dan doğrudan okunup makaleye eklenecektir.
"""

import os
import time
import json
import requests
import numpy as np
import pandas as pd

# Colab ortamında Google Drive'ı otomatik bağla
try:
    from google.colab import drive
    if not os.path.exists("/content/drive/MyDrive"):
        drive.mount("/content/drive")
except ImportError:
    pass

OPENALEX_EMAIL = "elfnrkck@gmail.com"
BASE_URL = "https://api.openalex.org/works"
OUT_DIR = "/content/drive/MyDrive/Q1_Makale_Ek_Deneyler"
os.makedirs(OUT_DIR, exist_ok=True)

DISCIPLINE_FILTERS = {
    "01_Muhendislik": "primary_topic.field.id:22",
    "02_Saglik_ve_Tip": "primary_topic.field.id:27",
    "03_Egitim_Bilimleri": "primary_topic.field.id:3304",
    "04_Bilgisayar_ve_Bilisim": "primary_topic.field.id:17",
    "05_Doga_ve_Fen_Bilimleri": "primary_topic.domain.id:3",
    "06_Sosyal_ve_Beseri_Bilimler": "primary_topic.domain.id:2",
    "07_Iktisat_ve_Isletme": "primary_topic.field.id:14|20",
    "08_Cevre_ve_Tarim_Bilimleri": "primary_topic.field.id:11|23",
    "09_Psikoloji_ve_Davranis": "primary_topic.field.id:32",
    "10_Hukuk_ve_Siyaset_Bilimi": "primary_topic.field.id:3308|3320"
}


def safe_get(url, params, max_retries=4):
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** (attempt + 1))
                continue
            print(f"   HTTP {resp.status_code} uyarısı: {resp.text[:120]}")
            return None
        except Exception as e:
            print(f"   Bağlantı denemesi {attempt+1}/{max_retries} hatası: {e}")
            time.sleep(2 ** (attempt + 1))
    return None


def fetch_openalex_random_sample(filter_str, n_target=1500, seed=42):
    records = []
    per_page = 200
    n_pages = int(np.ceil(n_target / per_page))
    for page in range(1, n_pages + 1):
        params = {
            "filter": filter_str,
            "sample": n_target,
            "seed": seed,
            "per-page": per_page,
            "page": page,
            "mailto": OPENALEX_EMAIL
        }
        data = safe_get(BASE_URL, params)
        if not data or not data.get("results"):
            break
        records.extend(data["results"])
        time.sleep(0.12)
    return records[:n_target]


def extract_work_features(w, disc_code):
    pub_year = w.get("publication_year", 2020)
    c_by_year = {item["year"]: item["cited_by_count"] for item in (w.get("counts_by_year") or [])}
    c1 = sum(c_by_year.get(y, 0) for y in [pub_year, pub_year + 1])
    c5 = sum(c_by_year.get(y, 0) for y in range(pub_year, pub_year + 6))
    source = (w.get("primary_location") or {}).get("source") or {}
    authorships = w.get("authorships") or []
    countries, insts = set(), set()
    for a in authorships:
        for inst in (a.get("institutions") or []):
            if inst.get("id"):
                insts.add(inst["id"])
            if inst.get("country_code"):
                countries.add(inst["country_code"])
    return {
        "makale_id": w.get("id"),
        "doi": w.get("doi"),
        "disiplin_kodu": disc_code,
        "yayin_yili": pub_year,
        "yayin_tarihi": w.get("publication_date"),
        "referans_sayisi": w.get("referenced_works_count", 0),
        "yazar_sayisi": len(authorships),
        "farkli_kurum_sayisi": len(insts),
        "farkli_ulke_sayisi": len(countries),
        "fon_hibe_var_mi": 1 if len(w.get("grants") or []) > 0 else 0,
        "acik_erisim_mi": 1 if (w.get("open_access") or {}).get("is_oa") else 0,
        "dergi_id": source.get("id"),
        "dergi_adi": source.get("display_name"),
        "atif_1_yil": c1,
        "atif_5_yil": c5,
        "atif_toplam": w.get("cited_by_count", 0),
        "fwci": w.get("fwci"),
        "pct_norm": ((w.get("citation_normalized_percentile") or {}).get("value")),
        "is_top10": 1 if ((w.get("citation_normalized_percentile") or {}).get("is_in_top_10_percent")) else 0,
        "is_top1": 1 if ((w.get("citation_normalized_percentile") or {}).get("is_in_top_1_percent")) else 0,
    }


def run_experiment_a_probabilistic_sample():
    print("\n[1/3] DENEY A: Olasılıklı OpenAlex `sample` Referans Örneklemi (N = 30.000)...")
    all_rows = []
    for dk, f_field in DISCIPLINE_FILTERS.items():
        for yr in [2020, 2021]:
            f_str = f"{f_field},publication_year:{yr},type:article,is_paratext:false"
            works = fetch_openalex_random_sample(f_str, n_target=1500, seed=42 + yr)
            for w in works:
                all_rows.append(extract_work_features(w, dk))
            print(f"   {dk} ({yr}): {len(works)} rastgele makale çekildi.")
    df_prob = pd.DataFrame(all_rows)
    df_prob.to_csv(os.path.join(OUT_DIR, "openalex_probabilistic_reference_30k.csv"), index=False)

    c_tot = df_prob["atif_toplam"].fillna(0)
    pi_emp = {
        "n_total": int(len(df_prob)),
        "pi_0_2": round(float((c_tot <= 2).mean()), 5),
        "pi_3_buffer": round(float((c_tot == 3).mean()), 5),
        "pi_4_25": round(float(((c_tot >= 4) & (c_tot <= 25)).mean()), 5),
        "pi_gt25": round(float((c_tot > 25).mean()), 5)
    }
    with open(os.path.join(OUT_DIR, "deney_a_evren_oranlari.json"), "w") as f:
        json.dump(pi_emp, f, indent=2)
    print("   Deney A tamamlandı! Evren oranları:", pi_emp)
    return df_prob


def run_experiment_b_historical_t0_journal(source_ids, cutoff_year=2019, max_journals=1000):
    print(f"\n[2/3] DENEY B: {min(len(source_ids), max_journals)} Dergi için {cutoff_year} Sonu Tarihsel t0 Prestiji...")
    hist_records = []
    for idx, sid in enumerate(source_ids[:max_journals], 1):
        if not sid:
            continue
        clean_sid = str(sid).replace("https://openalex.org/", "")
        sdata = safe_get(f"https://api.openalex.org/sources/{clean_sid}", {"mailto": OPENALEX_EMAIL})
        if sdata:
            cby = sdata.get("counts_by_year") or []
            pre_cits = sum(x["cited_by_count"] for x in cby if x["year"] <= cutoff_year)
            pre_works = sum(x["works_count"] for x in cby if x["year"] <= cutoff_year)
            hist_records.append({
                "dergi_id": sid,
                f"tarihsel_atif_le_{cutoff_year}": pre_cits,
                f"tarihsel_yayin_le_{cutoff_year}": pre_works,
                f"tarihsel_ort_atif_t0": pre_cits / max(1, pre_works)
            })
        if idx % 100 == 0:
            print(f"   {idx}/{min(len(source_ids), max_journals)} dergi işlendi...")
        time.sleep(0.10)
    df_hist = pd.DataFrame(hist_records)
    df_hist.to_csv(os.path.join(OUT_DIR, f"dergi_tarihsel_t0_{cutoff_year}.csv"), index=False)
    print(f"   Deney B tamamlandı! ({len(df_hist)} dergi kaydedildi.)")
    return df_hist


def run_experiment_c_external_2022_cohort():
    print("\n[3/3] DENEY C: 2022 Gelecek Dış Kohort Doğrulama Verisi (N = 10.000)...")
    rows_22 = []
    for dk, f_field in DISCIPLINE_FILTERS.items():
        f_str = f"{f_field},publication_year:2022,type:article,is_paratext:false"
        works = fetch_openalex_random_sample(f_str, n_target=1000, seed=2022)
        for w in works:
            rows_22.append(extract_work_features(w, dk))
        print(f"   {dk} (2022): {len(works)} makale çekildi.")
    df22 = pd.DataFrame(rows_22)
    df22.to_csv(os.path.join(OUT_DIR, "openalex_external_2022_cohort_10k.csv"), index=False)
    print("   Deney C tamamlandı!")
    return df22


if __name__ == "__main__":
    print("=== GOOGLE COLAB OPENALEX Q1 EK DOĞRULAMA PAKETİ BAŞLATILDI ===")
    df_prob = run_experiment_a_probabilistic_sample()
    unique_journals = df_prob["dergi_id"].dropna().unique().tolist()
    df_hist = run_experiment_b_historical_t0_journal(unique_journals, cutoff_year=2019, max_journals=1000)
    df_2022 = run_experiment_c_external_2022_cohort()
    print("\n✅ TÜM COLAB DENEYLERİ BAŞARIYLA TAMAMLANDI!")
    print(f"Çıktılar şu Drive klasörüne kaydedildi: {OUT_DIR}")
    print("Artık sohbete dönüp 'Colab deneyleri bitti' yazabilirsiniz.")
