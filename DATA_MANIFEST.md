# Data Manifest & Reproducibility Guide

Because the 60 raw OpenAlex stratified CSV files and the 5,750-work full-text cache exceed GitHub file-size limits, they are not committed directly to this Git repository.

## Option A: Inspect Precomputed Verified Results Immediately (No Raw CSVs Required)
```bash
python3 run_all_q1_analyses.py --verify-json
```
All exact manuscript numbers (Tables 1B–9, 500-rep cluster bootstrap CIs, Table 6 3x3 confusion matrices) are stored in `model_artifacts/q1_v6_master_results.json`.

## Option B: Re-Run Full Pipeline from Raw Snapshot
1. Request the frozen November 2026 snapshot archive (`downloads/`) from the corresponding author (`elfnrkck@gmail.com`) or regenerate fresh OpenAlex cohorts via `src/Colab_OpenAlex_Q1_Ek_Deneyler.py`.
2. Place the files into `downloads/` matching the schema below and run `python3 run_all_q1_analyses.py`.

## Expected Input Files in `downloads/`
- `alt_parti_01_*.csv` .. `alt_parti_10_*.csv`: 60 files (90,000 rows, 87,095 unique `clean_id`)
- `openalex_probabilistic_reference_30k.csv`: 4.78 MB (SHA-256 prefix: `ac24a52ff3aae727`)
- `openalex_external_2022_cohort_10k.csv`: 1.67 MB (SHA-256 prefix: `9c4fedc7e6c39518`)
- `dergi_tarihsel_t0_2019.csv`: 0.06 MB (SHA-256 prefix: `21a7a49451626056`)
- `tam_metin_derin_analiz_v2_5_cache.json`: 7.36 MB (SHA-256 prefix: `fa374a4414123cbe`)
