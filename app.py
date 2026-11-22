#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hugging Face Spaces (Gradio) & Local Web App for Academic Citation Trajectory Forecasting
==========================================================================================
Predicts 5-year citation count (with 80% prediction interval), Field-Normalized Top 10%
probability, and Top 1% Elite Impact probability for a new manuscript using either:
  (A) Pre-Publication Mode (Strict Historical t0 features only)
  (B) Dynamic Trajectory Mode (Pre-Publication t0 + Year 1 Early Citations C1)

Note: Outputs represent statistical expectations calibrated on the OpenAlex 2020-2021
probabilistic reference cohort and should NOT be used as a substitute for peer review
or as a direct measure of intrinsic scientific quality (DORA / Leiden Manifesto compliant).
"""

import json
import math
import os
import numpy as np
import gradio as gr

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_artifacts", "trained_citation_models_v1.json")
with open(MODEL_PATH, "r", encoding="utf-8") as f:
    ARTIFACT = json.load(f)

DISCIPLINE_LABELS = {
    "01_Muhendislik": "Engineering (Mühendislik)",
    "02_Saglik_ve_Tip": "Health & Medical Sciences (Sağlık ve Tıp)",
    "04_Bilgisayar_ve_Bilisim": "Computer & Information Sciences (Bilgisayar ve Bilişim)",
    "05_Doga_ve_Fen_Bilimleri": "Natural & Physical Sciences (Doğa ve Fen Bilimleri)",
    "06_Sosyal_ve_Beseri_Bilimler": "Social Sciences & Humanities (Sosyal ve Beşeri Bilimler)",
    "07_Iktisat_ve_Isletme": "Economics, Business & Management (İktisat ve İşletme)",
    "08_Cevre_ve_Tarim_Bilimleri": "Environmental & Agricultural Sciences (Çevre ve Tarım)",
    "09_Psikoloji_ve_Davranis": "Psychology & Behavioral Sciences (Psikoloji ve Davranış)"
}
LABEL_TO_CODE = {v: k for k, v in DISCIPLINE_LABELS.items()}


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, float(x)))))


def predict_logistic(model_dict, x_vec):
    mu = np.array(model_dict["mu"], dtype=float)
    sd = np.array(model_dict["sd"], dtype=float)
    beta = np.array(model_dict["beta"], dtype=float)
    z = np.clip((x_vec - mu) / sd, -5.0, 5.0)
    z_aug = np.concatenate([[1.0], z])
    return sigmoid(float(z_aug @ beta))


def predict_ridge(model_dict, x_vec):
    mu = np.array(model_dict["mu"], dtype=float)
    sd = np.array(model_dict["sd"], dtype=float)
    beta = np.array(model_dict["beta"], dtype=float)
    z = (x_vec - mu) / sd
    z_aug = np.concatenate([[1.0], z, np.sign(z) * np.log1p(np.abs(z))])
    return max(0.0, float(z_aug @ beta))


def forecast_citations(
    discipline_label,
    journal_mean_cit,
    journal_total_works,
    ref_count,
    author_count,
    inst_count,
    country_count,
    is_open_access,
    use_year1_citations,
    year1_citations
):
    disc_code = LABEL_TO_CODE[discipline_label]
    disc_ohe = [1.0 if d == disc_code else 0.0 for d in ARTIFACT["disciplines"]]
    oa_val = 1.0 if is_open_access else 0.0

    base_feats = [
        float(journal_mean_cit),
        float(journal_total_works),
        float(ref_count),
        float(author_count),
        float(inst_count),
        float(country_count),
        oa_val
    ]

    if use_year1_citations:
        x_vec = np.array(base_feats + [math.log1p(max(0.0, float(year1_citations)))] + disc_ohe, dtype=float)
        p_top10 = predict_logistic(ARTIFACT["models"]["dyn_top10_clf"], x_vec)
        p_top1 = predict_logistic(ARTIFACT["models"]["dyn_top1_clf"], x_vec)
        pred_logc5 = predict_ridge(ARTIFACT["models"]["dyn_logc5_reg"], x_vec)
        sigma = 0.55
        mode_str = "Dynamic Trajectory Mode (Pre-Publication t0 + Year 1 Citations C1)"
    else:
        x_vec = np.array(base_feats + disc_ohe, dtype=float)
        p_top10 = predict_logistic(ARTIFACT["models"]["t0_top10_clf"], x_vec)
        p_top1 = predict_logistic(ARTIFACT["models"]["t0_top1_clf"], x_vec)
        pred_logc5 = predict_ridge(ARTIFACT["models"]["t0_logc5_reg"], x_vec)
        sigma = float(ARTIFACT.get("residual_std_logc5_t0", 0.90))
        mode_str = "Strict Pre-Publication Mode (t0 Bibliographic & Historical Journal Profile)"

    pred_c5 = max(0.0, math.expm1(pred_logc5))
    ci80_low = max(0.0, math.expm1(max(0.0, pred_logc5 - 1.28 * sigma)))
    ci80_high = max(0.0, math.expm1(pred_logc5 + 1.28 * sigma))

    summary_md = f"""### Citation Trajectory Forecast ({mode_str})

| Metric | Estimate | Interpretation |
| :--- | :---: | :--- |
| **Expected 5-Year Cumulative Citations ($\\hat{{C}}_5$)** | **{pred_c5:.1f} citations** | Point expectation on $\\log(1+C_5)$ scale |
| **80% Prediction Interval** | **{ci80_low:.0f} – {ci80_high:.0f} citations** | Accounts for inherent right-skewed citation variance |
| **Field-Normalized Top 10% Probability** | **{p_top10*100:.1f}%** | Probability of reaching top decile in field & year |
| **Field-Normalized Top 1% Probability** | **{p_top1*100:.1f}%** | Probability of reaching elite top 1% impact tier |

> **Responsible Research Assessment Notice:** This statistical forecast reflects historical OpenAlex associations and is intended strictly for exploratory scientometric analysis. Citation counts do not measure intrinsic scientific quality (Leiden Manifesto / DORA).
"""
    return summary_md


with gr.Blocks(theme=gr.themes.Soft(), title="Academic Citation Trajectory Forecaster") as demo:
    gr.Markdown(
        "# Academic Citation Trajectory Forecaster\n"
        "Open-source reproducible companion tool for *Bibliographic Characteristics and Journal Context Associated with Early and Five-Year Citation Accumulation*."
    )
    with gr.Row():
        with gr.Column():
            disc_in = gr.Dropdown(choices=list(DISCIPLINE_LABELS.values()), value="Computer & Information Sciences (Bilgisayar ve Bilişim)", label="Discipline (OpenAlex Field)")
            j_cit_in = gr.Number(value=12.5, label="Target Journal Historical Mean Citations/Paper (Pre-t0)", minimum=0)
            j_wrk_in = gr.Number(value=5000, label="Target Journal Historical Publication Volume (Pre-t0)", minimum=0)
            ref_in = gr.Slider(0, 200, value=45, step=1, label="Reference Count (Bibliographic Depth)")
            auth_in = gr.Slider(1, 50, value=4, step=1, label="Number of Authors")
            inst_in = gr.Slider(1, 30, value=2, step=1, label="Number of Distinct Institutions")
            ctry_in = gr.Slider(1, 20, value=2, step=1, label="Number of Distinct Countries")
            oa_in = gr.Checkbox(value=True, label="Open Access Publication")
            use_c1_in = gr.Checkbox(value=False, label="Include Year 1 Early Citations (Dynamic Mode)")
            c1_in = gr.Number(value=3, label="Observed Year 1 Citations (C1, if Dynamic Mode checked)", minimum=0)
            btn = gr.Button("Predict 5-Year Citation Trajectory", variant="primary")
        with gr.Column():
            out_md = gr.Markdown(label="Forecast Results")

    btn.click(
        fn=forecast_citations,
        inputs=[disc_in, j_cit_in, j_wrk_in, ref_in, auth_in, inst_in, ctry_in, oa_in, use_c1_in, c1_in],
        outputs=out_md
    )

if __name__ == "__main__":
    demo.launch()
