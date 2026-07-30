# Diagnostic Findings: Root-Cause Analysis of Model Underperformance

**Date:** 27 July 2026 | **Analyst verification:** static code inspection + per-sample output analysis
**Evaluated artefacts:** `results/{mimic,iu}_real_sample_predictions.csv` (11,688 + 4,720 rows)

---

## D1. The vision encoder is NOT BioMedCLIP

`models/baseline_vlm.py`, `_load_vision_encoder()` lines 74–79:

```python
try:
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    return model.visual
```

The `model_name` argument (`microsoft/BiomedCLIP-...`) is **logged but never used**. The
loaded backbone is general-domain **OpenAI CLIP ViT-B/16**. `logs/execution.log` contains
no "Using standard ViT fallback" warning, confirming this branch succeeded and is the
live code path on every run.

**Consequence:** no domain-specific medical pretraining. General CLIP features are weak
for subtle radiographic findings, which is the primary suspected cause of poor visual
grounding. All prior documentation claiming a BioMedCLIP encoder is inaccurate.

## D2. Structured Label Guidance leaks ground-truth labels

`models/label_guided_vlm.py` passes `report_texts[b]` (the **reference report**) into
`construct_slg_prompt(report_text=...)`, which calls
`StructuredLabelEncoder.extract_labels_from_text()`.

**Consequence:** SLG conditions receive oracle labels derived from the target at test
time, not labels predicted from the image. The SLG and Combined conditions are therefore
**upper bounds under label supervision**, not deployable systems. The proposal specified
image-predicted labels.

## D3. Validation "BLEU-4" is a fabricated proxy

`training/trainer.py` line 124:

```python
mock_bleu4 = max(0.01, round(1.0 / (1.0 + avg_val_loss), 4))
```

Verified: val_loss 0.9319 -> 0.5176; 0.9911 -> 0.5022; 1.0191 -> 0.4953 — matching the
logged "Val BLEU-4" exactly. This is a monotonic transform of the loss, not BLEU.

**Consequence:** `best_bleu4` and `best_loss` checkpoints select on the same criterion.
Checkpoint selection never optimised generation quality. Reported validation BLEU of
~0.50 is not comparable to the test BLEU-4 of 0.0726 (MIMIC) / 0.0885 (IU).

## D4. Template collapse — the decoder ignores the image

| Dataset | Test images | Unique generated reports (Baseline) | Unique (Combined) |
|---|---:|---:|---:|
| MIMIC-CXR | 2,922 | **57** | 143 |
| IU X-Ray  | 1,180 | **10** | 8 |

65.7% of MIMIC baseline outputs are the identical sentence "no acute cardiopulmonary
process". **99.5% (MIMIC) / 100% (IU)** of POSITIVE conditions injected into the SLG
prompt are ignored or negated by the decoder.

**Consequence:** the model emits a small set of memorised normal-report templates
largely independent of visual input. CheXbert Micro-F1 ~0.55 reflects majority-class
"No Finding" agreement (IU precision 0.87–0.90 vs recall 0.40), not pathology detection.

---

## Recommended remediation (future work)

1. Load genuine BioMedCLIP weights via `open_clip.create_model_and_transforms(
   'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')`; assert the
   loaded checkpoint identity at construction time.
2. Train an image-conditioned CheXbert label classifier; feed **predicted** labels to
   the SLG prompt. Retain the oracle-label run as a declared upper bound.
3. Replace `mock_bleu4` with true corpus BLEU computed over generated text.
4. Address collapse: scale-match the visual prefix to the T5 embedding norm, unfreeze
   later encoder blocks, and add repetition/diversity penalties at decode time.
