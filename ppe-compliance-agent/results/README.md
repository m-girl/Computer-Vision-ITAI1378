# Results

Everything the project produced — metrics, diagnostics, and example agent
outputs (success, false positive, false negative) — lives here.

```
results/
└── smoke_test_10ep_320px/     ← the training run used for every result in this repo
    ├── metrics.csv               per-class precision/recall/mAP
    ├── evaluation_metrics.txt    raw Ultralytics eval summary
    ├── diagnostics/              full Ultralytics eval output (see below)
    ├── dataset_samples/          ground-truth labeled training images
    ├── success_cases/            verified correct multi-class detection
    └── failure_cases/            3 verified failure modes (see below)
```

This folder is named `smoke_test_10ep_320px` because that's what this run
started as — a quick check that the full pipeline (train → export → agents →
reports) worked end-to-end before committing to a longer run. A 50-epoch/640px
run was attempted but never completed: Colab's runtime was too unstable across
repeated sessions to finish it in the available time (see the main
[README](../README.md#5-model-training) and
[`docs/AI_usage_log.md`](../docs/AI_usage_log.md) for the full account). **This
10-epoch/320px run is therefore the final result**, not a placeholder — every
number and image in this repo comes from it.

## metrics.csv + evaluation_metrics.txt + diagnostics/

Standard Ultralytics `.val()` output on the 143-image validation set (1,172
instances): `results.png` (loss/precision/recall/mAP curves), `confusion_matrix.png`
and `confusion_matrix_normalized.png`, the four Box PR/P/R/F1-confidence curves,
`labels.jpg` (dataset class balance + box size/position distribution), and
`train_batch0-2.jpg` (training-data augmentation sanity check). `evaluation_metrics.txt`
is the raw summary Ultralytics printed for this run.

Headline numbers: **mAP50 = 0.524, mAP50-95 = 0.253**. Full per-class breakdown
in `metrics.csv` and the main [README](../README.md#6-evaluation-results).

## Three verified cases (ground truth vs. prediction, side by side)

These come directly from Ultralytics' `val_batch*_labels.jpg` /
`val_batch*_pred.jpg` mosaics — real held-out validation images with both the
ground-truth boxes and the model's predicted boxes rendered, so every claim
below is checkable against the source image.

**Success — `success_cases/success_case_helmet_vest.png`**
Ground truth: `Person, helmet, vest` (plus two `none` boxes marking bare
hands/face). Prediction: `Person 0.7, helmet 0.9, vest 0.9` — all three main
PPE classes correctly detected with high confidence, boxes well-aligned. This
is representative of the model's best-case behavior: a single, unoccluded
worker in typical construction-site framing.

**Failure (misclassification) — `failure_cases/failure_case_helmet_goggles_confusion.png`**
Ground truth: `Person, helmet, vest, gloves x2, boots x2`. Prediction: `Person
0.8, goggles 0.3` (wrong — should be `helmet`), `vest`, `gloves 0.3`, `boots
0.7/0.8`. The model mistook the worker's helmet for goggles. This isn't a one-off:
the confusion matrix shows real cross-talk between these two classes (`helmet`
row shows 7 instances confused with `goggles`, and `goggles`->`helmet` shows a
comparable number in the other direction) — both are small, head-region objects
that this model hasn't learned to separate reliably at this amount of training.

**Failure (false negative) — `failure_cases/failure_case_missed_no_gloves.png`**
Ground truth: `Person, no_gloves x2`. Prediction: `Person 0.8` only — both
violation boxes were missed entirely. This is the most consequential kind of
failure for a compliance system: the Compliance Agent would report this scene
as fully compliant when it isn't. It's consistent with `no_gloves` being one of
the weakest classes in the metrics (recall 0.071, mAP50 0.126) — see the
"Honest analysis" section of the main README for why.

**Also included (from an earlier single-image demo, out-of-domain):**
`failure_cases/false_positive_helmet_on_light.png` — a `helmet` box placed on a
stage light in a non-construction photo, with no person underneath it.
