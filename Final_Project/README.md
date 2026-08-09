# PPE Compliance Checker — Multi-Agent Computer Vision System

**ITAI 1378 — Computer Vision Capstone Project (Tier 3)**

**Team Members**
- Eva Abou Harb
- Mary Ann Mastri
- Zain Ahmed

---

## 1. Overview

This project is a three-agent computer vision system that checks construction-site
photos for PPE (personal protective equipment) compliance. It doesn't just run a
detector on an image — it perceives (detects PPE items), reasons (decides whether
the scene is compliant, and why), and acts (produces an annotated image, a JSON
report, and a console alert for violations).

This satisfies **Tier 3** on two independent grounds:
1. **True multi-agent system** — three agents with distinct roles that pass
   structured messages (plain Python dicts / JSON) to one another.
2. **Custom-trained CV model** — a YOLO11n object detector fine-tuned from scratch
   on a PPE dataset, exported to ONNX, and used for inference.

## 2. Architecture

```
   image
     │
     ▼
┌─────────────────────┐   list of {class, confidence, bbox}
│  Perception Agent    │ ─────────────────────────────────────┐
│  (YOLO11 ONNX model) │                                       │
└─────────────────────┘                                       ▼
                                                    ┌───────────────────────┐
                                                    │  Compliance Agent      │
                                                    │  (rule-based reasoning)│
                                                    └───────────────────────┘
                                                                │
                                          {compliant, violations, missing, detected}
                                                                │
                                                                ▼
                                                    ┌───────────────────────┐
                                                    │  Reporter Agent        │
                                                    │  (produces output)     │
                                                    └───────────────────────┘
                                                                │
                                     annotated image + JSON report + console alert
```

| Agent | Role | Input → Output |
|---|---|---|
| **Perception Agent** | Runs the fine-tuned YOLO11 ONNX model on an image | image → list of `{class, confidence, bbox}` detections |
| **Compliance Agent** | Rule-based reasoning over detections | detections → `{compliant, violations, missing, detected}` |
| **Reporter Agent** | Produces the useful output | compliance result → annotated image + JSON report (+ console alert for violations) |

Each agent only ever sees the *structured output* of the agent before it — never
raw pixels it didn't produce itself — except the Perception Agent, which is the
only one that touches the image. This is enforced by the message schema, not
just convention: `compliance_agent()` takes a list of detection dicts, not an
image; `reporter_agent()` takes the compliance dict, not the raw model output.

## 3. Repository Structure

```
ppe-compliance-agent/
├── README.md                     ← you are here
├── docs/
│   ├── AI_usage_log.md           ← how AI tools were used while building this
│   └── images/                   ← figures referenced in this README
├── notebooks/
│   └── PPE_Compliance_Agent.ipynb ← full pipeline: setup → train → export → agents → eval
└── outputs/
    ├── sample_reports/           ← example JSON reports produced by the Reporter Agent
    └── sample_annotated_images/  ← example annotated output images
```

## 4. Dataset

Trained on Ultralytics' own
[Construction-PPE dataset](https://docs.ultralytics.com/datasets/detect/construction-ppe)
— 1,416 images (1,132 train / 143 val / 141 test), 178.4 MB, 11 classes:

- **Worn:** `helmet, gloves, vest, boots, goggles`
- **Missing/violation:** `no_helmet, no_gloves, no_boots, no_goggle`
- **Other:** `Person, none`

No API keys, tokens, or logins are required. Referencing `construction-ppe.yaml`
in Ultralytics' training code downloads and unpacks the dataset automatically
from a public GitHub release.

<p align="center">
  <img src="docs/images/dataset_samples.png" width="800"><br>
  <em>Sample training images with ground-truth PPE labels.</em>
</p>

## 5. Model Training

- **Base checkpoint:** `yolo11n.pt` (COCO-pretrained, fine-tuned on Construction-PPE)
- **Framework:** Ultralytics YOLO11, exported to ONNX for inference (`onnxruntime`, CPU provider)

Two training configurations were run:

| Run | Epochs | Image size | Purpose |
|---|---|---|---|
| Smoke test | 10 | 320px | Verify the full pipeline (train → export → agents → reports) runs end-to-end before committing GPU hours to a longer run |
| Final | 50 | 640px | Production model used for the results below |

> **Fill in before submitting:** replace the metrics table and training-curve image
> in Section 6 with the output of your 50-epoch/640px run once it finishes. The
> numbers currently shown below are from the 10-epoch/320px smoke test.

## 6. Evaluation Results

Validation set: 143 images, 1,172 ground-truth instances.

**Smoke-test run (10 epochs, 320px):**

| Metric | Value |
|---|---|
| mAP50 | 0.524 |
| mAP50-95 | 0.253 |
| Mean Precision | 0.710 |
| Mean Recall | 0.504 |

| Class | Precision | Recall | mAP50 | mAP50-95 | Instances |
|---|---|---|---|---|---|
| helmet | 0.786 | 0.746 | 0.768 | 0.403 | 201 |
| gloves | 0.698 | 0.632 | 0.672 | 0.308 | 136 |
| vest | 0.768 | 0.830 | 0.842 | 0.495 | 171 |
| boots | 0.735 | 0.702 | 0.733 | 0.401 | 151 |
| goggles | 0.736 | 0.723 | 0.745 | 0.266 | 47 |
| none | 0.578 | 0.531 | 0.506 | 0.207 | 81 |
| Person | 0.775 | 0.879 | 0.876 | 0.515 | 239 |
| no_helmet | 0.473 | 0.378 | 0.331 | 0.109 | 45 |
| no_goggle | 1.000 | 0.048 | 0.161 | 0.046 | 41 |
| no_gloves | 0.256 | 0.071 | 0.126 | 0.035 | 56 |
| no_boots | 1.000 | 0.000 | 0.000 | 0.000 | 4 |

<p align="center">
  <img src="docs/images/training_curves_smoketest.png" width="800"><br>
  <em>Loss and precision/recall/mAP curves, 10-epoch smoke test.</em>
</p>

### Honest analysis

The **worn-PPE classes** (`helmet`, `vest`, `boots`, `goggles`, `Person`) trained
reasonably well even in 10 epochs, with mAP50 in the 0.67–0.88 range. The
**explicit violation classes** (`no_helmet`, `no_goggle`, `no_gloves`, `no_boots`)
are much weaker — recall is especially poor (as low as 0.0 for `no_boots`, which
only has 4 validation instances). This is expected: those classes are rarer in
the dataset, visually more ambiguous (the model has to notice an *absence*, not
a distinct object), and 10 epochs is not enough to learn them reliably. This is
exactly why the smoke test was run at low epochs/resolution first — to confirm
the pipeline plumbing works before spending GPU hours on a real run. The 50-epoch/
640px run is expected to substantially improve recall on the violation classes.

## 7. Example Outputs

### Success case

*(Add an annotated image here from `outputs/sample_annotated_images/` showing a
correctly detected, clearly compliant or clearly non-compliant scene, plus a
one-line description of what the agent got right.)*

### Failure case

<p align="center">
  <img src="docs/images/failure_case_false_positive.png" width="500">
</p>

The detector produced a **false positive**: a `helmet 0.50` box over a stage
light with no person underneath it, plus a low-confidence `none 0.32` box that
doesn't correspond to a real object. This happens because the smoke-test model
was trained for only 10 epochs at a reduced 320px resolution on a domain
(construction sites) quite different from this test image (a drummer on stage) —
the model is generalizing poorly to out-of-domain scenes and bright circular
light fixtures are being confused with helmets. **Mitigations:** raising the
confidence threshold, training longer/at higher resolution (the 50-epoch/640px
run), and restricting deployment to in-domain imagery.

## 8. How to Run

1. Open `notebooks/PPE_Compliance_Agent.ipynb` in Google Colab.
2. `Runtime → Run all`. No API keys, tokens, or accounts are required anywhere —
   the dataset download is fully public and automatic (~178MB).
3. Section 4's Google Drive mount is optional — only used to persist trained
   weights across Colab sessions. Skip it if you don't need that.
4. Training defaults to 50 epochs / 640px in the notebook (`Cell 5.0`). Reduce
   `epochs` there for a quicker smoke-test run.
5. Sections 8–9 run the three-agent pipeline over a batch of validation images
   and produce the dashboard shown above. Reports land in
   `/content/ppe_multi_agent_outputs/` inside Colab — copy the ones you want
   to keep into `outputs/sample_reports/` and `outputs/sample_annotated_images/`
   in this repo.

## 9. Known Limitations

- Violation classes (`no_helmet`, `no_gloves`, `no_goggle`, `no_boots`) are
  underrepresented in the dataset and are the weakest part of the model —
  see Section 6.
- The Compliance Agent's rules are simple set-membership logic (required PPE
  present, no explicit violation class detected) — it doesn't reason about
  *which person* is missing *which* item in a multi-person scene; it flags the
  scene as a whole.
- Trained and evaluated on a single public dataset; not validated against a
  different construction-site camera/lighting setup.

## 10. AI Usage

See [`docs/AI_usage_log.md`](docs/AI_usage_log.md) for a full account of how AI
tools were used throughout this project.

## 11. Reproducing This Project

Same as Section 8 above — clone this repo, open the notebook in Colab, `Run all`.
Total cost: no paid APIs required; only Colab GPU compute time.
