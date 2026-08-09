# AI Usage Log

This project made extensive use of AI coding assistants during development. This
log documents what was used, for what, and how outputs were checked.

## Tools used

- **GitHub Copilot** — inline code completion while writing notebook cells
  (preprocessing utilities, ONNX decode logic, matplotlib plotting boilerplate).
- **ChatGPT** — early drafting of the YOLO training/export cells and general
  troubleshooting of Colab/Ultralytics/ONNX Runtime version issues.
- **Gemini** — used in parallel with ChatGPT for the same kind of drafting and
  debugging support.
- **Claude (Anthropic)** — used specifically for debugging and code review once
  the notebook was largely built, and for putting together this repository's
  documentation (README, this log, repo structure).

## How each tool was used

**Drafting.** The bulk of the notebook — dataset setup, the YOLO11 training
loop, ONNX export, the preprocessing/decoding utilities, and the three-agent
pipeline (Perception / Compliance / Reporter agents) — was written iteratively
with Copilot, ChatGPT, and Gemini suggesting code that was then run, checked
against actual Colab output, and revised. Every cell was executed and its
output inspected before moving on; nothing was accepted purely on the AI's say-so.

**Debugging with Claude.** Colab's runtime was unstable throughout the project
(frequent disconnects, session resets), which made it costly to iterate on bugs
that only appeared deep in the pipeline. When ChatGPT and Gemini weren't
resolving a runtime error effectively, Claude was used to diagnose it directly
from the notebook and full traceback. Claude identified and fixed a bug in
`decode_yolo_onnx()` (Cell 7.0) where bounding-box coordinates, derived from a
raw NumPy array in the ONNX model output, stayed as `numpy.float32` instead of
native Python floats. `round()` on a NumPy scalar returns another NumPy scalar,
so the box coordinates in each detection dict were still `float32` by the time
the Reporter Agent tried to `json.dump()` them in Cell 8.2 — which NumPy types
cannot serialize to JSON. This was causing the orchestrator cell (Cell 8.3) to
crash with `TypeError: Object of type float32 is not JSON serializable` on
every run. The fix casts each coordinate to `float()` before rounding.

**Documentation.** Claude was also used to draft this repository's README and
this usage log, based on the finished notebook and a description of the
project's history and results.

## What we'd flag for the reader

- Given how much of this notebook was written with AI assistance, the team
  focused verification effort on the parts most likely to silently produce
  wrong results: the box-coordinate math in `decode_yolo_onnx` (letterboxing,
  scale/offset correction back to original image coordinates), the compliance
  rule logic (set membership against `REQUIRED_PPE` / `VIOLATION_CLASSES`), and
  the evaluation metrics reported by Ultralytics' own `.val()` call, which were
  used as-is rather than recomputed.
- Colab instability (frequent disconnects and session limits) meant the free
  daily compute allotment (~5h40m) was exhausted repeatedly; the team purchased
  100 Colab compute units (~$10) and used roughly 24 of them getting a clean
  end-to-end smoke-test run (10 epochs, 320px) before committing to the longer
  50-epoch, 640px training run used for the final reported results.
- The smoke test's own metrics and failure case (see README Sections 6–7) are
  reported honestly rather than omitted, even though they're weaker than the
  final run — they were a real, useful checkpoint that validated the pipeline
  before spending further compute.
