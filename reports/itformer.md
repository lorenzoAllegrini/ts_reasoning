# ITFormer — smoke test report

**Located via:** arXiv 2506.20093 (Lei et al., ICML 2025) → project site
`pandalin98.github.io/itformer_site` → official repo
**https://github.com/Pandalin98/ITFormer-ICML25** (public, cloned shallow).
Checkpoints on HF: `pandalin98/ITFormer-{0.5B,3B,7B}`; dataset `pandalin98/EngineMT-QA`.
**Hardware:** Apple M1 Max, 64 GB — no CUDA GPU; CPU inference.

## Environment (`.venv-itformer`, uv, repo's pinned requirements)

torch 2.5.1 · transformers 4.47.1 · datasets 2.4.0 · accelerate 1.6.0 ·
safetensors 0.4.5 · numpy 1.26.4 · timm 1.0.15 · tokenizers 0.21.0 (+ h5py/fsspec for
the smoke dataset). **Deviations from their requirements.txt, all documented:**
- `xformers==0.0.28.post3` **skipped** — no macOS-arm64 wheels, and verified the code
  never imports it (requirements-only entry).
- `nltk`, `rouge-score`, `scikit-learn` **added** — imported by `utils/metrics.py`
  but missing from requirements.txt.

## Checkpoints (in `./models/itformer/`, HF_TOKEN from `.env`, never printed)

- `models/itformer/ITFormer-0.5B/` — **2.2 GB** snapshot of `pandalin98/ITFormer-0.5B`
  (training states excluded). The smallest of the three published sizes.
  Note: the repo's custom `TLM.from_pretrained` accepts **local paths only** (raises on
  HF ids), hence the explicit snapshot download.
- `models/itformer/Qwen2.5-0.5B-Instruct/` — base LLM (the code hardcodes the relative
  path `LLM/Qwen2.5-0.5B-Instruct`; satisfied via symlink into `./models/itformer/`).

## Inference entry point

`inference.py --config yaml/infer.yaml` — batch evaluation over EngineMT-QA
(`time_series_data.h5` + `test_qa.jsonl`). The full h5 is **18.84 GB** (shape read
remotely via HTTP range requests: `seq_data (118921, 600, 33)` — 600 steps × 33
sensors), far too heavy for a smoke test — so the smoke reuses their pipeline UNCHANGED
on a **1-sample mini-dataset I built**: `smoke.h5` with the synthetic series (600 pts,
sin period 60 → 10 cycles, +3.0 step after t=300, tiled on all 33 channels) and
`smoke_qa.jsonl` with the prompt as an open (stage-1) question containing `<ts>`.

Run: `python inference.py --config yaml/infer_smoke.yaml --model_checkpoint
models/itformer/ITFormer-0.5B --max_new_tokens 160 --num_workers 0`
(smoke yaml: `fp16: false` for CPU; `--num_workers 0` because workers>0 crashes —
`Can't pickle local object 'main_inference.<locals>.SimpleConfig'`, an upstream bug).

## Smoke test — outcome: ✅ runs end-to-end (fast), ❌ description factually wrong

```
[total wall: 20.6 s incl. model load — by far the fastest of the three]
Question:  Describe the shape of this signal: overall form, any sudden changes,
           any periodic pattern.
Prediction: "The signal consists of a straight path with consistent changes in shape.
             There are no sudden changes or periodic patterns observed."
```

Fluent natural language, but **wrong on both required elements** — it explicitly denies
the periodic pattern (10 clean sine cycles) and the sudden change (a +3σ step at
midpoint). Fairness note: ITFormer-0.5B is trained exclusively on aviation-engine QA
(EngineMT); a synthetic sinusoid+step is far out of its training domain.

## Verdict on the description criterion

Installed: **yes** · Inference: **yes** (exit 0, results JSON written) ·
Time: 20.6 s total on CPU · Criterion (mentions BOTH periodicity and step): **NO (0/2,
actively denies both)**.
