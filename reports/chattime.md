# ChatTime — smoke test report

**Repo:** https://github.com/ForestsKing/ChatTime (cloned shallow into `./ChatTime`)
**Model:** `ChengsenWang/ChatTime-1-7B-Chat` (7B, LLaMA-based; time series serialized as text tokens)
**Hardware:** Apple M1 Max, 64 GB RAM — **no CUDA GPU** (`nvidia-smi` not present). Inference on **CPU**.

## Environment (`.venv-chattime`, uv)

Installed per instructions (`torch transformers pandas numpy matplotlib`) **plus 4 packages
the instruction list was missing but the repo requires**: `scikit-learn` (imported by
`utils/tools.py`), `accelerate` (required by `device_map="auto"`), `sentencepiece` +
`protobuf` (required by `LlamaTokenizer`).

| package | installed | repo pins (requirements.txt) |
|---|---|---|
| torch | 2.13.0 | 2.7.1+cu118 |
| transformers | 5.14.1 | 4.53.0 |
| numpy | 2.5.1 | 2.1.2 |
| pandas | 3.0.3 | 2.3.0 |
| scikit-learn | 1.9.0 | 1.7.0 |
| accelerate | 1.14.0 | 1.8.1 |
| matplotlib | 3.11.1 | — |
| sentencepiece / protobuf / safetensors | 0.2.2 / 7.35.1 / 0.8.0 | — |

## Checkpoint

Downloaded with HF_TOKEN from `.env` (never printed), **not** in the default HF cache:
`HF_HOME=./models/chattime` →
`./models/chattime/hub/models--ChengsenWang--ChatTime-1-7B-Chat/snapshots/20baa88d.../`
3 safetensors shards, **13.64 GB** total, integrity verified (headers open, 291 tensors = 104+109+78).

## Smoke test — outcome: ✅ PASS (on CPU retry)

1. **First attempt (byte-exact `smoke_chattime.py`)**: ❌ crashed with **SIGBUS (exit 138)**
   at "Loading weights: 0/291" — `device_map="auto"` mapped the fp16 weights to **MPS** and
   the load path crashed (torch 2.13.0 + transformers 5.14.1; a macOS crash report was
   generated). Not an OOM (plenty of free RAM) and not corruption (shards verified intact).
   No full Python traceback exists: the process died on a signal.
2. **CPU retry (`smoke_chattime_cpu.py`)**, per the task instructions ("if GPU goes OOM,
   retry on CPU"): identical logic, two documented deltas — a load shim forcing
   `device_map="cpu"` (the class hardcodes `"auto"`) and `float32` (fp16 matmuls are
   emulated and glacial on ARM CPUs). Outcome:

```
PREDICT OK, output type/shape: <class 'numpy.ndarray'> (24,)
```

**Inference time:** 502 s wall-clock total for the whole script (load 13.6 GB fp16→fp32
+ 2 autoregressive rounds × 8 sampled sequences on CPU). `/usr/bin/time -p`:
real 502.42, user 8.69, sys 253.12.

## Textual QA interface — YES, it exists (`ChatTime.analyze`)

Beyond `predict()`, the class exposes **`analyze(question, series)`** (used by the official
`demo.ipynb` on the bundled TSQA dataset). Caveat discovered by reading the code: it is
built for **multiple-choice** QA — it post-processes generations with
`re.findall(r"\([abc]\)", ...)[0]` and majority-votes 8 samples, so a free-form question
crashes the regex. The free-form test below therefore replicates `analyze()`'s internals
(same discretizer → serializer → analysis prompt → pipeline) and reports the RAW text.

### Native `analyze()` on the repo's own TSQA row — ✅ works

Question (from `dataset/TSQA.csv`, definitions of outlier / sudden spike / level shift,
"Select one of the following answers … Only answer (a), (b), or (c)"):

```
TSQA analyze() -> (c)   (ground truth: (c))   [436 s on CPU — 8 sampled generations + majority vote]
```

### Free-form: "Describe the shape of this signal" (sinusoid + step series, raw samples)

The model does **not** produce natural language: asked an open question it falls back to
**continuing the serialized series** (its training format). Raw generations (3 samples,
each captured up to 500 chars):

```
--- sample 1 ---
###0.2693### ###0.3517### ###0.4001### ###0.4415### ###0.4573### ###0.4733### ###0.4999###
###0.4999### ###0.4733### ###0.4415### ###0.4001### ###0.3409### ###0.3001### ###0.2553###
###0.2033### ###0.1609### ###0.1571### ###0.1483### ###0.1571### ###0.1857### … ###-0.
--- sample 2 ---
###0.2999### ###0.3673### ###0.3999### ###0.4415### ###0.4733### ###0.4931### ###0.4999###
… then a long flat run of ###0.0999### repeated
--- sample 3 ---
###0.2999### ###0.3517### ###0.3999### ###0.4451### ###0.4733### ###0.4999### … " Bel "
… then a flat run of ###-0.1001### repeated
```

(42 s for the 3 samples. Note the stray non-numeric token " Bel " in sample 3.)

## Verdict on the description criterion

**Textual QA interface: YES, but multiple-choice only.** `analyze()` answered the TSQA
question correctly ((c) = ground truth). **Free-form description: NOT supported** — the
"Chat" variant degenerates to serialized value continuation, producing no words at all.
It therefore **cannot state that the series contains a periodic oscillation and a
midpoint step** → description criterion for the SUMMARY table: **NO**.
