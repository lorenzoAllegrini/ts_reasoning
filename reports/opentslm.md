# OpenTSLM (Stanford) — smoke test report

**Repo:** https://github.com/StanfordBDHG/OpenTSLM @ `104013b` (checkout already present in
the project; verified `origin` = official repo — reused, not re-cloned)
**Hardware:** Apple M1 Max, 64 GB — no CUDA GPU. `device="cpu"` forced **explicitly**
(README states: "MPS does NOT work for pretrained HF checkpoints"; same warning in the
task sheet).

## Environment (`.venv-opentslm`, uv, `uv pip install -r requirements.txt`)

Key versions: torch 2.13.0 · transformers 5.14.1 · open-flamingo **0.0.2** (the version
the repo's own `uv.lock` pins) · peft 0.18.x · datasets 4.x · einops · wfdb.
Package installed editable (`-e .` in requirements.txt) as `opentslm`.

## Checkpoints

Pretrained checkpoints live on HF Hub under the `OpenTSLM` org
(`{base}-{task}-{sp|flamingo}`), loaded via the factory `OpenTSLM.load_pretrained`.
Downloaded with HF_TOKEN (never printed) into `HF_HOME=./models/opentslm`:

- `OpenTSLM/llama-3.2-1b-tsqa-flamingo` → `models/opentslm/hub/models--OpenTSLM--llama-3.2-1b-tsqa-flamingo/...model_checkpoint.pt`
- `OpenTSLM/llama-3.2-1b-m4-flamingo` → same layout (captioning-task variant)
- base LLM `meta-llama/Llama-3.2-1B` (gated; the token has Meta access)

**Backbone choice:** the task sheet says "Flamingo with the SMALLEST available backbone
(~3B)". Smallest on the org is actually **gemma-3-270m**, but `google/gemma-3-270m` is
**licence-gated for this token (403 GatedRepoError, verified)** — accepting the Gemma
licence is an account action I cannot perform. Smallest ACCESSIBLE backbone:
**Llama-3.2-1B**.

## Pure-inference entry point

`demo/huggingface/0X_test_hf_*.py` scripts (no training needed): load from Hub +
`model.generate(batch)`. My `smoke_opentslm.py` mirrors `TSQADataset`'s sample format
exactly (z-normalised series, caption "This is the time series, it has mean X and std Y.",
post-prompt "Answer:").

## Friction log — 3 upstream bugs/incompatibilities found and patched locally

1. **`AttributeError: 'SimpleNamespace' object has no attribute 'requires_grad_'`**
   (`OpenTSLMFlamingo.__init__`). HEAD wraps the time-series encoder as
   `SimpleNamespace(visual=encoder)`, but open-flamingo **0.0.2** (their own lock!)
   assigns the argument directly to `self.vision_encoder`. **Proof the module form is
   what was trained:** the pretrained checkpoints contain `model.vision_encoder.*` keys
   (pos_embed / patch_embed / input_norm). → local 1-line patch: pass the bare encoder.
2. **`TypeError: Flamingo.generate() got an unexpected keyword argument 'eos_token_id'`**
   — 0.0.2's `generate()` has a fixed signature without eos/pad and no `**kwargs`.
   → local patch: drop the two kwargs (0.0.2 handles EOS internally).
3. **`RuntimeError: expected m1 and m2 to have the same dtype: BFloat16 != float`** —
   the base Llama loads in bf16 while Flamingo add-ons (perceiver/cross-attn/encoder)
   are fp32; mixed-dtype matmul fails on CPU. → `model.llm.float()` after load.

Checkpoint load reports only *Unexpected* keys (`lang_encoder.old_decoder_blocks.*`,
open-flamingo 2.x-style duplicates of the FROZEN base decoder — harmless) and **zero
Missing keys**: cross-attention, perceiver, TS encoder and embeddings all load.

## Smoke test — outcome: ✅ runs end-to-end; description quality varies by checkpoint

Series: same as Model 1 (240 pts, sin(2πt/24) + step of +3.0 after t=120).
Prompt: "Describe the shape of this signal: overall form, any sudden changes, any
periodic pattern."

### `llama-3.2-1b-tsqa-flamingo` (QA-finetuned)

```
[load: 11 s] [generate: 53 s]   (total wall 70.9 s)
 (b) (b)) (b) (b): (b) (b))) (b)))))))) (b)) (b))))a)))a))): (b): (b): (b)b)a): (b)c)a)c)…
```
Degenerate: the TSQA checkpoint is finetuned on multiple-choice QA, so an open
descriptive prompt collapses into answer-option tokens. Mechanically the pipeline works;
descriptively unusable.

### `llama-3.2-1b-m4-flamingo` (M4 = captioning-finetuned)

```
[load: 229 s (incl. checkpoint download)] [generate: 57 s]   (total wall 292.9 s)
 (0, 0)b0 on the x-axis, the values range from 0 to 1000 on the y-axis. Initially, the
data shows a stable trend around 1000, followed by a sharp decline to approximately 900.
This is succeeded by a rapid increase, peaking near 1000 again. The pattern repeats with
another steep drop and subsequent rise, maintaining a similar pattern of peaks and
troughs. The data suggests periodic fluctuations with a consistent frequency, indicating
potential cyclical behavior or seasonal variation. The overall trend appears stable,
with no significant long-term upward or downward movement.b0 is plotted on both axes,
providing a clear visualization of the periodic nature of the data.b0 is the x-axis,
the y-axis, and the observed values range from 0 to 1000.a grid for enhanced
readability, aiding in the analysis of the periodicity and amplitude of the
fluctuations.b0 is
```

Real natural language (unlike the TSQA variant), with artifacts: stray "b0" tokens and
hallucinated axis ranges (0–1000; the input was z-normalised).

## Verdict on the description criterion

- **Periodic oscillation: ✅ mentioned clearly** ("pattern repeats", "periodic
  fluctuations with a consistent frequency", "cyclical behavior").
- **Sharp step at midpoint: ❌ missed — and actively denied**: "The overall trend
  appears stable, with no significant long-term upward or downward movement", while the
  series contains a sustained +3σ level shift at t=120.

**Criterion (must mention BOTH): NO — partial (periodicity only).**
Installed: **yes** · Inference: **yes** (after 3 documented local patches) ·
Time: TSQA 70.9 s total (gen 53 s) · M4 292.9 s total (gen 57 s), CPU.
