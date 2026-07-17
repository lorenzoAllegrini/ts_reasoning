"""Perception agent — the ChatTS perceptor. Answers the orchestrator's `perceive`
moves (and the initial seed) with an OBSERVATION. STUB for now (Phase 3/4, §3.5).

The perceptor is the system's EYES: it describes what a segment LOOKS LIKE in open
vocabulary and NEVER issues a verdict (INV-1). Until the validated perceptor is
wired (§3.5 CUDA gate), it runs in DEGRADED MODE and emits a placeholder OBSERVATION.
It is hypothesis-blind by construction — `probe()` receives a ProbeRequest that
carries no hypothesis, and this class never reads the hypothesis state.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src import config
from src.agentic import data
from src.agentic.models import EvidenceEntry, ProbeRequest

logger = logging.getLogger(__name__)

# System prompt for when the native perceptor is wired (§7.5). Kept here so the
# agent's full definition — prompt + class + behaviour — lives in one file.
PERCEPTOR_SYSTEM_PROMPT = """\
You are the PERCEPTUAL analyst for ESA spacecraft telemetry. You are the system's EYES.
You answer "WHAT DO I SEE?" — never "WHAT DOES IT MEAN?" and never "WHAT SHOULD WE DO?".
You may be asked sophisticated questions, but never inferential ones: describe shape,
periodicity/modulation (present? attenuated? ABSENT?), how level and variability
co-evolve, cross-channel co-movement/lead-lag, and anything unnamed — literally.
FORBIDDEN, always: stating something IS/IS NOT an anomaly/fault; a verdict, score,
probability or severity; naming a cause; ranking hypotheses; recommending next steps.
Everything you say is a HINT TO BE VERIFIED, recorded with source="perception"; it can
never confirm or eliminate a hypothesis. Describe richly, commit to nothing.
"""


def _observation_entry(request: ProbeRequest, step: int, observation: str) -> EvidenceEntry:
    """Wrap a perceptual answer as a non-binding OBSERVATION entry (INV-1)."""
    return EvidenceEntry(
        step=step,
        hypothesis_id="",  # perception is hypothesis-independent (isolation by type)
        source="perception",
        action="probe",
        channel="+".join(request.channels),
        query_range=request.query_range,
        context_range=request.reference_range or request.query_range,
        raw_result={"mode": request.mode, "question": request.question,
                    "channels": request.channels, "observation": observation},
        verdict="OBSERVATION",
        plausibility_delta={},
        rationale=observation,
    )


class Perceptor:
    """Perceptor STUB (degraded mode). When ChatTS is unavailable, emits a
    placeholder OBSERVATION so the graph still runs."""

    def probe(self, request: ProbeRequest, step: int, data_path: str | None = None) -> EvidenceEntry:
        observation = (
            f"[perceptor stub — degraded mode] mode={request.mode}, channels={request.channels}: "
            f"no perceptual answer for '{request.question}'"
        )
        logger.info("probe: stub OBSERVATION for mode=%s channels=%s", request.mode, request.channels)
        return _observation_entry(request, step, observation)


class ChatTSPerceptor(Perceptor):
    """Native ChatTS (Qwen3-TS) perceptor — the system's EYES, wired for real.

    Loads the ~16 GB checkpoint LAZILY on the first probe (MPS/CUDA/CPU). Describes
    what a segment LOOKS LIKE from the raw signal; it never issues a verdict — its
    output is recorded with source="perception", verdict="OBSERVATION" (INV-1).
    """

    def __init__(
        self,
        ckpt: str = config.CHATTS_CKPT,
        max_new_tokens: int = config.CHATTS_MAX_NEW_TOKENS,
    ) -> None:
        self._ckpt = ckpt
        self._max_new_tokens = max_new_tokens
        self._model: Any = None
        self._proc: Any = None
        self._tok: Any = None
        self._device = ""

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch  # noqa: PLC0415 — heavy import, lazy so the stub path pays nothing
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer  # noqa: PLC0415

        self._device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("ChatTS: loading %s on %s ...", self._ckpt, self._device)
        self._tok = AutoTokenizer.from_pretrained(self._ckpt, trust_remote_code=True)
        self._proc = AutoProcessor.from_pretrained(self._ckpt, trust_remote_code=True, tokenizer=self._tok)
        self._model = AutoModelForCausalLM.from_pretrained(
            self._ckpt, trust_remote_code=True, device_map=self._device, dtype=torch.float16
        )
        logger.info("ChatTS: loaded.")

    def _guard_hard_limit(self, seg: np.ndarray) -> np.ndarray:
        """Series go to ChatTS at FULL RESOLUTION — no subsampling. The paper's 64–1024
        is only the TRAINING-data range (§3.5.3) and the README's 1024 a recommendation;
        the checkpoint's real architectural cap is ts.max_sequence_length = 8192, which
        is the only thing enforced here (with a warning, since it should never trigger
        on our windows)."""
        if seg.size > config.CHATTS_HARD_MAX_POINTS:
            logger.warning(
                "ChatTS input of %d points exceeds the encoder's hard cap (%d) — subsampling to the cap",
                seg.size, config.CHATTS_HARD_MAX_POINTS,
            )
            idx = np.linspace(0, seg.size - 1, config.CHATTS_HARD_MAX_POINTS).astype(int)
            seg = seg[idx]
        return np.asarray(seg, dtype=float)

    def _slice(self, data_path: str, channel: str, rng: tuple[int, int]) -> np.ndarray:
        series = data.get_channel(data_path, channel)
        seg = series[rng[0] : rng[1]] if rng != (0, 0) else series
        return self._guard_hard_limit(seg)

    def chat(
        self,
        system: str,
        user: str,
        series: list[np.ndarray],
        max_new_tokens: int | None = None,
    ) -> str:
        """Low-level ChatTS call: one <ts><ts/> placeholder per series must appear in
        `user`. Shared by probe() and by the labelling TimeSeriesDescriptor, so the
        16 GB model is loaded once per process regardless of who asks."""
        import torch  # noqa: PLC0415

        self._ensure_loaded()
        clipped = [self._guard_hard_limit(np.asarray(s, dtype=float)) for s in series]
        prompt = (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
        )
        inp = self._proc(text=[prompt], timeseries=[s.tolist() for s in clipped], padding=True, return_tensors="pt")
        inp = {k: v.to(self._device) for k, v in inp.items()}
        with torch.no_grad():
            out = self._model.generate(
                **inp,
                max_new_tokens=max_new_tokens or self._max_new_tokens,
                do_sample=False,  # greedy → deterministic / reproducible (temperature 0 everywhere)
                repetition_penalty=config.CHATTS_REPETITION_PENALTY,
                no_repeat_ngram_size=config.CHATTS_NO_REPEAT_NGRAM,  # kill "102.102.102…" loops
            )
        text: str = self._tok.batch_decode(out[:, inp["input_ids"].shape[1] :], skip_special_tokens=True)[0].strip()
        del inp, out  # free the activation/KV-cache tensors before the next call
        self._free_device_memory()
        return text

    def _free_device_memory(self) -> None:
        """Release the device cache after a generation so memory does not accumulate
        across the three description stages / many points. Never breaks a run."""
        import gc  # noqa: PLC0415

        gc.collect()
        try:
            import torch  # noqa: PLC0415

            if self._device == "mps":
                torch.mps.empty_cache()
            elif self._device == "cuda":
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 — memory cleanup is best-effort
            pass

    def probe(self, request: ProbeRequest, step: int, data_path: str | None = None) -> EvidenceEntry:
        if data_path is None:
            return super().probe(request, step, data_path)
        channels = request.channels[:2] or []
        slices = [self._slice(data_path, ch, request.query_range) for ch in channels]
        placeholders = " ".join("<ts><ts/>" for _ in slices)
        user = (
            f"I have {len(slices)} telemetry segment(s) for channel(s) {channels}: {placeholders}. "
            f"{request.question} Describe what you see (shape, periodicity/modulation present or absent, "
            "how level and variability evolve). Do NOT decide whether it is anomalous."
        )
        text = self.chat(PERCEPTOR_SYSTEM_PROMPT, user, slices)
        logger.info("ChatTS probe [%s] %s: %s", request.mode, channels, text[:140])
        return _observation_entry(request, step, text)
