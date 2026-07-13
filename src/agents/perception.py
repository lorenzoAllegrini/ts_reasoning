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

from src import config, data
from src.models import EvidenceEntry, ProbeRequest

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
        max_points: int = config.CHATTS_MAX_POINTS,
        max_new_tokens: int = config.CHATTS_MAX_NEW_TOKENS,
    ) -> None:
        self._ckpt = ckpt
        self._max_points = max_points
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

    def _slice(self, data_path: str, channel: str, rng: tuple[int, int]) -> np.ndarray:
        series = data.get_channel(data_path, channel)
        seg = series[rng[0] : rng[1]] if rng != (0, 0) else series
        if seg.size > self._max_points:  # uniform subsample to keep ChatTS's context small
            idx = np.linspace(0, seg.size - 1, self._max_points).astype(int)
            seg = seg[idx]
        return np.asarray(seg, dtype=float)

    def probe(self, request: ProbeRequest, step: int, data_path: str | None = None) -> EvidenceEntry:
        if data_path is None:
            return super().probe(request, step, data_path)
        import torch  # noqa: PLC0415

        self._ensure_loaded()
        channels = request.channels[:2] or []
        slices = [self._slice(data_path, ch, request.query_range) for ch in channels]
        placeholders = " ".join("<ts><ts/>" for _ in slices)
        user = (
            f"I have {len(slices)} telemetry segment(s) for channel(s) {channels}: {placeholders}. "
            f"{request.question} Describe what you see (shape, periodicity/modulation present or absent, "
            "how level and variability evolve). Do NOT decide whether it is anomalous."
        )
        prompt = (
            f"<|im_start|>system\n{PERCEPTOR_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
        )
        inp = self._proc(text=[prompt], timeseries=[s.tolist() for s in slices], padding=True, return_tensors="pt")
        inp = {k: v.to(self._device) for k, v in inp.items()}
        with torch.no_grad():
            out = self._model.generate(**inp, max_new_tokens=self._max_new_tokens)
        text = self._tok.batch_decode(out[:, inp["input_ids"].shape[1] :], skip_special_tokens=True)[0].strip()
        logger.info("ChatTS probe [%s] %s: %s", request.mode, channels, text[:140])
        return _observation_entry(request, step, text)
