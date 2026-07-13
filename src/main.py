"""Entrypoint — run the interpretation loop and print the evidence log (INV-3).

    uv run python -m src.main            # synthetic worked-example signal, real LLM
    uv run python -m src.main --stub     # synthetic signal, no LLM (deterministic path)
    uv run python -m src.main --smoke    # ping Groq and verify the model string (§12 Step 0)

The evidence log IS the deliverable: every hypothesis, query, verdict and belief
update is printed as the audit trail.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

from data.demo_ts_generator import save
from src import config
from src.graph import TelemetryInterpretationGraph, make_initial_state
from src.llm import get_llm
from src.models import AgentState
from src.observability import init_tracing


def _smoke_test() -> int:
    """Ping the Groq model with a trivial prompt; list available models on failure."""
    print(f"Pinging Groq model '{config.GROQ_MODEL}' ...")
    try:
        response = get_llm().invoke("Reply with exactly one word: pong")
    except Exception as exc:  # noqa: BLE001 — smoke test must report, not crash
        print(f"ERROR calling '{config.GROQ_MODEL}': {exc}")
        try:
            from groq import Groq

            client = Groq(api_key=os.environ["GROQ_API_KEY"])
            print("Available Groq models:")
            for model in sorted(client.models.list().data, key=lambda m: m.id):
                print(f"  - {model.id}")
        except Exception as list_exc:  # noqa: BLE001
            print(f"(could not list models: {list_exc})")
        return 1
    print(f"OK — response: {response.content!r}")
    return 0


def _print_report(state: AgentState) -> None:
    report = state["report"] or {}
    print("\n" + "=" * 70)
    print("EVIDENCE LOG (the deliverable)")
    print("=" * 70)
    for e in state["evidence_log"]:
        print(
            f"[step {e.step}] {e.source:<13} H={e.hypothesis_id} {e.action} "
            f"ch={e.channel} q={e.query_range} c={e.context_range} -> {e.verdict}"
        )
        print(f"           {e.rationale}")
        if e.plausibility_delta:
            print(f"           Δplausibility: {{" + ", ".join(f'{k}: {v:+.2f}' for k, v in e.plausibility_delta.items()) + "}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print(f"concluded (ACCEPT): {report.get('concluded')}   iterations: {report.get('n_iterations')}")
    if report.get("conclusion"):
        print(f"  {report['conclusion']['type']} — {report.get('conclusion_reasoning', '')}")
    for row in report.get("ranking", []):
        marker = "→" if row["status"] == "open" and row == report.get("ranking", [None])[0] else " "
        print(f"  {marker} {row['id']} [{row['type']}] p={row['plausibility']:.3f} ({row['status']})")


def _build_graph(use_stub: bool) -> TelemetryInterpretationGraph:
    if not use_stub:
        return TelemetryInterpretationGraph()  # real LLM agents
    from data.worked_example import worked_example_graph  # noqa: PLC0415

    return worked_example_graph()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Continuous Telemetry Interpretation loop")
    parser.add_argument("--stub", action="store_true", help="run the deterministic path (no LLM)")
    parser.add_argument("--smoke", action="store_true", help="ping Groq and verify the model string")
    parser.add_argument("--data", type=str, default=None, help="path to an .npz signal (default: synthetic demo)")
    parser.add_argument("--trace", action="store_true", help="emit OpenTelemetry/GenAI traces (or set ESA_TRACING=1)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for noisy in ("httpx", "groq", "urllib3", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    init_tracing(force=args.trace)  # opt-in; instruments the LLM calls that follow

    if args.smoke:
        return _smoke_test()

    data_path = args.data or save(str(Path(tempfile.mkdtemp()) / "demo.npz"))
    state = make_initial_state(
        data_path,
        "TEMP_BATT jumps and its modulation stops after t=1200. What happened?",
        channels=["TEMP_BATT", "HEATER_CURRENT"],
        global_query_range=(0, 2000),
    )
    final = _build_graph(args.stub).run(state)
    _print_report(final)
    return 0


if __name__ == "__main__":
    sys.exit(main())
