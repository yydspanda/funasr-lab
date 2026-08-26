#!/usr/bin/env python3
"""Run or inspect one explicit CPU-first FunASR baseline smoke path."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path
from typing import Any


TRACKS: dict[str, dict[str, Any]] = {
    "paraformer": {
        "model": "paraformer-zh",
        "vad_model": "fsmn-vad",
        "punc_model": "ct-punc",
        "mode": "offline",
    },
    "sensevoice": {
        "model": "iic/SenseVoiceSmall",
        "vad_model": "fsmn-vad",
        "mode": "offline",
    },
    "streaming": {
        "model": "paraformer-zh-streaming",
        "mode": "streaming",
        "chunk_size": [0, 10, 5],
        "encoder_chunk_look_back": 4,
        "decoder_chunk_look_back": 1,
    },
}


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    plan = dict(TRACKS[args.track])
    plan.update(
        {
            "track": args.track,
            "audio": str(args.audio),
            "audio_exists": args.audio.is_file(),
            "device": args.device,
            "model_revision": args.model_revision,
            "vad_model_revision": args.vad_model_revision,
            "punc_model_revision": args.punc_model_revision,
            "disable_update": True,
        }
    )
    return plan


def _offline_inference(plan: dict[str, Any]) -> Any:
    from funasr import AutoModel

    model_kwargs = {
        key: plan[key]
        for key in ("model", "vad_model", "punc_model")
        if key in plan
    }
    model = AutoModel(
        **model_kwargs,
        model_revision=plan["model_revision"],
        vad_model_revision=plan["vad_model_revision"],
        punc_model_revision=plan["punc_model_revision"],
        device=plan["device"],
        disable_update=True,
    )
    return model.generate(input=plan["audio"])


def _streaming_inference(plan: dict[str, Any]) -> Any:
    import soundfile as sf
    from funasr import AutoModel

    audio, sample_rate = sf.read(plan["audio"], dtype="float32")
    if sample_rate != 16_000:
        raise ValueError(f"streaming smoke requires 16 kHz audio, got {sample_rate}")
    if getattr(audio, "ndim", 1) != 1:
        raise ValueError("streaming smoke requires mono audio")

    model = AutoModel(
        model=plan["model"],
        model_revision=plan["model_revision"],
        device=plan["device"],
        disable_update=True,
    )
    chunk_size = plan["chunk_size"]
    chunk_stride = chunk_size[1] * 960
    cache: dict[str, Any] = {}
    partials: list[str] = []
    chunk_count = (len(audio) - 1) // chunk_stride + 1
    for index in range(chunk_count):
        chunk = audio[index * chunk_stride : (index + 1) * chunk_stride]
        result = model.generate(
            input=chunk,
            cache=cache,
            is_final=index == chunk_count - 1,
            chunk_size=chunk_size,
            encoder_chunk_look_back=plan["encoder_chunk_look_back"],
            decoder_chunk_look_back=plan["decoder_chunk_look_back"],
        )
        text = result[0].get("text", "") if result else ""
        if text:
            partials.append(text)
    return {"partials": partials, "text": "".join(partials)}


def _peak_rss_mb() -> float:
    # Linux reports ru_maxrss in KiB; this project currently targets Linux/WSL.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect or run a pinned FunASR baseline smoke path."
    )
    parser.add_argument("--track", choices=sorted(TRACKS), required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-revision", default="master")
    parser.add_argument("--vad-model-revision", default="master")
    parser.add_argument("--punc-model-revision", default="master")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the resolved plan without importing FunASR or downloading models",
    )
    args = parser.parse_args()
    plan = build_plan(args)
    if args.dry_run:
        print(json.dumps({"kind": "baseline-smoke-plan", "plan": plan}, indent=2))
        return 0
    if not args.audio.is_file():
        parser.error(f"audio file does not exist: {args.audio}")
    if args.model_revision == "master":
        parser.error("actual runs require a pinned --model-revision")
    if "vad_model" in plan and args.vad_model_revision == "master":
        parser.error("actual runs with VAD require a pinned --vad-model-revision")
    if "punc_model" in plan and args.punc_model_revision == "master":
        parser.error("actual runs with punctuation require --punc-model-revision")

    started = time.perf_counter()
    result = (
        _streaming_inference(plan)
        if plan["mode"] == "streaming"
        else _offline_inference(plan)
    )
    elapsed = time.perf_counter() - started
    output = {
        "kind": "baseline-smoke-result",
        "plan": plan,
        "elapsed_seconds": elapsed,
        "peak_rss_mb": _peak_rss_mb(),
        "result": result,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
