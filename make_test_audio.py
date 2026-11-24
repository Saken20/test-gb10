#!/usr/bin/env python3
"""Synthesizes the llm_prompts questions with the configured TTS: 16 kHz mono WAV files + asr_manifest.csv.

Synthetic voices give optimistic WER; use real recordings for ASR quality.
"""
import argparse
import asyncio
import csv
import os
from pathlib import Path

import httpx
import numpy as np

import bench
from adapters import pcm16_to_wav, wav_to_f32

RATE = 16000  # what most ASR models expect


def write_manifest(manifest: Path, rows):
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["audio", "text"])
        writer.writeheader()
        writer.writerows(rows)


async def make_audio(cfg, base: Path, out_dir: Path, manifest: Path) -> int:
    if not isinstance(cfg.get("tts"), dict):
        raise SystemExit("make_test_audio needs a tts section in the config")
    run = cfg.get("run") or {}
    try:
        prompts = [p["prompt"] for p in bench.load_inputs(cfg, base, print)["prompts"]]
    except bench.CONFIG_ERRORS as e:
        raise SystemExit(f"Config error: {e}") from None
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        async with httpx.AsyncClient(timeout=run.get("request_timeout_s", 300)) as client:
            # only TTS is needed here, so other stages are not built
            try:
                svc = await bench.Services.create({"tts": cfg["tts"], "run": run}, client, base)
            except bench.CONFIG_ERRORS as e:
                raise SystemExit(f"Config error: {e}") from None
            try:
                for i, text in enumerate(prompts, 1):
                    audio, _, _ = await svc.tts(text)
                    samples = np.frombuffer(wav_to_f32(audio, RATE), "<f4")
                    path = out_dir / f"{i:03d}.wav"
                    path.write_bytes(pcm16_to_wav((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes(), RATE))
                    rel = Path(os.path.relpath(path, manifest.parent)).as_posix()  # out_dir may sit outside it
                    rows.append({"audio": rel, "text": text})
                    print(f"{path.name}: {len(samples) / RATE:.1f} s - {text}")
            finally:
                await svc.aclose()
    finally:
        if rows:  # keep what was synthesized before a failure, but never blank an existing manifest
            write_manifest(manifest, rows)
    return len(rows)


def main():
    bench.safe_console()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="data/audio", help="where to write the WAV files (relative to the config)")
    ap.add_argument("--manifest", help="manifest path relative to the config (default: data.asr_manifest from it, "
                                       "else data/asr_manifest.csv)")
    args = ap.parse_args()

    cfg_path, cfg = bench.load_config(ap, args.config)
    base = cfg_path.parent
    manifest = base / (args.manifest or (cfg.get("data") or {}).get("asr_manifest") or "data/asr_manifest.csv")
    n = asyncio.run(make_audio(cfg, base, base / args.out, manifest))
    print(f"Done: {n} files, manifest {manifest}")


if __name__ == "__main__":
    main()
