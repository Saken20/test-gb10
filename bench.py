#!/usr/bin/env python3
"""Voice-pipeline benchmark (ASR -> NLU -> LLM -> TTS or any subset): speed, concurrency, quality, memory."""
import argparse
import asyncio
import contextlib
import csv
import io
import json
import random
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
import wave
from datetime import datetime
from pathlib import Path

import httpx
import jiwer
import psutil
import yaml

import adapters

# CJK terminators are not followed by a space
SENT_END = re.compile(r"[.!?…](\s|$)|[。！？]")
FILLER_USER = "Could you tell me a bit more about the options you mentioned earlier?"
FILLER_ASSISTANT = ("Sure. There are three options: a basic one, a standard one and an extended one; "
                    "most people pick the standard option.")
DEFAULT_PROMPT = "Can you suggest a plan for a relaxed weekend in a new city, and roughly what it would cost?"
VALID_TESTS = ("latency", "load", "quality")
CONFIG_SECTIONS = ("data", "run", "thresholds", "expect")
KNOWN_SECTIONS = ("llm", "asr", "nlu", "tts", *CONFIG_SECTIONS)
DATA_STAGE = {"asr_manifest": "asr", "llm_prompts": "llm", "nlu_set": "nlu", "tts_texts": "tts"}
MAX_ERROR_RATE = 0.01
SECRET_HINTS = ("key", "token", "secret", "password", "passwd", "auth", "credential", "bearer", "headers")
CONFIG_ERRORS = (ValueError, TypeError, FileNotFoundError, AttributeError, ImportError)
_URL_USERINFO = re.compile(r"(\b[a-zA-Z][\w+.-]*://)[^/\s@'\"]+@")
_URL_QUERY = re.compile(r"(\b[a-zA-Z][\w+.-]*://[^\s?#'\"]*)\?[^\s#'\"]*")


def pct(values, p):
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    k = (len(vals) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(vals) - 1)
    return vals[f] + (vals[c] - vals[f]) * (k - f)


def stats(values):
    vals = [v for v in values if v is not None]
    return {
        "n": len(vals),
        "p50": pct(vals, 50),
        "p95": pct(vals, 95),
        "p10": pct(vals, 10),
        "mean": statistics.fmean(vals) if vals else None,
    }


def fmt(x, nd=2):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def wav_duration(data: bytes):
    try:
        with wave.open(io.BytesIO(data)) as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return None


def normalize(text: str) -> str:
    # ё/е are used interchangeably in Russian transcripts; harmless for other languages
    text = (text or "").lower().replace("ё", "е")
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def error_rates(refs, hyps):
    """WER/CER over pairs with a non-empty reference (jiwer rejects empty ones)."""
    pairs = [(r, h) for r, h in zip(refs, hyps) if r]
    if not pairs:
        return None, None
    r, h = [p[0] for p in pairs], [p[1] for p in pairs]
    return jiwer.wer(r, h), jiwer.cer(r, h)


def verdict(measured, expected, tol=0.3, higher_better=True):
    if measured is None:
        return "no data"
    if not expected:
        return "no expectation"
    ratio = measured / expected
    if 1 - tol <= ratio <= 1 + tol:
        return f"as expected (±{int(tol * 100)}%)"
    return "better than expected" if (ratio > 1) == higher_better else "worse than expected"


def llm_text(cfg, key, default):
    return (cfg.get("llm") or {}).get(key) or default


def _is_secret_key(key):
    k = str(key).lower()
    return any(h in k for h in SECRET_HINTS)


def _sensitive(v):
    # numbers are never secrets; this keeps max_tokens, history_tokens etc. visible
    return v not in (None, "", {}, []) and not isinstance(v, (bool, int, float))


def scrub_urls(text: str) -> str:
    return _URL_QUERY.sub(r"\1", _URL_USERINFO.sub(r"\1", text))


def redact(obj):
    """Config copy for results.json: secret-looking keys masked, URL credentials and queries removed."""
    if isinstance(obj, dict):
        return {k: ("***" if _is_secret_key(k) and _sensitive(v) else redact(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact(v) for v in obj]
    if isinstance(obj, str):
        return scrub_urls(obj)
    return obj


def _strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v)


def secret_values(obj):
    found = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if _is_secret_key(k) and _sensitive(v):
                found.update(s for s in _strings(v) if len(s) >= 4)
            else:
                found |= secret_values(v)
    elif isinstance(obj, list):
        for v in obj:
            found |= secret_values(v)
    return found


def make_scrubber(cfg):
    secrets = sorted(secret_values(cfg), key=len, reverse=True)

    def scrub(text):
        text = str(text)
        for s in secrets:
            text = text.replace(s, "***")
        return scrub_urls(text)

    return scrub


class Monitor(threading.Thread):
    """Logs GPU load and system RAM each second; on unified memory (GB10) nvidia-smi shows no memory."""

    def __init__(self, path: Path, interval=1.0):
        super().__init__(daemon=True)
        self.path, self.interval = path, interval
        self.phase = "idle"
        self.stop_ev = threading.Event()
        self.has_smi = shutil.which("nvidia-smi") is not None
        self.peaks = {}

    def gpu(self):
        if not self.has_smi:
            return [None] * 4
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5).stdout.splitlines()[0]
            vals = []
            for v in out.split(","):
                try:
                    vals.append(float(v.strip()))
                except ValueError:
                    vals.append(None)
            return vals
        except Exception:
            return [None] * 4

    def run(self):
        with open(self.path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time", "phase", "ram_used_gb", "gpu_util_pct", "gpu_mem_used_mb", "power_w", "temp_c"])
            t0 = time.time()
            while not self.stop_ev.is_set():
                phase = self.phase  # before the slow reads, so a sample is not credited to the next phase
                ram = psutil.virtual_memory().used / 1e9
                util, mem, power, temp = self.gpu()
                w.writerow([round(time.time() - t0, 1), phase, round(ram, 2), util, mem, power, temp])
                f.flush()
                p = self.peaks.setdefault(phase, {})
                for k, v in (("ram_used_gb", ram), ("gpu_util_pct", util), ("gpu_mem_used_mb", mem),
                             ("power_w", power), ("temp_c", temp)):
                    if v is not None:
                        p[k] = max(p.get(k, v), v)
                self.stop_ev.wait(self.interval)


class Services:
    """Prefer `await Services.create(...)`: it closes adapters built before a failing one."""

    def __init__(self, cfg, client: httpx.AsyncClient, base_dir=None, build=True):
        self.cfg, self.c, self.base_dir = cfg, client, base_dir
        self.stages = {}
        self.llm_usage_missing = False
        if build:
            for s in self.configured(cfg):
                self.stages[s] = adapters.build(s, cfg[s], cfg, client, base_dir)

    @staticmethod
    def configured(cfg):
        return [s for s in adapters.STAGES if isinstance(cfg.get(s), dict)]

    @classmethod
    async def create(cls, cfg, client, base_dir=None):
        svc = cls(cfg, client, base_dir, build=False)
        try:
            for s in cls.configured(cfg):
                svc.stages[s] = adapters.build(s, cfg[s], cfg, client, base_dir)
        except BaseException:
            with contextlib.suppress(Exception):
                await svc.aclose()
            raise
        return svc

    def has(self, *stages):
        return all(s in self.stages for s in stages)

    def missing(self, *stages):
        return [s for s in stages if s not in self.stages]

    def _stage(self, name):
        if name not in self.stages:
            raise RuntimeError(f"stage '{name}' is not configured")
        return self.stages[name]

    async def aclose(self):
        errors = []
        for a in self.stages.values():
            close = getattr(a, "aclose", None)
            if close is None:
                continue
            try:
                await close()
            except Exception as e:
                errors.append(e)
        if errors:
            raise errors[0]

    async def asr(self, audio: bytes) -> str:
        return await self._stage("asr").transcribe(audio)

    async def nlu(self, text: str):
        if "nlu" not in self.stages:
            return None
        return await self.stages["nlu"].classify(text)

    async def tts(self, text: str):
        """Returns (WAV, time to first audio byte, total time)."""
        a = self._stage("tts")
        t0 = time.perf_counter()
        first, chunks = None, []
        async for ch in a.stream(text):
            if first is None:
                first = time.perf_counter() - t0
            chunks.append(ch)
        total = time.perf_counter() - t0
        raw = b"".join(chunks)
        if not raw:
            raise RuntimeError("TTS returned an empty response")
        wav = a.to_wav(raw)
        # an error page or JSON must not count as a successful synthesis
        if not isinstance(wav, (bytes, bytearray)) or wav[:4] != b"RIFF" or wav[8:12] != b"WAVE":
            raise RuntimeError("TTS response is not WAV audio (check tts.response)")
        if not getattr(a, "streams_audio", True):
            first = total
        return bytes(wav), (first if first is not None else total), total

    async def llm(self, messages, on_text=None):
        """Streams the answer; on_text(accumulated_text, t) is called on every piece."""
        usage = {}
        t0 = time.perf_counter()
        t_first, parts, n_chunks = None, [], 0
        async for piece, u in self._stage("llm").stream(messages):
            if u:
                usage.update(u)
                if not piece:  # usage-only chunk, not a decode step
                    continue
            n_chunks += 1
            if t_first is None:
                t_first = time.perf_counter() - t0
            if piece:
                parts.append(piece)
                if on_text:
                    on_text("".join(parts), time.perf_counter() - t0)
        total = time.perf_counter() - t0
        # without usage from the server, tokens are counted as stream chunks
        n_tok = usage.get("completion_tokens")
        if not n_tok:
            n_tok = n_chunks
            self.llm_usage_missing = True
        decode = total - (t_first if t_first is not None else total)
        return {
            "text": "".join(parts),
            "ttft_s": t_first,
            "total_s": total,
            "completion_tokens": n_tok,
            "prompt_tokens": usage.get("prompt_tokens"),
            "decode_tps": (n_tok - 1) / decode if n_tok and n_tok > 1 and decode > 0 else None,
        }


def build_messages(llm_cfg, user_text, history_tokens, chars_per_token=4):
    msgs = []
    if llm_cfg.get("system_prompt"):
        msgs.append({"role": "system", "content": llm_cfg["system_prompt"]})
    if history_tokens > 0:
        filler_user = llm_cfg.get("history_filler_user") or FILLER_USER
        filler_assistant = llm_cfg.get("history_filler_assistant") or FILLER_ASSISTANT
        # random session number in every filler turn defeats the prefix cache, which would flatter TTFT/memory;
        # every turn, not just the first, because some services keep only the last few
        user = f"Session {random.randint(1, 10**9)}. {filler_user}"
        acc = 0
        while acc < history_tokens * chars_per_token or not acc:
            msgs += [{"role": "user", "content": user}, {"role": "assistant", "content": filler_assistant}]
            acc += len(user) + len(filler_assistant)
    msgs.append({"role": "user", "content": user_text})
    return msgs


def history_args(cfg):
    run = cfg.get("run") or {}
    return run.get("history_tokens", 0), run.get("chars_per_token", 4)


async def llm_turn(svc, cfg, prompt):
    msgs = build_messages(cfg.get("llm") or {}, prompt, *history_args(cfg))
    return await svc.llm(msgs)


async def voice_turn(svc, cfg, audio: bytes):
    """pause_s: end of the user's utterance -> first sound of the answer (TTS starts on the first sentence)."""
    t0 = time.perf_counter()
    text = await svc.asr(audio)
    t_asr = time.perf_counter() - t0
    t_nlu = None
    if svc.has("nlu"):
        t1 = time.perf_counter()
        await svc.nlu(text)
        t_nlu = time.perf_counter() - t1

    first = {}

    def on_text(acc, _t):
        if "task" in first:
            return
        m = SENT_END.search(acc)
        if m or len(acc) >= 80:
            first["text"] = acc[:m.end()] if m else acc
            first["start"] = time.perf_counter()
            first["task"] = asyncio.create_task(svc.tts(first["text"]))

    prompt = text or llm_text(cfg, "default_prompt", DEFAULT_PROMPT)
    msgs = build_messages(cfg.get("llm") or {}, prompt, *history_args(cfg))
    try:
        llm = await svc.llm(msgs, on_text)
    except BaseException:
        if "task" in first:  # do not leave the first-sentence TTS running
            first["task"].cancel()
            await asyncio.gather(first["task"], return_exceptions=True)
        raise

    if "task" not in first:  # answer shorter than one sentence
        first["text"] = llm["text"] or "OK."
        first["start"] = time.perf_counter()
        first["task"] = asyncio.create_task(svc.tts(first["text"]))
    _, tts_first, _ = await first["task"]
    pause = (first["start"] - t0) + tts_first

    rest = llm["text"][len(first["text"]):].strip()
    if rest:  # synthesize the rest too, so TTS loads the GPU as in real use
        await svc.tts(rest)

    # turn_s: from the end of the user's utterance until the whole answer is synthesized
    return {**llm, "asr_s": t_asr, "nlu_s": t_nlu, "tts_first_s": tts_first, "pause_s": pause,
            "turn_s": time.perf_counter() - t0, "asr_text": text}


VOICE_STAGES = ("asr", "llm", "tts")


def _read_csv(path, name, columns):
    # utf-8-sig: spreadsheet apps save CSV with a BOM that would hide the first column name
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        if not set(columns) <= set(fields):
            raise ValueError(f"{name}: header must be {','.join(columns)} (got {','.join(fields) or 'an empty file'})")
        return list(reader)


def _read_prompts(path, name):
    prompts = []
    with open(path, encoding="utf-8-sig") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                obj = None
            if not isinstance(obj, dict) or not isinstance(obj.get("prompt"), str):
                raise ValueError(f'{name}:{n}: expected JSON with a "prompt" key')
            keys = obj.get("must_contain")
            if keys is not None and not (isinstance(keys, list) and all(isinstance(k, str) for k in keys)):
                raise ValueError(f"{name}:{n}: must_contain must be a list of strings")
            prompts.append(obj)
    return prompts


def load_inputs(cfg, base: Path, log=None):
    """Raises ValueError with a one-line reason when a data file has the wrong format."""
    say = log or (lambda _msg: None)
    d = cfg.get("data") or {}
    used = set(Services.configured(cfg))
    inp = {"asr": [], "prompts": [], "nlu": [], "tts": [], "missing_audio": 0}

    def existing(key):
        if not d.get(key):
            return None
        path = base / d[key]
        if path.exists():
            return path
        if key != "llm_prompts" and DATA_STAGE[key] in used:
            say(f"  Warning: data.{key} file not found: {path}")
        return None

    mpath = existing("asr_manifest")
    if mpath:
        name = d["asr_manifest"]
        for line, row in enumerate(_read_csv(mpath, name, ("audio", "text")), 2):
            if not row.get("audio"):
                raise ValueError(f"{name}:{line}: empty audio path")
            apath = (mpath.parent / row["audio"]).resolve()
            if apath.exists():
                inp["asr"].append({"audio": apath.read_bytes(), "text": row.get("text") or "", "file": row["audio"]})
            else:
                inp["missing_audio"] += 1
        if inp["missing_audio"] and "asr" in used:
            say(f"  Warning: {inp['missing_audio']} audio file(s) listed in {mpath.name} are missing")
    path = existing("llm_prompts")
    if path:
        inp["prompts"] = _read_prompts(path, d["llm_prompts"])
    path = existing("nlu_set")
    if path:
        inp["nlu"] = _read_csv(path, d["nlu_set"], ("text", "intent"))
    path = existing("tts_texts")
    if path:
        inp["tts"] = [l.strip() for l in path.read_text(encoding="utf-8-sig").splitlines() if l.strip()]
    if not inp["prompts"]:
        if "llm" in used:
            where = f" ({base / d['llm_prompts']})" if d.get("llm_prompts") else ""
            say(f"  No data.llm_prompts file{where}, using the built-in prompt")
        inp["prompts"] = [{"prompt": llm_text(cfg, "default_prompt", DEFAULT_PROMPT)}]
    return inp


def config_warnings(cfg):
    out = [f"Warning: unknown section {k!r} ignored; expected {', '.join(KNOWN_SECTIONS)}"
           for k in cfg if k not in KNOWN_SECTIONS]
    llm = cfg.get("llm")
    if isinstance(llm, dict) and not llm.get("model"):
        with contextlib.suppress(ValueError):
            if adapters.stage_type("llm", llm) == "openai":
                out.append("Warning: llm.model is empty; most servers reject it (GET /v1/models lists the names)")
    return out


def voice_skip_reason(svc, inp):
    missing = svc.missing(*VOICE_STAGES)
    if missing:
        return f"stage(s) not configured: {', '.join(missing)}"
    if not inp["asr"]:
        n = inp.get("missing_audio", 0)
        extra = f" ({n} audio file(s) missing - run make_test_audio.py)" if n else ""
        return "no audio in data.asr_manifest" + extra
    return None


async def test_latency(svc, cfg, inp, log):
    run = cfg.get("run") or {}
    res = {}
    prompts = [p["prompt"] for p in inp["prompts"]]
    if svc.has("llm"):
        for i in range(run.get("warmup", 3)):
            await llm_turn(svc, cfg, prompts[i % len(prompts)])
        llm = [await llm_turn(svc, cfg, prompts[i % len(prompts)]) for i in range(run.get("repeats", 20))]
        res["llm"] = {k: stats([r[k] for r in llm]) for k in ("ttft_s", "decode_tps", "completion_tokens", "prompt_tokens")}
        log(f"  LLM, 1 user: {fmt(res['llm']['decode_tps']['p50'], 1)} tok/s, "
            f"first token {fmt(res['llm']['ttft_s']['p50'])} s")
    else:
        log("  LLM latency skipped: stage 'llm' not configured")

    reason = voice_skip_reason(svc, inp)
    if reason is None:
        await voice_turn(svc, cfg, inp["asr"][0]["audio"])  # not measured
        e2e = [await voice_turn(svc, cfg, inp["asr"][i % len(inp["asr"])]["audio"])
               for i in range(run.get("repeats", 20))]
        res["e2e"] = {k: stats([r[k] for r in e2e])
                      for k in ("asr_s", "nlu_s", "ttft_s", "tts_first_s", "pause_s", "turn_s", "decode_tps")}
        log(f"  Voice turn: pause before answer {fmt(res['e2e']['pause_s']['p50'])} s "
            f"(p95 {fmt(res['e2e']['pause_s']['p95'])}), full turn {fmt(res['e2e']['turn_s']['p50'])} s")
    else:
        log(f"  Voice-turn latency skipped: {reason}")
    return res


async def run_level(svc, cfg, inp, n, mode, duration):
    stop_at = time.perf_counter() + duration
    results, errors = [], []

    async def worker(i):
        await asyncio.sleep(i * 0.3)  # stagger starts like real users
        k = i
        while time.perf_counter() < stop_at:
            try:
                if mode == "llm":
                    p = inp["prompts"][k % len(inp["prompts"])]["prompt"]
                    results.append(await llm_turn(svc, cfg, p))
                else:
                    results.append(await voice_turn(svc, cfg, inp["asr"][k % len(inp["asr"])]["audio"]))
            except Exception as e:
                errors.append(repr(e)[:200])
                await asyncio.sleep(0.5)  # a failing service answers instantly; don't spin on it
            k += n

    t0 = time.perf_counter()
    await asyncio.gather(*(worker(i) for i in range(n)))
    wall = time.perf_counter() - t0
    out = {
        "n": n, "mode": mode, "turns": len(results), "errors": len(errors), "error_samples": errors[:3],
        "decode_tps": stats([r["decode_tps"] for r in results]),
        "ttft_s": stats([r["ttft_s"] for r in results]),
        "total_tokens_per_s": sum(r["completion_tokens"] or 0 for r in results) / wall if wall else None,
    }
    if mode == "pipeline":
        for k in ("pause_s", "turn_s", "asr_s", "nlu_s", "tts_first_s"):
            out[k] = stats([r[k] for r in results])
    return out


async def test_load(svc, cfg, inp, log, mon):
    run = cfg.get("run") or {}
    res = {"llm": [], "pipeline": [], "stopped": {}}
    modes = []
    if svc.has("llm"):
        modes.append("llm")
    else:
        log("  LLM-only load skipped: stage 'llm' not configured")
    reason = voice_skip_reason(svc, inp)
    if reason is None:
        modes.append("pipeline")
    for mode in modes:
        for n in run.get("concurrency_levels", [1, 2, 5, 10]):
            if mon:
                mon.phase = f"load_{mode}_{n}"
            r = await run_level(svc, cfg, inp, n, mode, run.get("concurrency_duration_s", 60))
            res[mode].append(r)
            msg = (f"  {mode:8s} N={n:>2}: {fmt(r['decode_tps']['p50'], 1)} tok/s per user (p10 "
                   f"{fmt(r['decode_tps']['p10'], 1)}), first token p95 {fmt(r['ttft_s']['p95'])} s")
            if mode == "pipeline":
                msg += f", pause p95 {fmt(r['pause_s']['p95'])} s, full turn p50 {fmt(r['turn_s']['p50'])} s"
            log(msg + f", errors {r['errors']}")
            stop = None
            if r["errors"] and (r["turns"] == 0 or r["errors"] / (r["turns"] + r["errors"]) > 0.5):
                sample = r["error_samples"][0] if r["error_samples"] else ""
                stop = f"{r['errors']} errors vs {r['turns']} turns at N={n} (e.g. {sample})"
            elif r["decode_tps"]["p50"] is not None and r["decode_tps"]["p50"] < run.get("stop_below_tps", 1):
                stop = f"speed fell below stop_below_tps at N={n}"
            if stop:
                res["stopped"][mode] = stop
                log(f"  {mode} load stopped: {stop}; higher levels skipped")
                break
    if reason is not None:
        log(f"  Full-pipeline load skipped: {reason}")
    return res


async def test_quality(svc, cfg, inp, log, out_dir: Path):
    res = {}
    if inp["asr"] and svc.has("asr"):
        refs, hyps, rtf = [], [], []
        for item in inp["asr"]:
            t0 = time.perf_counter()
            hyp = await svc.asr(item["audio"])
            dt = time.perf_counter() - t0
            dur = wav_duration(item["audio"])
            if dur:
                rtf.append(dt / dur)
            refs.append(normalize(item["text"]))
            hyps.append(normalize(hyp))
        res["asr"] = {"files": len(refs), "with_reference": sum(1 for r in refs if r), "rtf": stats(rtf)}
        wer, cer = error_rates(refs, hyps)
        if wer is not None:
            res["asr"]["wer"], res["asr"]["cer"] = wer, cer
        log(f"  ASR: WER {fmt(res['asr'].get('wer'), 3)}, CER {fmt(res['asr'].get('cer'), 3)} (0 = perfect), "
            f"RTF {fmt(res['asr']['rtf']['p50'], 3)} ({res['asr']['with_reference']} files with reference text)")
    elif inp["asr"]:
        log("  ASR quality skipped: stage 'asr' not configured")

    if inp["tts"] and svc.has("tts"):
        rtf, refs, hyps, firsts, asr_failures = [], [], [], [], []
        for text in inp["tts"]:
            audio, first, total = await svc.tts(text)
            firsts.append(first)
            dur = wav_duration(audio)
            if dur:
                rtf.append(total / dur)
            if svc.has("asr"):
                try:
                    hyps.append(normalize(await svc.asr(audio)))
                    refs.append(normalize(text))
                except Exception as e:
                    asr_failures.append(repr(e)[:200])
        res["tts"] = {"texts": len(inp["tts"]), "rtf": stats(rtf), "first_audio_s": stats(firsts)}
        cer = error_rates(refs, hyps)[1]
        if cer is not None:
            res["tts"]["roundtrip_cer"] = cer
        if asr_failures:
            res["tts"]["asr_failures"] = len(asr_failures)
            res["tts"]["asr_failure_samples"] = asr_failures[:3]
            log(f"  TTS intelligibility: ASR failed on {len(asr_failures)} of {len(inp['tts'])} clips "
                f"(e.g. {asr_failures[0]})")
        log(f"  TTS: RTF {fmt(res['tts']['rtf']['p50'], 3)}, first audio {fmt(res['tts']['first_audio_s']['p50'])} s, "
            f"intelligibility (CER after ASR, 0 = perfect) {fmt(res['tts'].get('roundtrip_cer'), 3)}")
    elif inp["tts"]:
        log("  TTS quality skipped: stage 'tts' not configured")

    if inp["nlu"] and svc.has("nlu"):
        ok, errs = 0, []
        for row in inp["nlu"]:
            pred = await svc.nlu(row["text"])
            if str(pred) == row["intent"]:
                ok += 1
            else:
                errs.append({"text": row["text"], "expected": row["intent"], "got": pred})
        res["nlu"] = {"samples": len(inp["nlu"]), "accuracy": ok / len(inp["nlu"]), "errors": errs[:20]}
        log(f"  NLU: accuracy {fmt(res['nlu']['accuracy'] * 100, 1)}%")
    elif inp["nlu"]:
        log("  NLU quality skipped: stage 'nlu' not configured")

    with_keys = [p for p in inp["prompts"] if p.get("must_contain")]
    if inp["prompts"] and svc.has("llm"):
        ok = 0
        with open(out_dir / "llm_answers.jsonl", "w", encoding="utf-8") as f:
            for p in inp["prompts"]:
                r = await llm_turn(svc, cfg, p["prompt"])
                hit = all(normalize(k) in normalize(r["text"]) for k in p.get("must_contain", []))
                if p.get("must_contain") and hit:
                    ok += 1
                f.write(json.dumps({"prompt": p["prompt"], "answer": r["text"], "must_contain": p.get("must_contain"),
                                    "ok": hit if p.get("must_contain") else None}, ensure_ascii=False) + "\n")
        res["llm"] = {"prompts": len(inp["prompts"]), "with_keys": len(with_keys),
                      "keys_ok_share": ok / len(with_keys) if with_keys else None}
        if res["llm"]["keys_ok_share"] is None:
            log("  LLM: no must_contain keys in llm_prompts, check skipped (answers saved to llm_answers.jsonl)")
        else:
            log(f"  LLM: key facts in {keys_share_text(res['llm']['keys_ok_share'])} of answers "
                f"(answers saved to llm_answers.jsonl)")
    return res


def keys_share_text(share):
    return "check skipped (no must_contain keys in llm_prompts)" if share is None else f"{fmt(share * 100, 1)}%"


def level_ok(r, tps_min, pause_max=None):
    attempts = r["turns"] + r["errors"]
    tps = r["decode_tps"]["p50"]
    ok = attempts > 0 and r["errors"] / attempts <= MAX_ERROR_RATE and tps is not None and tps >= tps_min
    if ok and pause_max is not None and "pause_s" in r:
        ok = r["pause_s"]["p95"] is not None and r["pause_s"]["p95"] <= pause_max
    return ok


def max_level(levels, tps_min, pause_max=None):
    """Highest concurrency before the first failing level; "N+" means even the top tested level passed."""
    best = 0
    levels = sorted(levels, key=lambda r: r["n"])
    for r in levels:
        if not level_ok(r, tps_min, pause_max):
            return str(best)
        best = r["n"]
    return f"{best}+" if best else "0"


def stage_label(cfg, stage):
    s = cfg.get(stage)
    if not isinstance(s, dict):
        return None
    try:
        t = adapters.stage_type(stage, s)
    except ValueError:
        t = "invalid"
    name = s.get("model") or (s.get("plugin") if t == "plugin" else None)
    return f"{stage} ({t}{', ' + str(name) if name else ''})"


def answer_lines(results, th):
    lat = results.get("latency") or {}
    load = results.get("load") or {}
    single = ((lat.get("llm") or {}).get("decode_tps") or {}).get("p50")
    pause = ((lat.get("e2e") or {}).get("pause_s") or {}).get("p50")
    parts = []
    if single is not None:
        parts.append(f"{fmt(single, 1)} tok/s")
    if pause is not None:
        parts.append(f"{fmt(pause)} s pause")
    lines = [f"**Answer:** 1 user: {', '.join(parts)}." if parts else "**Answer:** no single-user measurements."]
    limit = th.get("limit_tps", 10)
    if load.get("pipeline"):
        pause_max = float(th.get("pause_max_s", 2.0))
        levels, noun = load["pipeline"], "concurrent dialogs"
        cond = f"≥ {limit} tok/s per user with pause p95 ≤ {fmt(pause_max, 1)} s"
        n = max_level(levels, limit, pause_max)
    elif load.get("llm"):
        levels, noun, cond = load["llm"], "concurrent answers (LLM only)", f"≥ {limit} tok/s per user"
        n = max_level(levels, limit)
    else:
        return lines
    top = max(r["n"] for r in levels)
    if n.endswith("+"):
        lines.append(f"All tested levels up to {top} {noun} stay {cond}; the real limit is higher.")
    elif n == "0":
        lines.append(f"No tested level stays {cond} with ≤ 1% errors ({top} was the most tested).")
    else:
        lines.append(f"Up to {n} {noun} stay {cond} ({top} was the most tested).")
    return lines


def write_summary(out_dir, machine, cfg, results, peaks, scrub=str):
    """peaks=None means the monitor was off (--no-monitor)."""
    exp = cfg.get("expect") or {}
    th = cfg.get("thresholds") or {}
    run = cfg.get("run") or {}
    llm_cfg = cfg.get("llm") or {}
    stages = [lbl for lbl in (stage_label(cfg, s) for s in adapters.STAGES) if lbl]
    lines = [f"# Benchmark results: {machine}", "", f"Date: {datetime.now():%Y-%m-%d %H:%M}",
             f"Stages: {', '.join(stages) or 'none'}"]
    if llm_cfg:
        name = llm_cfg.get("model") or llm_cfg.get("plugin") or "-"
        lines.append(f"LLM: {name}, dialog history ≈ {run.get('history_tokens', 0)} tokens")
    for note in results.get("notes", []):
        lines.append(f"Note: {note}")
    lines += ["", *answer_lines(results, th), "",
              "p50 = median, p95 = 95% of turns were faster, N+ = even the highest tested level passed", ""]

    rows = []
    lat = results.get("latency") or {}
    single = (lat.get("llm") or {}).get("decode_tps", {}).get("p50")
    rows.append(("LLM decode speed, 1 user, tok/s", fmt(exp.get("single_tps")), fmt(single, 1),
                 verdict(single, exp.get("single_tps"))))
    if lat.get("e2e"):
        pause, turn = lat["e2e"]["pause_s"], lat["e2e"]["turn_s"]
        rows.append(("Pause before answer, 1 user, s (p50 / p95)", exp.get("pause_s", "-"),
                     f"{fmt(pause.get('p50'))} / {fmt(pause.get('p95'))}", "compare manually"))
        rows.append(("Full turn until the whole answer is voiced, 1 user, s (p50 / p95)", "-",
                     f"{fmt(turn.get('p50'))} / {fmt(turn.get('p95'))}", "compare manually"))

    load = results.get("load") or {}
    if load.get("llm"):
        comf = max_level(load["llm"], th.get("comfortable_tps", 20))
        lim = max_level(load["llm"], th.get("limit_tps", 10))
        rows.append((f"Concurrent answers, comfortable (≥ {th.get('comfortable_tps', 20)} tok/s), LLM only",
                     exp.get("comfortable_n", "-"), comf, "compare manually"))
        rows.append((f"Concurrent answers, limit (≥ {th.get('limit_tps', 10)} tok/s), LLM only",
                     exp.get("limit_n", "-"), lim, "compare manually"))
    if load.get("pipeline"):
        comf_p = max_level(load["pipeline"], th.get("comfortable_tps", 20), th.get("pause_max_s", 2.0))
        rows.append((f"Concurrent dialogs, full pipeline (≥ {th.get('comfortable_tps', 20)} tok/s, "
                     f"pause p95 ≤ {fmt(float(th.get('pause_max_s', 2.0)), 1)} s)", exp.get("comfortable_n", "-"), comf_p,
                     "compare manually"))
        for n in (1, 2):
            a = next((r for r in load.get("llm", []) if r["n"] == n), None)
            b = next((r for r in load["pipeline"] if r["n"] == n), None)
            if a and b and a["decode_tps"]["p50"] and b["decode_tps"]["p50"]:
                drop = 1 - b["decode_tps"]["p50"] / a["decode_tps"]["p50"]
                rows.append((f"LLM slowdown caused by ASR/TTS at N={n}, %", fmt(exp.get("pipeline_drop_pct")),
                             fmt(drop * 100, 1), verdict(drop * 100, exp.get("pipeline_drop_pct"), tol=0.5,
                                                         higher_better=False)))
    if peaks is None:
        peak_ram = "not measured (--no-monitor)"
    else:
        peak_ram = fmt(max((p.get("ram_used_gb", 0) for p in peaks.values()), default=None), 1)
    rows.append(("Peak RAM, GB (includes GPU memory on unified-memory machines)", fmt(exp.get("memory_gb")),
                 peak_ram, "compare manually"))

    lines += ["## Expected vs measured", "", "| Metric | Expected | Measured | Verdict |", "| --- | --- | --- | --- |"]
    lines += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows]

    if load.get("llm"):
        lines += ["", "## Load: LLM only", "",
                  "| Concurrent | Tok/s per user, p50 | Tok/s, worst 10% | First token p95, s | Total tok/s | Turns | Errors |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for r in load["llm"]:
            lines.append(f"| {r['n']} | {fmt(r['decode_tps']['p50'], 1)} | {fmt(r['decode_tps']['p10'], 1)} | "
                         f"{fmt(r['ttft_s']['p95'])} | {fmt(r['total_tokens_per_s'], 0)} | {r['turns']} | {r['errors']} |")
        if (load.get("stopped") or {}).get("llm"):
            lines += ["", f"Stopped early: {load['stopped']['llm']}"]

    if load.get("pipeline"):
        lines += ["", "## Load: full pipeline", "",
                  "Pause: from the end of the user's utterance to the first sound of the answer; "
                  "full turn: until the whole answer is synthesized.", "",
                  "| Concurrent | Tok/s per user, p50 | Tok/s, worst 10% | Pause p50 / p95, s | Full turn p50 / p95, s | "
                  "ASR p50, s | First token p50, s | TTS first sentence p50, s | Total tok/s | Turns | Errors |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for r in load["pipeline"]:
            lines.append(f"| {r['n']} | {fmt(r['decode_tps']['p50'], 1)} | {fmt(r['decode_tps']['p10'], 1)} | "
                         f"{fmt(r['pause_s']['p50'])} / {fmt(r['pause_s']['p95'])} | "
                         f"{fmt(r['turn_s']['p50'])} / {fmt(r['turn_s']['p95'])} | {fmt(r['asr_s']['p50'])} | "
                         f"{fmt(r['ttft_s']['p50'])} | {fmt(r['tts_first_s']['p50'])} | "
                         f"{fmt(r['total_tokens_per_s'], 0)} | {r['turns']} | {r['errors']} |")
        if (load.get("stopped") or {}).get("pipeline"):
            lines += ["", f"Stopped early: {load['stopped']['pipeline']}"]

    q = results.get("quality") or {}
    if q:
        lines += ["", "## Quality", "", "| Metric | Value |", "| --- | --- |"]
        if "asr" in q:
            lines += [f"| ASR: WER (0 = perfect) | {fmt(q['asr'].get('wer'), 3)} |",
                      f"| ASR: CER (0 = perfect) | {fmt(q['asr'].get('cer'), 3)} |",
                      f"| ASR: RTF (processing time / audio duration) | {fmt(q['asr']['rtf']['p50'], 3)} |"]
        if "tts" in q:
            lines += [f"| TTS: RTF | {fmt(q['tts']['rtf']['p50'], 3)} |",
                      f"| TTS: first audio, s | {fmt(q['tts']['first_audio_s']['p50'])} |",
                      f"| TTS: intelligibility (CER after ASR, 0 = perfect) | {fmt(q['tts'].get('roundtrip_cer'), 3)} |"]
            if q["tts"].get("asr_failures"):
                lines.append(f"| TTS: clips ASR failed on | {q['tts']['asr_failures']} |")
        if "nlu" in q:
            lines.append(f"| NLU: intent accuracy | {fmt(q['nlu']['accuracy'] * 100, 1)}% |")
        if "llm" in q:
            lines.append(f"| LLM: answers with key facts | {keys_share_text(q['llm']['keys_ok_share'])} |")

    if peaks:
        lines += ["", "## Resources by phase (peaks)", "",
                  "| Phase | RAM, GB | GPU util, % | GPU memory, MB | Power, W | Temperature, °C |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for ph, p in peaks.items():
            lines.append(f"| {ph} | {fmt(p.get('ram_used_gb'), 1)} | {fmt(p.get('gpu_util_pct'), 0)} | "
                         f"{fmt(p.get('gpu_mem_used_mb'), 0)} | {fmt(p.get('power_w'), 0)} | {fmt(p.get('temp_c'), 0)} |")

    (out_dir / "summary.md").write_text(scrub("\n".join(lines) + "\n"), encoding="utf-8")


def _request_url(e, fallback):
    try:
        return str(e.request.url)
    except RuntimeError:
        return fallback


def describe_error(name, e, url=None):
    if isinstance(e, (httpx.ConnectError, httpx.ConnectTimeout)):
        return f"{name}: cannot connect to {_request_url(e, url)} - is the server running?"
    if isinstance(e, httpx.HTTPStatusError):
        try:
            body = " ".join(e.response.text.split())[:200]
        except httpx.ResponseNotRead:
            body = ""
        return f"{name}: HTTP {e.response.status_code} from {_request_url(e, url)}: {body}"
    return f"{name}: ERROR {e!r}"[:300]


async def check_services(svc, cfg, inp, log):
    ok = True
    nlu_probe = inp["nlu"][0]["text"] if inp["nlu"] else "Hello"
    tts_probe = inp["tts"][0] if inp["tts"] else "Hello, how can I help you?"
    for name, stage, coro in (
        ("LLM", "llm", lambda: svc.llm([{"role": "user", "content": "Hello"}])),
        ("ASR", "asr", (lambda: svc.asr(inp["asr"][0]["audio"])) if inp["asr"] else None),
        ("NLU", "nlu", lambda: svc.nlu(nlu_probe)),
        ("TTS", "tts", lambda: svc.tts(tts_probe)),
    ):
        if not svc.has(stage):
            log(f"  {name}: not configured, skipped")
            continue
        if coro is None:
            log(f"  {name}: no test data, check skipped")
            continue
        try:
            await coro()
            log(f"  {name}: OK")
        except Exception as e:
            ok = False
            log("  " + describe_error(name, e, (cfg.get(stage) or {}).get("url")))
    return ok


def safe_console():
    """Never crash on characters the console encoding cannot print."""
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.reconfigure(errors="replace")


def load_config(ap, path):
    cfg_path = Path(path).resolve()
    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        ap.error(f"config not found: {path}. Start with: cp config.example.yaml config.yaml")
    except yaml.YAMLError as e:
        ap.error(f"{path}: invalid YAML: {e}")
    if not isinstance(cfg, dict):
        ap.error(f"{path} must contain a YAML mapping")
    return cfg_path, cfg


class RunLog:
    """Prints at once; lines go to log.txt once the results folder exists (none is made on config errors)."""

    def __init__(self, scrub):
        self.scrub, self.f, self.pending = scrub, None, []

    def __call__(self, msg):
        msg = self.scrub(msg)
        print(msg, flush=True)
        if self.f:
            self.f.write(msg + "\n")
            self.f.flush()
        else:
            self.pending.append(msg)

    def open(self, path):
        self.f = open(path, "w", encoding="utf-8")
        self.f.writelines(m + "\n" for m in self.pending)
        self.f.flush()

    def close(self):
        if self.f:
            self.f.close()


async def main():
    safe_console()
    ap = argparse.ArgumentParser(description="Voice-pipeline benchmark (ASR -> NLU -> LLM -> TTS, any subset)")
    ap.add_argument("--config", default="config.yaml", help="path to the YAML config")
    ap.add_argument("--machine", default="local", help="machine name used in the results folder name")
    ap.add_argument("--tests", default="latency,load,quality", help="comma-separated: latency, load, quality")
    ap.add_argument("--no-monitor", action="store_true",
                    help="skip GPU/RAM monitoring (use when the models run on another machine)")
    args = ap.parse_args()
    args.tests = [t.strip() for t in args.tests.split(",") if t.strip()]
    unknown = [t for t in args.tests if t not in VALID_TESTS]
    if unknown or not args.tests:
        ap.error(f"--tests: unknown {', '.join(unknown) or '(empty)'}; choose from {', '.join(VALID_TESTS)}")

    cfg_path, cfg = load_config(ap, args.config)
    for k in CONFIG_SECTIONS:  # `run:` with no value parses as None
        cfg[k] = cfg.get(k) or {}
    log = RunLog(make_scrubber(cfg))
    try:
        await run_bench(args, cfg, cfg_path.parent, log)
    finally:
        log.close()


def config_error(log, msg):
    log(f"Config error: {msg}")
    raise SystemExit(2)


async def run_bench(args, cfg, base, log):
    scrub = log.scrub
    for w in config_warnings(cfg):
        log(w)
    if not Services.configured(cfg):
        config_error(log, "no stages configured: add at least one of llm, asr, nlu, tts to the config")
    try:
        inp = load_inputs(cfg, base, log)
    except CONFIG_ERRORS as e:
        config_error(log, e)
    log(f"Data: audio {len(inp['asr'])}, prompts {len(inp['prompts'])}, NLU {len(inp['nlu'])}, TTS {len(inp['tts'])}")

    timeout = httpx.Timeout(cfg["run"].get("request_timeout_s", 300))
    limits = httpx.Limits(max_connections=1000, max_keepalive_connections=200)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        try:
            svc = await Services.create(cfg, client, base)
        except CONFIG_ERRORS as e:
            config_error(log, e)
        out_dir = base / "results" / f"{args.machine}_{datetime.now():%Y%m%d_%H%M%S}"
        out_dir.mkdir(parents=True, exist_ok=True)
        log.open(out_dir / "log.txt")
        results = {"machine": args.machine, "config": redact(cfg), "notes": []}

        def note(msg):
            log(msg)
            results["notes"].append(scrub(msg))

        def save():
            text = json.dumps(results, ensure_ascii=False, indent=2, default=str)
            (out_dir / "results.json").write_text(scrub(text), encoding="utf-8")

        mon = None
        try:
            try:
                if not args.no_monitor:
                    mon = Monitor(out_dir / "monitor.csv")
                    mon.start()
                    idle = float(cfg["run"].get("idle_s", 5))
                    if idle > 0:  # baseline memory with the models loaded but no requests
                        log(f"Recording the idle baseline for {idle:g} s")
                        await asyncio.sleep(idle)
                log("Service check:")
                if not await check_services(svc, cfg, inp, log):
                    note("Some services are unreachable - check the addresses in the config")
                    log(f"See {out_dir / 'log.txt'}")
                    raise SystemExit(1)
                for name, title, phase, fn in (
                    ("latency", "Test 1. Single user", "latency", lambda: test_latency(svc, cfg, inp, log)),
                    ("load", "Test 2. Concurrent dialogs", None, lambda: test_load(svc, cfg, inp, log, mon)),
                    ("quality", "Test 3. Quality", "quality", lambda: test_quality(svc, cfg, inp, log, out_dir)),
                ):
                    if name not in args.tests:
                        continue
                    log(title)
                    if mon and phase:
                        mon.phase = phase
                    try:
                        results[name] = await fn()
                    except Exception as e:  # one broken test must not cancel the others
                        note(f"Test '{name}' failed: {e!r}"[:500])
                    save()
            finally:
                try:
                    await svc.aclose()
                except Exception as e:
                    log(f"Error while closing adapters: {e!r}")
                if svc.llm_usage_missing:
                    results["notes"].append("LLM completion tokens were counted as stream chunks (the server sent "
                                            "no usage); prompt tokens are unknown")
        finally:
            if mon:
                mon.stop_ev.set()
                mon.join(timeout=5)
            results["resource_peaks"] = mon.peaks if mon else {}
            save()
            write_summary(out_dir, args.machine, cfg, results, mon.peaks if mon else None, scrub)
    log(f"Done: {out_dir / 'summary.md'}")


if __name__ == "__main__":
    asyncio.run(main())
