"""Stage adapters. A plugin is a class built as Cls(stage_cfg, cfg, client), optional `async aclose()`, with:
    ASR: async transcribe(wav) -> str          NLU: async classify(text) -> str | None
    LLM: async-gen stream(messages) -> (piece, usage | None)
    TTS: stream(text) -> AsyncIterator[bytes], to_wav(raw) -> bytes, streams_audio: bool
"""
import base64
import importlib
import importlib.util
import inspect
import io
import json
import sys
import wave
from pathlib import Path

STAGES = ("asr", "nlu", "llm", "tts")
DEFAULT_TYPE = {"asr": "openai", "nlu": "http", "llm": "openai", "tts": "http"}
# legacy `mode:` values -> `type:`
MODE_ALIASES = {"openai": "openai", "json": "http", "http": "http", "plugin": "plugin"}


def get_path(obj, dotted):
    """Value at a dotted path such as `result.text` or `items.0.audio`."""
    for key in dotted.split("."):
        obj = obj[int(key)] if isinstance(obj, list) else obj[key]
    return obj


def pcm16_to_wav(pcm: bytes, rate: int, channels: int = 1) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def wav_to_f32(data: bytes, rate: int = 16000) -> bytes:
    """WAV (8/16/32-bit, any rate and channel count) -> float32 LE mono at `rate`."""
    import numpy as np  # only audio conversion needs numpy

    with wave.open(io.BytesIO(data)) as w:
        sr, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        frames = w.readframes(w.getnframes())
    if width == 1:
        x = (np.frombuffer(frames, np.uint8).astype(np.float32) - 128) / 128
    elif width == 2:
        x = np.frombuffer(frames, "<i2").astype(np.float32) / 32768
    elif width == 4:
        x = np.frombuffer(frames, "<i4").astype(np.float32) / 2**31
    else:
        raise ValueError(f"{8 * width}-bit WAV samples are not supported")
    x = x.reshape(-1, channels).mean(axis=1)
    if sr != rate and len(x) > 1:
        # linear interpolation is enough for speech (band up to 8 kHz)
        n = round(len(x) * rate / sr)
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)
    return np.clip(x, -1, 1).astype("<f4").tobytes()


async def _raise_for_status(r):
    if r.is_error:
        await r.aread()  # so the error message can quote the body, also for streamed responses
    r.raise_for_status()


def _headers(stage_cfg):
    h = dict(stage_cfg.get("headers") or {})
    if stage_cfg.get("api_key"):
        h.setdefault("Authorization", f"Bearer {stage_cfg['api_key']}")
    return h


class OpenAILLM:
    """OpenAI-compatible /v1/chat/completions with SSE streaming (vLLM, SGLang, llama.cpp, TGI, ...)."""

    def __init__(self, stage_cfg, cfg, client):
        self.s, self.c = stage_cfg, client

    async def stream(self, messages):
        s = self.s
        body = {
            "model": s.get("model", ""), "messages": messages,
            "max_tokens": s.get("max_tokens", 200), "temperature": s.get("temperature", 0.7),
            "stream": True, "stream_options": {"include_usage": True},
            **(s.get("extra_body") or {}),
        }
        async with self.c.stream("POST", s["url"], json=body, headers=_headers(s)) as r:
            await _raise_for_status(r)
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                for ch in obj.get("choices", []):
                    d = ch.get("delta", {}) or {}
                    piece = d.get("content") or ""
                    # a reasoning delta is a decode step with no visible text
                    if piece or d.get("reasoning_content") or d.get("reasoning"):
                        yield piece, None
                if obj.get("usage"):
                    yield "", obj["usage"]


class OpenAIASR:
    def __init__(self, stage_cfg, cfg, client):
        self.s, self.c = stage_cfg, client

    async def transcribe(self, wav: bytes) -> str:
        s = self.s
        data = {"model": s.get("model", ""), **(s.get("extra") or {})}
        if s.get("language"):
            data["language"] = s["language"]
        r = await self.c.post(s["url"], files={"file": ("audio.wav", wav, "audio/wav")}, data=data,
                              headers=_headers(s))
        await _raise_for_status(r)
        return r.json().get("text", "")


class HttpASR:
    """POST {audio_field: base64 WAV, ...extra} -> response[text_field]."""

    def __init__(self, stage_cfg, cfg, client):
        self.s, self.c = stage_cfg, client

    async def transcribe(self, wav: bytes) -> str:
        s = self.s
        body = {s.get("audio_field", "audio_base64"): base64.b64encode(wav).decode(), **(s.get("extra") or {})}
        r = await self.c.post(s["url"], json=body, headers=_headers(s))
        await _raise_for_status(r)
        return get_path(r.json(), s.get("text_field", "text"))


class HttpNLU:
    """POST {text_field: text, ...extra} -> response[intent_field]."""

    def __init__(self, stage_cfg, cfg, client):
        self.s, self.c = stage_cfg, client

    async def classify(self, text: str):
        s = self.s
        r = await self.c.post(s["url"], json={s.get("text_field", "text"): text, **(s.get("extra") or {})},
                              headers=_headers(s))
        await _raise_for_status(r)
        return get_path(r.json(), s.get("intent_field", "intent"))


class HttpTTS:
    """POST {text_field: text, ...extra}; the response is WAV bytes, JSON with base64 WAV, or raw PCM16."""

    def __init__(self, stage_cfg, cfg, client):
        self.s, self.c = stage_cfg, client
        self.response = stage_cfg.get("response", "wav")
        if self.response not in ("wav", "json_base64", "pcm16"):
            raise ValueError(f"tts.response must be wav, json_base64 or pcm16, got {self.response!r}")
        # JSON arrives in one piece, so the first audio is only usable at the end
        self.streams_audio = self.response != "json_base64"

    async def stream(self, text: str):
        s = self.s
        body = {s.get("text_field", "text"): text, **(s.get("extra") or {})}
        async with self.c.stream("POST", s["url"], json=body, headers=_headers(s)) as r:
            await _raise_for_status(r)
            async for ch in r.aiter_bytes():
                yield ch

    def to_wav(self, raw: bytes) -> bytes:
        if self.response == "json_base64":
            return base64.b64decode(get_path(json.loads(raw), self.s.get("audio_field", "audio")))
        if self.response == "pcm16":
            return pcm16_to_wav(raw, int(self.s.get("sample_rate", 24000)), int(self.s.get("channels", 1)))
        return raw


BUILTIN = {
    "llm": {"openai": OpenAILLM},
    "asr": {"openai": OpenAIASR, "http": HttpASR},
    "nlu": {"http": HttpNLU},
    "tts": {"http": HttpTTS},
}


_plugin_modules = {}


def stage_type(stage: str, stage_cfg: dict) -> str:
    mode = stage_cfg.get("mode")
    if stage_cfg.get("type") == "grpc" or mode == "grpc":
        raise ValueError(f"{stage}: 'grpc' is not a built-in adapter any more; use type: plugin "
                         "with a plugin class that speaks your gRPC API")
    t = stage_cfg.get("type")
    if t is None and mode is not None:
        if mode not in MODE_ALIASES:
            raise ValueError(f"{stage}: unknown mode {mode!r}")
        t = mode
    if t is None:
        return "plugin" if stage_cfg.get("plugin") else DEFAULT_TYPE[stage]
    return MODE_ALIASES.get(t, t)


def _load_file(path: Path):
    module = _plugin_modules.get(path)
    if module is not None:
        return module
    # lets the plugin import helper packages that sit next to it
    added = str(path.parent) not in sys.path
    if added:
        sys.path.append(str(path.parent))
    name = f"bench_plugin_{path.stem}_{len(_plugin_modules)}"
    mod_spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(mod_spec)
    sys.modules[name] = module
    try:
        mod_spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        if added and str(path.parent) in sys.path:
            sys.path.remove(str(path.parent))
        raise
    _plugin_modules[path] = module
    return module


def load_plugin(spec: str, base_dir=None):
    """Class from `path/to/file.py:Class` or `module:Class` (both looked up next to the config first)."""
    if not isinstance(spec, str) or ":" not in spec:
        raise ValueError(f"plugin must look like file.py:ClassName or module:ClassName, got {spec!r}")
    target, cls_name = spec.rsplit(":", 1)
    base = Path(base_dir or Path.cwd())
    if target.endswith(".py") or "/" in target or "\\" in target:
        path = Path(target).expanduser()
        if not path.is_absolute():
            path = base / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"plugin file not found: {path}")
        module, where = _load_file(path), str(path)
    else:
        local = base / (target.replace(".", "/") + ".py")
        if local.is_file():
            module, where = _load_file(local.resolve()), str(local.resolve())
        else:
            module, where = importlib.import_module(target), target
    try:
        return getattr(module, cls_name)
    except AttributeError:
        raise AttributeError(f"plugin {spec!r}: {where} has no class {cls_name!r}") from None


REQUIRED = {"asr": ("transcribe",), "nlu": ("classify",), "llm": ("stream",), "tts": ("stream", "to_wav")}


def check_adapter(stage: str, obj):
    name = type(obj).__name__
    for method in REQUIRED[stage]:
        if not callable(getattr(obj, method, None)):
            raise TypeError(f"{stage} adapter {name} has no method {method}() (see adapters.py)")
    if stage == "llm" and not inspect.isasyncgenfunction(obj.stream):
        raise TypeError(f"llm adapter {name}.stream must be an async generator (async def ... yield piece, usage)")


def build(stage: str, stage_cfg: dict, cfg: dict, client, base_dir=None):
    if stage not in BUILTIN:
        raise ValueError(f"unknown stage {stage!r}; expected one of {', '.join(STAGES)}")
    t = stage_type(stage, stage_cfg)
    if t == "plugin":
        if not stage_cfg.get("plugin"):
            raise ValueError(f"{stage}: type: plugin needs `plugin: file.py:ClassName` or `module:ClassName`")
        cls = load_plugin(stage_cfg["plugin"], base_dir)
    else:
        cls = BUILTIN[stage].get(t)
        if cls is None:
            choices = ", ".join([*BUILTIN[stage], "plugin"])
            raise ValueError(f"{stage}: unknown type {t!r}; expected one of {choices}")
        if not stage_cfg.get("url"):
            raise ValueError(f"{stage}: type {t} needs `url:`")
    obj = cls(stage_cfg, cfg, client)
    check_adapter(stage, obj)
    return obj
