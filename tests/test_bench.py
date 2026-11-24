import asyncio
import io
import json
import shutil
import socket
import sys
import wave
from pathlib import Path

import httpx
import numpy as np
import pytest
import yaml

import adapters
import bench
import fake_server

ROOT = Path(__file__).resolve().parent.parent
SYSTEM = "You are a helpful voice assistant."
FAST_RUN = {"warmup": 1, "repeats": 2, "concurrency_levels": [1, 2], "concurrency_duration_s": 1,
            "history_tokens": 100, "request_timeout_s": 30}


def make_wav(seconds, rate=16000, channels=1, amplitude=0.3):
    t = np.arange(int(seconds * rate)) / rate
    mono = amplitude * np.sin(2 * np.pi * 300 * t)
    pcm = (np.repeat(mono[:, None], channels, axis=1) * 32767).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


@pytest.fixture
def server():
    srv = fake_server.start()
    yield srv
    srv.shutdown()
    srv.server_close()


def full_cfg(url):
    return {
        "llm": {"type": "openai", "url": f"{url}/v1/chat/completions", "model": "fake-llm",
                "system_prompt": SYSTEM, "api_key": "secret-key"},
        "asr": {"type": "openai", "url": f"{url}/v1/audio/transcriptions", "model": "fake-asr"},
        "nlu": {"type": "http", "url": f"{url}/predict"},
        "tts": {"type": "http", "url": f"{url}/tts", "response": "wav"},
        "run": dict(FAST_RUN),
    }


async def with_services(cfg, fn, base_dir=None):
    async with httpx.AsyncClient(timeout=30) as client:
        svc = bench.Services(cfg, client, base_dir)
        try:
            return await fn(svc)
        finally:
            await svc.aclose()


def write_data(tmp_path, audio=True):
    d = tmp_path / "data"
    d.mkdir()
    data = {}
    if audio:
        (d / "001.wav").write_bytes(make_wav(1.0))
        (d / "manifest.csv").write_text(f"audio,text\n001.wav,{fake_server.TRANSCRIPT}\n", encoding="utf-8")
        data["asr_manifest"] = "data/manifest.csv"
    (d / "prompts.jsonl").write_text(json.dumps({"prompt": "Weather?", "must_contain": ["sunny"]}) + "\n",
                                     encoding="utf-8")
    (d / "nlu.csv").write_text("text,intent\nWhat is the weather,weather\nHello there,other\n", encoding="utf-8")
    (d / "tts.txt").write_text("Good morning!\n", encoding="utf-8")
    data.update({"llm_prompts": "data/prompts.jsonl", "nlu_set": "data/nlu.csv", "tts_texts": "data/tts.txt"})
    return data


def run_main(tmp_path, cfg, monkeypatch, tests="latency,load,quality", exit_code=None):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["bench.py", "--config", str(config), "--machine", "fake",
                                      "--no-monitor", "--tests", tests])
    if exit_code is None:
        asyncio.run(bench.main())
    else:
        with pytest.raises(SystemExit) as e:
            asyncio.run(bench.main())
        assert e.value.code == exit_code
    out = next((tmp_path / "results").glob("fake_*"), None)
    if out is None:
        return None, None
    results_file = out / "results.json"
    results = json.loads(results_file.read_text(encoding="utf-8")) if results_file.exists() else None
    return out, results


def test_full_pipeline_voice_turn(server):
    cfg = full_cfg(server.url)

    async def fn(svc):
        return {
            "asr": await svc.asr(make_wav(1.0)),
            "asr_silence": await svc.asr(make_wav(1.0, amplitude=0)),
            "nlu": await svc.nlu("what is the weather"),
            "llm": await svc.llm(bench.build_messages(cfg["llm"], "Weather?", 100)),
            "tts": await svc.tts("Good morning"),
            "turn": await bench.voice_turn(svc, cfg, make_wav(1.0)),
        }

    out = asyncio.run(with_services(cfg, fn))
    assert out["asr"] == fake_server.TRANSCRIPT
    assert out["asr_silence"] == ""
    assert out["nlu"] == "weather"

    llm = out["llm"]
    assert llm["text"] == fake_server.ANSWER
    assert llm["completion_tokens"] == len(fake_server.PIECES) + 1  # from usage
    assert llm["prompt_tokens"] == 7
    assert llm["ttft_s"] > 0 and llm["decode_tps"] > 0
    req = server.llm_requests[0]
    assert req["stream"] is True and req["stream_options"] == {"include_usage": True}
    assert req["messages"][0] == {"role": "system", "content": SYSTEM}
    assert req["messages"][1]["content"].startswith("Session ")
    llm_headers = next(h for p, h, _ in server.requests if p == "/v1/chat/completions")
    assert llm_headers.get("Authorization") == "Bearer secret-key"

    audio, first, total = out["tts"]
    assert bench.wav_duration(audio) == pytest.approx(fake_server.TTS_SEC_PER_CHAR * len("Good morning"), rel=0.01)
    assert 0 < first <= total

    turn = out["turn"]
    assert turn["asr_text"] == fake_server.TRANSCRIPT
    assert turn["turn_s"] >= turn["pause_s"] > 0 and turn["tts_first_s"] > 0
    tts_texts = [json.loads(b)["text"].strip() for p, _, b in server.requests if p == "/tts"]
    assert tts_texts[-2:] == ["It is sunny and warm today.", "A light jacket is enough for the evening."]


def test_llm_without_usage_counts_stream_chunks(server):
    server.send_usage = False
    cfg = {"llm": {"url": f"{server.url}/v1/chat/completions", "model": "m"}}

    async def fn(svc):
        return await svc.llm([{"role": "user", "content": "hi"}]), svc.llm_usage_missing

    r, missing = asyncio.run(with_services(cfg, fn))
    assert r["completion_tokens"] == len(fake_server.PIECES) + 1  # the reasoning step counts too
    assert r["prompt_tokens"] is None and missing


def test_tts_pcm16_is_wrapped_into_wav(server):
    cfg = {"tts": {"url": f"{server.url}/tts_pcm", "response": "pcm16", "sample_rate": fake_server.TTS_RATE}}
    audio, first, total = asyncio.run(with_services(cfg, lambda svc: svc.tts("Hello world")))
    with wave.open(io.BytesIO(audio)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (fake_server.TTS_RATE, 1, 2)
    assert bench.wav_duration(audio) == pytest.approx(fake_server.TTS_SEC_PER_CHAR * len("Hello world"), rel=0.01)
    assert 0 < first <= total


def test_tts_json_base64_first_audio_is_total(server):
    cfg = {"tts": {"url": f"{server.url}/tts_json", "response": "json_base64", "audio_field": "audio"}}
    audio, first, total = asyncio.run(with_services(cfg, lambda svc: svc.tts("Hello")))
    assert bench.wav_duration(audio) == pytest.approx(fake_server.TTS_SEC_PER_CHAR * 5, rel=0.01)
    assert first == total


def test_pcm16_to_wav_helper():
    pcm = np.zeros(2400, "<i2").tobytes()
    wav = adapters.pcm16_to_wav(pcm, 24000)
    assert bench.wav_duration(wav) == pytest.approx(0.1)


def test_wav_to_f32_resamples_to_16k_mono():
    samples = np.frombuffer(adapters.wav_to_f32(make_wav(1.0, rate=24000, channels=2)), "<f4")
    assert samples.size == 16000
    assert 0.25 < np.abs(samples).max() <= 1.0


def test_mode_alias(server):
    assert adapters.stage_type("asr", {"mode": "json"}) == "http"
    assert adapters.stage_type("asr", {"mode": "openai"}) == "openai"
    assert adapters.stage_type("asr", {}) == "openai"
    assert adapters.stage_type("tts", {}) == "http"
    assert adapters.stage_type("llm", {"plugin": "x.py:Y"}) == "plugin"
    cfg = {"asr": {"mode": "json", "url": f"{server.url}/asr", "text_field": "result.text"},
           "llm": {"mode": "openai", "url": f"{server.url}/v1/chat/completions", "model": "m"}}

    async def fn(svc):
        assert isinstance(svc.stages["asr"], adapters.HttpASR)
        assert isinstance(svc.stages["llm"], adapters.OpenAILLM)
        return await svc.asr(make_wav(0.5))

    assert asyncio.run(with_services(cfg, fn)) == fake_server.TRANSCRIPT


@pytest.mark.parametrize("section", [{"mode": "grpc", "addr": "localhost:1"}, {"type": "grpc"}])
def test_grpc_is_not_builtin(section):
    with pytest.raises(ValueError, match="type: plugin"):
        adapters.build("llm", section, {}, None)


def test_unknown_type_is_a_clear_error():
    with pytest.raises(ValueError, match="unknown type 'websocket'"):
        adapters.build("tts", {"type": "websocket"}, {}, None)


def test_plugin_from_file_relative_to_config(tmp_path):
    (tmp_path / "plugins").mkdir()
    shutil.copy(ROOT / "examples" / "plugin_echo.py", tmp_path / "plugins" / "echo.py")
    cfg = {"llm": {"type": "plugin", "plugin": "plugins/echo.py:EchoLLM", "delay_s": 0.001},
           "nlu": {"type": "plugin", "plugin": "plugins/echo.py:EchoNLU"}}

    async def fn(svc):
        return await svc.llm([{"role": "user", "content": "one two three"}]), await svc.nlu("hello there")

    r, intent = asyncio.run(with_services(cfg, fn, base_dir=tmp_path))
    assert r["text"] == "one two three"
    assert r["completion_tokens"] == 3 and r["prompt_tokens"] == 3
    assert intent == "greeting"


def test_plugin_from_module(server):
    cfg = {"nlu": {"type": "plugin", "plugin": "adapters:HttpNLU", "url": f"{server.url}/predict"}}
    assert asyncio.run(with_services(cfg, lambda svc: svc.nlu("the weather"))) == "weather"


def test_plugin_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        adapters.load_plugin("missing.py:X", tmp_path)
    with pytest.raises(AttributeError, match="plugin 'json:NoSuchClass': json has no class 'NoSuchClass'"):
        adapters.load_plugin("json:NoSuchClass")
    (tmp_path / "plugs.py").write_text("X = 1\n", encoding="utf-8")
    with pytest.raises(AttributeError) as e:
        adapters.load_plugin("plugs.py:Nope", tmp_path)
    assert str(e.value) == f"plugin 'plugs.py:Nope': {(tmp_path / 'plugs.py').resolve()} has no class 'Nope'"
    with pytest.raises(ValueError, match="needs `plugin:"):
        adapters.build("llm", {"type": "plugin"}, {}, None)


def test_main_full_pipeline(tmp_path, monkeypatch, server):
    cfg = full_cfg(server.url)
    cfg["data"] = write_data(tmp_path)
    out, results = run_main(tmp_path, cfg, monkeypatch)

    assert results["latency"]["e2e"]["pause_s"]["n"] == 2
    assert all(level["errors"] == 0 for mode in ("llm", "pipeline") for level in results["load"][mode])
    assert [lvl["n"] for lvl in results["load"]["pipeline"]] == [1, 2]
    for level in results["load"]["pipeline"]:
        assert level["turn_s"]["p50"] >= level["pause_s"]["p50"] > 0
        assert level["asr_s"]["n"] == level["turns"]
    q = results["quality"]
    assert q["asr"]["wer"] == 0
    assert q["nlu"]["accuracy"] == 1.0
    assert q["llm"]["keys_ok_share"] == 1.0
    assert q["tts"]["texts"] == 1
    assert results["config"]["llm"]["api_key"] == "***"

    summary = (out / "summary.md").read_text(encoding="utf-8")
    for text in ("## Expected vs measured", "## Load: LLM only", "## Load: full pipeline", "## Quality",
                 "**Answer:** 1 user: ", "p50 = median", "Pause before answer", "| Pause p50 / p95, s |",
                 "Peak RAM, GB (includes GPU memory on unified-memory machines) | - | not measured (--no-monitor)",
                 "ASR: WER (0 = perfect)", "caused by ASR/TTS"):
        assert text in summary, text
    assert "response time by number of users" not in summary
    assert "secret-key" not in (out / "results.json").read_text(encoding="utf-8")


def test_main_llm_only(tmp_path, monkeypatch, server):
    cfg = {"llm": full_cfg(server.url)["llm"], "run": dict(FAST_RUN)}
    cfg["data"] = write_data(tmp_path)  # audio/nlu/tts data present but their stages are not
    out, results = run_main(tmp_path, cfg, monkeypatch)

    assert results["latency"]["llm"]["decode_tps"]["n"] == 2
    assert "e2e" not in results["latency"]
    assert len(results["load"]["llm"]) == 2 and results["load"]["pipeline"] == []
    assert set(results["quality"]) == {"llm"}
    log = (out / "log.txt").read_text(encoding="utf-8")
    assert "ASR: not configured, skipped" in log
    assert "Voice-turn latency skipped: stage(s) not configured: asr, tts" in log
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "## Load: LLM only" in summary and "## Load: full pipeline" not in summary
    assert "Pause" not in summary and "Full turn" not in summary
    assert "concurrent answers (LLM only) stay" in summary


def test_main_stt_only_quality(tmp_path, monkeypatch, server):
    cfg = {"asr": full_cfg(server.url)["asr"], "run": dict(FAST_RUN), "data": write_data(tmp_path)}
    out, results = run_main(tmp_path, cfg, monkeypatch, tests="quality")
    assert results["quality"]["asr"]["wer"] == 0
    assert set(results["quality"]) == {"asr"}
    assert not (out / "llm_answers.jsonl").exists()


def test_main_with_plugin_llm(tmp_path, monkeypatch):
    shutil.copy(ROOT / "examples" / "plugin_echo.py", tmp_path / "my_llm.py")
    cfg = {"llm": {"type": "plugin", "plugin": "my_llm.py:EchoLLM", "delay_s": 0.001},
           "run": dict(FAST_RUN, history_tokens=0), "data": write_data(tmp_path, audio=False)}
    out, results = run_main(tmp_path, cfg, monkeypatch, tests="latency")
    assert results["latency"]["llm"]["completion_tokens"]["p50"] == 1  # echoes the one-word prompt
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "llm (plugin, my_llm.py:EchoLLM)" in summary and "LLM: my_llm.py:EchoLLM," in summary


def closed_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_main_stops_when_a_service_is_down(tmp_path, monkeypatch, capsys):
    port = closed_port()
    cfg = {"llm": {"url": f"http://127.0.0.1:{port}/v1/chat/completions", "model": "m"},
           "run": dict(FAST_RUN, request_timeout_s=2)}
    out, results = run_main(tmp_path, cfg, monkeypatch, exit_code=1)
    assert "latency" not in results
    assert any("unreachable" in n for n in results["notes"])
    assert (f"LLM: cannot connect to http://127.0.0.1:{port}/v1/chat/completions - is the server running?"
            in (out / "log.txt").read_text(encoding="utf-8"))
    assert (out / "summary.md").exists()
    assert f"See {out / 'log.txt'}" in capsys.readouterr().out


def test_http_error_quotes_the_body_of_a_streamed_response(tmp_path, monkeypatch, server):
    server.llm_fail_after = 0
    cfg = {"llm": full_cfg(server.url)["llm"], "run": dict(FAST_RUN)}
    out, _ = run_main(tmp_path, cfg, monkeypatch, exit_code=1)
    log = (out / "log.txt").read_text(encoding="utf-8")
    assert f"LLM: HTTP 500 from {server.url}/v1/chat/completions: " in log and "overloaded" in log


def test_http_500_records_failure_and_stops_load(tmp_path, monkeypatch, server):
    server.llm_fail_after = 1  # the service check passes, every later request fails
    cfg = {"llm": full_cfg(server.url)["llm"], "run": dict(FAST_RUN)}
    out, results = run_main(tmp_path, cfg, monkeypatch, tests="latency,load")
    assert "latency" not in results  # failed, but the load test still ran
    assert any(n.startswith("Test 'latency' failed") for n in results["notes"])
    levels = results["load"]["llm"]
    assert len(levels) == 1 and levels[0]["turns"] == 0 and levels[0]["errors"] >= 1
    assert levels[0]["errors"] <= 4  # 0.5 s back-off instead of a tight loop
    assert "500" in levels[0]["error_samples"][0]
    assert "errors vs 0 turns" in results["load"]["stopped"]["llm"]
    assert "Stopped early" in (out / "summary.md").read_text(encoding="utf-8")


def test_secrets_never_reach_results_log_or_summary(tmp_path, monkeypatch, server):
    host = server.url.split("//")[1]
    cfg = {
        "llm": {"url": f"http://user:pw-secret-1@{host}/v1/chat/completions?key=query-secret-2", "model": "m",
                "extra_body": {"x_api_token": "body-secret-3"}, "headers": {"X-Custom": "header-secret-4"}},
        "asr": {"url": f"{server.url}/fail?key=query-secret-5", "extra": {"password": "extra-secret-6"}},
        "run": dict(FAST_RUN, history_tokens=0, max_tokens_like=7),
    }
    cfg["data"] = write_data(tmp_path)
    out, results = run_main(tmp_path, cfg, monkeypatch, exit_code=1)
    text = "".join((out / f).read_text(encoding="utf-8") for f in ("results.json", "log.txt", "summary.md"))
    assert f"ASR: HTTP 500 from {server.url}/fail" in text and "boom" in text
    for secret in ("pw-secret-1", "query-secret-2", "body-secret-3", "header-secret-4", "query-secret-5",
                   "extra-secret-6", "user:pw"):
        assert secret not in text, secret
    assert results["config"]["run"]["max_tokens_like"] == 7  # numbers stay visible


def test_empty_yaml_sections(tmp_path, monkeypatch, server):
    config = tmp_path / "config.yaml"
    config.write_text(f"llm:\n  url: {server.url}/v1/chat/completions\n  model: m\n"
                      "data:\nrun:\nthresholds:\nexpect:\nnlu:\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["bench.py", "--config", str(config), "--machine", "fake", "--no-monitor",
                                      "--tests", "quality"])
    asyncio.run(bench.main())
    out = next((tmp_path / "results").glob("fake_*"))
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert results["quality"]["llm"]["keys_ok_share"] is None
    assert "check skipped (no must_contain keys in llm_prompts)" in (out / "summary.md").read_text(encoding="utf-8")
    assert "LLM: no must_contain keys in llm_prompts, check skipped" in (out / "log.txt").read_text(encoding="utf-8")


def test_unknown_test_name_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["bench.py", "--config", str(tmp_path / "c.yaml"), "--tests", "latncy"])
    with pytest.raises(SystemExit) as e:
        asyncio.run(bench.main())
    assert e.value.code == 2


def test_config_error_exits_nonzero_without_a_results_folder(tmp_path, monkeypatch, capsys):
    cfg = {"llm": {"type": "plugin", "plugin": "nope.py:X"}, "run": dict(FAST_RUN)}
    run_main(tmp_path, cfg, monkeypatch, exit_code=2)
    assert "Config error: plugin file not found" in capsys.readouterr().out
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("content, message", [
    (None, "config not found: {path}. Start with: cp config.example.yaml config.yaml"),
    ("llm: [", "{path}: invalid YAML: "),
])
def test_config_file_errors(tmp_path, monkeypatch, capsys, content, message):
    config = tmp_path / "config.yaml"
    if content is not None:
        config.write_text(content, encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["bench.py", "--config", str(config)])
    with pytest.raises(SystemExit) as e:
        asyncio.run(bench.main())
    assert e.value.code == 2
    assert message.format(path=config) in capsys.readouterr().err


@pytest.mark.parametrize("key, name, content, message", [
    ("asr_manifest", "m.csv", "path,transcript\na.wav,hi\n",
     "data/m.csv: header must be audio,text (got path,transcript)"),
    ("asr_manifest", "m.csv", "audio,text\n,hi\n", "data/m.csv:2: empty audio path"),
    ("llm_prompts", "p.jsonl", '{"prompt": "a"}\n\n["b"]\n', 'data/p.jsonl:3: expected JSON with a "prompt" key'),
    ("llm_prompts", "p.jsonl", '{"prompt": "a", "must_contain": "x"}\n', "data/p.jsonl:1: must_contain must be a list"),
    ("nlu_set", "n.csv", "\ufefftext,label\nhi,greet\n", "data/n.csv: header must be text,intent (got text,label)"),
])
def test_bad_data_files_are_config_errors(tmp_path, key, name, content, message):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / name).write_text(content, encoding="utf-8")
    with pytest.raises(ValueError) as e:
        bench.load_inputs({"data": {key: f"data/{name}"}}, tmp_path)
    assert str(e.value).startswith(message)


def test_bad_data_file_exits_2_before_any_request(tmp_path, monkeypatch, capsys, server):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "m.csv").write_text("path,transcript\n", encoding="utf-8")
    cfg = {"asr": full_cfg(server.url)["asr"], "data": {"asr_manifest": "data/m.csv"}}
    run_main(tmp_path, cfg, monkeypatch, exit_code=2)
    assert "Config error: data/m.csv: header must be audio,text" in capsys.readouterr().out
    assert not server.requests and not (tmp_path / "results").exists()


def test_data_warnings_only_for_configured_stages(tmp_path, monkeypatch, server):
    cfg = {"llm": full_cfg(server.url)["llm"], "run": dict(FAST_RUN),
           "data": {"asr_manifest": "data/a.csv", "nlu_set": "data/n.csv", "tts_texts": "data/t.txt",
                    "llm_prompts": "data/p.jsonl"}}
    out, _ = run_main(tmp_path, cfg, monkeypatch, tests="latency")
    log = (out / "log.txt").read_text(encoding="utf-8")
    assert "Warning" not in log
    assert f"No data.llm_prompts file ({tmp_path / 'data' / 'p.jsonl'}), using the built-in prompt" in log


def test_unknown_section_and_empty_model_warnings(tmp_path, monkeypatch, server):
    llm = dict(full_cfg(server.url)["llm"], model="")
    out, _ = run_main(tmp_path, {"llm": llm, "lm": {}, "run": dict(FAST_RUN)}, monkeypatch, tests="latency")
    log = (out / "log.txt").read_text(encoding="utf-8")
    assert "unknown section 'lm' ignored; expected llm, asr, nlu, tts, data, run, thresholds, expect" in log
    assert "llm.model is empty; most servers reject it (GET /v1/models lists the names)" in log


def level(n, tps, turns=100, errors=0):
    return {"n": n, "decode_tps": {"p50": tps}, "turns": turns, "errors": errors}


def test_max_level_stops_at_first_failing_level():
    assert bench.max_level([level(1, 30), level(2, 25), level(5, 15)], 20) == "2"
    assert bench.max_level([level(1, 30), level(2, 5), level(5, 25)], 20) == "1"
    assert bench.max_level([level(1, 30), level(2, 25)], 20) == "2+"
    assert bench.max_level([level(1, 30, turns=99, errors=1)], 20) == "1+"  # 1% errors still passes
    assert bench.max_level([level(1, 30, errors=2)], 20) == "0"
    assert bench.max_level([level(1, 30, turns=0)], 20) == "0"


def test_pipeline_without_nlu(tmp_path, monkeypatch, server):
    cfg = full_cfg(server.url)
    del cfg["nlu"]
    cfg["data"] = write_data(tmp_path)
    out, results = run_main(tmp_path, cfg, monkeypatch, tests="latency")
    e2e = results["latency"]["e2e"]
    assert e2e["pause_s"]["n"] == 2 and e2e["nlu_s"]["n"] == 0  # no NLU -> None, not ~0 s


def test_tts_only(tmp_path, monkeypatch, server):
    cfg = {"tts": full_cfg(server.url)["tts"], "run": dict(FAST_RUN), "data": write_data(tmp_path)}
    out, results = run_main(tmp_path, cfg, monkeypatch)
    assert set(results["quality"]) == {"tts"}
    assert "roundtrip_cer" not in results["quality"]["tts"]
    assert results["latency"] == {} and results["load"]["llm"] == [] and results["load"]["pipeline"] == []


def test_asr_llm_without_tts_skips_voice_tests(tmp_path, monkeypatch, server):
    full = full_cfg(server.url)
    cfg = {"asr": full["asr"], "llm": full["llm"], "run": dict(FAST_RUN), "data": write_data(tmp_path)}
    out, results = run_main(tmp_path, cfg, monkeypatch, tests="latency,load")
    assert "llm" in results["latency"] and "e2e" not in results["latency"]
    assert results["load"]["pipeline"] == []
    log = (out / "log.txt").read_text(encoding="utf-8")
    assert "Voice-turn latency skipped: stage(s) not configured: tts" in log
    assert "Full-pipeline load skipped: stage(s) not configured: tts" in log


def test_missing_audio_files_are_reported(tmp_path, monkeypatch, server):
    cfg = full_cfg(server.url)
    cfg["data"] = write_data(tmp_path)
    (tmp_path / "data" / "001.wav").unlink()
    cfg["data"]["llm_prompts"] = "data/typo.jsonl"
    out, results = run_main(tmp_path, cfg, monkeypatch, tests="latency")
    log = (out / "log.txt").read_text(encoding="utf-8")
    assert "No data.llm_prompts file (" in log
    assert "1 audio file(s) missing - run make_test_audio.py" in log


def test_empty_reference_text_does_not_crash(tmp_path, monkeypatch, server):
    d = tmp_path / "data"
    d.mkdir()
    (d / "a.wav").write_bytes(make_wav(1.0))
    (d / "b.wav").write_bytes(make_wav(1.0))
    (d / "m.csv").write_text(f"audio,text\na.wav,\nb.wav,{fake_server.TRANSCRIPT}\n", encoding="utf-8")
    cfg = {"asr": full_cfg(server.url)["asr"], "run": dict(FAST_RUN), "data": {"asr_manifest": "data/m.csv"}}
    out, results = run_main(tmp_path, cfg, monkeypatch, tests="quality")
    q = results["quality"]["asr"]
    assert q["files"] == 2 and q["with_reference"] == 1 and q["wer"] == 0
    (d / "m.csv").write_text("audio,text\na.wav,\n", encoding="utf-8")
    shutil.rmtree(tmp_path / "results")
    out, results = run_main(tmp_path, cfg, monkeypatch, tests="quality")
    assert "wer" not in results["quality"]["asr"]


def test_tts_empty_or_non_audio_body_is_an_error(server):
    for cfg in ({"tts": {"url": f"{server.url}/tts_empty"}},
                {"tts": {"url": f"{server.url}/predict", "text_field": "text"}}):  # JSON, not WAV
        with pytest.raises(RuntimeError, match="TTS"):
            asyncio.run(with_services(cfg, lambda svc: svc.tts("Hello weather")))


PLUGINS = '''
import asyncio

class Good:
    closed = []
    def __init__(self, stage_cfg, cfg, client):
        self.name = stage_cfg.get("name")
    async def transcribe(self, wav):
        return "x"
    async def aclose(self):
        Good.closed.append(self.name)

class FailingClose(Good):
    async def aclose(self):
        raise RuntimeError("close failed")

class Broken:
    def __init__(self, stage_cfg, cfg, client):
        raise RuntimeError("boom")

class NotAGenerator:
    def __init__(self, stage_cfg, cfg, client):
        pass
    async def stream(self, messages):
        return []
'''


def test_plugin_aclose_and_failed_build_cleanup(tmp_path):
    (tmp_path / "plugs.py").write_text(PLUGINS, encoding="utf-8")
    good = adapters.load_plugin("plugs.py:Good", tmp_path)

    # build failure on llm closes the asr adapter built before it
    cfg = {"asr": {"type": "plugin", "plugin": "plugs.py:Good", "name": "asr"},
           "llm": {"type": "plugin", "plugin": "plugs.py:Broken"}}

    async def create():
        async with httpx.AsyncClient() as client:
            await bench.Services.create(cfg, client, tmp_path)

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(create())
    assert good.closed == ["asr"]

    # one failing aclose does not stop the others
    good.closed.clear()
    cfg = {"asr": {"type": "plugin", "plugin": "plugs.py:FailingClose", "name": "asr"},
           "nlu": {"type": "plugin", "plugin": "plugs.py:Good", "name": "nlu"}}
    with pytest.raises(TypeError, match="no method classify"):  # Good has no classify()
        asyncio.run(with_services(cfg, lambda svc: None, base_dir=tmp_path))
    cfg = {"asr": cfg["asr"], "tts": {"type": "plugin", "plugin": "adapters:HttpTTS", "url": "http://x"}}

    async def close_all():
        async with httpx.AsyncClient() as client:
            svc = await bench.Services.create(cfg, client, tmp_path)
            svc.stages["nlu"] = good({"name": "nlu"}, {}, None)
            with pytest.raises(RuntimeError, match="close failed"):
                await svc.aclose()

    asyncio.run(close_all())
    assert good.closed == ["nlu"]


def test_plugin_shape_is_checked(tmp_path):
    (tmp_path / "plugs.py").write_text(PLUGINS, encoding="utf-8")
    with pytest.raises(TypeError, match="async generator"):
        adapters.build("llm", {"type": "plugin", "plugin": "plugs.py:NotAGenerator"}, {}, None, tmp_path)
    with pytest.raises(ValueError, match="needs `url:`"):
        adapters.build("llm", {"type": "openai", "model": "m"}, {}, None)


def test_module_plugin_found_next_to_config(tmp_path):
    (tmp_path / "localplug.py").write_text(PLUGINS, encoding="utf-8")
    obj = adapters.build("asr", {"type": "plugin", "plugin": "localplug:Good"}, {}, None, tmp_path)
    assert type(obj).__name__ == "Good"


def test_plugin_import_error_cleans_up(tmp_path):
    (tmp_path / "badplug.py").write_text("raise ImportError('missing dep')\n", encoding="utf-8")
    before = set(sys.modules)
    with pytest.raises(ImportError, match="missing dep"):
        adapters.load_plugin("badplug.py:X", tmp_path)
    assert not [m for m in set(sys.modules) - before if m.startswith("bench_plugin_badplug")]
    assert str(tmp_path) not in sys.path


def test_type_json_alias():
    assert adapters.stage_type("asr", {"type": "json"}) == "http"
    assert adapters.stage_type("asr", {"type": "http"}) == "http"


def test_sentence_end_cjk():
    assert bench.SENT_END.search("你好。我们")
    assert not bench.SENT_END.search("3.5 percent")


def test_make_test_audio_writes_16k_wavs_and_manifest(tmp_path, server):
    import make_test_audio

    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "prompts.jsonl").write_text(json.dumps({"prompt": "How warm is it?"}) + "\n",
                                                     encoding="utf-8")
    cfg = {"tts": {"url": f"{server.url}/tts_pcm", "response": "pcm16", "sample_rate": 24000},
           "data": {"llm_prompts": "data/prompts.jsonl"}}
    n = asyncio.run(make_test_audio.make_audio(cfg, tmp_path, tmp_path / "data" / "audio",
                                               tmp_path / "data" / "asr_manifest.csv"))
    assert n == 1
    cfg["data"]["asr_manifest"] = "data/asr_manifest.csv"
    item = bench.load_inputs(cfg, tmp_path)["asr"][0]
    assert item["text"] == "How warm is it?"
    with wave.open(io.BytesIO(item["audio"])) as w:
        assert (w.getframerate(), w.getnchannels()) == (16000, 1)
        assert w.getnframes() / 16000 == pytest.approx(fake_server.TTS_SEC_PER_CHAR * len("How warm is it?"), rel=0.01)


def test_make_test_audio_manifest_defaults_to_config(tmp_path, monkeypatch, server):
    import make_test_audio

    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "prompts.jsonl").write_text(json.dumps({"prompt": "Hi"}) + "\n", encoding="utf-8")
    cfg = {"tts": {"url": f"{server.url}/tts"},
           "data": {"llm_prompts": "data/prompts.jsonl", "asr_manifest": "data/custom.csv"}}
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["make_test_audio.py", "--config", str(config)])
    make_test_audio.main()
    assert (tmp_path / "data" / "custom.csv").read_text(encoding="utf-8").startswith("audio,text")
