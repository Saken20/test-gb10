# Voice pipeline benchmark

[Русская версия](docs/README.ru.md)

Measures how fast, for how many users and how well a speech + LLM pipeline (**ASR → NLU → LLM → TTS**, or any part of
it) runs on your hardware. For anyone choosing a machine, a model or server settings for a voice assistant or chat bot.
Works with any OpenAI-compatible server (vLLM, SGLang, llama.cpp, TGI, ...), plain HTTP/JSON services and plugins.

## Quick start: measure an LLM in 3 steps

You need a running OpenAI-compatible LLM server, e.g. `vllm serve Qwen/Qwen3-8B --port 8000`
(scripts in [`setup/`](setup/README.md)). No GPU at hand? Try the [dry run](#dry-run-without-a-gpu).
Commands are bash (Linux, macOS, Git Bash, WSL); on Windows use Git Bash or WSL rather than PowerShell.

```bash
# 1. install
pip install -r requirements.txt
# 2. point at your LLM
printf 'llm: {url: http://localhost:8000/v1/chat/completions, model: Qwen/Qwen3-8B}\n' > config.yaml
# 3. run
python bench.py --config config.yaml --machine my-box
```

`type` defaults to `openai`; `model` must be the name the server serves (see `GET /v1/models`). The run takes
~5 minutes (4 load levels × 60 s); `--tests latency` checks your setup in ~30 s. Without your own prompts a built-in
one is used; add `data/llm_prompts.jsonl` ([test data](#test-data)) to measure on your own questions.

## What you get

Each run writes `results/<machine>_<date>_<time>/` next to the config. **Open `summary.md` first**: a two-line
**Answer:** at the top (speed for one user and how many users fit your thresholds), a legend, then one-user latency,
load levels, quality, resource peaks and, if you set `expect:`, an "Expected vs measured" table. In the legend, p50/p95
are the median and the 95th percentile, a load level passes at ≤1% failed requests, and "N+" means even the highest
tested level N passed.

| File | Contents |
| --- | --- |
| `summary.md` | the tables above |
| `results.json` | all numbers (p10/p50/p95/mean) and the config used |
| `monitor.csv` | per second: `time, phase, ram_used_gb, gpu_util_pct, gpu_mem_used_mb, power_w, temp_c`; not written with `--no-monitor` |
| `llm_answers.jsonl` | quality-test answers with their `must_contain` verdict (quality test with an `llm` stage) |
| `log.txt` | the console log |

What is measured:

- **Latency, one user:** time to first token (TTFT), decode tokens/s, pause from the end of the user's phrase to the
  first sound of the answer, full turn until the answer is voiced.
- **Load:** 1, 2, 5, 10... simultaneous dialogs, a fixed time per level, stopping when per-user speed falls below a
  threshold. Shows how many users fit your thresholds and how much ASR/TTS slow the LLM down.
- **Quality:** ASR WER/CER and real-time factor, TTS real-time factor and intelligibility (TTS → ASR round-trip CER),
  NLU intent accuracy, required words in LLM answers (`must_contain`).
- **Resources:** GPU utilization, GPU memory, power, temperature and **system RAM**; the `idle` phase is a
  `run.idle_s` baseline with the models loaded. On unified memory (GB10 / DGX Spark, Apple silicon) RAM is the GPU memory.

Secrets are masked in `results.json`, `log.txt` and `summary.md`: keys containing `key`, `token`, `secret`,
`password`, `passwd`, `auth`, `credential`, `bearer` or `headers`, URL credentials and URL query strings.

## Full pipeline

1. Serve the models: LLM and ASR on any OpenAI-compatible server, TTS and NLU on any HTTP endpoint (or plugins).
2. Fill in the `asr`, `llm`, `tts` and (optionally) `nlu` sections of `config.yaml`.
3. Copy the example data into `data/`, English or Russian (for Russian also set `run.chars_per_token: 3`):
   ```bash
   for f in data/*.example.*; do [[ $f == *.ru.example.* ]] || cp "$f" "${f/.example/}"; done   # English
   for f in data/*.ru.example.*; do cp "$f" "${f/.ru.example/}"; done                           # Russian
   ```
   No recordings? `python make_test_audio.py --config config.yaml` voices the prompts with your TTS into
   `data/audio/*.wav` and `data/asr_manifest.csv` (`--out`, `--manifest` to change). Enough for speed; WER on a
   synthetic voice looks better than on real users.
4. Run `python bench.py --config config.yaml --machine my-box`.

Options: `--tests latency,load,quality` (any subset; an unknown name is an error); `--no-monitor` when the models run
on another machine (the monitor reads the local GPU and RAM). A full voice-pipeline run on real models with up to
10 users takes 20–30 minutes.
If WAV files in the manifest are missing, the voice tests are skipped and the LLM tests still run.

To compare machines, run the same config and data under different `--machine` names (e.g. `gb10-fp8`,
`rtx5090-fp8`), changing one thing at a time, and compare the `summary.md` files.

## Dry run without a GPU

`tests/fake_server.py` fakes every stage on one port. Run these one after another:

```bash
python tests/fake_server.py 18080 &                                    # leave it running
for f in data/*.example.*; do [[ $f == *.ru.example.* ]] || cp "$f" "${f/.example/}"; done
sed 's#localhost:800[0-3]#127.0.0.1:18080#' config.example.yaml > config.yaml
python make_test_audio.py --config config.yaml
python bench.py --config config.yaml --no-monitor --tests latency,quality   # ~1 min; with load ~8 min
```

The numbers mean nothing; it checks the tool and shows the output files. Endpoints, to try other stage types:

| Endpoint | Fakes | Stage config |
| --- | --- | --- |
| `/v1/chat/completions` | streaming LLM | `llm: {type: openai}` |
| `/v1/audio/transcriptions` | OpenAI-style ASR | `asr: {type: openai}` |
| `/asr` | JSON ASR, `{"result": {"text": ...}}` | `asr: {type: http, text_field: result.text}` |
| `/predict` | NLU, `{"intent": ...}` | `nlu: {type: http}` |
| `/tts`, `/tts_json`, `/tts_pcm` | TTS: streamed WAV, `{"audio": base64 WAV}`, raw PCM16 at 24 kHz | `tts: {type: http}` with `response: wav`, `json_base64`, or `pcm16` + `sample_rate: 24000` |

## Config

Every key is commented in `config.example.yaml`. Omit a stage section to skip that stage.

```yaml
llm:
  type: openai                # openai | plugin
  url: http://localhost:8000/v1/chat/completions
  model: Qwen/Qwen3-8B
  max_tokens: 200
  temperature: 0.7
  system_prompt: "You are a helpful voice assistant. Answer in 1-3 sentences."
  extra_body: {}              # merged into the request, e.g. {chat_template_kwargs: {enable_thinking: false}}
asr:
  type: openai                # openai | http | plugin
  url: http://localhost:8001/v1/audio/transcriptions
  model: Qwen/Qwen3-ASR-1.7B
nlu:
  type: http                  # http | plugin
  url: http://localhost:8003/predict
  intent_field: intent        # dotted path in the response, e.g. result.intent
tts:
  type: http                  # http | plugin
  url: http://localhost:8002/tts
  response: wav               # wav | json_base64 | pcm16 (needs sample_rate)
data: {asr_manifest: data/asr_manifest.csv, llm_prompts: data/llm_prompts.jsonl, nlu_set: data/nlu.csv,
       tts_texts: data/tts_texts.txt}
run:
  warmup: 3
  repeats: 20                 # turns in the one-user test
  concurrency_levels: [1, 2, 5, 10]
  concurrency_duration_s: 60  # length of each load level
  history_tokens: 4000        # dialog history sent with every turn
  chars_per_token: 4          # ~4 for English, ~3 for Russian
  stop_below_tps: 1           # stop the load test below this per-user speed
  request_timeout_s: 300
  idle_s: 5                   # idle baseline before the tests
thresholds:
  comfortable_tps: 20         # per-user tokens/s that feels fluent
  limit_tps: 10               # lowest acceptable per-user tokens/s
  pause_max_s: 2.0            # acceptable p95 pause before the answer
# expect:                    # optional: your expectations, shown in "Expected vs measured"
#   single_tps: 25
#   pause_s: "1-2"
#   comfortable_n: "1-2"
#   limit_n: "~10"
#   pipeline_drop_pct: 20     # LLM slowdown caused by ASR/TTS, %
```

HTTP formats of the built-in types:

| Stage | `type` | Request | Response |
| --- | --- | --- | --- |
| LLM | `openai` | chat completions, `stream: true` | SSE; `usage` is read when sent |
| ASR | `openai` | multipart `file` + `model` (+ `language`, `extra`) | `{"text": ...}` |
| ASR | `http` | `{audio_field: base64 WAV, ...extra}` | `text_field` (dotted path) |
| NLU | `http` | `{text_field: text, ...extra}` | `intent_field` (dotted path) |
| TTS | `http` | `{text_field: text, ...extra}` | streamed WAV, `{audio_field: base64 WAV}` or raw PCM16 |

- Every built-in stage takes `api_key:` (sent as `Authorization: Bearer ...`) and `headers:`.
- The history is padded with filler turns up to `history_tokens`, each starting with a random "Session <n>." so the
  server cannot reuse a cached prefix. Override with `llm.history_filler_user`, `llm.history_filler_assistant`;
  the fallback prompt is `llm.default_prompt`.
- The old key `mode:` still works as `type:` (`mode: json` means `type: http`).

## Plugins

For anything else (gRPC, WebSocket, an in-process model) write a small Python class:

```yaml
asr:
  type: plugin
  plugin: plugins/my_asr.py:MyASR   # file relative to the config, or package.module:ClassName
  url: http://localhost:9000/recognize   # the section's other keys go to the class
```

The class is built as `Cls(stage_cfg, cfg, client)` (`client` is the shared `httpx.AsyncClient`), may define
`async def aclose(self)`, and implements its stage's method. Start from
[`examples/plugin_echo.py`](examples/plugin_echo.py) or the built-in adapters in `adapters.py`.

| Stage | Method |
| --- | --- |
| ASR | `async def transcribe(self, wav: bytes) -> str` |
| NLU | `async def classify(self, text: str) -> str \| None` |
| LLM | `def stream(self, messages: list[dict])`: async generator of `(piece, usage)`. `piece` may be `""` (reasoning or keep-alive); `usage` (`prompt_tokens`, `completion_tokens`) only on the chunk that carries it, else `None`. Without usage, stream chunks count as tokens. |
| TTS | `def stream(self, text: str)`: async generator of raw bytes; `def to_wav(self, raw: bytes) -> bytes`; `streams_audio: bool` (`False`: time to first audio = full synthesis time) |

A plugin LLM without `model:` is shown in the summary header by its `plugin:` spec. Keep private plugins outside
the repository or in `plugins_private/` (ignored by git).

## Test data

UTF-8 files; examples in `data/*.example.*` (English) and `data/*.ru.example.*` (Russian).

| File | Format | Used for |
| --- | --- | --- |
| `asr_manifest.csv` | CSV `audio,text`: WAV path relative to the manifest, reference text | WER/CER; the user's phrase in pipeline tests |
| `llm_prompts.jsonl` | `{"prompt": "...", "must_contain": ["word", ...]}` per line | LLM prompts, `must_contain` check |
| `nlu.csv` | CSV `text,intent` | intent accuracy |
| `tts_texts.txt` | one phrase per line | TTS speed and intelligibility |

Text is compared lower-cased, without punctuation, `ё` → `е`. `must_contain` is a substring check (`cancel` matches
"cancelled"). Real users' 16 kHz mono WAV recordings are best; 30–100 files give a stable WER, NLU needs 20+
phrases per intent.

## Hardware notes

- [docs/gb10.md](docs/gb10.md): NVIDIA GB10 / DGX Spark: what is different, memory planning, reading the results.
- [setup/README.md](setup/README.md): install and serve any Hugging Face model with vLLM (`gb10`, `rtx` profiles).

## Tests

```bash
pip install -r requirements.txt pytest
python -m pytest
```

No GPU or models needed: the tests use the fake server from the [dry run](#dry-run-without-a-gpu).

## License

MIT, see [LICENSE](LICENSE).
