# Serving models with vLLM for the benchmark

Two scripts get any Hugging Face model behind an OpenAI-compatible endpoint that `bench.py` can measure:

- `install.sh` creates a conda env with a pinned torch + vLLM stack and checks it on the GPU;
- `serve.sh` starts `vllm serve` for one model with the settings of a hardware profile.

The same pins are used on every machine, so numbers from different hardware compare like for like.
Profiles live in `profiles/`:

| Profile | Machine | What it changes |
|---|---|---|
| `gb10` | NVIDIA GB10 (DGX Spark class, see [docs/gb10.md](../docs/gb10.md)): aarch64, sm_121, 128 GB unified memory | memory fraction 0.25 of all memory, page cache freed before each start, CUDA 13 toolkit on `PATH` |
| `rtx` | x86_64 with a discrete RTX GPU (4090, 5090, RTX PRO 6000, ...) | memory fraction 0.45 of the card, x86 glibc check |

`--profile auto` (the default) picks `gb10` on aarch64 and `rtx` on x86_64. To add a machine, copy a profile and
adjust it. Every value in a profile is a default: a variable exported in the shell wins.

## Requirements

- Linux with an NVIDIA driver **R580 or newer** (`nvidia-smi`): torch and vLLM here are CUDA 13 builds.
- glibc 2.34+ on aarch64, 2.35+ on x86_64 (Ubuntu 22.04 or 24.04).
- GB10 only: CUDA toolkit 12.9+ (13.x recommended) for FlashInfer kernels; the script finds the newest
  `/usr/local/cuda-13*`.
- About 25 GB of disk for the env, plus model weights (an 8B bf16 model is about 16 GB).
- conda. If none is found, Miniforge is installed into `~/miniforge3`.

## Install

```bash
bash setup/install.sh --check                  # profile auto, env "bench-vllm"
bash setup/install.sh --profile rtx --check    # explicit profile
```

| Flag | Meaning |
|---|---|
| `--profile gb10\|rtx\|auto` | hardware profile (default `auto`) |
| `--env NAME` | conda env name (default `bench-vllm`) |
| `--python X.Y` | Python version of a new env (default 3.11) |
| `--extra "pkg ..."` | more pip packages, resolved together with the pins |
| `--recreate` | delete the env and create it again |
| `--check` | generate a few tokens with `Qwen/Qwen3-0.6B` (override with `SMOKE_MODEL=...`) |

What the install checks every time: `pip check`; that torch is a CUDA build, sees the GPU and runs a bf16
matmul; driver version, CPU architecture and glibc; on GB10 also the CUDA toolkit version.

### Pinned versions

| Package | Version | Why |
|---|---|---|
| torch | 2.13.0 (cu130) | the pin of vLLM 0.30.0; torch on PyPI for aarch64 is CPU-only up to 2.10 |
| torchaudio | 2.11.0 (cu130) | the matching cu130 build available for aarch64 |
| vllm | 0.30.0 | official CUDA 13 aarch64 build with GB10/sm_121 fixes |
| flashinfer-cubin | 0.6.18.post1 | prebuilt kernels for vLLM's FlashInfer; only on `flashinfer.ai/whl` since 0.6.14 |
| transformers | 5.14.1 | vLLM 0.30 needs >= 5.10.4 and has no upper bound |

Override with `TORCH_VERSION`, `VLLM_VERSION`, `TRANSFORMERS_VERSION`, `FLASHINFER_VERSION`, `TORCHAUDIO_VERSION`
(then re-check that the set still resolves). Heavy packages are installed from wheels only (`--only-binary`):
if a version has no aarch64 wheel, pip fails at once instead of compiling for hours. Envs are created from
conda-forge only, because the Anaconda `defaults` channel needs an interactive ToS acceptance.

## Serve an LLM and an ASR model

One model per process, one terminal (or tmux window) each. Start them **one at a time**: vLLM measures free
memory at startup, and two simultaneous starts confuse each other.

```bash
bash setup/serve.sh Qwen/Qwen3-8B --port 8000 --max-len 16384
# wait for "Application startup complete", then in a second window:
bash setup/serve.sh openai/whisper-large-v3-turbo --port 8001 --mem 0.10
```

| Flag | Meaning |
|---|---|
| `--port N` | default 8000 |
| `--host ADDR` | default `127.0.0.1`; `0.0.0.0` to accept remote clients |
| `--mem FRACTION` | `--gpu-memory-utilization`; default from the profile |
| `--max-len N` | `--max-model-len`; a shorter context leaves more memory for the KV cache |
| `--profile`, `--env` | as in `install.sh` |
| `--drop-caches` | free the page cache before start even on the `rtx` profile |
| `-- ...` | anything after `--` goes to `vllm serve` as is |

LLMs answer on `/v1/chat/completions`. Speech recognition models that your vLLM version supports (Whisper,
and other ASR architectures listed in the vLLM docs, e.g. `Qwen/Qwen3-ASR-1.7B`) answer on
`/v1/audio/transcriptions`. Gated models need `export HF_TOKEN=...` and an accepted license on the model page.

Point `config.yaml` at the servers; `model` must be the id you served:

```yaml
llm:
  type: openai
  url: http://localhost:8000/v1/chat/completions
  model: Qwen/Qwen3-8B
asr:
  type: openai
  url: http://localhost:8001/v1/audio/transcriptions
  model: openai/whisper-large-v3-turbo
```

Quick check without the benchmark:

```bash
curl -s localhost:8000/v1/models
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "Qwen/Qwen3-8B", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 30}'
curl -s localhost:8001/v1/audio/transcriptions -F model=openai/whisper-large-v3-turbo -F file=@sample.wav
free -g      # memory taken by the models
```

## Memory

`--gpu-memory-utilization` is the share of GPU memory vLLM may take for weights, activations and KV cache.

- **GB10:** the fraction is of **all** 128 GB of unified memory. vLLM's default 0.9 would take the whole
  machine, so the profile uses 0.25 (about 32 GB, enough for an 8B bf16 model with a 16K context); two models
  at 0.25 each still leave room for the OS and the benchmark. Why, and a budget for a full voice pipeline:
  [docs/gb10.md](../docs/gb10.md#memory-budget).
- **RTX:** the fraction is of the card's VRAM. The default 0.45 leaves the other half for a second model; for a
  single model you can pass `--mem 0.9`.

## Rented GB10 in an unprivileged container

A rented GB10 is usually a Docker container on someone else's machine. The scripts work there with these notes:

| Topic | What to do |
|---|---|
| Choosing a machine | driver >= 580 (vast.ai filter "Max CUDA" >= 13.0); disk >= 100 GB |
| Image | `nvidia/cuda:13.0.3-devel-ubuntu24.04`: arm64, CUDA 13.0, ships `nvcc`. Images with CUDA 12.8 are too old for sm_121: `apt-get install -y cuda-toolkit-13-0` |
| Rights | you are root without `sudo`; run `apt-get` directly. Docker inside the container is not available |
| Packages | `apt-get update && apt-get install -y curl tmux` |
| Long-running servers | run each `serve.sh` in its own tmux window (`tmux`, then `Ctrl+b c` for a new window) so they survive SSH disconnects |
| Page cache | `drop_caches` is not allowed, yet cached files count as used GPU memory. `serve.sh` and `install.sh --check` then evict files via `posix_fadvise` (about a minute) and print the free memory |
| Cost | a stopped instance still pays for disk: copy `results/` off and destroy it |

Typical session on such a box:

```bash
apt-get update && apt-get install -y curl tmux
bash setup/install.sh --check
tmux new -d -s llm 'bash setup/serve.sh Qwen/Qwen3-8B --port 8000 --max-len 16384 2>&1 | tee llm.log'
until curl -sf localhost:8000/v1/models >/dev/null; do sleep 5; done   # one vLLM start at a time
tmux new -d -s asr 'bash setup/serve.sh openai/whisper-large-v3-turbo --port 8001 --mem 0.10 2>&1 | tee asr.log'
tmux attach -t asr                                                      # Ctrl+b d to detach
```

On an RTX box the commands are the same; `auto` picks the `rtx` profile.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `driver ...: CUDA 13 wheels ... need R580 or newer` | old driver | update the driver; on vast.ai pick a machine with Max CUDA >= 13.0 |
| `torch does not see the GPU: CPU-only build` | torch came from PyPI instead of the cu130 index | `bash setup/install.sh --recreate` |
| `glibc ..., need 2.34 or newer` | OS older than Ubuntu 22.04 | use Ubuntu 22.04/24.04 (DGX OS is 24.04) |
| pip: `No matching distribution` for a package in `--only-binary` | no wheel for this platform at that version | do not build from source: pick a version with a wheel and override the pin variable |
| `pip check: platform tag mismatch ... ignored` | some `nvidia-*` wheels carry an x86 tag in their metadata on aarch64 | harmless; `check_torch` proves the libraries work |
| `nvcc (CUDA toolkit) not found` or `older than 12.9` (GB10) | FlashInfer needs nvcc 12.9+ to build sm_121 kernels | `apt-get install -y cuda-toolkit-13-0`; the profile picks the newest `/usr/local/cuda-13*` |
| `Free memory on device ... is less than desired` | page cache counted as used (GB10), fraction too high, or two vLLM starting at once | start one at a time, lower `--mem`, `--drop-caches` |
| vLLM hangs after `Using V2 Model Runner` (GB10 container) | page cache holds the memory and cannot be dropped | `serve.sh` evicts it automatically on `gb10`; check `free -g` |
| EngineCore hangs on its first pinned-memory allocation | `fork` inherits a held torch lock | keep `VLLM_WORKER_MULTIPROC_METHOD=spawn` (set by both profiles) |
| FlashInfer sampler fails to build kernels | sampler JIT needs nvcc for sm_121 | keep `VLLM_USE_FLASHINFER_SAMPLER=0` (set by both profiles) |
| `No available memory for the cache blocks` or `... KV cache is needed, which is larger than the available` | KV cache does not fit | raise `--mem` or lower `--max-len` |
| `401`/`403` from Hugging Face | gated or private model | `export HF_TOKEN=...` and accept the license on the model page |
| `$'\r': command not found` | scripts copied from Windows with CRLF line endings | `sed -i 's/\r$//' setup/*.sh setup/profiles/*.env` |
