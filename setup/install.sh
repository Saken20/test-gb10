#!/usr/bin/env bash
# Creates a conda env with the pinned torch + vLLM stack and checks it on the GPU.
#
#   bash setup/install.sh [--profile gb10|rtx|auto] [--env NAME] [--python X.Y]
#                         [--extra "pkg1 pkg2"] [--recreate] [--check]
#
#   --profile   hardware profile from setup/profiles/ (default: auto by CPU architecture)
#   --env       conda env name (default: bench-vllm, or $ENV_NAME)
#   --python    Python version for a new env (default: 3.11)
#   --extra     more pip packages installed in the same resolve as the pinned stack
#   --recreate  delete the env and create it again
#   --check     after install, generate a few tokens with a small public model
#               ($SMOKE_MODEL, default Qwen/Qwen3-0.6B) through vLLM's Python API
#
# Version pins can be overridden: TORCH_VERSION, VLLM_VERSION, TRANSFORMERS_VERSION, ...

source "$(dirname "$0")/common.sh"

profile=auto
python_version=3.11
extra=""
recreate=0
check=0
while [ $# -gt 0 ]; do
    case "$1" in
        --profile)  profile="${2:?--profile needs a value}"; shift 2 ;;
        --env)      ENV_NAME="${2:?--env needs a value}"; shift 2 ;;
        --python)   python_version="${2:?--python needs a value}"; shift 2 ;;
        --extra)    extra="${2:?--extra needs a value}"; shift 2 ;;
        --recreate) recreate=1; shift ;;
        --check)    check=1; shift ;;
        -h|--help)  usage; exit 0 ;;
        *) die "unknown argument: $1 (see --help)" ;;
    esac
done

load_profile "$profile"
check_gpu
check_glibc "$PROFILE_GLIBC"
[ "$PROFILE_NEEDS_NVCC" != 1 ] || check_cuda_toolkit
check_disk 25
make_tmp
ensure_conda
# ffmpeg: torchcodec (pulled in by vLLM) loads its libraries on import, and audio decoding needs it
ensure_env "$ENV_NAME" "$python_version" "$recreate" ffmpeg

# vLLM does not cap transformers from above; without a pin the newest one arrives
cat >"$SETUP_TMP/constraints.txt" <<EOF
torch==$TORCH_VERSION
torchaudio==$TORCHAUDIO_VERSION
vllm==$VLLM_VERSION
transformers==$TRANSFORMERS_VERSION
flashinfer-cubin==$FLASHINFER_VERSION
EOF
extra_pkgs=()
[ -z "$extra" ] || read -r -a extra_pkgs <<<"$extra"
log "installing torch + vLLM (several GB of CUDA libraries)"
# vllm[audio] serves /v1/audio/transcriptions; flashinfer-cubin avoids a kernel compile on first start
pip_install --extra-index-url "$FLASHINFER_INDEX" \
    torch torchaudio "vllm[audio]" transformers flashinfer-cubin "${extra_pkgs[@]}" \
    -c "$SETUP_TMP/constraints.txt"

check_pip
check_torch
"$ENV_PY" -c 'import transformers, vllm; print("vllm", vllm.__version__, "| transformers", transformers.__version__)'

if [ "$check" = 1 ]; then
    model="${SMOKE_MODEL:-Qwen/Qwen3-0.6B}"
    log "smoke test: $model through vLLM (downloads the model on first run)"
    # A file, not stdin: with spawn the EngineCore child re-imports __main__
    cat >"$SETUP_TMP/vllm_smoke.py" <<'PY'
import os
import time

from vllm import LLM, SamplingParams


def main():
    llm = LLM(
        model=os.environ["SMOKE_MODEL"],
        gpu_memory_utilization=float(os.environ["GPU_MEMORY_UTILIZATION"]),
        max_model_len=2048,
    )
    params = SamplingParams(max_tokens=64, temperature=0.0)
    start = time.perf_counter()
    out = llm.generate(["Name three primary colors."], params)[0].outputs[0]
    took = time.perf_counter() - start
    print(f"vLLM answered ({len(out.token_ids)} tokens in {took:.2f}s): {out.text.strip()!r}")
    if not out.text.strip():
        raise SystemExit("vLLM returned an empty answer")


if __name__ == "__main__":
    main()
PY
    [ "$FREE_PAGE_CACHE" != 1 ] || free_page_cache
    # env bin first on PATH, as after conda activate: JIT builds look for ninja there
    PATH="$ENV_PREFIX/bin:$PATH" SMOKE_MODEL="$model" "$ENV_PY" "$SETUP_TMP/vllm_smoke.py" \
        || die "vLLM smoke test failed (see above)"
fi

log "done: env $ENV_NAME, profile $PROFILE"
log "serve a model: bash $SETUP_DIR/serve.sh <hf-model-id> --env $ENV_NAME --profile $PROFILE"
