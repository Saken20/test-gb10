#!/usr/bin/env bash
# Serves a Hugging Face model with vLLM's OpenAI-compatible server, using a hardware profile.
#
#   bash setup/serve.sh <hf-model-id> [--port N] [--host ADDR] [--mem FRACTION] [--max-len N]
#                       [--profile gb10|rtx|auto] [--env NAME] [--drop-caches] [-- extra vllm args]
#
#   --port         default 8000
#   --host         default 127.0.0.1 (use 0.0.0.0 to accept remote clients)
#   --mem          vLLM --gpu-memory-utilization (default from the profile; on GB10 it is a
#                  fraction of ALL unified memory)
#   --max-len      vLLM --max-model-len (default: the model's own limit)
#   --drop-caches  free the page cache first even if the profile does not ask for it
#   -- ...         everything after -- goes to `vllm serve` unchanged
#
# LLMs answer on /v1/chat/completions, speech models vLLM supports on /v1/audio/transcriptions.
# Gated models read HF_TOKEN from the environment. Start one vLLM at a time.

source "$(dirname "$0")/common.sh"

model="${1:-}"
case "$model" in
    -h|--help) usage; exit 0 ;;
    ""|-*) usage; exit 2 ;;
esac
shift

profile=auto
port=8000
host=127.0.0.1
mem=""
max_len=""
drop=0
while [ $# -gt 0 ]; do
    case "$1" in
        --port)        port="${2:?--port needs a value}"; shift 2 ;;
        --host)        host="${2:?--host needs a value}"; shift 2 ;;
        --mem)         mem="${2:?--mem needs a value}"; shift 2 ;;
        --max-len)     max_len="${2:?--max-len needs a value}"; shift 2 ;;
        --profile)     profile="${2:?--profile needs a value}"; shift 2 ;;
        --env)         ENV_NAME="${2:?--env needs a value}"; shift 2 ;;
        --drop-caches) drop=1; shift ;;
        --)            shift; break ;;
        *) die "unknown argument: $1 (vLLM flags go after --)" ;;
    esac
done

load_profile "$profile"
[ "$(uname -m)" = "${PROFILE_ARCH:-$(uname -m)}" ] || warn "profile $PROFILE is meant for $PROFILE_ARCH, this machine is $(uname -m)"
find_conda || die "conda not found: run setup/install.sh first"
set +u
conda activate "$ENV_NAME" || die "no env $ENV_NAME: run setup/install.sh --env $ENV_NAME"
set -u
ENV_PY="$(command -v python)"

if [ "$drop" = 1 ] || [ "$FREE_PAGE_CACHE" = 1 ]; then
    free_page_cache
fi

args=(serve "$model" --host "$host" --port "$port"
      --gpu-memory-utilization "${mem:-$GPU_MEMORY_UTILIZATION}")
[ -z "$max_len" ] || args+=(--max-model-len "$max_len")
log "vllm ${args[*]}${*:+ $*}"
exec vllm "${args[@]}" "$@"
