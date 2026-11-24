#!/usr/bin/env bash
# Sourced by install.sh and serve.sh; why these pins and checks exist: README.md next to this file.
# shellcheck disable=SC2034  # variables are read by the sourcing scripts

set -euo pipefail

SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_HOME="${CONDA_HOME:-$HOME/miniforge3}"
ENV_NAME="${ENV_NAME:-bench-vllm}"

# One tested stack for every profile, so GB10 and RTX numbers compare like for like
TORCH_VERSION="${TORCH_VERSION:-2.13.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.11.0}"
VLLM_VERSION="${VLLM_VERSION:-0.30.0}"
FLASHINFER_VERSION="${FLASHINFER_VERSION:-0.6.18.post1}"
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-5.14.1}"

TORCH_BACKEND="${TORCH_BACKEND:-cu130}"
TORCH_INDEX="https://download.pytorch.org/whl/${TORCH_BACKEND}"
# flashinfer-cubin is not published on PyPI since 0.6.14
FLASHINFER_INDEX="https://flashinfer.ai/whl/"
# Wheels only: building these from source on aarch64 takes hours and usually fails
ONLY_BINARY="torch,torchaudio,torchvision,vllm,tilelang,flashinfer-python,xgrammar,triton,numpy,av"

log()  { printf '[setup] %s\n' "$*"; }
warn() { printf '[setup] WARNING: %s\n' "$*" >&2; }
die()  { printf '[setup] ERROR: %s\n' "$*" >&2; exit 1; }

# --help text is the header comment of the calling script
usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; }

make_tmp() {
    SETUP_TMP="$(mktemp -d -t bench-setup.XXXXXX)"
    trap 'rm -rf "$SETUP_TMP"' EXIT
}

detect_profile() {
    case "$(uname -m)" in
        aarch64) echo gb10 ;;
        x86_64)  echo rtx ;;
        *) die "no profile for $(uname -m); pass --profile explicitly" ;;
    esac
}

load_profile() {
    PROFILE="$1"
    [ "$PROFILE" != auto ] || PROFILE="$(detect_profile)"
    local file="$SETUP_DIR/profiles/$PROFILE.env"
    [ -f "$file" ] || die "unknown profile '$PROFILE', see $SETUP_DIR/profiles/"
    # shellcheck disable=SC1090
    . "$file"
    log "profile: $PROFILE"
}

check_gpu() {
    local arch
    arch="$(uname -m)"
    [ "$arch" = "${PROFILE_ARCH:-$arch}" ] \
        || die "profile $PROFILE expects $PROFILE_ARCH, this machine is $arch; pick another --profile"

    command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found: the NVIDIA driver is not installed"
    local name driver cap mem
    IFS=',' read -r name driver cap mem < <(nvidia-smi \
        --query-gpu=name,driver_version,compute_cap,memory.total \
        --format=csv,noheader,nounits | head -n1 | sed 's/, */,/g')
    log "GPU: $name, driver $driver, compute capability $cap, memory $mem MiB"
    [ "${driver%%.*}" -ge 580 ] \
        || die "driver $driver: CUDA 13 wheels (torch $TORCH_VERSION, vLLM $VLLM_VERSION) need R580 or newer"
    if [ -n "${PROFILE_CAP:-}" ] && [ "$cap" != "$PROFILE_CAP" ]; then
        warn "compute capability $cap, profile $PROFILE was tuned for $PROFILE_CAP"
    fi
    # unified-memory GPUs report N/A here, so compare only numbers
    if [ -n "${PROFILE_MIN_VRAM_MIB:-}" ] && [[ "$mem" =~ ^[0-9]+$ ]] \
        && [ "$mem" -lt "$PROFILE_MIN_VRAM_MIB" ]; then
        warn "only $mem MiB of GPU memory: an 8B bf16 LLM plus an ASR model may not fit, consider smaller models"
    fi
}

# FlashInfer JIT-compiles sm_121 kernels with nvcc, and nvcc older than 12.9 does not know sm_121
check_cuda_toolkit() {
    local fix="apt-get install -y cuda-toolkit-13-0 (or use the nvidia/cuda:13.0.3-devel-ubuntu24.04 image)"
    local home nvcc ver
    home="$(printf '%s\n' /usr/local/cuda-13* | sort -V | tail -n1)"
    [ -d "$home" ] || home="${CUDA_HOME:-/usr/local/cuda}"
    nvcc="$home/bin/nvcc"
    if [ ! -x "$nvcc" ]; then
        warn "nvcc (CUDA toolkit) not found: vLLM will fail at startup if it has to build a kernel. $fix"
        return 0
    fi
    ver="$("$nvcc" --version | sed -n 's/.*release \([0-9.]*\),.*/\1/p')"
    if [ "$(printf '%s\n%s\n' 12.9 "$ver" | sort -V | head -n1)" != 12.9 ]; then
        warn "CUDA toolkit $ver ($nvcc) is older than 12.9 and cannot build sm_121 FlashInfer kernels. $fix"
    else
        log "CUDA toolkit for kernel builds: $ver ($nvcc)"
    fi
}

# vLLM's aarch64 deps (tilelang) ship only manylinux_2_34 wheels; x86 llguidance needs 2.35
check_glibc() {
    local have
    have="$(getconf GNU_LIBC_VERSION | awk '{ print $2 }')"
    [ "$(printf '%s\n%s\n' "$1" "$have" | sort -V | head -n1)" = "$1" ] \
        || die "glibc $have, need $1 or newer (Ubuntu 22.04 has 2.35, 24.04 has 2.39)"
}

check_disk() {
    local free_gb
    free_gb="$(df -BG --output=avail "$HOME" | tail -n1 | tr -dc '0-9')"
    [ "$free_gb" -ge "$1" ] || warn "$free_gb GB free in $HOME, the env needs about $1 GB plus model weights"
}

# In a non-interactive shell conda is only available through conda.sh
find_conda() {
    local prefix
    for prefix in "$CONDA_HOME" "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3" \
                  /opt/conda /opt/miniforge3; do
        if [ -f "$prefix/etc/profile.d/conda.sh" ]; then
            set +u
            # shellcheck disable=SC1091
            . "$prefix/etc/profile.d/conda.sh"
            set -u
            return 0
        fi
    done
    if command -v conda >/dev/null 2>&1; then
        set +u
        # shellcheck disable=SC1091
        . "$(conda info --base)/etc/profile.d/conda.sh"
        set -u
        return 0
    fi
    return 1
}

# Miniforge, not Miniconda: the defaults channel needs an Anaconda ToS accept that breaks non-interactive create
install_miniforge() {
    command -v curl >/dev/null 2>&1 || die "curl is required: apt-get install -y curl"
    local installer
    installer="Miniforge3-Linux-$(uname -m).sh"
    log "conda not found, installing Miniforge into $CONDA_HOME"
    curl -fsSL "https://github.com/conda-forge/miniforge/releases/latest/download/$installer" \
        -o "$SETUP_TMP/$installer"
    bash "$SETUP_TMP/$installer" -b -p "$CONDA_HOME"
    find_conda || die "Miniforge installed but conda.sh is missing in $CONDA_HOME"
}

ensure_conda() {
    find_conda || install_miniforge
    log "conda: $(conda --version)"
}

env_prefix() {
    conda env list | awk -v n="$1" '$1 == n { print $NF }'
}

ensure_env() {
    local name="$1" python="$2" recreate="$3"
    shift 3
    if [ "$recreate" = 1 ] && [ -n "$(env_prefix "$name")" ]; then
        log "removing env $name (--recreate)"
        conda env remove -y -n "$name"
    fi
    if [ -n "$(env_prefix "$name")" ]; then
        log "env $name exists, bringing it to the pinned versions"
        [ $# -eq 0 ] || conda install -y -n "$name" --override-channels -c conda-forge "$@"
    else
        log "creating env $name (Python $python, conda-forge only)"
        conda create -y -n "$name" --override-channels -c conda-forge "python=$python" pip "$@"
    fi
    ENV_PREFIX="$(env_prefix "$name")"
    ENV_PY="$ENV_PREFIX/bin/python"
    [ -x "$ENV_PY" ] || die "env interpreter not found: $ENV_PY"

    local actual
    actual="$("$ENV_PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
    [ "$actual" = "$python" ] || die "env $name has Python $actual, need $python: rerun with --recreate"
    # old pip fails on torch wheels (pypa/pip#12038)
    "$ENV_PY" -m pip install --quiet --upgrade pip
}

# One pip call: constraint pins join a single resolve, so torch is never swapped silently later
pip_install() {
    "$ENV_PY" -m pip install --only-binary "$ONLY_BINARY" --extra-index-url "$TORCH_INDEX" "$@"
}

check_pip() {
    local out real
    out="$("$ENV_PY" -m pip check 2>&1)" && return 0
    # some aarch64 nvidia-* wheels carry an x86 tag in metadata; check_torch proves they work
    real="$(grep -v 'is not supported on this platform' <<<"$out" || true)"
    if [ -n "$real" ]; then
        printf '%s\n' "$out" >&2
        die "pip check found incompatible dependencies (see above)"
    fi
    warn "pip check: platform tag mismatch in metadata of $(cut -d' ' -f1 <<<"$out" | tr '\n' ' ')- ignored"
}

check_torch() {
    "$ENV_PY" - <<'PY' || die "torch in the env does not work with the GPU (see above)"
import sys

import torch

print(f"torch {torch.__version__}, CUDA {torch.version.cuda}")
if torch.version.cuda is None or not torch.cuda.is_available():
    sys.exit("torch does not see the GPU: CPU-only build (needs the cu130 index) or no driver")
major, minor = torch.cuda.get_device_capability(0)
archs = torch.cuda.get_arch_list()
print(f"GPU {torch.cuda.get_device_name(0)}, capability {major}.{minor}, kernels: {' '.join(archs)}")
if not torch.version.cuda.startswith("13"):
    print(f"WARNING: CUDA {torch.version.cuda}, the pinned stack expects 13.x")
if not any(a.startswith(f"sm_{major}") for a in archs):
    print(f"WARNING: no sm_{major}x kernels in this build, only PTX JIT is left")
a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
b = (a @ a).float()
torch.cuda.synchronize()
if not torch.isfinite(b).all():
    sys.exit("bf16 matmul on the GPU returned nan/inf")
print("bf16 matmul on the GPU: ok")
PY
}

# On unified memory (GB10) page cache counts as used GPU memory and vLLM refuses to start;
# unprivileged containers cannot drop_caches, so evict files via posix_fadvise instead
free_page_cache() {
    sync
    if { echo 3 > /proc/sys/vm/drop_caches; } 2>/dev/null \
        || { command -v sudo >/dev/null 2>&1 && sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; }; then
        log "page cache dropped"
        return 0
    fi
    log "no rights for drop_caches, evicting readable files from the page cache (about a minute)"
    "${ENV_PY:-python3}" - <<'PY' || warn "could not evict the page cache: vLLM may not find enough free memory"
import os
import stat
import sys

if not hasattr(os, "posix_fadvise"):
    sys.exit("posix_fadvise is Linux-only")
skip = {"/proc", "/sys", "/dev", "/run"}
count = 0
for folder, dirs, files in os.walk("/"):
    dirs[:] = [d for d in dirs if os.path.join(folder, d) not in skip]
    for name in files:
        path = os.path.join(folder, name)
        try:
            # regular files only: opening a named pipe would hang
            if not stat.S_ISREG(os.lstat(path).st_mode):
                continue
            fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        except OSError:
            continue
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            count += 1
        except OSError:
            pass
        finally:
            os.close(fd)
print(f"[setup] files evicted from the page cache: {count}")
PY
    log "memory now: $(free -g | awk '/^Mem/ { print $4 " GB free, " $6 " GB cache" }')"
}
