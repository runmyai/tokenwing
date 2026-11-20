#!/usr/bin/env bash
# Builds and packages a Linux release archive for TokenWing (llama-server + autotuner).
# Every archive carries the CPU code for every CPU level (picked at start: AVX-512 machines get their kernels), HTTPS built in
# (BoringSSL, no system OpenSSL needed) and our web page. The page is built once by the workflow into tools/ui/dist
# (npm run build); a build without it fails. --smoke starts the packed server from its own folder, checks the page and writes
# a short reply with the model in SMOKE_MODEL (any small GGUF).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

VARIANT=""
BACKEND=""
OUT_DIR="$REPO_DIR/dist"
WORK_DIR=""
JOBS="$(nproc 2>/dev/null || echo 4)"
RUN_SMOKE=0

while [ $# -gt 0 ]; do
  case "$1" in
    --variant) VARIANT="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --work-dir) WORK_DIR="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --smoke) RUN_SMOKE=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

: "${VARIANT:?--variant is required}"
: "${BACKEND:?--backend is required}"

if [ -z "$WORK_DIR" ]; then
  WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/tw-${VARIANT}-XXXXXX")"
  CLEAN_WORK=1
else
  mkdir -p "$WORK_DIR"
  CLEAN_WORK=0
fi
trap '[ "$CLEAN_WORK" -eq 1 ] && rm -rf "$WORK_DIR"' EXIT

ROCM_ARCH="${AMDGPU_TARGETS:-gfx908;gfx90a;gfx942;gfx1030;gfx1100;gfx1101;gfx1102;gfx1151;gfx1150;gfx1200;gfx1201}"

CMAKE_ARGS=(
  -S "$REPO_DIR"
  -B "$WORK_DIR/build"
  -G Ninja
  -DCMAKE_BUILD_TYPE=Release
  -DGGML_NATIVE=OFF
  -DBUILD_SHARED_LIBS=ON
  -DGGML_BACKEND_DL=ON
  -DGGML_CPU_ALL_VARIANTS=ON
  -DGGML_CUDA_NCCL=OFF
  -DLLAMA_BUILD_TESTS=OFF
  -DLLAMA_BUILD_EXAMPLES=OFF
  -DLLAMA_BUILD_APP=OFF
  -DLLAMA_BUILD_SERVER=ON
  -DLLAMA_BUILD_TOOLS=ON
  -DLLAMA_BUILD_BORINGSSL=ON
  -DLLAMA_BUILD_UI=ON
  -DLLAMA_USE_PREBUILT_UI=OFF
  '-DCMAKE_INSTALL_RPATH=$ORIGIN:$ORIGIN/..'
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
)

case "$BACKEND" in
  cpu)
    ;;
  cuda)
    CMAKE_ARGS+=(-DGGML_CUDA=ON)
    ;;
  vulkan)
    CMAKE_ARGS+=(-DGGML_VULKAN=ON)
    ;;
  rocm)
    export HIPCXX="${HIPCXX:-$(hipconfig -l)/clang}"
    export HIP_PATH="${HIP_PATH:-$(hipconfig -R)}"
    export AMDGPU_TARGETS="$ROCM_ARCH"
    CMAKE_ARGS+=(-DGGML_HIP=ON "-DAMDGPU_TARGETS=$ROCM_ARCH")
    ;;
  *)
    echo "Unsupported --backend '$BACKEND' (expected cpu, cuda, vulkan, rocm)" >&2
    exit 2
    ;;
esac

[ -f "$REPO_DIR/tools/ui/dist/index.html" ] || { echo "No web page in tools/ui/dist: build it first (npm ci && npm run build in tools/ui)" >&2; exit 1; }

cmake "${CMAKE_ARGS[@]}"
# everything: the backend modules (one CPU module per CPU level, the GPU module) are loaded at start, so no target needs them
cmake --build "$WORK_DIR/build" -j "$JOBS"

STAGE="$WORK_DIR/stage/tokenwing"
rm -rf "$WORK_DIR/stage"
mkdir -p "$STAGE"
cp -f "$WORK_DIR/build/bin/llama-server" "$STAGE/"
cp -f "$WORK_DIR/build/bin/autotuner" "$STAGE/"
cp -a "$WORK_DIR/build/bin/"*.so* "$STAGE/"
if [ "$BACKEND" = cuda ]; then
  # the CUDA runtime and cuBLAS the CUDA module needs (the user's driver supplies the rest)
  CUDA_LIB="${CUDA_HOME:-/usr/local/cuda}/lib64"
  cp -a "$CUDA_LIB"/libcudart.so* "$CUDA_LIB"/libcublas.so* "$CUDA_LIB"/libcublasLt.so* "$STAGE/"
fi
chmod +x "$STAGE/llama-server" "$STAGE/autotuner"
ls "$STAGE"/libggml-cpu-*.so >/dev/null || { echo "No CPU modules in the archive" >&2; exit 1; }

echo "=== Binary inspection ($VARIANT) ==="
file "$STAGE/llama-server" "$STAGE/autotuner"
ldd "$STAGE/llama-server" || true
ldd "$STAGE/autotuner" || true

if [ "$RUN_SMOKE" -eq 1 ]; then
  echo "=== Smoke test ($VARIANT) ==="
  "$STAGE/llama-server" --version
  "$STAGE/autotuner" --self-test
  : "${SMOKE_MODEL:?SMOKE_MODEL (a small GGUF) is needed for --smoke}"
  python3 "$SCRIPT_DIR/smoke.py" "$STAGE/llama-server" "$SMOKE_MODEL"
fi

mkdir -p "$OUT_DIR"
TAR_PATH="$OUT_DIR/${VARIANT}.tar.gz"
tar -C "$WORK_DIR/stage" -czf "$TAR_PATH" tokenwing
ls -lh "$TAR_PATH"
