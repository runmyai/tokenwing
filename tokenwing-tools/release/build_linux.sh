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

# the desktop and laptop GPU types (as the Windows package): the ROCm runtime is bundled, and the datacenter types' rocBLAS /
# hipBLASLt kernels alone would take the archive past GitHub's 2 GiB per release file
ROCM_ARCH="${AMDGPU_TARGETS:-gfx1030;gfx1100;gfx1101;gfx1102;gfx1150;gfx1151;gfx1200;gfx1201}"

CMAKE_ARGS=(
  -S "$REPO_DIR"
  -B "$WORK_DIR/build"
  -G Ninja
  -DCMAKE_BUILD_TYPE=Release
  -DGGML_NATIVE=OFF
  -DBUILD_SHARED_LIBS=ON
  -DGGML_BACKEND_DL=ON
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

# every x86 CPU level as its own module (AVX-512 machines get their kernels); ARM64 keeps one CPU build: the ARM levels need
# a compiler newer than the one this release is built with (Ubuntu 22.04, so the engine runs on older Linux too)
[ "$(uname -m)" = x86_64 ] && CMAKE_ARGS+=(-DGGML_CPU_ALL_VARIANTS=ON)

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
# only the server and the autotuner are shipped: the other tools' libraries stay out
find "$STAGE" -name 'libllama-*-impl.so*' ! -name 'libllama-server-impl.so*' -delete
if [ "$BACKEND" = cuda ]; then
  # the CUDA runtime and cuBLAS the CUDA module needs (the user's driver supplies the rest)
  CUDA_LIB="${CUDA_HOME:-/usr/local/cuda}/lib64"
  cp -a "$CUDA_LIB"/libcudart.so* "$CUDA_LIB"/libcublas.so* "$CUDA_LIB"/libcublasLt.so* "$STAGE/"
fi
if [ "$BACKEND" = rocm ]; then
  # the ROCm runtime the AMD module needs, so a user needs only the kernel's amdgpu driver: every library ldd finds under the
  # ROCm folder, plus comgr and the HSA runtime (HIP may open them by name), plus libnuma and libelf (not on every desktop)
  ROCM_REAL="$(readlink -f "$HIP_PATH")"
  ROCM_LIB="$ROCM_REAL/lib"
  seeds=("$STAGE/libggml-hip.so")
  rocm_libs=()
  for n in libamd_comgr.so.3 libhsa-runtime64.so.1; do
    [ -e "$ROCM_LIB/$n" ] || { echo "No $n in $ROCM_LIB" >&2; exit 1; }
    seeds+=("$ROCM_LIB/$n")
    rocm_libs+=("$n")
    cp -L "$ROCM_LIB/$n" "$STAGE/$n"
  done
  while read -r soname path; do
    case "$(readlink -f "$path")" in
      "$ROCM_REAL"/*) cp -L "$path" "$STAGE/$soname"; rocm_libs+=("$soname") ;;
      *) case "$soname" in libnuma.so.1|libelf.so.1) cp -L "$path" "$STAGE/$soname" ;; esac ;;
    esac
  done < <(ldd "${seeds[@]}" | awk '$2 == "=>" && $3 ~ /^\// {print $1, $3}' | sort -u)
  # each ROCm library looks for the others next to itself: one whose search path has no plain $ORIGIN (librocroller has only
  # $ORIGIN/../lib) would look in the system's ROCm
  for n in $(printf '%s\n' "${rocm_libs[@]}" | sort -u); do
    readelf -d "$STAGE/$n" | grep -E 'R(UN)?PATH' | grep -qE '\[(.*:)?\$ORIGIN(:.*)?\]' || patchelf --set-rpath '$ORIGIN' "$STAGE/$n"
  done
  # rocBLAS and hipBLASLt read their kernels from rocblas/library and hipblaslt/library next to their library: only the built
  # GPU types (and the files of no GPU type) are taken
  targets=" ${ROCM_ARCH//;/ } "
  for sub in rocblas hipblaslt; do
    mkdir -p "$STAGE/$sub/library"
    for f in "$ROCM_LIB/$sub/library"/*; do
      [ -f "$f" ] || continue
      keep=1
      for a in $(basename "$f" | grep -oE 'gfx[0-9a-z]+' | sort -u); do keep=0; [[ "$targets" == *" $a "* ]] && { keep=1; break; }; done
      [ "$keep" = 1 ] && cp -L "$f" "$STAGE/$sub/library/"
    done
  done
  for a in $targets; do
    ls "$STAGE/rocblas/library/"*"$a"* >/dev/null 2>&1 || { echo "No rocBLAS kernels for $a" >&2; exit 1; }
    case "$a" in gfx11*|gfx12*)
      ls "$STAGE/hipblaslt/library/"*"$a"* >/dev/null 2>&1 || { echo "No hipBLASLt kernels for $a" >&2; exit 1; } ;;
    esac
  done
  # every library of the package finds what it needs, and nothing in the system's ROCm
  missing="$(ldd "$STAGE"/*.so* 2>&1 | grep 'not found' || true)"
  [ -z "$missing" ] || { echo "Libraries not found:" >&2; echo "$missing" >&2; exit 1; }
  for p in $(ldd "$STAGE"/*.so* | awk '$2 == "=>" && $3 ~ /^\// {print $3}' | sort -u); do
    case "$(readlink -f "$p")" in "$ROCM_REAL"/*) echo "Taken from the system's ROCm, not the package: $p" >&2; exit 1 ;; esac
  done
  echo "=== ROCm runtime in the archive ==="
  ls -l "$STAGE"/lib*.so* | grep -vE 'libggml|libllama|libmtmd'
  du -sh "$STAGE/rocblas" "$STAGE/hipblaslt"
fi
chmod +x "$STAGE/llama-server" "$STAGE/autotuner"
# the build folder is not needed any more: its room goes to the archive (the ROCm one is over 1 GB)
[ "$CLEAN_WORK" -eq 1 ] && rm -rf "$WORK_DIR/build"
ls "$STAGE"/libggml-cpu*.so >/dev/null || { echo "No CPU module in the archive" >&2; exit 1; }

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
