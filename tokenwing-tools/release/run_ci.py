#!/usr/bin/env python3
"""Cross-platform CMake build and core CTest runner for TokenWing CI."""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import tempfile
from pathlib import Path

from build_windows import find_vs_installation, run_in_env

COMMON_TARGETS = (
    "llama-server",
    "autotuner",
    "test-cpu-track",
    "test-gguf",
    "test-alloc",
    "test-unicode",
    "test-sampling",
    "test-chat",
    "test-jinja",
    "test-arg-parser",
    "test-log",
)

COMMON_CTESTS = (
    "test-autotuner",
    "test-cpu-track",
    "test-gguf",
    "test-alloc",
    "test-unicode",
    "test-sampling",
    "test-chat",
    "test-jinja",
    "test-arg-parser",
    "test-log",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Configure, build, and run offline core tests for TokenWing.")
    parser.add_argument("--jobs", type=int, default=max(1, os.cpu_count() or 4))
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    is_windows = platform.system() == "Windows"
    targets = list(COMMON_TARGETS)
    ctests = list(COMMON_CTESTS)
    if not is_windows:
        # test-server-component.cpp line 355 fails MSVC static compilation on server_pipe<int>::write.
        targets.append("test-server-component")
        ctests.append("test-server-component")
    ctest_regex = f"^({'|'.join(ctests)})$"

    with tempfile.TemporaryDirectory(prefix="tw-ci-") as tmp:
        bld = Path(tmp) / "build"
        cmake_flags = [
            "-G",
            "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DGGML_NATIVE=OFF",
            "-DGGML_STATIC=ON",
            "-DBUILD_SHARED_LIBS=OFF",
            "-DGGML_CUDA_NCCL=OFF",
            "-DLLAMA_OPENSSL=OFF",
            "-DLLAMA_BUILD_TESTS=ON",
            "-DLLAMA_BUILD_SERVER=ON",
            "-DLLAMA_BUILD_TOOLS=ON",
            "-DLLAMA_BUILD_EXAMPLES=OFF",
            "-DLLAMA_BUILD_APP=OFF",
        ]
        if is_windows:
            vs = find_vs_installation()
            vcvarsall = vs / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"
            cfg_str = f'cmake -S "{repo.as_posix()}" -B "{bld.as_posix()}" {" ".join(cmake_flags)}'
            bld_str = f'cmake --build "{bld.as_posix()}" -j {args.jobs} --target {" ".join(targets)}'
            tst_str = f'ctest --test-dir "{bld.as_posix()}" --output-on-failure -R "{ctest_regex}"'
            run_in_env(vcvarsall, "x64", [cfg_str, bld_str, tst_str], Path(tmp), {})
            exe = bld / "bin" / "llama-server.exe"
        else:
            subprocess.run(["cmake", "-S", str(repo), "-B", str(bld), *cmake_flags], check=True)
            subprocess.run(["cmake", "--build", str(bld), "-j", str(args.jobs), "--target", *targets], check=True)
            subprocess.run(["ctest", "--test-dir", str(bld), "--output-on-failure", "-R", ctest_regex], check=True)
            exe = bld / "bin" / "llama-server"
        subprocess.run([str(exe), "--version"], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
