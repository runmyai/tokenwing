#!/usr/bin/env python3
"""Builds and packages Windows release archives for TokenWing (llama-server + autotuner).

x64 archives carry the CPU code for every CPU level (picked at start), HTTPS built in (BoringSSL) and our web page, built once
by the workflow into tools/ui/dist (a build without it fails). --smoke starts the packed server from its own folder, checks
the page and writes a short reply with the model in SMOKE_MODEL."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

PE_MACHINE = {"x64": 0x8664, "arm64": 0xAA64}
VC_CRT_DLLS = (
    "msvcp140.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll",
)
ROCM_ARCH_DEFAULT = "gfx1030;gfx1100;gfx1101;gfx1102;gfx1151;gfx1150;gfx1200;gfx1201"
# what the AMD library and the ROCm libraries it loads need from the ROCm SDK; none of them loads LLVM's OpenMP runtime
# (libomp140): OpenMP comes only through ggml-base.dll of the main MSVC build, which uses vcomp140.dll (staged with the VC runtime)
ROCM_RUNTIME_DLLS = (
    "amd_comgr.dll",
    "amdhip64_7.dll",
    "hipblas.dll",
    "libhipblaslt.dll",
    "origami.dll",
    "rocblas.dll",
    "rocm_kpack.dll",
    "rocsolver.dll",
)


def find_vs_installation() -> Path:
    pf86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    vswhere = Path(pf86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if vswhere.is_file():
        out = subprocess.check_output(
            [str(vswhere), "-products", "*", "-sort", "-format", "json"],
            text=True,
            encoding="utf-8",
        )
        for item in json.loads(out or "[]"):
            p = Path(item.get("installationPath", ""))
            if (p / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat").is_file():
                return p
    vs_env = os.environ.get("VSINSTALLDIR")
    if vs_env:
        p = Path(vs_env.rstrip("\\/"))
        if (p / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat").is_file():
            return p
    raise FileNotFoundError("Could not locate Visual Studio with VC\\Auxiliary\\Build\\vcvarsall.bat")


def collect_vc_redist_dlls(vs_dir: Path, include_omp: bool) -> list[Path]:
    roots: list[Path] = []
    if os.environ.get("VCToolsRedistDir"):
        roots.append(Path(os.environ["VCToolsRedistDir"]))
    base = vs_dir / "VC" / "Redist" / "MSVC"
    if base.is_dir():
        roots.extend(sorted(base.iterdir(), reverse=True))
    found: dict[str, Path] = {}
    for ver in roots:
        for d in sorted((ver / "x64").glob("Microsoft.VC*.CRT"), reverse=True):
            for name in VC_CRT_DLLS:
                if name not in found and (d / name).is_file():
                    found[name] = d / name
        if include_omp:
            for d in sorted((ver / "x64").glob("Microsoft.VC*.OpenMP"), reverse=True):
                omp = d / "vcomp140.dll"
                if "vcomp140.dll" not in found and omp.is_file():
                    found["vcomp140.dll"] = omp
    return [found[k] for k in sorted(found)]


def find_rocm_paths() -> tuple[Path, Path, Path, Path, list[Path]]:
    subprocess.run([sys.executable, "-m", "rocm_sdk", "init"], check=True)
    root = Path(subprocess.check_output([sys.executable, "-m", "rocm_sdk", "path", "--root"], text=True).strip())
    cmake_dir = Path(subprocess.check_output([sys.executable, "-m", "rocm_sdk", "path", "--cmake"], text=True).strip())
    clang_c = next(
        (p for p in (root / "lib" / "llvm" / "bin" / "clang.exe", root / "bin" / "amdclang.exe", root / "bin" / "clang.exe") if p.is_file()),
        None,
    )
    clang_cxx = next(
        (p for p in (root / "lib" / "llvm" / "bin" / "clang++.exe", root / "bin" / "amdclang++.exe", root / "bin" / "clang++.exe") if p.is_file()),
        None,
    )
    if not clang_c or not clang_cxx:
        raise FileNotFoundError(f"Could not locate Clang in ROCm devel root: {root}")
    rocm_sdk_mod = Path(subprocess.check_output([sys.executable, "-c", "import rocm_sdk, pathlib; print(pathlib.Path(rocm_sdk.__file__).parent.parent)"], text=True).strip())
    search_dirs = [root / "bin", root / "lib" / "llvm" / "bin"]
    search_dirs.extend(p / "bin" for p in sorted(rocm_sdk_mod.glob("_rocm_sdk*")) if (p / "bin").is_dir())
    return root, cmake_dir, clang_c, clang_cxx, search_dirs


def run_in_env(vcvarsall: Path | None, vcvars_arg: str, cmds: list[str], cwd: Path, extra_env: dict[str, str]) -> None:
    env = os.environ.copy()
    env.update(extra_env)
    bat = cwd / "_build_step.cmd"
    lines = ["@echo off"]
    if vcvarsall is not None:
        lines.append(f'call "{vcvarsall}" {vcvars_arg} >nul')
        lines.append("if errorlevel 1 exit /b %errorlevel%")
    for c in cmds:
        lines.append(c)
        lines.append("if errorlevel 1 exit /b %errorlevel%")
    bat.write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")
    try:
        subprocess.run(["cmd.exe", "/d", "/c", str(bat)], cwd=str(cwd), env=env, check=True)
    finally:
        bat.unlink(missing_ok=True)


def verify_pe_machine(path: Path, arch: str) -> None:
    data = path.read_bytes()
    if data[:2] != b"MZ":
        raise ValueError(f"Invalid DOS header in {path}")
    pe_off = struct.unpack_from("<I", data, 0x3C)[0]
    machine = struct.unpack_from("<H", data, pe_off + 4)[0]
    expected = PE_MACHINE[arch]
    if machine != expected:
        raise ValueError(f"PE machine mismatch for {path}: 0x{machine:04x} != 0x{expected:04x}")


def stage_rocm_blas_libraries(stage: Path, rocm_dirs: list[Path], targets: str) -> None:
    target_list = [t.strip().split(":")[0] for t in targets.split(";") if t.strip()]
    gfx_re = re.compile(r"gfx[0-9a-z]+")
    for rel_sub in (("rocblas", "library"), ("hipblaslt", "library")):
        dst_root = stage.joinpath(*rel_sub)
        seen_src: set[Path] = set()
        for d in rocm_dirs:
            src_root = d.joinpath(*rel_sub)
            if not src_root.is_dir():
                continue
            for src in sorted(src_root.rglob("*")):
                if not src.is_file():
                    continue
                real = src.resolve()
                if real in seen_src:
                    continue
                rel = src.relative_to(src_root)
                found_archs = gfx_re.findall(rel.as_posix())
                if found_archs and not any(a in target_list for a in found_archs):
                    continue
                seen_src.add(real)
                dst = dst_root / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(real, dst)

    rb_dir = stage / "rocblas" / "library"
    lt_dir = stage / "hipblaslt" / "library"
    for arch_name in target_list:
        if not rb_dir.is_dir() or not any(rb_dir.glob(f"*{arch_name}*")):
            raise FileNotFoundError(
                f"Missing rocblas/library Tensile resources for {arch_name} (install rocm[device-{arch_name}])"
            )
        if arch_name.startswith(("gfx11", "gfx12")):
            arch_lt = lt_dir / arch_name
            if not arch_lt.is_dir() or not any(arch_lt.iterdir()):
                raise FileNotFoundError(
                    f"Missing hipblaslt/library/{arch_name} Tensile resources (install rocm[device-{arch_name}])"
                )


def stage_dlls(
    backend: str,
    stage: Path,
    build_bin: Path,
    vs_dir: Path | None,
    cuda_root: Path | None,
    mingw_root: Path | None,
    rocm_dirs: list[Path],
    rocm_targets: str = ROCM_ARCH_DEFAULT,
) -> None:
    for built_dll in sorted(build_bin.glob("*.dll")):
        # only the server and the autotuner are shipped: the other tools' libraries stay out
        if re.fullmatch(r"llama-.*-impl\.dll", built_dll.name) and built_dll.name != "llama-server-impl.dll":
            continue
        shutil.copy2(built_dll, stage / built_dll.name)

    if backend == "arm64-cpu":
        assert mingw_root is not None
        omp = next(mingw_root.rglob("aarch64-w64-mingw32/bin/libomp.dll"), None) or next(mingw_root.rglob("libomp.dll"), None)
        if not omp:
            raise FileNotFoundError(f"Could not find libomp.dll under {mingw_root}")
        shutil.copy2(omp, stage / "libomp.dll")
        return

    assert vs_dir is not None
    for dll in collect_vc_redist_dlls(vs_dir, include_omp=True):
        shutil.copy2(dll, stage / dll.name)

    if backend in ("cuda12", "cuda13"):
        assert cuda_root is not None
        major = "12" if backend == "cuda12" else "13"
        for name in (f"cudart64_{major}.dll", f"cublas64_{major}.dll", f"cublasLt64_{major}.dll"):
            # a full CUDA 13 toolkit holds both bin\x64 and bin\arm64; the CUDA packages are x64
            matches = sorted(p for p in (cuda_root / "bin").rglob(name) if "arm64" not in p.parts)
            if not matches:
                raise FileNotFoundError(f"Missing {name} under {cuda_root / 'bin'}")
            shutil.copy2(matches[0], stage / name)
    elif backend == "hip":
        for name in ROCM_RUNTIME_DLLS:
            hit = next((d / name for d in rocm_dirs if (d / name).is_file()), None)
            if not hit:
                raise FileNotFoundError(f"Missing ROCm runtime DLL {name} in {rocm_dirs}")
            shutil.copy2(hit, stage / name)
        stage_rocm_blas_libraries(stage, rocm_dirs, rocm_targets)


def build_zip(stage_parent: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in sorted(stage_parent.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(stage_parent).as_posix())


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and package TokenWing Windows release archives.")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--backend", required=True, choices=("cpu", "cuda12", "cuda13", "vulkan", "hip", "arm64-cpu"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--cuda-root", type=Path, default=None)
    parser.add_argument("--vulkan-sdk", type=Path, default=None)
    parser.add_argument("--mingw-root", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--check-env", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    out_dir = (args.out_dir or (repo / "dist")).resolve()
    arch = "arm64" if args.backend == "arm64-cpu" else "x64"
    vs_dir = None if args.backend == "arm64-cpu" else find_vs_installation()
    if args.check_env:
        dlls = [] if vs_dir is None else [d.name for d in collect_vc_redist_dlls(vs_dir, include_omp=True)]
        print(f"variant={args.variant} backend={args.backend} arch={arch} vs={vs_dir} vc_dlls={dlls}")
        return 0

    with tempfile.TemporaryDirectory(prefix=f"tw-{args.variant}-") as tmp:
        work = Path(tmp)
        bld = work / "build"
        if not (repo / "tools" / "ui" / "dist" / "index.html").is_file():
            raise FileNotFoundError("No web page in tools/ui/dist: build it first (npm ci && npm run build in tools/ui)")
        # every CPU level as a module loaded at start; the ARM64 cross build stays one static program
        modules = args.backend != "arm64-cpu"
        cmake_args = [
            f'-S "{repo.as_posix()}"',
            f'-B "{bld.as_posix()}"',
            "-G Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DGGML_NATIVE=OFF",
            "-DBUILD_SHARED_LIBS=ON" if modules else "-DBUILD_SHARED_LIBS=OFF",
            *(["-DGGML_BACKEND_DL=ON", "-DGGML_CPU_ALL_VARIANTS=ON"] if modules else []),
            "-DGGML_CUDA_NCCL=OFF",
            "-DLLAMA_BUILD_TESTS=OFF",
            "-DLLAMA_BUILD_EXAMPLES=OFF",
            "-DLLAMA_BUILD_APP=OFF",
            "-DLLAMA_BUILD_SERVER=ON",
            "-DLLAMA_BUILD_TOOLS=ON",
            "-DLLAMA_BUILD_BORINGSSL=ON",
            "-DLLAMA_BUILD_UI=ON",
            "-DLLAMA_USE_PREBUILT_UI=OFF",
        ]
        extra_env: dict[str, str] = {}
        rocm_dirs: list[Path] = []
        rocm_targets = os.environ.get("AMDGPU_TARGETS", ROCM_ARCH_DEFAULT)
        vcvarsall = None if vs_dir is None else vs_dir / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"

        if args.backend == "cpu":
            pass
        elif args.backend in ("cuda12", "cuda13"):
            if not args.cuda_root:
                raise ValueError("--cuda-root is required for CUDA builds")
            cr = args.cuda_root.resolve()
            cmake_args.extend([
                "-DGGML_CUDA=ON",
                f'-DCMAKE_CUDA_COMPILER="{(cr / "bin" / "nvcc.exe").as_posix()}"',
                f'-DCUDAToolkit_ROOT="{cr.as_posix()}"',
                "-DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler",
            ])
            extra_env["PATH"] = os.pathsep.join(
                [str(cr / "bin"), str(cr / "bin" / "x64"), str(cr / "nvvm" / "bin"), str(cr / "nvvm" / "bin" / "x64"), os.environ.get("PATH", "")]
            )
        elif args.backend == "vulkan":
            vk = (args.vulkan_sdk or Path(os.environ.get("VULKAN_SDK", ""))).resolve()
            if not vk.is_dir():
                raise ValueError("Valid --vulkan-sdk or VULKAN_SDK env var is required")
            cmake_args.append("-DGGML_VULKAN=ON")
            extra_env["VULKAN_SDK"] = str(vk)
            extra_env["PATH"] = f"{vk / 'Bin'}{os.pathsep}{os.environ.get('PATH', '')}"
        elif args.backend == "hip":
            # as llama.cpp's own Windows releases: the server, the autotuner and the CPU modules are built with Microsoft's
            # compiler (this build), and only the GPU module ggml-hip.dll with AMD's clang (the second build below); the
            # modules talk through ggml's C interface, so they work together
            pass
        elif args.backend == "arm64-cpu":
            if not args.mingw_root:
                raise ValueError("--mingw-root is required for arm64-cpu builds")
            mr = args.mingw_root.resolve()
            cmake_args.extend([
                "-DGGML_STATIC=ON",
                "-DCMAKE_SYSTEM_NAME=Windows",
                "-DCMAKE_SYSTEM_PROCESSOR=ARM64",
                f'-DCMAKE_C_COMPILER="{(mr / "bin" / "aarch64-w64-mingw32-gcc.exe").as_posix()}"',
                f'-DCMAKE_CXX_COMPILER="{(mr / "bin" / "aarch64-w64-mingw32-g++.exe").as_posix()}"',
            ])
            extra_env["PATH"] = f"{mr / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"

        jobs = max(1, os.cpu_count() or 4)
        run_in_env(
            vcvarsall,
            "x64",
            # every target: the backend modules are loaded at start, so no target needs them
            [f"cmake {' '.join(cmake_args)}", f'cmake --build "{bld.as_posix()}" -j {jobs}'],
            work,
            extra_env,
        )

        hip_dll = None
        if args.backend == "hip":
            root, cmake_dir, clang_c, clang_cxx, rocm_dirs = find_rocm_paths()
            bld_hip = work / "build-hip"
            hip_args = [
                f'-S "{repo.as_posix()}"', f'-B "{bld_hip.as_posix()}"', "-G Ninja", "-DCMAKE_BUILD_TYPE=Release",
                "-DGGML_NATIVE=OFF", "-DBUILD_SHARED_LIBS=ON", "-DGGML_BACKEND_DL=ON", "-DGGML_CPU=OFF", "-DGGML_HIP=ON",
                "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF", "-DLLAMA_BUILD_TOOLS=OFF", "-DLLAMA_BUILD_SERVER=OFF",
                "-DLLAMA_BUILD_UI=OFF", "-DLLAMA_OPENSSL=OFF",
                f'-DAMDGPU_TARGETS="{rocm_targets}"', f'-DGPU_TARGETS="{rocm_targets}"',
                f'-DCMAKE_C_COMPILER="{clang_c.as_posix()}"', f'-DCMAKE_CXX_COMPILER="{clang_cxx.as_posix()}"',
                f'-DCMAKE_PREFIX_PATH="{root.as_posix()};{cmake_dir.as_posix()};{(root / "lib" / "llvm").as_posix()}"',
                # the GPU code's compile ignores Windows' dllimport on every declaration: one warning each, millions in all
                '"-DCMAKE_HIP_FLAGS=-Wno-ignored-attributes"',
            ]
            hip_env = {"HIP_PATH": str(root), "PATH": os.pathsep.join([str(d) for d in rocm_dirs] + [os.environ.get("PATH", "")])}
            run_in_env(vcvarsall, "x64", [f"cmake {' '.join(hip_args)}", f'cmake --build "{bld_hip.as_posix()}" -j {jobs} --target ggml-hip'],
                       work, hip_env)
            hip_dll = bld_hip / "bin" / "ggml-hip.dll"
            if not hip_dll.is_file():
                raise FileNotFoundError(f"The AMD build made no {hip_dll}")

        stage_parent = work / "stage"
        stage = stage_parent / "tokenwing"
        stage.mkdir(parents=True)
        build_bin = bld / "bin"
        if hip_dll is not None:
            shutil.copy2(hip_dll, stage / hip_dll.name)
        shutil.copy2(build_bin / "llama-server.exe", stage / "llama-server.exe")
        shutil.copy2(build_bin / "autotuner.exe", stage / "autotuner.exe")
        stage_dlls(args.backend, stage, build_bin, vs_dir, args.cuda_root, args.mingw_root, rocm_dirs, rocm_targets)
        shutil.rmtree(bld, ignore_errors=True)

        for pe in [*stage.glob("*.exe"), *stage.glob("*.dll")]:
            verify_pe_machine(pe, arch)

        if modules and not any(stage.glob("ggml-cpu-*.dll")):
            raise FileNotFoundError("No CPU modules in the archive")
        if args.smoke:
            subprocess.run([str(stage / "llama-server.exe"), "--version"], check=True)
            subprocess.run([str(stage / "autotuner.exe"), "--self-test"], check=True)
            model = os.environ.get("SMOKE_MODEL")
            if not model:
                raise ValueError("SMOKE_MODEL (a small GGUF) is needed for --smoke")
            subprocess.run([sys.executable, str(Path(__file__).with_name("smoke.py")), str(stage / "llama-server.exe"), model],
                           check=True)

        zip_path = out_dir / f"{args.variant}.zip"
        build_zip(stage_parent, zip_path)
        print(f"Built {zip_path} ({zip_path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
