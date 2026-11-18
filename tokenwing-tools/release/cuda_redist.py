#!/usr/bin/env python3
"""Download and verify NVIDIA CUDA redistributable components into a toolkit root."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

BASE_URL = "https://developer.download.nvidia.com/compute/cuda/redist"

# Union of CUDA 12.x and CUDA 13.x redistributable component names (13.x split
# cuda_cccl -> cccl and moved nvvm/crt/ptxcompiler/tileiras out of cuda_nvcc).
COMPONENTS = (
    "cuda_cudart",
    "cuda_nvcc",
    "cuda_cccl",
    "cccl",
    "cuda_crt",
    "cuda_nvrtc",
    "cuda_nvtx",
    "cuda_nvml_dev",
    "libcublas",
    "libnvvm",
    "libnvptxcompiler",
    "libnvjitlink",
    "cuda_tileiras",
)


def fetch_manifest(version: str) -> dict:
    url = f"{BASE_URL}/redistrib_{version}.json"
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.load(resp)


def resolve_packages(manifest: dict, platform: str) -> list[tuple[str, str, str]]:
    pkgs: list[tuple[str, str, str]] = []
    for name in COMPONENTS:
        entry = manifest.get(name)
        if not isinstance(entry, dict) or platform not in entry:
            continue
        meta = entry[platform]
        rel = meta["relative_path"]
        sha = meta["sha256"].lower()
        pkgs.append((name, f"{BASE_URL}/{rel}", sha))
    if not pkgs:
        raise ValueError(f"No CUDA packages resolved for platform={platform}")
    return pkgs


def download_verified(url: str, dst: Path, expected_sha256: str) -> None:
    h = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=300) as resp, dst.open("wb") as out:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
            out.write(chunk)
    actual = h.hexdigest().lower()
    if actual != expected_sha256:
        raise ValueError(f"SHA-256 mismatch for {url}: got {actual}, expected {expected_sha256}")


def extract_zip_strip_root(zip_path: Path, dest_root: Path) -> int:
    count = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            parts = Path(info.filename).parts
            rel = Path(*parts[1:]) if len(parts) > 1 else Path(parts[0])
            target = (dest_root / rel).resolve()
            if dest_root.resolve() not in target.parents and target != dest_root.resolve():
                raise ValueError(f"Unsafe zip path: {info.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Install NVIDIA CUDA toolkit subset from official redist JSON.")
    parser.add_argument("--version", required=True, help="CUDA redist version (e.g. 12.8.1 or 13.4.1)")
    parser.add_argument("--platform", default="windows-x86_64")
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true", help="Print resolved URLs and SHA-256 sums without downloading.")
    args = parser.parse_args()

    manifest = fetch_manifest(args.version)
    pkgs = resolve_packages(manifest, args.platform)
    for name, url, sha in pkgs:
        print(f"Resolved {name}: {url} sha256={sha[:16]}...")

    if args.check_only:
        return 0

    dest = args.dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cuda-redist-") as tmp:
        tmp_dir = Path(tmp)
        for name, url, sha in pkgs:
            archive = tmp_dir / Path(url).name
            print(f"Downloading {name}...")
            download_verified(url, archive, sha)
            if archive.suffix != ".zip":
                raise ValueError(f"Unsupported archive format for Windows redist: {archive.name}")
            n = extract_zip_strip_root(archive, dest)
            print(f"  Extracted {n} files into {dest}")
            archive.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
