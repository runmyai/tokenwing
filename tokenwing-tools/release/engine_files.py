#!/usr/bin/env python3
"""The file list of an engine release, for updates that download only the files that changed.

    python3 engine_files.py <folder with the packages> <tag> [--upload <hf repo>]

Reads every tokenwing-*.tar.gz / .zip in the folder and writes `engine-files.json` there: per build (the package name without
`tokenwing-` and its extension), every file of the package's `tokenwing/` folder with its size, SHA-256 and whether it is a
program, and every link with its target. With --upload, each file whose SHA-256 the Hugging Face repo does not have yet is
uploaded to it, named by its SHA-256 (needs huggingface_hub and a token with write access, HF_TOKEN). The release's SHA256SUMS
covers engine-files.json, and SHA256SUMS itself is signed, so every file the app downloads is checked against a signed checksum.
"""

import hashlib
import json
import os
import stat
import sys
import tarfile
import tempfile
import zipfile


def sha256(data_or_file):
    h = hashlib.sha256()
    if isinstance(data_or_file, bytes):
        h.update(data_or_file)
    else:
        for block in iter(lambda: data_or_file.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def entries(path):
    """(path inside tokenwing/, kind, data or link target, is program) for each member of a package."""
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                if info.is_dir() or not info.filename.startswith("tokenwing/"):
                    continue
                yield info.filename[len("tokenwing/"):], "file", z.read(info), info.filename.endswith(".exe")
    else:
        with tarfile.open(path, "r:gz") as t:
            for m in t.getmembers():
                if not m.name.startswith("tokenwing/") or m.isdir():
                    continue
                name = m.name[len("tokenwing/"):]
                if m.issym():
                    yield name, "link", m.linkname, False
                elif m.isfile():
                    yield name, "file", t.extractfile(m).read(), bool(m.mode & stat.S_IXUSR) and not name.endswith(".so") and ".so." not in name


def main():
    folder, tag = sys.argv[1], sys.argv[2]
    upload = sys.argv[sys.argv.index("--upload") + 1] if "--upload" in sys.argv else None
    builds, blobs = {}, set()
    # each distinct file once, named by its SHA-256, for the upload
    stash = tempfile.TemporaryDirectory(dir=folder) if upload else None
    for pkg in sorted(os.listdir(folder)):
        if not pkg.startswith("tokenwing-") or not pkg.endswith((".tar.gz", ".zip")):
            continue
        build = pkg[len("tokenwing-"):].removesuffix(".tar.gz").removesuffix(".zip")
        files = []
        for name, kind, data, program in entries(os.path.join(folder, pkg)):
            if kind == "link":
                files.append({"path": name, "link": data})
                continue
            digest = sha256(data)
            if stash and digest not in blobs:
                with open(os.path.join(stash.name, digest), "wb") as f:
                    f.write(data)
            blobs.add(digest)
            entry = {"path": name, "size": len(data), "sha256": digest}
            if program:
                entry["program"] = True
            files.append(entry)
        if not any(f["path"] in ("llama-server", "llama-server.exe") for f in files):
            sys.exit(f"{pkg}: no llama-server in it")
        builds[build] = sorted(files, key=lambda f: f["path"])
    if not builds:
        sys.exit(f"no packages in {folder}")
    out = {"format": 1, "version": tag, "store": f"https://huggingface.co/{upload or 'runmyai/tokenwing-files'}/resolve/main/",
           "builds": builds}
    with open(os.path.join(folder, "engine-files.json"), "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
        f.write("\n")
    size = sum(f.get("size", 0) for fs in builds.values() for f in fs)
    print(f"engine-files.json: {len(builds)} builds, {len(blobs)} distinct files ({size / 2**30:.1f} GiB in all packages)")
    if upload:
        with stash:
            upload_blobs(stash.name, upload, blobs, tag)


def upload_blobs(stash, repo, blobs, tag):
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    missing = sorted(blobs - set(api.list_repo_files(repo)))
    print(f"{repo}: {len(blobs) - len(missing)} files already there, {len(missing)} to upload")
    if missing:
        ops = [CommitOperationAdd(path_in_repo=d, path_or_fileobj=os.path.join(stash, d)) for d in missing]
        api.create_commit(repo, ops, commit_message=f"engine files of {tag}")
        print(f"{repo}: uploaded {len(missing)} files")


if __name__ == "__main__":
    main()
