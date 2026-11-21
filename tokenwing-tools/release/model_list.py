#!/usr/bin/env python3
"""Publish a new version of a model's files on Hugging Face, with the signed list the Run My AI app updates from.

    python3 model_list.py <repo> --version <v> --notes <text> [--upload FILE ...] [--key PEM] [--dry-run]

Uploads the given files to the repo (data files such as a new correction or VRAM list), then writes `runmyai.json`: the repo's
versions, oldest first, each with every model file's SHA-256 and size (README, LICENSE and the list itself left out), and signs it
(Ed25519, `runmyai.json.sig`, base64) with the update key (default ~/.config/runmyai/update-signing-key.pem). The app downloads
only the files whose checksum changed, and only after the signature verifies against its built-in public key. A version is added
only when a file changed. Needs huggingface_hub and a token with write access to the repo.
"""

import argparse
import base64
import hashlib
import json
import os
import subprocess
import sys
import tempfile

SKIP = {".gitattributes", "README.md", "LICENSE", "runmyai.json", "runmyai.json.sig"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("repo")
    ap.add_argument("--version", required=True)
    ap.add_argument("--notes", required=True)
    ap.add_argument("--upload", nargs="*", default=[])
    ap.add_argument("--key", default=os.path.expanduser("~/.config/runmyai/update-signing-key.pem"))
    ap.add_argument("--dry-run", action="store_true", help="write the list and its signature here, upload nothing")
    args = ap.parse_args()

    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    api = HfApi()
    if args.upload and not args.dry_run:
        api.create_commit(args.repo, [CommitOperationAdd(os.path.basename(f), f) for f in args.upload],
                          commit_message=f"model data for {args.version}")
    files = {}
    for e in api.list_repo_tree(args.repo, recursive=True):
        if getattr(e, "size", None) is None or e.path in SKIP:
            continue
        if e.lfs is not None:
            files[e.path] = {"sha256": e.lfs.sha256, "size": e.size}
        else:
            local = hf_hub_download(args.repo, e.path)
            files[e.path] = {"sha256": hashlib.sha256(open(local, "rb").read()).hexdigest(), "size": e.size}
    if args.dry_run:
        for f in args.upload:
            data = open(f, "rb").read()
            files[os.path.basename(f)] = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}

    versions = []
    if "runmyai.json" in api.list_repo_files(args.repo):
        versions = json.load(open(hf_hub_download(args.repo, "runmyai.json")))["versions"]
    if versions and versions[-1]["files"] == files:
        sys.exit("no file changed since the newest version: nothing to publish")
    if any(v["version"] == args.version for v in versions):
        sys.exit(f"version {args.version} is already in the list")
    versions.append({"version": args.version, "notes": args.notes, "files": files})
    text = json.dumps({"format": 1, "versions": versions}, indent=1, sort_keys=True) + "\n"

    with tempfile.TemporaryDirectory() as tmp:
        lst, sig = os.path.join(tmp, "runmyai.json"), os.path.join(tmp, "sig.bin")
        open(lst, "w").write(text)
        subprocess.run(["openssl", "pkeyutl", "-sign", "-inkey", args.key, "-rawin", "-in", lst, "-out", sig], check=True)
        sig_text = base64.b64encode(open(sig, "rb").read()).decode()
        if args.dry_run:
            open("runmyai.json", "w").write(text)
            open("runmyai.json.sig", "w").write(sig_text)
            print(f"dry run: runmyai.json ({len(versions)} versions, {len(files)} files) and its signature written here")
            return
        open(os.path.join(tmp, "runmyai.json.sig"), "w").write(sig_text)
        api.create_commit(args.repo, [CommitOperationAdd("runmyai.json", lst),
                                      CommitOperationAdd("runmyai.json.sig", os.path.join(tmp, "runmyai.json.sig"))],
                          commit_message=f"update list: {args.version}")
    print(f"{args.repo}: version {args.version} published ({len(files)} files)")


if __name__ == "__main__":
    main()
