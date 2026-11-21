#!/usr/bin/env python3
"""Copy the data appended to a TokenWing model file out into separate files next to where the model lives.

    python3 extract_model_data.py <model.tokenwing.gguf or its https:// URL> -o <folder>

A model file made by make_model_file.py carries its data after the last tensor (tokenwing.blob.<name>.offset / .size in its
metadata). This writes each into its own file, named as the engine and the app look for them next to the model:
  correction   -> <model file name>.correction        (--moe-stream-correction <file>)
  vram_experts -> <model file name>.vram-experts.json (--moe-stream-vram-experts-file <file>)
  usage_seed   -> <model file name>.usage-seed.json   (read by the engine when it is there)
  layer_order  -> <model file name>.layer-order.json  (read by the engine when it is there)
A URL is read with range requests: only the metadata and the data blocks are fetched, never the tensors. Separate files let an
update of the data download a few MB instead of the whole model.
"""

import argparse
import os
import struct
import sys
import urllib.request

SUFFIX = {"correction": ".correction", "vram_experts": ".vram-experts.json", "usage_seed": ".usage-seed.json",
          "layer_order": ".layer-order.json"}


class Source:
    def __init__(self, where):
        self.where, self.pos = where, 0
        self.file = None if where.startswith("https://") else open(where, "rb")
        self.buf, self.buf_start = b"", 0

    def read_at(self, off, n):
        if self.file:
            self.file.seek(off)
            return self.file.read(n)
        req = urllib.request.Request(self.where, headers={"Range": f"bytes={off}-{off + n - 1}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        if len(data) != n:
            raise IOError(f"asked {n} bytes at {off}, got {len(data)}")
        return data

    def read(self, n):
        # the metadata is read in 4 MiB windows, so a URL needs only a few requests
        while self.pos + n > self.buf_start + len(self.buf):
            if self.pos >= self.buf_start + len(self.buf):
                self.buf, self.buf_start = self.read_at(self.pos, max(n, 4 << 20)), self.pos
            else:
                self.buf += self.read_at(self.buf_start + len(self.buf), max(n, 4 << 20))
        start = self.pos - self.buf_start
        self.pos += n
        return self.buf[start:start + n]


SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def blobs_of(src):
    if src.read(4) != b"GGUF":
        sys.exit("not a GGUF file")
    version, n_tensors, n_kv = struct.unpack("<IQQ", src.read(20))
    if version < 2:
        sys.exit(f"GGUF version {version} is not read")
    text = lambda: src.read(struct.unpack("<Q", src.read(8))[0]).decode("utf-8", "replace")

    def value(t):
        if t in SIZES:
            raw = src.read(SIZES[t])
            return struct.unpack("<Q", raw)[0] if t == 10 else raw
        if t == 8:
            return text()
        if t == 9:
            sub, count = struct.unpack("<IQ", src.read(12))
            for _ in range(count):
                value(sub)
            return None
        sys.exit(f"unknown metadata type {t}")

    found = {}
    for _ in range(n_kv):
        key = text()
        t = struct.unpack("<I", src.read(4))[0]
        v = value(t)
        if key.startswith("tokenwing.blob.") and t == 10:
            name, field = key[len("tokenwing.blob."):].rsplit(".", 1)
            found.setdefault(name, {})[field] = v
    return {n: (f["offset"], f["size"]) for n, f in found.items() if "offset" in f and "size" in f}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()
    src = Source(args.model)
    blobs = blobs_of(src)
    if not blobs:
        sys.exit("the file carries no appended data")
    base = os.path.basename(args.model.split("?")[0])
    os.makedirs(args.output, exist_ok=True)
    for name, (off, size) in sorted(blobs.items()):
        if name not in SUFFIX:
            print(f"{name}: not a data file the engine reads separately, skipped")
            continue
        out = os.path.join(args.output, base + SUFFIX[name])
        with open(out + ".tmp", "wb") as f:
            done = 0
            while done < size:
                chunk = src.read_at(off + done, min(64 << 20, size - done))
                f.write(chunk)
                done += len(chunk)
        os.replace(out + ".tmp", out)
        print(f"{name}: {size} bytes -> {out}")


if __name__ == "__main__":
    main()
