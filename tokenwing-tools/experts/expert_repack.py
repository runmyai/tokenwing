#!/usr/bin/env python3
"""Repack the routed experts of a GGUF into one 4096-aligned block per expert.

Layout of the output sidecar: for each MoE layer (in order), for each expert
(0..n_expert-1), the gate, up and down slices exactly as they appear in the GGUF
(no re-quantization), with each expert block starting on a 4096-byte boundary.

The source is read as three sequential streams per layer (one per projection
tensor) in large chunks, never expert by expert, so a repack stays I/O bound.

Example: python3 expert_repack.py model.gguf -o experts.bin
The metadata is written next to it as experts.bin.json.
"""

import argparse
import hashlib
import json
import mmap
import os
import sys
import time
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

from expert_index import IO_ALIGN, PROJS, build_index, human  # noqa: E402
from expert_io import open_file, pread_bytes, pread_into, pwrite_from  # noqa: E402

SIDECAR_FORMAT = "tokenwing expert sidecar v1"
DEFAULT_CHUNK = 32 << 20
DEFAULT_WRITE_CHUNK = 32 << 20


class SeqReader:
    """Sequential byte stream over one GGUF tensor, refilled in big chunks.

    Logical position `cursor` walks the tensor once, forward only. Bytes are served
    out of a window [win_start, win_start + win_len) that is reloaded whenever the
    next slice does not fit in it; the window is (re)loaded at a 4096-aligned offset
    in direct mode, so every pread is a valid O_DIRECT read and the throughput
    measured is device-to-device instead of page-cache-to-page-cache.
    """

    def __init__(self, fd, offset, nbytes, chunk=DEFAULT_CHUNK, direct=False, align=IO_ALIGN):
        self.fd = fd
        self.direct = direct
        self.align = align
        self.start = offset
        self.end = offset + nbytes
        self.cursor = offset
        # A direct read is a whole number of pages, and a tensor's size is usually not one, so the
        # last window of a tensor has to read past the tensor's end (into the next tensor's bytes, which
        # the file has) - see _load_window
        self.file_end = os.fstat(fd).st_size
        self.chunk = max(-(-chunk // align) * align, align)
        self.buf = mmap.mmap(-1, self.chunk)
        self.view = memoryview(self.buf)
        self.win_start = 0
        self.win_len = 0

    def read(self, n):
        """Return a memoryview of the next n bytes; valid until the next read."""
        if self.cursor + n > self.end:
            raise EOFError(f"{self.start}: wanted {n} bytes at {self.cursor}, past tensor end {self.end}")
        if not self.win_start <= self.cursor or self.cursor + n > self.win_start + self.win_len:
            self._load_window()
            if self.cursor + n > self.win_start + self.win_len:
                # without this the slice below would come out of the untouched part of the mapping
                raise EOFError(f"{self.start}: wanted {n} bytes at {self.cursor}, the file only reaches "
                                f"{self.win_start + self.win_len}")
        at = self.cursor - self.win_start
        self.cursor += n
        return self.view[at:at + n]

    def _load_window(self):
        start = self.cursor - self.cursor % self.align if self.direct else self.cursor
        # The window may run past the tensor's end, or the last partial page of the tensor is
        # never loaded and the caller gets the mapping's zeros instead of its bytes
        limit = self.file_end if self.direct else self.end
        avail = min(self.chunk, limit - start)
        if avail <= 0:
            raise EOFError(f"{self.start}: cannot read a full window at {self.cursor}")
        if self.direct:
            if start + avail == self.file_end:
                want = -(-avail // self.align) * self.align
                need = avail
            else:
                want = avail - avail % self.align
                need = want
        else:
            want = avail
            need = avail
        if want <= 0:
            raise EOFError(f"{self.start}: cannot read a full window at {self.cursor}")
        got = 0
        while got < need:
            sub = self.view[got:got + want - got]
            try:
                n = pread_into(self.fd, sub, start + got)
            finally:
                sub.release()
            if n == 0:
                raise EOFError(f"{self.start}: short read at {start + got}")
            got += n
        self.win_start = start
        self.win_len = got

    def close(self):
        self.view.release()
        self.buf.close()


def header_fingerprint(path, nbytes):
    """sha256 of the GGUF header region, to tie a sidecar to its source model."""
    h = hashlib.sha256()
    fd = open_file(path, os.O_RDONLY)
    try:
        left = nbytes
        off = 0
        while left:
            blk = pread_bytes(fd, min(left, 1 << 22), off)
            if not blk:
                raise EOFError(f"{path}: header region is only {off} bytes, wanted {nbytes}")
            h.update(blk)
            off += len(blk)
            left -= len(blk)
    finally:
        os.close(fd)
    return h.hexdigest()


def check_complete(idx, path):
    """Refuse to repack a model whose tensor data is not all on disk."""
    m = idx["model"]
    shards = m.get("shards") or [{"path": m["path"], "size": m["file_size"]}]
    if m["incomplete"]:
        raise SystemExit(f"refusing to repack {path}: a shard's header describes more tensor data than the "
                         f"file holds (the shards are {m['file_size']:,} B in total)")
    # A split GGUF is several files, so every projection is checked against the one it is in
    for il, lay in idx["layers"].items():
        for proj, d in lay["projections"].items():
            end = d["offset"] + d["n_bytes"]
            size = shards[int(d.get("shard", 0))]["size"]
            if end > size:
                raise SystemExit(f"refusing to repack {path}: tensor {d['tensor']} ends at {end:,} B "
                                 f"but shard {d.get('shard', 0)} is {size:,} B")


# llama.cpp's repacked CPU layout for MXFP4 (block_mxfp4x8, ggml/src/ggml-cpu/repack.h):
# eight rows of 32 elements share one block of 136 bytes - the eight e8m0 scales first, then their
# quants interleaved in 32-byte chunks. It is exactly the same size as the plain layout (8 x 17 B) and
# an expert stays contiguous at the same offset, so a sidecar in this layout needs no conversion when it
# is read. The reference is ggml's own repack_mxfp4_to_mxfp4_8_bl().
REPACK_ROWS = 8

try:
    import numpy as _np
except ImportError:
    _np = None


def cpu_repack_mxfp4(blob, n_rows, interleave=REPACK_ROWS):
    """One expert slice, [n_rows, ncols*32] MXFP4 bytes in, the same bytes repacked out.

    The plain layout is 17 bytes per 32 elements: 1 scale byte + 16 quants. The repacked block is
    8 + 4*32 = 136 bytes for the same 8 x 32 elements: the eight scales, then the quants of the eight
    rows in the order the kernel reads them (row r's first 8 bytes, row r+1's first 8 bytes, ...).
    """
    if _np is None:
        raise SystemExit("numpy is needed for --cpu-repack")

    qk = 32
    block_size = 17
    nblk = len(blob) // block_size
    if nblk % interleave:
        raise SystemExit(f"an expert slice of {len(blob)} bytes is not a multiple of "
                         f"{interleave} blocks of 32 elements")

    a        = _np.frombuffer(blob, dtype=_np.uint8).reshape(nblk, block_size)
    ncols    = nblk // n_rows                       # blocks per row (ne0 / 32)
    groups   = n_rows // interleave                  # row groups
    assert groups*ncols*interleave == nblk

    # (groups, cols, rows) for the scales and for each 8-byte half of the quants
    scales   = a[:, 0].reshape(groups, interleave, ncols).transpose(0, 2, 1)
    # (groups, cols, half, row, 8): the kernel reads the first 8 bytes of all eight rows, then the
    # second 8 bytes of all eight rows, so the half is the outer of the two
    halves   = a[:, 1:].reshape(groups, interleave, ncols, 2, 8).transpose(0, 2, 3, 1, 4)

    out = _np.zeros((groups, ncols, 8 + 2*interleave*8), dtype=_np.uint8)
    out[:, :, :interleave] = scales                                  # the eight e8m0 scales first
    out[:, :, interleave:] = halves.reshape(groups, ncols, 2*interleave*8)

    return out.reshape(-1).tobytes()


def write_layer(dst, buf, readers, sizes, block, useful, per_write, base, n_expert, layer, cpu_repack=False,
                n_rows=None):
    """Fill buf with per_write expert blocks and pwritev them; returns the bytes written."""
    written = 0
    pad = bytes(block - useful) if block > useful else b""
    e = 0
    while e < n_expert:
        n = min(per_write, n_expert - e)
        for i in range(n):
            at = i * block
            for p in PROJS:
                blob = readers[p].read(sizes[p])
                if cpu_repack:
                    # Each projection's slice on its own is [n_rows, ncols*32]; the rows are the
                    # tensor's ne[1] of the expert, so the slice is repacked as a whole
                    blob = cpu_repack_mxfp4(blob, n_rows[p])
                buf[at:at + sizes[p]] = blob
                at += sizes[p]
            if pad:
                buf[i * block + useful:(i + 1) * block] = pad
        want = n * block
        if pwrite_from(dst, buf[:want], base + e * block) != want:
            raise OSError(f"short write for layer {layer} at expert {e}")
        written += want
        e += n
    return written


def repack(model_path, out_path, index=None, align=IO_ALIGN, chunk=DEFAULT_CHUNK, use_direct=False,
           direct_read=False, write_chunk=DEFAULT_WRITE_CHUNK, progress=True, cpu_repack=False):
    """Write the expert sidecar for model_path to out_path; returns the sidecar metadata."""
    idx = index if index is not None else build_index(model_path, align)
    if index is None and Path(model_path).resolve() != Path(idx["model"]["path"]).resolve():
        raise SystemExit(f"index was built for {idx['model']['path']}, not {model_path}")
    check_complete(idx, model_path)

    plan = idx["repack_plan"]
    if plan["io_align"] != align:
        raise SystemExit(f"index was built for align {plan['io_align']}, not {align}")
    m = idx["model"]
    n_expert = m["n_expert"]
    moe_layers = m["moe_layers"]

    layouts = {}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    # A split GGUF is several files; one fd each, and a layer reads from the shard its tensors are in
    srcs = [open_file(sh["path"], os.O_RDONLY, direct=direct_read)
            for sh in (m.get("shards") or [{"path": model_path}])]
    dst = open_file(out_path, flags, 0o644, direct=use_direct)

    block_bytes = plan["read_bytes_per_expert"]
    single_block = len(block_bytes) == 1
    total = plan["sidecar_bytes"]
    t0 = time.perf_counter()
    written = 0
    try:
        for n, il in enumerate(moe_layers):
            pl = plan["per_layer"][str(il)]
            lay = idx["layers"][str(il)]["projections"]
            sizes = {p: lay[p]["slice_size"] for p in PROJS}
            block = pl["block_bytes"]
            # Only MXFP4 has a repacked layout (8x8 blocks of 136 B for 8 rows of 32 elements);
            # a layer stored in another type (a Q4_0 next-token-prediction block, for example) stays
            # plain. Repacking it would produce bytes that mean nothing, so each block records what
            # it holds.
            repack_here = cpu_repack and all(lay[p]["ggml_type"] == "MXFP4" for p in PROJS)
            layouts[str(il)] = "mxfp4_8x8" if repack_here else "plain"
            # how many rows an expert slice has, i.e. the tensor's ne[1]: the slice holds
            # ne[0]/32 blocks per row
            n_rows = {p: (sizes[p] // 17) // (lay[p]["shape_ggml"][0] // 32) for p in PROJS}
            per_write = max(1, write_chunk // block)
            readers = {p: SeqReader(srcs[int(lay[p].get("shard", 0))], lay[p]["offset"],
                                    lay[p]["n_bytes"], chunk, direct_read, align)
                       for p in PROJS}
            outbuf = mmap.mmap(-1, block * per_write)
            try:
                view = memoryview(outbuf)
                written += write_layer(dst, view, readers, sizes, block, sum(sizes.values()),
                                       per_write, pl["base_offset"], n_expert, il,
                                       cpu_repack=repack_here, n_rows=n_rows)
                view.release()
            finally:
                outbuf.close()
                for r in readers.values():
                    r.close()
            if progress:
                dt = time.perf_counter() - t0
                rate = written / dt
                left = total - written
                print(f"  [{n + 1:3d}/{len(moe_layers)}] layer {il:3d}  "
                      f"{written / 2**30:6.2f}/{total / 2**30:.2f} GiB  "
                      f"{rate / 1e9:4.2f} GB/s  eta {left / rate if rate else 0:5.0f} s",
                      flush=True)
        os.fsync(dst)
    finally:
        for fd in srcs:
            os.close(fd)
        os.close(dst)
    dt = time.perf_counter() - t0

    meta = {
        "format": SIDECAR_FORMAT,
        "io_align": align,
        # What the experts in this file are stored in. Absent means
        # plain. A separate key from "layout" below, which has always described how the blocks sit in
        # the file. Per block, because only some layers can be repacked (MXFP4 only); "mixed" here
        # means look at the blocks. The engine refuses a repacked block unless the pools are repacked.
        "expert_layout": next(iter(set(layouts.values()))) if len(set(layouts.values())) == 1 else "mixed",
        "layout": "for each MoE layer, for each expert: gate|up|down contiguous, block 4096-aligned",
        "offset_formula": "base_offset[layer] + e * block_bytes; gate at +0, up at +slice_gate, "
                          "down at +slice_gate + slice_up",
        "source": {
            "path": m["path"],
            "size": m["file_size"],
            "arch": m["arch"],
            "gguf_version": m["gguf_version"],
            "gguf_alignment": m["alignment"],
            "data_section_offset": m["data_section_offset"],
            "header_sha256": header_fingerprint(model_path, m["data_section_offset"]),
            "header_bytes_hashed": m["data_section_offset"],
        },
        "n_expert": n_expert,
        "n_expert_used": m["n_expert_used"],
        "n_layer": m["n_layer"],
        "n_layer_nextn": m["n_layer_nextn"],
        "moe_layers": moe_layers,
        "block_bytes": block_bytes[0] if single_block else None,
        "block_bytes_per_layer": {str(il): plan["per_layer"][str(il)]["block_bytes"] for il in moe_layers},
        "sidecar_bytes": total,
        "useful_bytes": plan["useful_bytes"],
        "padding_bytes": plan["padding_bytes"],
        "repack_seconds": dt,
        "repack_bytes_per_second": written / dt if dt else 0,
        "blocks": {},
    }
    for il in moe_layers:
        pl = plan["per_layer"][str(il)]
        lay = idx["layers"][str(il)]["projections"]
        at = 0
        projs = {}
        for p in PROJS:
            d = lay[p]
            projs[p] = {
                "offset": at,
                "size": d["slice_size"],
                "ggml_type": d["ggml_type"],
                "shape_ggml": d["shape_ggml"],
                "source_tensor": d["tensor"],
                "source_offset": d["offset"],
                # Where this tensor is in the model, so the engine can read the same bytes
                # again and compare a streamed slot against the file it claims to come from. The expert
                # e of the tensor is at source_shard:source_offset + e*size.
                "source_shard": int(d.get("shard", 0)),
            }
            at += d["slice_size"]
        meta["blocks"][str(il)] = {
            "base_offset": pl["base_offset"],
            "block_bytes": pl["block_bytes"],
            "padding_bytes": pl["block_bytes"] - at,
            "is_nextn": idx["layers"][str(il)]["is_nextn"],
            "expert_layout": layouts[str(il)],
            "projections": projs,
        }

    meta_path = out_path.with_name(out_path.name + ".json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=1)
    return meta, meta_path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", help="path to a .gguf")
    ap.add_argument("-o", "--out", required=True, help="sidecar file to write")
    ap.add_argument("--index", help="expert index JSON (default: build it from --model)")
    ap.add_argument("--align", type=int, default=IO_ALIGN, help="block alignment (default 4096)")
    ap.add_argument("--chunk-mb", type=float, default=DEFAULT_CHUNK / 2**20,
                    help="sequential read chunk per projection tensor (default 32)")
    ap.add_argument("--write-chunk-mb", type=float, default=DEFAULT_WRITE_CHUNK / 2**20,
                    help="blocks are batched into writes of this size (default 32); measured on ntfs3: "
                         "a sidecar written in small chunks reads back ~2x slower, so keep it large")
    ap.add_argument("--direct-write", action="store_true",
                    help="O_DIRECT the output too; measured on ntfs3: a sidecar written that way "
                         "reads back ~30%% slower, so buffered + fsync is the default")
    ap.add_argument("--direct-read", action="store_true",
                    help="O_DIRECT the source too, for a true device-to-device throughput number")
    ap.add_argument("--cpu-repack", action="store_true",
                    help="write the experts in llama.cpp's repacked CPU layout (mxfp4_8x8), which the "
                         "engine's repacked MUL_MAT_ID reads without converting anything on the way in. "
                         "Same size, same offsets; the metadata says so in its \"layout\" field")
    ap.add_argument("--quiet", action="store_true", help="no per-layer progress")
    args = ap.parse_args()

    index = None
    if args.index:
        from expert_io import ExpertIndex

        index = ExpertIndex.load(args.index).data
        print(f"using index {args.index} ({index['model']['path']})")

    meta, meta_path = repack(args.model, args.out, index, args.align,
                             int(args.chunk_mb * 2**20), args.direct_write, args.direct_read,
                             int(args.write_chunk_mb * 2**20), not args.quiet, args.cpu_repack)
    n_expert = meta["n_expert"]
    dt = meta["repack_seconds"]
    print(f"\nsource     {meta['source']['path']}")
    print(f"           {human(meta['source']['size'])}, header sha256 {meta['source']['header_sha256'][:16]}...")
    print(f"sidecar    {args.out}  {human(meta['sidecar_bytes'])} ({meta['sidecar_bytes']:,} B)")
    print(f"layout     {meta.get('expert_layout', 'plain')}")
    print(f"           {len(meta['moe_layers'])} MoE layers x {n_expert} experts x "
          f"{human(meta['block_bytes'] or meta['sidecar_bytes'] // (n_expert * len(meta['moe_layers'])))} blocks"
          f", padding {human(meta['padding_bytes'])}")
    print(f"repack     {dt:.1f} s  ({meta['repack_bytes_per_second'] / 1e9:.2f} GB/s written)")
    print(f"metadata   {meta_path}")


if __name__ == "__main__":
    main()
