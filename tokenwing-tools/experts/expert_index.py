#!/usr/bin/env python3
"""Map every routed-expert slice of a GGUF file to an absolute file offset.

Only the GGUF header is read, so this also works on a partially downloaded file
(an unfinished download) whose tensor data is missing.
Writes the index plus an O_DIRECT (4096-aligned) read plan to JSON.

Example: python3 expert_index.py model.gguf -o index.json
"""

import argparse
import json
import os
import re
import struct
import sys
from collections import Counter
from pathlib import Path

GGUF_PY = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gguf-py"))
if GGUF_PY not in sys.path:
    sys.path.insert(0, GGUF_PY)

from gguf.constants import (  # noqa: E402
    GGML_QUANT_SIZES,
    GGUF_DEFAULT_ALIGNMENT,
    GGUF_MAGIC,
    GGMLQuantizationType,
    GGUFValueType,
)

IO_ALIGN = 4096
PROJS = ("gate", "up", "down")
PROJ_TENSOR = {"gate": "ffn_gate_exps", "up": "ffn_up_exps", "down": "ffn_down_exps"}
EXPS_RE = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")
NEXTN_RE = re.compile(r"^blk\.(\d+)\.nextn\.")

_SCALAR_FMT = {
    GGUFValueType.UINT8: "<B",
    GGUFValueType.INT8: "<b",
    GGUFValueType.UINT16: "<H",
    GGUFValueType.INT16: "<h",
    GGUFValueType.UINT32: "<I",
    GGUFValueType.INT32: "<i",
    GGUFValueType.FLOAT32: "<f",
    GGUFValueType.UINT64: "<Q",
    GGUFValueType.INT64: "<q",
    GGUFValueType.FLOAT64: "<d",
    GGUFValueType.BOOL: "<B",
}

KEEP_ARRAYS = (
    "expert_feed_forward_length",
    "expert_shared_feed_forward_length",
    "swiglu_clamp_exp",
    "swiglu_clamp_shexp",
    "rope_sections",
)


class Truncated(Exception):
    pass


class Cursor:
    def __init__(self, buf):
        self.b = buf
        self.o = 0

    def _take(self, fmt):
        n = struct.calcsize(fmt)
        if self.o + n > len(self.b):
            raise Truncated()
        v = struct.unpack_from(fmt, self.b, self.o)[0]
        self.o += n
        return v

    def u8(self):
        return self._take("<B")

    def u32(self):
        return self._take("<I")

    def u64(self):
        return self._take("<Q")

    def string(self):
        n = self.u64()
        if self.o + n > len(self.b):
            raise Truncated()
        s = str(self.b[self.o:self.o + n], encoding="utf-8", errors="replace")
        self.o += n
        return s


class TensorInfo:
    def __init__(self, name, ggml_type, dims, rel_offset):
        self.name = name
        self.ggml_type = ggml_type
        self.dims = dims  # ggml order: dims[0] is the fastest moving (row) dim
        self.rel_offset = rel_offset
        block_size, type_size = GGML_QUANT_SIZES[ggml_type]
        self.block_size = block_size
        self.type_size = type_size
        self.n_elements = 1
        for d in dims:
            self.n_elements *= d
        self.n_bytes = self.n_elements * type_size // block_size


# A split GGUF (llama-gguf-split) is several files: every one of them carries the full KV section, and
# each carries only the tensors whose bytes are in it, with their offsets relative to *its own* data
# section. resolve_shards() finds the set a path belongs to, and GGUFSet presents it as one model.
SHARD_RE = re.compile(r"^(?P<stem>.+)-(?P<idx>\d{5})-of-(?P<count>\d{5})\.gguf$")


def resolve_shards(path):
    """The sorted list of the files of a (possibly split) GGUF. One file -> [path]."""
    path = Path(path)
    if path.is_dir():
        parts = sorted(p for p in path.glob("*-of-*.gguf"))
        if not parts:
            raise ValueError(f"{path}: no *-of-*.gguf in it")
        return parts
    m = SHARD_RE.match(path.name)
    if not m:
        return [path]
    stem, idx, count = m.group("stem"), int(m.group("idx")), int(m.group("count"))
    if idx != 1:
        raise ValueError(f"{path}: pass the first shard of a split GGUF (…-00001-of-{count:05d}.gguf), "
                         "not shard {idx}")
    parts = []
    for i in range(1, count + 1):
        p = path.parent / f"{stem}-{i:05d}-of-{count:05d}.gguf"
        if not p.exists():
            raise FileNotFoundError(f"{p}: the split GGUF of {path.name} is missing shard {i} of {count}")
        parts.append(p)
    return parts


class GGUFSet:
    """One model, one file or several: the KV of the first shard, the tensors of all of them."""

    def __init__(self, path):
        self.path = str(path)
        self.shards = [GGUFHeader(str(p)) for p in resolve_shards(path)]
        if not self.shards:
            raise ValueError(f"{path}: no GGUF")
        self.split = len(self.shards) > 1
        h0 = self.shards[0]
        self.kv = h0.kv
        self.version = h0.version
        self.alignment = h0.alignment
        self.header_end = h0.header_end
        self.data_offset = h0.data_offset
        self.file_size = sum(h.file_size for h in self.shards)
        # the global tensor order is the shards in order, and inside a shard the order of its header
        self.tensors = []
        self.shard_of = {}
        for i, h in enumerate(self.shards):
            for t in h.tensors:
                t.shard = i
                t.shard_data_offset = h.data_offset
                t.abs_offset = h.data_offset + t.rel_offset
                self.tensors.append(t)
                self.shard_of[t.name] = i
        # a shard that does not hold all of its own tensor data is a partial download
        self.incomplete = any(h.data_offset + sum(t.n_bytes for t in h.tensors) > h.file_size
                              for h in self.shards)
        self.shard_paths = [str(p) for p in resolve_shards(path)]


class GGUFHeader:
    def __init__(self, path):
        self.path = str(path)
        self.file_size = os.path.getsize(self.path)
        self.kv = {}
        self.tensors = []
        self.alignment = GGUF_DEFAULT_ALIGNMENT
        self.header_end = 0
        self.data_offset = 0
        self.parse()

    def parse(self):
        fd = os.open(self.path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        try:
            n = 1 << 22
            while True:
                try:
                    if hasattr(os, "pread"):
                        buf = os.pread(fd, n, 0)
                    else:
                        os.lseek(fd, 0, os.SEEK_SET)
                        buf = os.read(fd, n)
                    self._parse_buf(buf)
                    return
                except Truncated:
                    if n >= self.file_size:
                        raise ValueError(f"{self.path}: header does not fit in the file ({n} bytes)")
                    n = min(self.file_size, n * 4)
        finally:
            os.close(fd)

    def _parse_buf(self, buf):
        cur = Cursor(buf)
        if cur.u32() != GGUF_MAGIC:
            raise ValueError(f"{self.path}: not a GGUF file")
        self.version = cur.u32()
        n_tensors = cur.u64()
        n_kv = cur.u64()
        for _ in range(n_kv):
            key = cur.string()
            self.kv[key] = self._value(cur, cur.u32(), key)
        for _ in range(n_tensors):
            name = cur.string()
            n_dims = cur.u32()
            if n_dims > 4:
                raise ValueError(f"tensor {name}: {n_dims} dims (expected <= 4)")
            dims = [cur.u64() for _ in range(n_dims)]
            ggml_type = GGMLQuantizationType(cur.u32())
            rel_offset = cur.u64()
            self.tensors.append(TensorInfo(name, ggml_type, dims, rel_offset))
        self.header_end = cur.o
        self.alignment = int(self.kv.get("general.alignment", GGUF_DEFAULT_ALIGNMENT))
        if self.alignment == 0 or (self.alignment & (self.alignment - 1)) != 0:
            raise ValueError(f"general.alignment={self.alignment} is not a non-zero power of two")
        self.data_offset = cur.o + (-cur.o) % self.alignment

    def _value(self, cur, raw_type, key, decode=True):
        vtype = GGUFValueType(raw_type)
        if vtype == GGUFValueType.STRING:
            if not decode:
                n = cur.u64()
                if cur.o + n > len(cur.b):
                    raise Truncated()
                cur.o += n
                return None
            return cur.string()
        if vtype == GGUFValueType.ARRAY:
            elem = cur.u32()
            n = cur.u64()
            if decode and n <= (1 << 20) and key.rpartition(".")[2] in KEEP_ARRAYS:
                return [self._value(cur, elem, key, True) for _ in range(n)]
            for _ in range(n):
                self._value(cur, elem, key, False)
            return {"array_len": n, "array_type": GGUFValueType(elem).name}
        if vtype == GGUFValueType.BOOL:
            return bool(cur.u8())
        fmt = _SCALAR_FMT.get(vtype)
        if fmt is None:
            raise ValueError(f"key {key}: unhandled value type {vtype}")
        return cur._take(fmt)


def kv_get(kv, key, default=None):
    v = kv.get(key, default)
    return default if isinstance(v, dict) else v


def build_index(path, io_align=IO_ALIGN):
    hdr = GGUFSet(path)   # one file or a split set, same fields plus .shards and .tensors[i].shard
    arch = kv_get(hdr.kv, "general.architecture", "unknown")
    n_layer_all = int(kv_get(hdr.kv, f"{arch}.block_count", 0) or 0)
    n_layer_nextn = int(kv_get(hdr.kv, f"{arch}.nextn_predict_layers", 0) or 0)
    n_layer = n_layer_all - n_layer_nextn if n_layer_all else None
    n_dense_lead = int(kv_get(hdr.kv, f"{arch}.leading_dense_block_count", 0) or 0)

    exps = {}
    for t in hdr.tensors:
        m = EXPS_RE.match(t.name)
        if m:
            exps.setdefault(int(m.group(1)), {})[m.group(2)] = t
    nextn_layers = sorted({int(NEXTN_RE.match(t.name).group(1)) for t in hdr.tensors if NEXTN_RE.match(t.name)})

    if not exps:
        raise ValueError(f"{path}: no ffn_(gate|up|down)_exps tensors found, not a routed-MoE GGUF")
    n_expert = int(kv_get(hdr.kv, f"{arch}.expert_count", 0) or 0)
    sample = next(iter(exps.values()))["gate"]
    if n_expert == 0:
        n_expert = sample.dims[-1]
    if sample.dims[-1] != n_expert:
        raise ValueError(
            f"{sample.name}: expert dim {sample.dims[-1]} != <arch>.expert_count {n_expert}")

    model = {
        "path": hdr.path,
        "file_size": hdr.file_size,
        "gguf_version": hdr.version,
        "arch": arch,
        "name": kv_get(hdr.kv, "general.name"),
        "quantization_version": kv_get(hdr.kv, "general.quantization_version"),
        "n_layer": n_layer,
        "n_layer_all": n_layer_all,
        "n_layer_nextn": n_layer_nextn,
        "leading_dense_block_count": n_dense_lead,
        "n_expert": n_expert,
        "n_expert_used": int(kv_get(hdr.kv, f"{arch}.expert_used_count", 0) or 0),
        "n_expert_groups": int(kv_get(hdr.kv, f"{arch}.expert_group_count", 0) or 0),
        "n_expert_group_used": int(kv_get(hdr.kv, f"{arch}.expert_group_used_count", 0) or 0),
        "n_expert_shared": int(kv_get(hdr.kv, f"{arch}.expert_shared_count", 0) or 0),
        "n_embd": int(kv_get(hdr.kv, f"{arch}.embedding_length", 0) or 0),
        "n_ff_exp": kv_get(hdr.kv, f"{arch}.expert_feed_forward_length") or kv_get(hdr.kv, f"{arch}.feed_forward_length"),
        "n_ff_shexp": kv_get(hdr.kv, f"{arch}.expert_shared_feed_forward_length"),
        "alignment": hdr.alignment,
        "header_end": hdr.header_end,
        "data_section_offset": hdr.data_offset,
        # The files the tensor data lives in. One entry for a normal GGUF, several for a split
        # one; every projection below names the shard it is in, and its offset is inside that shard.
        "shards": [{"path": p, "size": os.path.getsize(p)} for p in hdr.shard_paths],
        "split": hdr.split,
        "moe_layers": sorted(exps),
        "nextn_layers": nextn_layers,
        "dense_layers": sorted(il for il in range(n_layer_all) if il not in exps),
        "incomplete": hdr.incomplete,
    }

    layers = {}
    for il in sorted(exps):
        projs = {}
        for proj in PROJS:
            t = exps[il].get(proj)
            if t is None:
                continue
            if t.n_bytes % n_expert:
                raise ValueError(f"{t.name}: {t.n_bytes} bytes is not divisible by n_expert={n_expert}")
            slice_size = t.n_bytes // n_expert
            projs[proj] = {
                "tensor": t.name,
                "ggml_type": t.ggml_type.name,
                "ggml_type_id": int(t.ggml_type),
                "block_size": t.block_size,
                "type_size": t.type_size,
                "shape_ggml": t.dims,
                "n_elements": t.n_elements,
                "n_bytes": t.n_bytes,
                "offset": t.abs_offset,
                "gguf_rel_offset": t.rel_offset,
                "shard": t.shard,
                "slice_size": slice_size,
                "offset_formula": f"{t.abs_offset} + e * {slice_size} (in shard {t.shard})",
                "residue_4096": t.abs_offset % io_align,
            }
        layers[str(il)] = {"is_nextn": n_layer is not None and il >= n_layer, "projections": projs}

    idx = {
        "format": "tokenwing expert index v1",
        "io_align": io_align,
        "model": model,
        "layers": layers,
    }
    idx["summary"] = summarize(idx)
    idx["aligned_plan"] = aligned_plan(idx, io_align)
    idx["repack_plan"] = repack_plan(idx, io_align)
    return idx


def summarize(idx):
    m = idx["model"]
    per_layer = {}
    for il, lay in idx["layers"].items():
        sizes = {p: d["slice_size"] for p, d in lay["projections"].items()}
        n_bytes = sum(d["n_bytes"] for d in lay["projections"].values())
        per_layer[il] = {
            "n_expert": m["n_expert"],
            "slice_size": sizes,
            "bytes_per_expert": sum(sizes.values()),
            "routed_expert_bytes": n_bytes,
            "bytes_per_token": sum(sizes.values()) * m["n_expert_used"],
        }
    residues = Counter()
    for lay in idx["layers"].values():
        for d in lay["projections"].values():
            residues[d["residue_4096"]] += 1
    slice_sizes = {tuple(sorted((p, d["slice_size"]) for p, d in lay["projections"].items()))
                   for lay in idx["layers"].values()}
    expert_bytes = {sum(v for _, v in s) for s in slice_sizes}
    return {
        "n_moe_layers": len(idx["layers"]),
        "n_expert": m["n_expert"],
        "n_expert_used": m["n_expert_used"],
        "routed_expert_bytes": sum(v["routed_expert_bytes"] for v in per_layer.values()),
        "routed_expert_share_of_file": (
            sum(v["routed_expert_bytes"] for v in per_layer.values()) / m["file_size"]),
        "distinct_expert_sizes": [{"proj": dict(s)} for s in sorted(slice_sizes)],
        "bytes_per_expert": sorted(expert_bytes),
        "bytes_per_token": sum(v["bytes_per_token"] for v in per_layer.values()),
        "tensor_offset_residue_4096": {str(k): v for k, v in sorted(residues.items())},
        "per_layer": per_layer,
    }


def aligned_plan(idx, io_align=IO_ALIGN):
    per_layer = {}
    n_expert = idx["model"]["n_expert"]
    n_layers = len(idx["layers"])
    all_over = []
    n_useful_total = 0
    n_over_total = 0
    n_merged_total = 0
    for il, lay in idx["layers"].items():
        n_useful = n_expert * sum(d["slice_size"] for d in lay["projections"].values())
        n_over = 0
        n_merged = 0
        for e in range(n_expert):
            ranges = []
            for proj in PROJS:
                d = lay["projections"][proj]
                start, end = aligned_range(d["offset"] + e * d["slice_size"], d["slice_size"], io_align)
                over = end - start - d["slice_size"]
                all_over.append(over)
                n_over += over
                ranges.append((start, end))
            n_merged += count_merged(ranges)
        n_useful_total += n_useful
        n_over_total += n_over
        n_merged_total += n_merged
        per_layer[il] = {
            "n_expert": n_expert,
            "useful_bytes": n_useful,
            "aligned_bytes": n_useful + n_over,
            "overhead_bytes": n_over,
            "overhead_pct": 100.0 * n_over / n_useful,
            "reads_per_expert": n_expert * len(PROJS),
            "reads_per_expert_after_range_merge": n_merged / n_expert,
        }
    n_slices = len(all_over)
    return {
        "io_align": io_align,
        "n_slices": n_slices,
        "n_experts_total": n_expert * n_layers,
        "useful_bytes": n_useful_total,
        "overhead_bytes": n_over_total,
        "overhead_pct": 100.0 * n_over_total / n_useful_total,
        "overhead_bytes_per_slice_avg": n_over_total / n_slices,
        "overhead_bytes_per_slice_max": max(all_over),
        "overhead_bytes_per_expert_avg": n_over_total / (n_expert * n_layers),
        "overhead_bytes_per_expert_max": len(PROJS) * max(all_over),
        "reads_per_expert": len(PROJS),
        "reads_per_expert_after_range_merge": n_merged_total / (n_expert * n_layers),
        "reads_per_expert_repacked": 1,
        "note": "reads_per_expert_after_range_merge = touching 4096 ranges of the 3 projections of one "
                "expert coalesced into single preadv calls (also fetches neighbour bytes)",
        "per_layer": per_layer,
    }


def aligned_range(off, size, io_align=IO_ALIGN):
    return off - off % io_align, -(-(off + size) // io_align) * io_align


def count_merged(ranges):
    ranges = sorted(ranges)
    n = 1
    end = ranges[0][1]
    for start, e in ranges[1:]:
        if start > end:
            n += 1
        end = max(end, e)
    return n


def repack_plan(idx, io_align=IO_ALIGN):
    per_layer = {}
    cursor = 0
    total_pad = 0
    total_pad_proj = 0
    for il, lay in idx["layers"].items():
        sizes = {p: d["slice_size"] for p, d in lay["projections"].items()}
        n_expert = lay["projections"]["gate"]["n_bytes"] // sizes["gate"]
        useful = sum(sizes.values())
        block = -(-useful // io_align) * io_align
        block_proj = sum(-(-v // io_align) * io_align for v in sizes.values())
        per_layer[il] = {
            "n_expert": n_expert,
            "slice_size": sizes,
            "useful_bytes_per_expert": useful,
            "block_bytes": block,
            "padding_bytes_per_expert": block - useful,
            "block_bytes_per_projection_aligned": block_proj,
            "base_offset": cursor,
            "offset_formula": f"{cursor} + e * {block}",
        }
        cursor += block * n_expert
        total_pad += (block - useful) * n_expert
        total_pad_proj += (block_proj - useful) * n_expert
    useful_total = sum(v["useful_bytes_per_expert"] * v["n_expert"] for v in per_layer.values())
    return {
        "io_align": io_align,
        "layout": "for each MoE layer, for each expert: gate|up|down contiguous, block 4096-aligned",
        "sidecar_bytes": cursor,
        "useful_bytes": useful_total,
        "padding_bytes": total_pad,
        "padding_pct": 100.0 * total_pad / useful_total,
        "padding_bytes_if_each_projection_aligned": total_pad_proj,
        "sidecar_bytes_if_each_projection_aligned": cursor + total_pad_proj - total_pad,
        "reads_per_expert": 1,
        "read_bytes_per_expert": sorted({v["block_bytes"] for v in per_layer.values()}),
        "per_layer": per_layer,
    }


def human(n):
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:,.0f} {unit}" if unit == "B" else f"{n:,.2f} {unit}"
        n /= 1024.0


def print_summary(idx):
    m, s, a, r = idx["model"], idx["summary"], idx["aligned_plan"], idx["repack_plan"]
    print(f"model            {m['path']}")
    print(f"                 {m['name']}  arch={m['arch']}  gguf_v{m['gguf_version']}  file={human(m['file_size'])}")
    print(f"header           end={m['header_end']:,}  alignment={m['alignment']}  "
          f"data_section_offset={m['data_section_offset']:,}  tensor data present: "
          f"{'no (partial download)' if m['incomplete'] else 'yes'}")
    print(f"layers           n_layer={m['n_layer']} (block_count={m['n_layer_all']}, nextn={m['n_layer_nextn']}), "
          f"leading_dense={m['leading_dense_block_count']}")
    print(f"                 MoE layers: {s['n_moe_layers']} "
          f"({m['moe_layers'][0] if m['moe_layers'] else '-'}..{m['moe_layers'][-1] if m['moe_layers'] else '-'})"
          + (f", nextn/MTP: {m['nextn_layers']}" if m["nextn_layers"] else ", nextn/MTP: none"))
    print(f"experts          n_expert={m['n_expert']}  top-k={m['n_expert_used']}  "
          f"groups={m['n_expert_groups']}x{m['n_expert_group_used']}  shared={m['n_expert_shared']}")
    print(f"                 n_embd={m['n_embd']}  n_ff_exp={m['n_ff_exp']}  n_ff_shexp={m['n_ff_shexp']}")
    sizes = s["distinct_expert_sizes"]
    print("expert slice     " + " | ".join(
        " ".join(f"{p}={v:,}" for p, v in sorted(d["proj"].items())) for d in sizes))
    print(f"bytes / expert   {[human(x) for x in s['bytes_per_expert']]}  (gate+up+down)")
    print(f"bytes / token    {human(s['bytes_per_token'])}  = {s['n_moe_layers']} layers x k={s['n_expert_used']} "
          f"x bytes/expert")
    print(f"routed experts   {human(s['routed_expert_bytes'])}  "
          f"({100 * s['routed_expert_share_of_file']:.1f}% of the file)")
    res = s["tensor_offset_residue_4096"]
    n_zero = res.get(0, 0)
    print(f"offsets mod 4096  {sum(res.values())} expert tensors, {len(res)} distinct residues, "
          f"{n_zero} of them 0, "
          f"range {min(map(int, res))}..{max(map(int, res))}, "
          f"all {m['alignment']}-aligned: {all(int(r) % m['alignment'] == 0 for r in res)}")
    print()
    print(f"O_DIRECT plan (align {a['io_align']})")
    print(f"  overhead/slice   avg {a['overhead_bytes_per_slice_avg']:,.0f} B   max {a['overhead_bytes_per_slice_max']:,} B")
    print(f"  overhead/expert  avg {a['overhead_bytes_per_expert_avg']:,.0f} B   "
          f"max {a['overhead_bytes_per_expert_max']:,} B")
    print(f"  overhead total   {a['overhead_pct']:.3f} % of useful bytes")
    print(f"  reads/expert     {a['reads_per_expert']} now, "
          f"{a['reads_per_expert_after_range_merge']:.2f} if touching ranges are merged, "
          f"{a['reads_per_expert_repacked']} after repacking "
          f"({a['n_slices']:,} slices in total)")
    print()
    print(f"repack plan (experts-only sidecar, {r['layout']})")
    print(f"  sidecar size     {human(r['sidecar_bytes'])}  ({r['sidecar_bytes']:,} B)")
    print(f"  padding          {human(r['padding_bytes'])}  ({r['padding_pct']:.3f} %)")
    print(f"  read/expert      {r['reads_per_expert']} read of "
          f"{', '.join(human(x) for x in r['read_bytes_per_expert'])}")
    ex = next(iter(r["per_layer"].values()))
    print(f"  example          layer {next(iter(r['per_layer']))}: {ex['offset_formula']} "
          f"(gate at +0, up at +{ex['slice_size']['gate']:,}, down at +{ex['slice_size']['gate'] + ex['slice_size']['up']:,})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", help="path to a .gguf (or a partially downloaded one)")
    ap.add_argument("-o", "--out", help="write the index as JSON to this path")
    ap.add_argument("--align", type=int, default=IO_ALIGN, help="O_DIRECT alignment (default 4096)")
    args = ap.parse_args()

    idx = build_index(args.model, args.align)
    print_summary(idx)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(idx, f, indent=1)
        print(f"\nwrote {args.out} ({os.path.getsize(args.out):,} B)")


if __name__ == "__main__":
    main()
