#!/usr/bin/env python3
"""Make a TokenWing model file: one GGUF that holds the model and everything the engine streams from.

The output is a normal GGUF with the same tensors in the same order with the same bytes and the same
metadata as the input, plus:

  general.alignment = 4096          every tensor starts on a 4096 boundary; with expert slices that are
                                    multiples of 512 (every common quant) each slice is then on a sector
                                    boundary, which is what direct (O_DIRECT) reads need. A drive that needs
                                    4096 reads a slice that is not a 4096-multiple through a copy
  tokenwing.format                  "tokenwing model v1"
  tokenwing.sums.*                  FNV-1a 64-bit checksums of sample experts, checked by --moe-stream-verify
  tokenwing.blob.<name>.offset/size optional extra data appended after the last tensor (4096-aligned):
                                    correction, vram_experts, usage_seed, layer_order

The trailing data keeps the file a valid GGUF: other programs ignore the bytes after the last tensor.
Run the engine on it with --moe-stream model.

A split GGUF (MODEL-00001-of-0000N.gguf ...) whose routed experts are all in its first file is converted file 1 only: the output
must be named OUT-00001-of-0000N.gguf, keeps the split keys, and the other files are used as they are - the tool puts a link
OUT-0000k-of-0000N.gguf next to the output for each of them (a symbolic link, else a hard link), so the loader finds them. A split
model with experts in other files has to be merged first (llama-gguf-split --merge).

The checksums are, for every MoE layer, those of the first, the middle (n_expert // 2) and the last expert's
gate, up and down slice, in that order (9 values per layer, tokenwing.sums.values; the layers are in
tokenwing.sums.layers). The data is streamed in chunks, so a 70 GB model needs no more RAM than a small one.
The file is written under a .tmp name and renamed when it is complete.

Example: python3 make_model_file.py model.gguf -o model.tokenwing.gguf --correction corr.bin
"""

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

GGUF_PY = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gguf-py"))
if GGUF_PY not in sys.path:
    sys.path.insert(0, GGUF_PY)

import gguf  # noqa: E402
from gguf.constants import GGUFValueType  # noqa: E402

IO_ALIGN = 4096
SLICE_ALIGN = 512   # the smallest sector a drive has: every expert slice must be a multiple of it
FORMAT_KEY = "tokenwing.format"
FORMAT_VALUE = "tokenwing model v1"
SUMS_LAYERS_KEY = "tokenwing.sums.layers"
SUMS_VALUES_KEY = "tokenwing.sums.values"
SUMS_ALGO_KEY = "tokenwing.sums.algorithm"
SUMS_ALGO = "fnv1a-64"
BLOB_NAMES = ("correction", "vram_experts", "usage_seed", "layer_order", "quant_plugin_aux")
BLOB_JSON_FORMATS = {
    "vram_experts": "tokenwing vram-experts v1",
    "usage_seed": "tokenwing usage v1",
    "layer_order": "tokenwing layer order v1",
}
CORRECTION_MAGIC = b"TWCORR01"
PROJS = ("gate", "up", "down")
EXPS_RE = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")
MERGED_RE = re.compile(r"^blk\.(\d+)\.ffn_gate_up_exps\.weight$")
SHARD_RE = re.compile(r"^(.+)-(\d{5})-of-(\d{5})\.gguf$")
CHUNK = 32 << 20


class Refuse(Exception):
    pass


def align_up(x, a=IO_ALIGN):
    return (x + a - 1) // a * a


def open_ro(path):
    return os.open(path, os.O_RDONLY | getattr(os, "O_BINARY", 0))   # (O_BINARY: Windows only)


def pread(fd, n, off):
    if hasattr(os, "pread"):
        return os.pread(fd, n, off)
    os.lseek(fd, off, os.SEEK_SET)   # no pread on this system: seek and read (each fd belongs to one thread of control here)
    return os.read(fd, n)


def drop_cache(fd):
    """Let the page cache go, where the system can say so: a 70 GB copy must not push the rest of the machine out of RAM."""
    if hasattr(os, "posix_fadvise"):
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)


def fnv1a64(data):
    h = 0xCBF29CE484222325
    for b in data:
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def sample_experts(n_expert):
    return [0, n_expert // 2, n_expert - 1]


def _sums_job(job):
    """One layer's 9 checksums from (path, layer, [(offset, size), ...] in the order expert-major, projection-minor)."""
    path, il, spans = job
    fd = open_ro(path)
    try:
        out = []
        for off, size in spans:
            buf = pread(fd, size, off)
            if len(buf) != size:
                raise Refuse(f"short read of {size} bytes at {off} in {path}")
            out.append(fnv1a64(buf))
        return il, out
    finally:
        os.close(fd)


def layer_spans(layers, offsets):
    """layers: il -> {proj: (n_expert, slice_bytes)}; offsets: il -> {proj: absolute offset of expert 0 in the file}."""
    jobs = {}
    for il, projs in layers.items():
        n_expert = projs["gate"][0]
        spans = []
        for e in sample_experts(n_expert):
            for p in PROJS:
                spans.append((offsets[il][p] + e * projs[p][1], projs[p][1]))
        jobs[il] = spans
    return jobs


def compute_sums(path, layers, offsets, n_jobs):
    jobs = [(path, il, spans) for il, spans in sorted(layer_spans(layers, offsets).items())]
    res = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as ex:
        for il, vals in ex.map(_sums_job, jobs):
            res[il] = vals
    return res


def find_experts(reader):
    layers = {}   # il -> {proj: (n_expert, slice_bytes)}
    tensors = {}  # il -> {proj: ReaderTensor}
    problems = []
    for t in reader.tensors:
        if MERGED_RE.match(t.name):
            raise Refuse(f"{t.name}: this model keeps gate and up in one tensor, which the engine does not stream")
        m = EXPS_RE.match(t.name)
        if not m:
            continue
        il, proj = int(m.group(1)), m.group(2)
        if len(t.shape) != 3:
            raise Refuse(f"{t.name} has {len(t.shape)} dimensions, an experts tensor has 3")
        n_expert = int(t.shape[2])
        if t.n_bytes % n_expert != 0:
            problems.append(f"{t.name}: {t.n_bytes} bytes do not divide into {n_expert} experts")
            continue
        slice_bytes = t.n_bytes // n_expert
        if slice_bytes % SLICE_ALIGN != 0:
            problems.append(f"{t.name}: an expert slice is {slice_bytes} bytes, not a multiple of {SLICE_ALIGN}")
        layers.setdefault(il, {})[proj] = (n_expert, slice_bytes)
        tensors.setdefault(il, {})[proj] = t
    if problems:
        shown = "\n  ".join(problems[:8])
        more = f"\n  ... and {len(problems) - 8} more" if len(problems) > 8 else ""
        raise Refuse("the expert slices cannot all be read with O_DIRECT, so this model cannot be streamed from its own file:\n  "
                     + shown + more)
    for il, projs in layers.items():
        missing = [p for p in PROJS if p not in projs]
        if missing:
            raise Refuse(f"layer {il} has no ffn_{missing[0]}_exps.weight next to its other expert tensors")
        if len({projs[p][0] for p in PROJS}) != 1:
            raise Refuse(f"layer {il}: gate, up and down disagree on the number of experts")
    if not layers:
        raise Refuse("no routed-expert tensors (blk.N.ffn_{gate,up,down}_exps.weight) in this file: nothing to stream")
    return layers, tensors


def shard_name(path, k, count):
    """The name the loader looks for: file k (1-based) of `count` next to `path` (which ends in -00001-of-NNNNN.gguf)."""
    m = SHARD_RE.match(Path(path).name)
    return str(Path(path).with_name(f"{m.group(1)}-{k:05d}-of-{count:05d}.gguf"))


def check_split(path, reader, out_path):
    """Returns [] for a model in one file, or the paths of files 2..N of a split model whose routed experts are all in file 1."""
    f = reader.fields.get("split.count")
    count = int(f.contents()) if f is not None else 1
    m = SHARD_RE.match(Path(path).name)
    if count <= 1:
        if m:
            raise Refuse(f"{path} is named like one file of a split GGUF but has no split.count: rename it or merge the shards")
        return []
    no = reader.fields.get("split.no")
    if m is None or int(m.group(2)) != 1 or int(m.group(3)) != count or no is None or int(no.contents()) != 0:
        raise Refuse(f"{path} is one file of a split GGUF (split.count = {count}): give the first one, named ...-00001-of-{count:05d}.gguf")
    om = SHARD_RE.match(Path(out_path).name)
    if om is None or int(om.group(2)) != 1 or int(om.group(3)) != count:
        raise Refuse(f"the model is split into {count} files, so the output's name must end in -00001-of-{count:05d}.gguf "
                     f"(the loader finds the other files by it), for example {Path(out_path).stem}-00001-of-{count:05d}.gguf")
    others = []
    for k in range(2, count + 1):
        sp = shard_name(path, k, count)
        if not os.path.isfile(sp):
            raise Refuse(f"file {k} of the split model is missing: {sp}")
        try:
            r = gguf.GGUFReader(sp)
        except (ValueError, OSError) as e:
            raise Refuse(f"{sp} cannot be read as a GGUF file: {e}")
        kno = r.fields.get("split.no")
        if kno is None or int(kno.contents()) != k - 1:
            raise Refuse(f"{sp} does not say it is file {k} of the split model (split.no)")
        exps = [t.name for t in r.tensors if EXPS_RE.match(t.name) or MERGED_RE.match(t.name)]
        if exps:
            raise Refuse(f"{sp} holds routed experts too ({exps[0]} ...): the engine streams from one file, so merge the shards "
                         "first (llama-gguf-split --merge)")
        others.append(sp)
    return others


def link_others(others, out_path, count):
    """Puts OUT-0000k-of-NNNNN.gguf next to the output for every other file of the split model: a symbolic link, else a hard link.
    A file already there is kept when it is that file."""
    for k, sp in enumerate(others, start=2):
        dst = shard_name(out_path, k, count)
        if os.path.lexists(dst):
            if os.path.exists(dst) and os.path.samefile(dst, sp):
                print(f"{dst}: already {sp}", flush=True)
                continue
            raise Refuse(f"{dst} exists and is not {sp}: move it away first")
        try:
            os.symlink(sp, dst)
            print(f"{dst}: a symbolic link to {sp}", flush=True)
        except OSError as e1:
            try:
                os.link(sp, dst)
                print(f"{dst}: a hard link to {sp}", flush=True)
            except OSError as e2:
                raise Refuse(f"cannot place {dst} next to the output ({e1}; {e2}): copy or link {sp} there yourself")


def check_blob(name, path):
    p = Path(path)
    if not p.is_file():
        raise Refuse(f"--{name.replace('_', '-')} {path}: no such file")
    size = p.stat().st_size
    if size == 0:
        raise Refuse(f"{path} is empty")
    with open(p, "rb") as f:
        head = f.read(8)
    if name == "correction":
        if head != CORRECTION_MAGIC:
            raise Refuse(f"{path} is not a TokenWing correction file (bad magic)")
    else:
        try:
            with open(p, "rb") as f:
                j = json.load(f)
        except (ValueError, UnicodeDecodeError) as e:
            raise Refuse(f"{path} is not valid JSON: {e}")
        if not isinstance(j, dict) or j.get("format") != BLOB_JSON_FORMATS[name]:
            raise Refuse(f"{path} is not a \"{BLOB_JSON_FORMATS[name]}\" file")
    return size


def build_writer(reader, arch, layers, sums, blobs):
    """blobs: name -> (offset, size). Returns a writer holding the KV data and the tensor infos (nothing written yet)."""
    w = gguf.GGUFWriter(Path("unused"), arch)   # (writes general.architecture itself)
    for field in reader.fields.values():
        name = field.name
        if name == gguf.Keys.General.ARCHITECTURE or name.startswith("GGUF.") or name == gguf.Keys.General.ALIGNMENT \
                or name.startswith("tokenwing."):
            continue
        vtype = field.types[0]
        sub = field.types[-1] if vtype == GGUFValueType.ARRAY else None
        w.add_key_value(name, field.contents(), vtype, sub_type=sub)
    w.add_custom_alignment(IO_ALIGN)
    w.add_string(FORMAT_KEY, FORMAT_VALUE)
    w.add_string(SUMS_ALGO_KEY, SUMS_ALGO)
    il_list = sorted(layers)
    w.add_key_value(SUMS_LAYERS_KEY, il_list, GGUFValueType.ARRAY, sub_type=GGUFValueType.UINT32)
    w.add_key_value(SUMS_VALUES_KEY, [v for il in il_list for v in sums[il]], GGUFValueType.ARRAY, sub_type=GGUFValueType.UINT64)
    for name in BLOB_NAMES:
        if name in blobs:
            off, size = blobs[name]
            w.add_uint64(f"tokenwing.blob.{name}.offset", off)
            w.add_uint64(f"tokenwing.blob.{name}.size", size)
    for t in reader.tensors:
        w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    return w


def header_length(w, tmp_path):
    w.write_header_to_file(Path(tmp_path))
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    n = w.fout[0].tell()
    w.close()
    os.remove(tmp_path)
    return n


def copy_range(src_fd, dst, off, n, hashers=()):
    done = 0
    while done < n:
        buf = pread(src_fd, min(CHUNK, n - done), off + done)
        if not buf:
            raise Refuse("the source file ended early")
        dst.write(buf)
        for h in hashers:
            h.update(buf)
        done += len(buf)
        if done % (1 << 30) < CHUNK:
            dst.flush()
            os.fsync(dst.fileno())
            drop_cache(dst.fileno())
            drop_cache(src_fd)


def hash_range(fd, off, n, hashers):
    done = 0
    while done < n:
        buf = pread(fd, min(CHUNK, n - done), off + done)
        if not buf:
            raise Refuse("a file ended early while it was checked")
        for h in hashers:
            h.update(buf)
        done += len(buf)
    drop_cache(fd)


def check_output(src_path, out_path, src_reader, layers, sums, blobs, blob_files, n_jobs):
    """Reads the finished file back: metadata, tensor table, every tensor's bytes, the checksums, the blobs."""
    print("check: reading the output back ...", flush=True)
    t0 = time.time()
    out = gguf.GGUFReader(out_path)
    bad = []
    if out.alignment != IO_ALIGN or out.data_offset % IO_ALIGN != 0:
        bad.append(f"alignment {out.alignment}, data offset {out.data_offset}")
    if len(out.tensors) != len(src_reader.tensors):
        bad.append(f"{len(out.tensors)} tensors, the source has {len(src_reader.tensors)}")
    for a, b in zip(src_reader.tensors, out.tensors):
        if (a.name, a.tensor_type, tuple(a.shape), a.n_bytes) != (b.name, b.tensor_type, tuple(b.shape), b.n_bytes):
            bad.append(f"tensor {a.name}: name, type, shape or size differs from the output's {b.name}")
        if b.data_offset % IO_ALIGN != 0:
            bad.append(f"tensor {b.name} starts at {b.data_offset}, not {IO_ALIGN}-aligned")
    n_kv = 0
    for field in src_reader.fields.values():
        name = field.name
        if name.startswith("GGUF.") or name == gguf.Keys.General.ALIGNMENT or name.startswith("tokenwing."):
            continue
        g = out.fields.get(name)
        if g is None:
            bad.append(f"key {name} is missing in the output")
            continue
        va, vb = field.contents(), g.contents()
        if (list(va) if hasattr(va, "__len__") and not isinstance(va, (str, bytes)) else va) != \
                (list(vb) if hasattr(vb, "__len__") and not isinstance(vb, (str, bytes)) else vb):
            bad.append(f"key {name} has another value in the output")
        n_kv += 1
    if out.fields[FORMAT_KEY].contents() != FORMAT_VALUE:
        bad.append(f"{FORMAT_KEY} is wrong")
    if bad:
        raise Refuse("check failed:\n  " + "\n  ".join(bad[:12]))
    print(f"check: {n_kv} metadata keys equal, {len(out.tensors)} tensors with the same names, types, shapes and sizes, all "
          f"{IO_ALIGN}-aligned", flush=True)

    # every tensor's bytes, both files, one running hash over all of them in file order
    hs, hd = hashlib.sha256(), hashlib.sha256()
    sfd, dfd = open_ro(src_path), open_ro(out_path)
    total = 0
    try:
        for a, b in zip(src_reader.tensors, out.tensors):
            ta, tb = hashlib.sha256(), hashlib.sha256()
            hash_range(sfd, a.data_offset, a.n_bytes, (hs, ta))
            hash_range(dfd, b.data_offset, b.n_bytes, (hd, tb))
            if ta.digest() != tb.digest():
                raise Refuse(f"check failed: the bytes of tensor {a.name} differ")
            total += a.n_bytes
    finally:
        os.close(sfd)
        os.close(dfd)
    if hs.digest() != hd.digest():
        raise Refuse("check failed: the whole-file tensor-data hash differs")
    print(f"check: all {total} tensor bytes identical, sha256 over all tensor data {hd.hexdigest()}", flush=True)

    # the sample checksums, recomputed from the output's own offsets
    offs = {il: {p: out.tensors[[t.name for t in out.tensors].index(f"blk.{il}.ffn_{p}_exps.weight")].data_offset for p in PROJS}
            for il in layers}
    again = compute_sums(out_path, layers, offs, n_jobs)
    for il in layers:
        if again[il] != sums[il]:
            raise Refuse(f"check failed: layer {il} checksums read back from the output differ")
    sums_in_file = [int(v) for v in out.fields[SUMS_VALUES_KEY].contents()]
    if sums_in_file != [v for il in sorted(layers) for v in sums[il]]:
        raise Refuse("check failed: the checksums stored in the metadata differ from the ones computed")
    print(f"check: checksums of {len(layers) * 9} sample slices read back from the output match the metadata", flush=True)

    ofd = open_ro(out_path)
    try:
        for name, (off, size) in blobs.items():
            if (int(out.fields[f"tokenwing.blob.{name}.offset"].contents()) != off or
                    int(out.fields[f"tokenwing.blob.{name}.size"].contents()) != size or off % IO_ALIGN != 0):
                raise Refuse(f"check failed: the metadata of blob {name} is wrong")
            ha, hb = hashlib.sha256(), hashlib.sha256()
            with open(blob_files[name], "rb") as f:
                for chunk in iter(lambda: f.read(CHUNK), b""):
                    ha.update(chunk)
            hash_range(ofd, off, size, (hb,))
            if ha.digest() != hb.digest():
                raise Refuse(f"check failed: blob {name} differs from its file")
            print(f"check: blob {name}: {size} bytes at {off}, identical to {blob_files[name]}", flush=True)
    finally:
        os.close(ofd)
    print(f"check: done in {time.time() - t0:.1f} s", flush=True)


def check_others(out_path, others, reader):
    """The loader's view of a converted split model: the other files are found next to the output and are the model's own."""
    n_split = len(others) + 1
    n_tensors = len(reader.tensors)
    for k, sp in enumerate(others, start=2):
        dst = shard_name(out_path, k, n_split)
        if not (os.path.exists(dst) and os.path.samefile(dst, sp)):
            raise Refuse(f"check failed: {dst} is not {sp}")
        n_tensors += len(gguf.GGUFReader(dst).tensors)
    f = reader.fields.get("split.tensors.count")
    if f is not None and int(f.contents()) != n_tensors:
        raise Refuse(f"check failed: the files hold {n_tensors} tensors, split.tensors.count says {int(f.contents())}")
    print(f"check: files 2..{n_split} found next to the output and identical to the model's; {n_tensors} tensors in all, as "
          f"split.tensors.count says", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Make a TokenWing model file (one GGUF to stream the experts from) out of a GGUF.")
    ap.add_argument("model", help="the model's GGUF (one file, or the first file of a split model whose experts are all in it)")
    ap.add_argument("-o", "--output", required=True, help="the file to write (written as OUTPUT.tmp, renamed when complete)")
    ap.add_argument("--correction", help="a TokenWing correction file to append (used by --moe-stream-correction model)")
    ap.add_argument("--vram-list", help="a 'tokenwing vram-experts v1' JSON to append (used by --moe-stream-vram-experts-file model)")
    ap.add_argument("--usage-seed", help="a 'tokenwing usage v1' JSON to append (the usage table's first seed)")
    ap.add_argument("--layer-order", help="a 'tokenwing layer order v1' JSON to append (the hot-first layer order)")
    ap.add_argument("--quant-plugin", help="shared library (.so / .dll) implementing ggml_quant_plugin_get_api for external quant conversion")
    ap.add_argument("--no-check", action="store_true", help="skip reading the finished file back and comparing it with the source")
    ap.add_argument("--jobs", type=int, default=0, help="processes for the checksums (default: up to 8)")
    args = ap.parse_args()

    src = os.path.abspath(args.model)
    out_path = os.path.abspath(args.output)
    tmp_path = out_path + ".tmp"
    n_jobs = args.jobs if args.jobs > 0 else max(1, min(8, os.cpu_count() or 1))

    try:
        if os.path.abspath(src) == out_path:
            raise Refuse("the output must be another file than the model")
        blob_files = {}
        for name, val in (("correction", args.correction), ("vram_experts", args.vram_list), ("usage_seed", args.usage_seed),
                          ("layer_order", args.layer_order)):
            if val:
                blob_files[name] = val
        blob_sizes = {name: check_blob(name, p) for name, p in blob_files.items()}

        t_start = time.time()
        if args.quant_plugin:
            from quant_plugin_convert import convert_with_quant_plugin
            convert_with_quant_plugin(
                sys.modules[__name__], src, out_path, tmp_path, args.quant_plugin,
                blob_files, blob_sizes, n_jobs, args.no_check, t_start,
            )
            return
        if not os.path.isfile(src):
            raise Refuse(f"{src}: no such file")
        try:
            reader = gguf.GGUFReader(src)
        except (ValueError, OSError) as e:
            raise Refuse(f"{src} cannot be read as a GGUF file: {e}")
        others = check_split(src, reader, out_path)
        n_split = len(others) + 1
        if reader.endianess != gguf.GGUFEndian.LITTLE:
            raise Refuse("a big-endian GGUF cannot be converted")
        arch = reader.get_field(gguf.Keys.General.ARCHITECTURE).contents()
        layers, ltensors = find_experts(reader)
        n_expert = {n for projs in layers.values() for n, _ in projs.values()}
        n_4k = sum(1 for projs in layers.values() for _, sb in projs.values() if sb % IO_ALIGN != 0)
        print(f"{src}: arch {arch}, {len(reader.tensors)} tensors, {len(layers)} MoE layers, {sorted(n_expert)} experts per layer; "
              f"every expert slice is a multiple of {SLICE_ALIGN} bytes"
              + (f" ({n_4k} of {3 * len(layers)} are not multiples of {IO_ALIGN}: a drive that needs {IO_ALIGN}-byte direct "
                 f"reads reads those through a copy)" if n_4k else f" and of {IO_ALIGN}"), flush=True)

        offsets = {il: {p: ltensors[il][p].data_offset for p in PROJS} for il in layers}
        print(f"checksums of the first, middle and last expert of every layer ({n_jobs} processes) ...", flush=True)
        sums = compute_sums(src, layers, offsets, n_jobs)

        # the header's length does not depend on the values of the blobs' offsets (fixed-width integers), so it is
        # measured once with placeholders and the real offsets are then known before anything is written
        placeholder = {name: (0, blob_sizes[name]) for name in blob_files}
        hlen = header_length(build_writer(reader, arch, layers, sums, placeholder), tmp_path)
        pos = align_up(hlen)
        for t in reader.tensors:
            pos = align_up(pos + t.data.nbytes) if t is not reader.tensors[-1] else pos + t.data.nbytes
        blobs = {}
        pos = align_up(pos)
        for name in BLOB_NAMES:
            if name in blob_files:
                blobs[name] = (pos, blob_sizes[name])
                pos = align_up(pos + blob_sizes[name])
        w = build_writer(reader, arch, layers, sums, blobs)

        if others:
            print(f"split model: file 1 of {n_split} holds every routed expert and is converted; the other files are used as they are",
                  flush=True)
            link_others(others, out_path, n_split)

        free = shutil.disk_usage(os.path.dirname(out_path) or ".").free
        need = pos + (1 << 20)
        if free < need:
            raise Refuse(f"{os.path.dirname(out_path)} has {free / 1e9:.1f} GB free, the file needs {need / 1e9:.1f} GB")

        w.write_header_to_file(Path(tmp_path))
        w.write_kv_data_to_file()
        w.write_ti_data_to_file()
        fout = w.fout[0]
        if fout.tell() != hlen:
            raise Refuse("internal error: the header length changed")
        print(f"writing {tmp_path} ({pos / 1e9:.2f} GB) ...", flush=True)
        sfd = open_ro(src)
        done = 0
        try:
            for t in reader.tensors:
                w.write_padding(fout, fout.tell())
                copy_range(sfd, fout, t.data_offset, t.n_bytes)
                done += t.n_bytes
            for name in BLOB_NAMES:
                if name in blobs:
                    w.write_padding(fout, fout.tell(), IO_ALIGN)
                    if fout.tell() != blobs[name][0]:
                        raise Refuse(f"internal error: blob {name} would start at {fout.tell()}, not {blobs[name][0]}")
                    bfd = open_ro(blob_files[name])
                    try:
                        copy_range(bfd, fout, 0, blobs[name][1])
                    finally:
                        os.close(bfd)
            fout.flush()
            os.fsync(fout.fileno())
        finally:
            os.close(sfd)
        w.close()
        os.replace(tmp_path, out_path)
        print(f"wrote {out_path}: {os.path.getsize(out_path)} bytes (source {os.path.getsize(src)}), {done} tensor bytes, "
              f"{time.time() - t_start:.1f} s", flush=True)

        if not args.no_check:
            check_output(src, out_path, reader, layers, sums, blobs, blob_files, n_jobs)
            if others:
                check_others(out_path, others, reader)
    except Refuse as e:
        print(f"make_model_file: {e}", file=sys.stderr)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        sys.exit(1)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


if __name__ == "__main__":
    main()
