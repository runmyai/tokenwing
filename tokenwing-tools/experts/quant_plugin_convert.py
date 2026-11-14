"""External GGML quantization plugin bridge for make_model_file.py."""

import ctypes
import hashlib
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import gguf
from gguf.constants import GGUFValueType

PLUGIN_ABI_VERSION = 1
PLUGIN_EXT_BASE = 43
PLUGIN_MAX_TYPES = 4
ACTION_KEEP = 0
ACTION_REHEADER = 1
ACTION_TRANSCODE = 2
SPLIT_KEYS = ("split.no", "split.count", "split.tensors.count")


class _PluginTypeInfo(ctypes.Structure):
    _fields_ = [
        ("enabled", ctypes.c_int32),
        ("type_name", ctypes.c_char_p),
        ("blck_size", ctypes.c_int64),
        ("type_size", ctypes.c_size_t),
        ("vec_dot_type", ctypes.c_int32),
        ("has_cuda", ctypes.c_int32),
        ("to_float", ctypes.c_void_p),
        ("from_float_ref", ctypes.c_void_p),
        ("vec_dot", ctypes.c_void_p),
    ]


class _PluginTensorPlan(ctypes.Structure):
    _fields_ = [
        ("action", ctypes.c_int32),
        ("out_type", ctypes.c_int32),
        ("out_n_dims", ctypes.c_int32),
        ("reserved", ctypes.c_int32),
        ("out_ne", ctypes.c_uint64 * 4),
        ("src_nbytes", ctypes.c_uint64),
        ("out_nbytes", ctypes.c_uint64),
    ]


class _PluginAPI(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("plugin_name", ctypes.c_char_p),
        ("plugin_version", ctypes.c_char_p),
        ("init", ctypes.CFUNCTYPE(ctypes.c_int32)),
        ("shutdown", ctypes.CFUNCTYPE(None)),
        ("last_error", ctypes.CFUNCTYPE(ctypes.c_char_p)),
        ("get_type_info", ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(_PluginTypeInfo))),
        ("load_aux_blob", ctypes.c_void_p),
        ("mul_mat_cuda", ctypes.c_void_p),
        ("mul_mat_id_cuda", ctypes.c_void_p),
        ("converter_create", ctypes.CFUNCTYPE(ctypes.c_void_p)),
        ("converter_destroy", ctypes.CFUNCTYPE(None, ctypes.c_void_p)),
        ("converter_probe_tensor", ctypes.CFUNCTYPE(
            ctypes.c_int32, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32,
            ctypes.c_int32, ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint64, ctypes.POINTER(_PluginTensorPlan))),
        ("converter_planned_aux_size", ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_void_p)),
        ("converter_transcode_tensor", ctypes.CFUNCTYPE(
            ctypes.c_int32, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32,
            ctypes.c_int32, ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t)),
        ("converter_finalize_aux", ctypes.CFUNCTYPE(
            ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t))),
    ]


class _RawTensor:
    def __init__(self, name, raw_dtype, dims, data_offset):
        self.name = name
        self.raw_dtype = int(raw_dtype)
        self.dims = tuple(int(d) for d in dims.tolist())
        self.data_offset = int(data_offset)


class _RawGGUFReader(gguf.GGUFReader):
    def _build_tensors(self, start_offs, fields):
        tensors, seen = [], set()
        for field in fields:
            _name_len, name_data, _n_dims, dims, raw_dtype, offset_tensor = field.parts
            name = str(bytes(name_data), encoding="utf-8")
            if name in seen:
                raise ValueError(f"Found duplicated tensor with name {name}")
            seen.add(name)
            tensors.append(_RawTensor(name, raw_dtype[0], dims, int(start_offs + offset_tensor[0])))
        self.tensors = tensors


class _PlannedTensor:
    def __init__(self, shard_path, name, raw_dtype, src_dims, data_offset, action, out_type, out_dims, src_nbytes, out_nbytes):
        self.shard_path = shard_path
        self.name = name
        self.raw_dtype = raw_dtype
        self.src_dims = src_dims
        self.data_offset = data_offset
        self.action = action
        self.out_type = out_type
        self.out_dims = out_dims
        self.src_nbytes = src_nbytes
        self.out_nbytes = out_nbytes


class QuantPluginBridge:
    def __init__(self, plugin_path, refuse_cls):
        self._refuse = refuse_cls
        try:
            self.lib = ctypes.CDLL(os.path.abspath(plugin_path))
        except OSError as e:
            raise refuse_cls(f"--quant-plugin {plugin_path}: cannot load shared library: {e}")
        try:
            get_api = self.lib.ggml_quant_plugin_get_api
        except AttributeError:
            raise refuse_cls(f"--quant-plugin {plugin_path}: missing ggml_quant_plugin_get_api symbol")
        get_api.restype = ctypes.POINTER(_PluginAPI)
        api_ptr = get_api()
        if not api_ptr:
            raise refuse_cls(f"--quant-plugin {plugin_path}: ggml_quant_plugin_get_api returned NULL")
        self.api = api_ptr.contents
        if self.api.abi_version != PLUGIN_ABI_VERSION:
            raise refuse_cls(f"--quant-plugin {plugin_path}: unsupported ABI version {self.api.abi_version}")
        if bool(self.api.init) and self.api.init() != 0:
            raise refuse_cls(f"--quant-plugin {plugin_path}: init failed: {self.last_error()}")
        for fn_name in ("converter_create", "converter_destroy", "converter_probe_tensor",
                        "converter_planned_aux_size", "converter_transcode_tensor", "converter_finalize_aux"):
            if not bool(getattr(self.api, fn_name)):
                raise refuse_cls(f"--quant-plugin {plugin_path}: plugin does not implement {fn_name}")
        for idx in range(PLUGIN_MAX_TYPES):
            ti = _PluginTypeInfo()
            if self.api.get_type_info(idx, ctypes.byref(ti)) == 0 and ti.enabled:
                gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType(PLUGIN_EXT_BASE + idx)] = (int(ti.blck_size), int(ti.type_size))

    def last_error(self):
        if bool(self.api.last_error):
            msg = self.api.last_error()
            if msg:
                return msg.decode("utf-8", errors="replace")
        return "unknown plugin error"


def _open_raw_reader(path, refuse_cls):
    try:
        return _RawGGUFReader(path)
    except (ValueError, OSError) as e:
        raise refuse_cls(f"{path} cannot be read as a GGUF file: {e}")


def _probe_shard_tensors(mmf, plugin, conv, shard_path, reader):
    fsz = os.path.getsize(shard_path)
    uniq_offs = sorted({t.data_offset for t in reader.tensors})
    next_map = {uniq_offs[i]: (uniq_offs[i + 1] if i + 1 < len(uniq_offs) else fsz) for i in range(len(uniq_offs))}
    fd = mmf.open_ro(shard_path)
    planned = []
    try:
        for t in reader.tensors:
            max_span = max(0, next_map[t.data_offset] - t.data_offset)
            head = mmf.pread(fd, min(65536, max_span), t.data_offset) if max_span > 0 else b""
            ne_arr = (ctypes.c_uint64 * max(1, len(t.dims)))(*t.dims) if t.dims else (ctypes.c_uint64 * 1)(0)
            head_buf = (ctypes.c_uint8 * len(head)).from_buffer_copy(head) if head else None
            head_ptr = ctypes.cast(head_buf, ctypes.c_void_p) if head_buf is not None else ctypes.c_void_p(0)
            plan = _PluginTensorPlan()
            rc = plugin.api.converter_probe_tensor(
                conv, t.name.encode("utf-8"), t.raw_dtype, len(t.dims), ne_arr,
                head_ptr, len(head), max_span, ctypes.byref(plan),
            )
            if rc != 0:
                raise mmf.Refuse(f"{t.name}: quant plugin probe failed: {plugin.last_error()}")
            if plan.action == ACTION_KEEP:
                out_type = gguf.GGMLQuantizationType(t.raw_dtype)
                out_dims = t.dims
                blck_size, type_size = gguf.GGML_QUANT_SIZES[out_type]
                if type_size == 0 or math.prod(out_dims) % blck_size != 0:
                    raise mmf.Refuse(f"{t.name}: type {out_type.name} has invalid block geometry")
                src_nb = out_nb = math.prod(out_dims) // blck_size * type_size
            elif plan.action in (ACTION_REHEADER, ACTION_TRANSCODE):
                out_type = gguf.GGMLQuantizationType(int(plan.out_type))
                out_dims = tuple(int(plan.out_ne[d]) for d in range(int(plan.out_n_dims)))
                src_nb, out_nb = int(plan.src_nbytes), int(plan.out_nbytes)
                blck_size, type_size = gguf.GGML_QUANT_SIZES[out_type]
                if type_size == 0 or not out_dims or math.prod(out_dims) % blck_size != 0 or \
                        math.prod(out_dims) // blck_size * type_size != out_nb:
                    raise mmf.Refuse(f"{t.name}: plugin returned mismatched shape {out_dims} / size {out_nb} for {out_type.name}")
            else:
                raise mmf.Refuse(f"{t.name}: unknown plugin tensor action {plan.action}")
            if t.data_offset + src_nb > fsz:
                raise mmf.Refuse(f"{t.name}: source span ({src_nb} bytes at {t.data_offset}) exceeds file size {fsz}")
            planned.append(_PlannedTensor(
                shard_path, t.name, t.raw_dtype, t.dims, t.data_offset,
                int(plan.action), out_type, out_dims, src_nb, out_nb,
            ))
    finally:
        os.close(fd)
    return planned


def _resolve_plugin_shards(mmf, src, reader, out_path):
    f = reader.fields.get("split.count")
    count = int(f.contents()) if f is not None else 1
    m = mmf.SHARD_RE.match(Path(src).name)
    if count <= 1:
        if m:
            raise mmf.Refuse(f"{src} is named like one file of a split GGUF but has no split.count: rename it or merge the shards")
        return [src], [], False
    no = reader.fields.get("split.no")
    if m is None or int(m.group(2)) != 1 or int(m.group(3)) != count or no is None or int(no.contents()) != 0:
        raise mmf.Refuse(f"{src} is one file of a split GGUF (split.count = {count}): give the first one, named ...-00001-of-{count:05d}.gguf")
    others = []
    for k in range(2, count + 1):
        sp = mmf.shard_name(src, k, count)
        if not os.path.isfile(sp):
            raise mmf.Refuse(f"file {k} of the split model is missing: {sp}")
        r = _open_raw_reader(sp, mmf.Refuse)
        kno = r.fields.get("split.no")
        if kno is None or int(kno.contents()) != k - 1:
            raise mmf.Refuse(f"{sp} does not say it is file {k} of the split model (split.no)")
        others.append((sp, r))
    om = mmf.SHARD_RE.match(Path(out_path).name)
    if om is None:
        return [src] + [sp for sp, _ in others], [], True
    if int(om.group(2)) != 1 or int(om.group(3)) != count:
        raise mmf.Refuse(f"the model is split into {count} files, so a split output must end in -00001-of-{count:05d}.gguf "
                         f"(or give a non-shard output filename to merge all {count} shards into one file)")
    for sp, r in others:
        exps = [t.name for t in r.tensors if mmf.EXPS_RE.match(t.name) or mmf.MERGED_RE.match(t.name)]
        if exps:
            raise mmf.Refuse(f"{sp} holds routed experts too ({exps[0]} ...): give a single output filename (not -00001-of-...) to merge")
    return [src], [sp for sp, _ in others], False


def _find_experts_planned(mmf, planned_tensors):
    layers, problems = {}, []
    for pt in planned_tensors:
        if mmf.MERGED_RE.match(pt.name):
            raise mmf.Refuse(f"{pt.name}: this model keeps gate and up in one tensor, which the engine does not stream")
        m = mmf.EXPS_RE.match(pt.name)
        if not m:
            continue
        il, proj = int(m.group(1)), m.group(2)
        if len(pt.out_dims) != 3:
            raise mmf.Refuse(f"{pt.name} has {len(pt.out_dims)} dimensions, an experts tensor has 3")
        n_expert = int(pt.out_dims[2])
        if pt.out_nbytes % n_expert != 0:
            problems.append(f"{pt.name}: {pt.out_nbytes} bytes do not divide into {n_expert} experts")
            continue
        slice_bytes = pt.out_nbytes // n_expert
        if slice_bytes % mmf.SLICE_ALIGN != 0:
            problems.append(f"{pt.name}: an expert slice is {slice_bytes} bytes, not a multiple of {mmf.SLICE_ALIGN}")
        layers.setdefault(il, {})[proj] = (n_expert, slice_bytes)
    if problems:
        shown = "\n  ".join(problems[:8])
        more = f"\n  ... and {len(problems) - 8} more" if len(problems) > 8 else ""
        raise mmf.Refuse("the expert slices cannot all be read with O_DIRECT, so this model cannot be streamed from its own file:\n  "
                         + shown + more)
    for il, projs in layers.items():
        missing = [p for p in mmf.PROJS if p not in projs]
        if missing:
            raise mmf.Refuse(f"layer {il} has no ffn_{missing[0]}_exps.weight next to its other expert tensors")
        if len({projs[p][0] for p in mmf.PROJS}) != 1:
            raise mmf.Refuse(f"layer {il}: gate, up and down disagree on the number of experts")
    if not layers:
        raise mmf.Refuse("no routed-expert tensors (blk.N.ffn_{gate,up,down}_exps.weight) in this file: nothing to stream")
    return layers


def _build_plugin_writer(mmf, reader, arch, planned_tensors, layers, sums, blobs, merge_split):
    w = gguf.GGUFWriter(Path("unused"), arch)
    for field in reader.fields.values():
        name = field.name
        if name == gguf.Keys.General.ARCHITECTURE or name.startswith("GGUF.") or name == gguf.Keys.General.ALIGNMENT \
                or name.startswith("tokenwing.") or (merge_split and name in SPLIT_KEYS):
            continue
        vtype = field.types[0]
        sub = field.types[-1] if vtype == GGUFValueType.ARRAY else None
        w.add_key_value(name, field.contents(), vtype, sub_type=sub)
    w.add_custom_alignment(mmf.IO_ALIGN)
    w.add_string(mmf.FORMAT_KEY, mmf.FORMAT_VALUE)
    w.add_string(mmf.SUMS_ALGO_KEY, mmf.SUMS_ALGO)
    il_list = sorted(layers)
    w.add_key_value(mmf.SUMS_LAYERS_KEY, il_list, GGUFValueType.ARRAY, sub_type=GGUFValueType.UINT32)
    w.add_key_value(mmf.SUMS_VALUES_KEY, [v for il in il_list for v in sums[il]], GGUFValueType.ARRAY, sub_type=GGUFValueType.UINT64)
    for name in mmf.BLOB_NAMES:
        if name in blobs:
            off, size = blobs[name]
            w.add_uint64(f"tokenwing.blob.{name}.offset", off)
            w.add_uint64(f"tokenwing.blob.{name}.size", size)
    for pt in planned_tensors:
        w.add_tensor_info(pt.name, tuple(reversed(pt.out_dims)), np.dtype(np.float32), pt.out_nbytes, raw_dtype=pt.out_type)
    return w


def _header_bytes(w, hdr_tmp_path):
    w.write_header_to_file(Path(hdr_tmp_path))
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    w.close()
    try:
        return Path(hdr_tmp_path).read_bytes()
    finally:
        if os.path.exists(hdr_tmp_path):
            os.remove(hdr_tmp_path)


def _check_plugin_output(mmf, out_path, src_reader, planned_tensors, layers, sums, blobs, blob_files, aux_bytes, merge_split, n_jobs):
    print("check: reading the output back ...", flush=True)
    t0 = time.time()
    out = gguf.GGUFReader(out_path)
    bad = []
    if out.alignment != mmf.IO_ALIGN or out.data_offset % mmf.IO_ALIGN != 0:
        bad.append(f"alignment {out.alignment}, data offset {out.data_offset}")
    if len(out.tensors) != len(planned_tensors):
        bad.append(f"{len(out.tensors)} tensors, expected {len(planned_tensors)}")
    for pt, b in zip(planned_tensors, out.tensors):
        b_dims = tuple(int(d) for d in b.shape.tolist())
        if (pt.name, pt.out_type, pt.out_dims, pt.out_nbytes) != (b.name, b.tensor_type, b_dims, b.n_bytes):
            bad.append(f"tensor {pt.name}: name, type, shape or size differs from output {b.name}")
        if b.data_offset % mmf.IO_ALIGN != 0:
            bad.append(f"tensor {b.name} starts at {b.data_offset}, not {mmf.IO_ALIGN}-aligned")
    n_kv = 0
    for field in src_reader.fields.values():
        name = field.name
        if name.startswith("GGUF.") or name == gguf.Keys.General.ALIGNMENT or name.startswith("tokenwing.") \
                or (merge_split and name in SPLIT_KEYS):
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
    if merge_split and any(k in out.fields for k in SPLIT_KEYS):
        bad.append("split metadata keys remained in merged single-file output")
    if out.fields[mmf.FORMAT_KEY].contents() != mmf.FORMAT_VALUE:
        bad.append(f"{mmf.FORMAT_KEY} is wrong")
    if bad:
        raise mmf.Refuse("check failed:\n  " + "\n  ".join(bad[:12]))
    print(f"check: {n_kv} metadata keys equal, {len(out.tensors)} tensors with expected types, shapes and sizes, all "
          f"{mmf.IO_ALIGN}-aligned", flush=True)

    dfd = mmf.open_ro(out_path)
    sfds = {}
    checked_raw = 0
    try:
        for pt, b in zip(planned_tensors, out.tensors):
            if pt.action not in (ACTION_KEEP, ACTION_REHEADER):
                continue
            if pt.shard_path not in sfds:
                sfds[pt.shard_path] = mmf.open_ro(pt.shard_path)
            ta, tb = hashlib.sha256(), hashlib.sha256()
            mmf.hash_range(sfds[pt.shard_path], pt.data_offset, pt.src_nbytes, (ta,))
            mmf.hash_range(dfd, b.data_offset, b.n_bytes, (tb,))
            if ta.digest() != tb.digest():
                raise mmf.Refuse(f"check failed: passthrough bytes of tensor {pt.name} differ")
            checked_raw += b.n_bytes
        for name, (off, size) in blobs.items():
            if (int(out.fields[f"tokenwing.blob.{name}.offset"].contents()) != off or
                    int(out.fields[f"tokenwing.blob.{name}.size"].contents()) != size or off % mmf.IO_ALIGN != 0):
                raise mmf.Refuse(f"check failed: the metadata of blob {name} is wrong")
            ha, hb = hashlib.sha256(), hashlib.sha256()
            if name == "quant_plugin_aux":
                ha.update(aux_bytes)
            else:
                with open(blob_files[name], "rb") as f:
                    for chunk in iter(lambda: f.read(mmf.CHUNK), b""):
                        ha.update(chunk)
            mmf.hash_range(dfd, off, size, (hb,))
            if ha.digest() != hb.digest():
                raise mmf.Refuse(f"check failed: blob {name} differs from its expected bytes")
            print(f"check: blob {name}: {size} bytes at {off} verified", flush=True)
    finally:
        for sfd in sfds.values():
            os.close(sfd)
        os.close(dfd)

    name_to_off = {t.name: t.data_offset for t in out.tensors}
    offs = {il: {p: name_to_off[f"blk.{il}.ffn_{p}_exps.weight"] for p in mmf.PROJS} for il in layers}
    again = mmf.compute_sums(out_path, layers, offs, n_jobs)
    for il in layers:
        if again[il] != sums[il]:
            raise mmf.Refuse(f"check failed: layer {il} checksums read back from the output differ")
    sums_in_file = [int(v) for v in out.fields[mmf.SUMS_VALUES_KEY].contents()]
    if sums_in_file != [v for il in sorted(layers) for v in sums[il]]:
        raise mmf.Refuse("check failed: the checksums stored in the metadata differ from the ones computed")
    print(f"check: {checked_raw} passthrough bytes identical, checksums of {len(layers) * 9} sample slices verified in "
          f"{time.time() - t0:.1f} s", flush=True)


def convert_with_quant_plugin(mmf, src, out_path, tmp_path, plugin_path, blob_files, blob_sizes, n_jobs, no_check, t_start):
    plugin = QuantPluginBridge(plugin_path, mmf.Refuse)
    if not os.path.isfile(src):
        raise mmf.Refuse(f"{src}: no such file")
    reader = _open_raw_reader(src, mmf.Refuse)
    if reader.endianess != gguf.GGUFEndian.LITTLE:
        raise mmf.Refuse("a big-endian GGUF cannot be converted")
    arch = reader.get_field(gguf.Keys.General.ARCHITECTURE).contents()
    shards_to_stream, others_to_link, merge_split = _resolve_plugin_shards(mmf, src, reader, out_path)

    if others_to_link:
        chk_conv = plugin.api.converter_create()
        if not chk_conv:
            raise mmf.Refuse("quant plugin converter_create returned NULL")
        try:
            for sp in others_to_link:
                pts = _probe_shard_tensors(mmf, plugin, chk_conv, sp, _open_raw_reader(sp, mmf.Refuse))
                changed = [p.name for p in pts if p.action != ACTION_KEEP]
                if changed:
                    raise mmf.Refuse(f"{sp} holds plugin-converted tensors ({changed[0]} ...): give a non-shard -o filename to merge")
        finally:
            plugin.api.converter_destroy(chk_conv)

    hdr_tmp_path = tmp_path + ".hdr"
    conv = plugin.api.converter_create()
    if not conv:
        raise mmf.Refuse("quant plugin converter_create returned NULL")
    try:
        planned_tensors = []
        for idx, sp in enumerate(shards_to_stream):
            sr = reader if idx == 0 else _open_raw_reader(sp, mmf.Refuse)
            planned_tensors.extend(_probe_shard_tensors(mmf, plugin, conv, sp, sr))

        layers = _find_experts_planned(mmf, planned_tensors)
        n_expert = {n for projs in layers.values() for n, _ in projs.values()}
        n_4k = sum(1 for projs in layers.values() for _, sb in projs.values() if sb % mmf.IO_ALIGN != 0)
        n_rehdr = sum(1 for pt in planned_tensors if pt.action == ACTION_REHEADER)
        n_trans = sum(1 for pt in planned_tensors if pt.action == ACTION_TRANSCODE)
        print(f"{src}: arch {arch}, {len(planned_tensors)} tensors ({n_trans} transcoded, {n_rehdr} reheadered"
              + (f", merged across {len(shards_to_stream)} shards" if merge_split else "")
              + f"), {len(layers)} MoE layers, {sorted(n_expert)} experts per layer; "
              f"every expert slice is a multiple of {mmf.SLICE_ALIGN} bytes"
              + (f" ({n_4k} of {3 * len(layers)} are not multiples of {mmf.IO_ALIGN})" if n_4k else f" and of {mmf.IO_ALIGN}"),
              flush=True)

        planned_aux = int(plugin.api.converter_planned_aux_size(conv))
        all_blob_sizes = dict(blob_sizes)
        if planned_aux > 0:
            all_blob_sizes["quant_plugin_aux"] = planned_aux

        zero_sums = {il: [0] * 9 for il in layers}
        placeholder = {name: (0, all_blob_sizes[name]) for name in all_blob_sizes}
        init_hdr = _header_bytes(
            _build_plugin_writer(mmf, reader, arch, planned_tensors, layers, zero_sums, placeholder, merge_split),
            hdr_tmp_path,
        )
        hlen = len(init_hdr)
        pos = mmf.align_up(hlen)
        out_tensor_offs = {}
        for i, pt in enumerate(planned_tensors):
            out_tensor_offs[pt.name] = pos
            pos = mmf.align_up(pos + pt.out_nbytes) if i + 1 < len(planned_tensors) else pos + pt.out_nbytes
        blobs = {}
        pos = mmf.align_up(pos)
        for name in mmf.BLOB_NAMES:
            if name in all_blob_sizes:
                blobs[name] = (pos, all_blob_sizes[name])
                pos = mmf.align_up(pos + all_blob_sizes[name])

        if others_to_link:
            mmf.link_others(others_to_link, out_path, len(others_to_link) + 1)

        free = shutil.disk_usage(os.path.dirname(out_path) or ".").free
        need = pos + (1 << 20)
        if free < need:
            raise mmf.Refuse(f"{os.path.dirname(out_path)} has {free / 1e9:.1f} GB free, the file needs {need / 1e9:.1f} GB")

        staged_hdr = _header_bytes(
            _build_plugin_writer(mmf, reader, arch, planned_tensors, layers, zero_sums, blobs, merge_split),
            hdr_tmp_path,
        )
        if len(staged_hdr) != hlen:
            raise mmf.Refuse("internal error: the header length changed")

        print(f"writing {tmp_path} ({pos / 1e9:.2f} GB) ...", flush=True)
        done = 0
        sfds = {}
        with open(tmp_path, "wb") as fout:
            fout.write(staged_hdr)
            try:
                for pt in planned_tensors:
                    pad = mmf.align_up(fout.tell(), mmf.IO_ALIGN) - fout.tell()
                    if pad:
                        fout.write(b"\x00" * pad)
                    if fout.tell() != out_tensor_offs[pt.name]:
                        raise mmf.Refuse(f"internal error: tensor {pt.name} offset {fout.tell()} != {out_tensor_offs[pt.name]}")
                    if pt.shard_path not in sfds:
                        sfds[pt.shard_path] = mmf.open_ro(pt.shard_path)
                    sfd = sfds[pt.shard_path]
                    if pt.action in (ACTION_KEEP, ACTION_REHEADER):
                        mmf.copy_range(sfd, fout, pt.data_offset, pt.out_nbytes)
                    else:
                        src_raw = mmf.pread(sfd, pt.src_nbytes, pt.data_offset)
                        if len(src_raw) != pt.src_nbytes:
                            raise mmf.Refuse(f"short read of {pt.src_nbytes} bytes for {pt.name}")
                        src_arr = (ctypes.c_uint8 * pt.src_nbytes).from_buffer_copy(src_raw)
                        dst_ba = bytearray(pt.out_nbytes)
                        dst_arr = (ctypes.c_uint8 * pt.out_nbytes).from_buffer(dst_ba)
                        ne_arr = (ctypes.c_uint64 * max(1, len(pt.src_dims)))(*pt.src_dims)
                        rc = plugin.api.converter_transcode_tensor(
                            conv, pt.name.encode("utf-8"), pt.raw_dtype, len(pt.src_dims), ne_arr,
                            ctypes.cast(src_arr, ctypes.c_void_p), pt.src_nbytes,
                            ctypes.cast(dst_arr, ctypes.c_void_p), pt.out_nbytes,
                        )
                        if rc != 0:
                            raise mmf.Refuse(f"{pt.name}: quant plugin transcode failed: {plugin.last_error()}")
                        fout.write(dst_ba)
                        del src_raw, src_arr, dst_ba, dst_arr
                    done += pt.out_nbytes

                aux_ba = bytearray(planned_aux)
                written_aux = ctypes.c_size_t(0)
                aux_arr = (ctypes.c_uint8 * planned_aux).from_buffer(aux_ba) if planned_aux > 0 else None
                aux_ptr = ctypes.cast(aux_arr, ctypes.c_void_p) if aux_arr is not None else ctypes.c_void_p(0)
                if plugin.api.converter_finalize_aux(conv, aux_ptr, planned_aux, ctypes.byref(written_aux)) != 0:
                    raise mmf.Refuse(f"quant plugin converter_finalize_aux failed: {plugin.last_error()}")
                if int(written_aux.value) != planned_aux:
                    raise mmf.Refuse(f"quant plugin auxiliary size mismatch: wrote {written_aux.value}, planned {planned_aux}")
                aux_bytes = bytes(aux_ba)

                for name in mmf.BLOB_NAMES:
                    if name in blobs:
                        pad = mmf.align_up(fout.tell(), mmf.IO_ALIGN) - fout.tell()
                        if pad:
                            fout.write(b"\x00" * pad)
                        if fout.tell() != blobs[name][0]:
                            raise mmf.Refuse(f"internal error: blob {name} would start at {fout.tell()}, not {blobs[name][0]}")
                        if name == "quant_plugin_aux":
                            fout.write(aux_bytes)
                        else:
                            bfd = mmf.open_ro(blob_files[name])
                            try:
                                mmf.copy_range(bfd, fout, 0, blobs[name][1])
                            finally:
                                os.close(bfd)
                fout.flush()
                os.fsync(fout.fileno())
            finally:
                for sfd in sfds.values():
                    os.close(sfd)

        out_expert_offs = {il: {p: out_tensor_offs[f"blk.{il}.ffn_{p}_exps.weight"] for p in mmf.PROJS} for il in layers}
        print(f"checksums of the first, middle and last expert of every layer ({n_jobs} processes) ...", flush=True)
        sums = mmf.compute_sums(tmp_path, layers, out_expert_offs, n_jobs)
        final_hdr = _header_bytes(
            _build_plugin_writer(mmf, reader, arch, planned_tensors, layers, sums, blobs, merge_split),
            hdr_tmp_path,
        )
        if len(final_hdr) != hlen:
            raise mmf.Refuse("internal error: final header length changed")
        with open(tmp_path, "r+b") as fout:
            fout.seek(0)
            fout.write(final_hdr)
            fout.flush()
            os.fsync(fout.fileno())
    finally:
        if os.path.exists(hdr_tmp_path):
            os.remove(hdr_tmp_path)
        plugin.api.converter_destroy(conv)

    os.replace(tmp_path, out_path)
    src_bytes_total = sum(os.path.getsize(sp) for sp in shards_to_stream)
    print(f"wrote {out_path}: {os.path.getsize(out_path)} bytes (source {src_bytes_total}), {done} tensor bytes, "
          f"{time.time() - t_start:.1f} s", flush=True)
    if not no_check:
        _check_plugin_output(mmf, out_path, reader, planned_tensors, layers, sums, blobs, blob_files, aux_bytes, merge_split, n_jobs)
        if others_to_link:
            mmf.check_others(out_path, others_to_link, reader)
