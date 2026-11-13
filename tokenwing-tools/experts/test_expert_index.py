#!/usr/bin/env python3
"""Correctness tests for expert_index.py / expert_io.py (no pytest needed).

Usage: python3 test_expert_index.py MODEL.gguf INDEX.json
  INDEX.json is the index of MODEL.gguf: python3 expert_index.py MODEL.gguf -o INDEX.json
  A small MoE model is enough. The partial-file test writes a copy of the model's header to the
  system temp directory (TMPDIR) and removes it again.
"""

import json
import os
import random
import sys
import tempfile
import traceback
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS.parent.parent / "gguf-py"))

import expert_index  # noqa: E402
from expert_io import ExpertIndex, PROJS, open_direct, read_expert, read_expert_buffered, read_expert_slice  # noqa: E402
from gguf import GGUFReader  # noqa: E402
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType  # noqa: E402

if len(sys.argv) < 3:
    sys.exit(__doc__)
MODEL = sys.argv[1]
INDEX = sys.argv[2]
TRUNC = Path(tempfile.gettempdir()) / f"partial-header-test-{os.getpid()}.gguf"
N_RANDOM = 24

_results = []


def test(fn):
    _results.append(fn)
    return fn


def bytes_of(vt, shape):
    block_size, type_size = GGML_QUANT_SIZES[vt]
    n = 1
    for d in shape:
        n *= d
    return n * type_size // block_size


@test
def test_header_matches_gguf_py():
    """The header-only parser must agree with gguf-py's GGUFReader on every offset."""
    hdr = expert_index.GGUFHeader(MODEL)
    rdr = GGUFReader(MODEL)
    assert hdr.data_offset == rdr.data_offset, (hdr.data_offset, rdr.data_offset)
    assert hdr.alignment == rdr.alignment
    by_name = {t.name: t for t in hdr.tensors}
    assert len(by_name) == len(rdr.tensors), (len(by_name), len(rdr.tensors))
    for t in rdr.tensors:
        mine = by_name[t.name]
        assert mine.n_bytes == t.n_bytes, t.name
        assert list(mine.dims) == [int(d) for d in t.shape], t.name
        assert hdr.data_offset + mine.rel_offset == t.data_offset, t.name
    print(f"  data_offset {hdr.data_offset:,} == gguf-py {rdr.data_offset:,}; "
          f"{len(rdr.tensors)} tensors, all absolute offsets identical")


@test
def test_slice_sizes():
    """index slice_size == gguf-py tensor n_bytes / n_expert, from type and shape alone."""
    idx = json.load(open(INDEX))
    n_expert = idx["model"]["n_expert"]
    n = 0
    for il, lay in idx["layers"].items():
        for proj, d in lay["projections"].items():
            vt = GGMLQuantizationType(d["ggml_type_id"])
            assert vt.name == d["ggml_type"]
            assert bytes_of(vt, d["shape_ggml"]) == d["n_bytes"], (il, proj)
            assert d["n_elements"] // n_expert * d["type_size"] // d["block_size"] == d["slice_size"]
            assert d["n_bytes"] == d["slice_size"] * n_expert
            assert d["n_elements"] == n_expert * d["slice_size"] * d["block_size"] // d["type_size"]
            n += 1
    sizes = {d["slice_size"] for lay in idx["layers"].values() for d in lay["projections"].values()}
    print(f"  {n} expert tensors, slice_size == n_bytes/n_expert == "
          f"quant_shape_to_byte_shape, sizes {sorted(sizes)}")


@test
def test_odirect_matches_everything():
    """O_DIRECT read == buffered read == the slice gguf-py exposes, for random experts."""
    idx = ExpertIndex.load(INDEX)
    rdr = GGUFReader(MODEL)
    tensors = {t.name: t for t in rdr.tensors}
    moe = idx.moe_layers
    picks = [(moe[0], 0, "gate"), (moe[0], idx.n_expert - 1, "down"),
             (moe[-1], 0, "up"), (moe[-1], idx.n_expert - 1, "gate")]
    rng = random.Random(1234)
    while len(picks) < N_RANDOM:
        picks.append((rng.choice(moe), rng.randrange(idx.n_expert), rng.choice(PROJS)))
    fd = open_direct(idx.path)
    n_cmp = 0
    try:
        for layer, expert, proj in picks:
            got = read_expert_slice(fd, idx, layer, expert, proj)
            ref = read_expert_buffered(idx.path, idx, layer, expert, proj)
            t = tensors[idx.tensor(layer, proj)["tensor"]]
            via_gguf = t.data[expert].tobytes()
            assert len(got) == idx.slice_size(layer, proj) == len(ref) == len(via_gguf)
            assert got == ref, f"buffered mismatch at {(layer, expert, proj)}"
            assert got == via_gguf, f"gguf-py mismatch at {(layer, expert, proj)}"
            assert idx.slice_offset(layer, expert, proj) == t.data_offset + expert * idx.slice_size(layer, proj)
            n_cmp += 1
    finally:
        os.close(fd)
    print(f"  {n_cmp} random (layer, expert, proj) slices: O_DIRECT == buffered == gguf-py data[expert]")


@test
def test_read_expert():
    idx = ExpertIndex.load(INDEX)
    fd = open_direct(idx.path)
    try:
        blob = read_expert(fd, idx, idx.moe_layers[min(3, len(idx.moe_layers) - 1)], min(42, idx.n_expert - 1))
    finally:
        os.close(fd)
    assert set(blob) == set(PROJS)
    assert all(len(v) == idx.slice_size(idx.moe_layers[3], p) for p, v in blob.items())
    print("  read_expert -> " + ", ".join(f"{p}={len(v):,} B" for p, v in blob.items()))


@test
def test_totals_are_independent():
    """Per-token and total bytes recomputed from the KV metadata only."""
    idx = json.load(open(INDEX))
    m = idx["model"]
    n_layer_moe = m["n_layer_all"] - m["leading_dense_block_count"]
    assert n_layer_moe == len(idx["layers"]), (n_layer_moe, len(idx["layers"]))
    n_embd, n_expert, n_used = m["n_embd"], m["n_expert"], m["n_expert_used"]
    per_expert_set = set()
    per_token = 0
    total = 0
    for il in m["moe_layers"]:
        lay = idx["layers"][str(il)]["projections"]
        n_ff = m["n_ff_exp"][il] if isinstance(m["n_ff_exp"], list) else m["n_ff_exp"]
        per_expert = 0
        for d in lay.values():
            bs, ts = GGML_QUANT_SIZES[GGMLQuantizationType(d["ggml_type_id"])]
            assert {d["shape_ggml"][0], d["shape_ggml"][1]} == {n_embd, n_ff}, d["tensor"]
            assert d["shape_ggml"][2] == n_expert, d["tensor"]
            block = d["shape_ggml"][0] * d["shape_ggml"][1]
            assert d["n_elements"] == block * n_expert, d["tensor"]
            assert d["n_bytes"] == block * n_expert * ts // bs
            per_expert += block * ts // bs
        per_expert_set.add(per_expert)
        assert sum(d["slice_size"] for d in lay.values()) == per_expert, il
        per_token += n_used * per_expert
        total += n_expert * per_expert
    assert idx["summary"]["bytes_per_expert"] == sorted(per_expert_set)
    assert idx["summary"]["bytes_per_token"] == per_token
    assert idx["summary"]["routed_expert_bytes"] == total
    assert idx["summary"]["routed_expert_share_of_file"] == total / os.path.getsize(MODEL)
    assert total < os.path.getsize(MODEL) - m["data_section_offset"]
    print(f"  bytes/expert {per_expert:,}, bytes/token {per_token:,} ({per_token / 2**20:.2f} MiB), "
          f"routed total {total:,} ({total / 2**30:.2f} GiB)")


@test
def test_repack_plan_arithmetic():
    idx = json.load(open(INDEX))
    r, s = idx["repack_plan"], idx["summary"]
    cursor = 0
    for il, lay in idx["layers"].items():
        p = r["per_layer"][il]
        useful = sum(p["slice_size"].values())
        assert p["base_offset"] == cursor
        assert p["offset_formula"] == f"{cursor} + e * {p['block_bytes']}"
        assert p["block_bytes"] % 4096 == 0 and p["block_bytes"] >= useful
        assert p["padding_bytes_per_expert"] == p["block_bytes"] - useful
        cursor += p["block_bytes"] * p["n_expert"]
    assert r["sidecar_bytes"] == cursor
    assert r["useful_bytes"] == s["routed_expert_bytes"]
    assert r["padding_bytes"] == cursor - s["routed_expert_bytes"]
    print(f"  sidecar {cursor:,} B, padding {r['padding_bytes']:,} B "
          f"({r['padding_pct']:.4f} %), 1 read per expert")


@test
def test_partial_file_header():
    """Same index from a truncated copy (a partially downloaded model)."""
    idx = json.load(open(INDEX))
    n = idx["model"]["data_section_offset"] + (1 << 20)
    with open(MODEL, "rb") as src, open(TRUNC, "wb") as dst:
        dst.write(src.read(n))
    try:
        part = expert_index.build_index(TRUNC)
        assert part["model"]["incomplete"] is True
        assert idx["model"]["incomplete"] is False
        assert part["model"]["data_section_offset"] == idx["model"]["data_section_offset"]
        assert part["layers"] == idx["layers"]
        for k in idx["summary"]:
            if k != "routed_expert_share_of_file":
                assert part["summary"][k] == idx["summary"][k], k
        assert part["aligned_plan"] == idx["aligned_plan"]
        assert part["repack_plan"] == idx["repack_plan"]
        with open(TRUNC, "r+b") as f:
            f.truncate(1 << 20)
        try:
            expert_index.build_index(TRUNC)
        except ValueError as e:
            assert "header does not fit" in str(e), e
            msg = str(e)
        else:
            raise AssertionError("a file shorter than the header should fail")
    finally:
        os.unlink(TRUNC)
    print(f"  header-only index identical from a {n:,} B partial file; "
          f"1 MiB file -> clean error: {msg}")


def main():
    print(f"model  {MODEL}\nindex  {INDEX}\n")
    failed = 0
    for fn in _results:
        print(f"{fn.__name__}:")
        try:
            fn()
        except Exception:
            failed += 1
            print("  FAIL")
            traceback.print_exc()
    print(f"\n{len(_results) - failed}/{len(_results)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
