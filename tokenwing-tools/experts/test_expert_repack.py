#!/usr/bin/env python3
"""Tests for the expert repack sidecar written by expert_repack.py (no pytest needed).

Usage: python3 test_expert_repack.py MODEL.gguf
  A small MoE model is enough. The test repacks the whole model into a sidecar, which is about as large
  as the model's routed experts. The sidecar and the other scratch files go to the system temp
  directory (TMPDIR) and are removed at the end; point TMPDIR at the drive whose read speed the
  benchmark test should measure.
"""

import os
import random
import shutil
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TOOLS.parent.parent / "gguf-py"))

import expert_repack  # noqa: E402
from expert_index import build_index, human  # noqa: E402
from expert_io import (  # noqa: E402
    PROJS,
    ExpertIndex,
    load_sidecar,
    open_direct,
    open_file,
    pread_into,
    read_expert,
    read_expert_block,
    sidecar_block_range,
)

if len(sys.argv) < 2:
    sys.exit(__doc__)
MODEL = sys.argv[1]
WORK = Path(tempfile.mkdtemp(prefix="expert_repack_test_"))
SIDECAR = WORK / "test.experts"
META = SIDECAR.with_name(SIDECAR.name + ".json")
N_RANDOM = 60
BENCH_N = 500
BENTHREADS = 8

_results = []
state = {}


def test(fn):
    _results.append(fn)
    return fn


def repack_seconds():
    return state["meta"]["repack_seconds"]


def blocks_aligned(meta):
    for blk in meta["blocks"].values():
        if blk["base_offset"] % meta["io_align"] or blk["block_bytes"] % meta["io_align"]:
            return False
    return True


@test
def test_repack_size_matches_plan():
    """The sidecar file is exactly the size the repack plan predicted."""
    raw = state["raw"]
    plan = raw["repack_plan"]
    size = os.path.getsize(SIDECAR)
    assert size == plan["sidecar_bytes"], (size, plan["sidecar_bytes"])
    assert size == plan["useful_bytes"] + plan["padding_bytes"]
    assert state["meta"]["sidecar_bytes"] == size
    assert state["meta"]["padding_bytes"] == plan["padding_bytes"], state["meta"]["padding_bytes"]
    assert state["meta"]["moe_layers"] == raw["model"]["moe_layers"]
    assert state["meta"]["n_expert"] == raw["model"]["n_expert"]
    blk_desc = (human(state["meta"]["block_bytes"])
                if state["meta"]["block_bytes"] is not None
                else "/".join(human(b) for b in plan["read_bytes_per_expert"]))
    print(f"  plan {plan['sidecar_bytes']:,} B == file {size:,} B  "
          f"({human(size)}), padding {plan['padding_bytes']}, {len(state['meta']['blocks'])} layers x "
          f"{state['meta']['n_expert']} experts x {blk_desc}")


@test
def test_metadata_describes_source():
    """The sidecar json pins the source model with a header fingerprint."""
    meta, raw = state["meta"], state["raw"]
    assert meta["format"] == expert_repack.SIDECAR_FORMAT
    assert meta["source"]["path"] == raw["model"]["path"]
    assert meta["source"]["size"] == os.path.getsize(MODEL)
    assert meta["source"]["header_sha256"] == expert_repack.header_fingerprint(
        MODEL, raw["model"]["data_section_offset"])
    assert len(meta["source"]["header_sha256"]) == 64
    assert blocks_aligned(meta)
    first = str(raw["model"]["moe_layers"][0])
    blk = meta["blocks"][first]
    assert blk["base_offset"] == raw["repack_plan"]["per_layer"][first]["base_offset"]
    off = 0
    for p in PROJS:
        d = blk["projections"][p]
        assert d["offset"] == off, (p, d["offset"], off)
        assert d["size"] == raw["layers"][first]["projections"][p]["slice_size"]
        assert d["ggml_type"] == raw["layers"][first]["projections"][p]["ggml_type"]
        assert d["source_tensor"].endswith(f"ffn_{p}_exps.weight")
        off += d["size"]
    assert off + blk["padding_bytes"] == blk["block_bytes"]
    for il in (meta["moe_layers"][0], meta["moe_layers"][-1]):
        lblk = meta["blocks"][str(il)]
        for e in (0, meta["n_expert"] // 2, meta["n_expert"] - 1):
            o, b = sidecar_block_range(meta, il, e)
            assert (o, b) == (lblk["base_offset"] + e * lblk["block_bytes"], lblk["block_bytes"])
            assert o % meta["io_align"] == 0 and b % meta["io_align"] == 0
    try:
        sidecar_block_range(meta, meta["moe_layers"][0], meta["n_expert"])
    except IndexError:
        pass
    else:
        raise AssertionError("expert id out of range should raise")
    print(f"  header sha256 {meta['source']['header_sha256'][:16]}..., gate|up|down at "
          f"+{[blk['projections'][p]['offset'] for p in PROJS]}, block {blk['block_bytes']:,} B, all blocks aligned")


@test
def test_random_blocks_match_gguf():
    """>= 50 random (layer, expert): 1 sidecar read == the 3 GGUF slices."""
    meta = load_sidecar(META)
    assert meta == state["meta"], "sidecar json does not round-trip"
    idx = state["index"]
    picks = [(il, e) for il in (meta["moe_layers"][0], meta["moe_layers"][-1]) for e in (0, meta["n_expert"] - 1)]
    rng = random.Random(20250929)
    while len(picks) < N_RANDOM:
        picks.append((rng.choice(meta["moe_layers"]), rng.randrange(meta["n_expert"])))
    fd_sc = open_direct(SIDECAR)
    fd_gg = open_direct(MODEL)
    try:
        for il, e in picks:
            got = read_expert_block(fd_sc, meta, il, e)
            ref = read_expert(fd_gg, idx, il, e)
            assert len(got) == 3
            for p, blob in zip(PROJS, got):
                assert len(blob) == len(ref[p]), f"layer {il} expert {e} {p}: {len(blob)} != {len(ref[p])}"
                assert blob == ref[p], f"layer {il} expert {e} {p} differs"
    finally:
        os.close(fd_sc)
        os.close(fd_gg)
    blk_desc = (f"{human(meta['block_bytes'])} B"
                if meta["block_bytes"] is not None
                else "/".join(human(b) for b in state["raw"]["repack_plan"]["read_bytes_per_expert"]))
    print(f"  {len(picks)} random (layer, expert) blocks identical to the GGUF slices, "
          f"1 O_DIRECT read each ({blk_desc})")


@test
def test_all_experts_match():
    """Every expert of every layer, compared byte for byte (mtime + seconds)."""
    meta, raw = state["meta"], state["raw"]
    sc = mmap_ro(SIDECAR)
    gg = mmap_ro(MODEL)
    t0 = time.perf_counter()
    n = 0
    for il in meta["moe_layers"]:
        lay = raw["layers"][str(il)]["projections"]
        base = meta["blocks"][str(il)]["base_offset"]
        block = meta["blocks"][str(il)]["block_bytes"]
        for e in range(meta["n_expert"]):
            blob = sc[base + e * block:base + (e + 1) * block]
            at = 0
            for p in PROJS:
                size = lay[p]["slice_size"]
                ref = gg[lay[p]["offset"] + e * size:lay[p]["offset"] + (e + 1) * size]
                assert blob[at:at + size] == ref, f"layer {il} expert {e} {p} differs"
                at += size
            n += 1
    dt = time.perf_counter() - t0
    sc.close()
    gg.close()
    state["full_check_seconds"] = dt
    total = meta["sidecar_bytes"]
    print(f"  {n:,} experts, {human(total)} compared in {dt:.1f} s "
          f"({2 * total / dt / 1e9:.2f} GB/s of compared data)")


def mmap_ro(path):
    import mmap

    fd = open_file(path, os.O_RDONLY)
    try:
        return mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
    finally:
        os.close(fd)


@test
def test_read_benchmark():
    """500 random experts: 1 read each (sidecar) vs 3 reads each (GGUF), 8 in flight.

    Run twice: once through the expert_io API (which allocates a fresh bounce buffer per
    read, so it also measures that cost) and once with preallocated buffers, which is
    what the streaming engine would do and which isolates the device behaviour.

    CAVEAT: the two files sit in different places on the volume, and on some
    filesystems (seen on ntfs3) identical reads of about 1.2 MB measure anywhere from
    1.0 to 5.9 GB/s depending on which file they come from (a byte-identical `dd` copy
    of the sidecar measured 3.4x faster than the sidecar itself). So the ratio printed
    here is NOT a clean measurement of the layout. The trustworthy comparison is the
    request-size sweep within a single file.
    """
    meta, idx = state["meta"], state["index"]
    raw = state["raw"]
    rng = random.Random(4242)
    picks = [(rng.choice(meta["moe_layers"]), rng.randrange(meta["n_expert"])) for _ in range(BENCH_N)]
    total_bytes = sum(meta["blocks"][str(il)]["block_bytes"] for il, _ in picks)

    def run_api(label, path, fetch):
        fds = [open_direct(path) for _ in range(BENTHREADS)]

        def work(i):
            try:
                return fetch(fds[i], picks[i::BENTHREADS])
            finally:
                os.close(fds[i])

        dt = timed([threading.Thread(target=work, args=(i,)) for i in range(BENTHREADS)])
        print(f"  {label:40} {dt:5.2f} s  {total_bytes / dt / 1e9:4.2f} GB/s  {dt / BENCH_N * 1e6:7.1f} us/expert")
        return dt

    def fetch_sidecar(fd, ps):
        return sum(len(x) for il, e in ps for x in read_expert_block(fd, meta, il, e))

    def fetch_gguf(fd, ps):
        return sum(len(x) for il, e in ps for x in read_expert(fd, idx, il, e).values())

    print(f"  through the expert_io API ({BENTHREADS} threads, {BENCH_N} experts, fresh bounce buffer per read)")
    sc_api = run_api("sidecar, 1 read/expert", SIDECAR, fetch_sidecar)
    gg_api = run_api("GGUF, 3 reads/expert", MODEL, fetch_gguf)

    def plan(kind):
        out = []
        for il, e in picks:
            p = raw["layers"][str(il)]["projections"]
            if kind == "sc":
                b = meta["blocks"][str(il)]
                out.append((b["base_offset"] + e * b["block_bytes"], b["block_bytes"]))
            else:
                for x in PROJS:
                    o = p[x]["offset"] + e * p[x]["slice_size"]
                    s = o - o % 4096
                    t = -(-(o + p[x]["slice_size"]) // 4096) * 4096
                    out.append((s, t - s))
        return out

    def run_io(label, path, reqs):
        buf = mmap_ro_nofill(max(l for _, l in reqs))
        fds = [open_direct(path) for _ in range(BENTHREADS)]
        per = [reqs[i::BENTHREADS] for i in range(BENTHREADS)]
        mv = memoryview(buf)

        def work(i):
            try:
                for o, l in per[i]:
                    sub = mv[:l]
                    try:
                        pread_into(fds[i], sub, o)
                    finally:
                        sub.release()
            finally:
                os.close(fds[i])

        dt = timed([threading.Thread(target=work, args=(i,)) for i in range(BENTHREADS)])
        mv.release()
        buf.close()
        print(f"  {label:40} {dt:5.2f} s  {total_bytes / dt / 1e9:4.2f} GB/s  {dt / BENCH_N * 1e6:7.1f} us/expert")
        return dt

    print(f"  preallocated buffer, no python copying (pure I/O, {BENTHREADS} threads, {BENTHREADS} in flight)")
    sc_io = run_io("sidecar, 1 read/expert", SIDECAR, plan("sc"))
    gg_io = run_io("GGUF, 3 reads/expert", MODEL, plan("gg"))
    state["bench"] = (sc_api, gg_api, sc_io, gg_io)
    print(f"  {human(total_bytes)} of useful bytes either way: sidecar is {gg_api / sc_api:.2f}x (API) "
          f"and {gg_io / sc_io:.2f}x (pure I/O)")


def timed(threads):
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return time.perf_counter() - t0


def mmap_ro_nofill(n):
    import mmap

    return mmap.mmap(-1, n)


@test
def test_refuses_incomplete_source():
    """A truncated GGUF is refused instead of producing a half sidecar."""
    trunc = WORK / "repack-incomplete.gguf"
    raw = state["raw"]
    n = raw["model"]["data_section_offset"] + (1 << 20)
    try:
        with open(MODEL, "rb") as src, open(trunc, "wb") as dst:
            dst.write(src.read(n))
        part = build_index(trunc)
        assert part["model"]["incomplete"] is True
        try:
            expert_repack.check_complete(part, trunc)
        except SystemExit as e:
            assert "refusing to repack" in str(e), e
            msg = str(e)
        else:
            raise AssertionError("a truncated GGUF should have been refused")
    finally:
        if trunc.exists():
            trunc.unlink()
    print(f"  {msg}")


def main():
    print(f"model    {MODEL}")
    print(f"sidecar  {SIDECAR}\n")
    print("repacking (this is the timed run)...")
    index_data = build_index(MODEL)
    state["raw"] = index_data
    state["index"] = ExpertIndex(index_data)
    t0 = time.perf_counter()
    meta, meta_path = expert_repack.repack(MODEL, SIDECAR, index_data, progress=True)
    wall = time.perf_counter() - t0
    state["meta"] = meta
    print(f"  {meta['sidecar_bytes'] / 2**30:.2f} GiB in {wall:.1f} s wall "
          f"({meta['sidecar_bytes'] / wall / 1e9:.2f} GB/s incl. index build)\n")

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
    try:
        sys.exit(main())
    finally:
        shutil.rmtree(WORK, ignore_errors=True)
