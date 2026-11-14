#!/usr/bin/env python3
"""Tests for make_model_file.py (no pytest needed, no model needed: the test writes a small GGUF of its own).

Usage: python3 test_make_model_file.py
  Writes into the system temp directory and removes everything at the end.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS.parent.parent / "gguf-py"))

import gguf  # noqa: E402

SCRIPT = TOOLS / "make_model_file.py"
WORK = Path(tempfile.mkdtemp(prefix="make_model_file_test_"))
N_EXPERT = 5


def write_gguf(path, slice_floats):
    """A model of 2 dense-ish tensors and 2 MoE layers; one expert slice is slice_floats * 4 bytes."""
    w = gguf.GGUFWriter(str(path), "bailingmoe3")
    w.add_string("general.name", "tiny test")
    w.add_array("tokenizer.ggml.tokens", ["a", "b", "c"])
    w.add_uint32("bailingmoe3.block_count", 3)
    w.add_custom_alignment(32)
    rng = np.random.default_rng(1)
    w.add_tensor("token_embd.weight", rng.standard_normal((3, 7)).astype(np.float32))
    for il in (1, 2):
        for p in ("gate", "up", "down"):
            w.add_tensor(f"blk.{il}.ffn_{p}_exps.weight", rng.standard_normal((N_EXPERT, 2, slice_floats // 2)).astype(np.float32))
        w.add_tensor(f"blk.{il}.attn_norm.weight", rng.standard_normal((9,)).astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def write_split(prefix, slice_floats, experts_in_second=False):
    """A model in two files: file 1 = write_gguf's tensors, file 2 = one big table (and, if asked, a layer's experts)."""
    rng = np.random.default_rng(2)
    p1, p2 = Path(f"{prefix}-00001-of-00002.gguf"), Path(f"{prefix}-00002-of-00002.gguf")
    n2 = 1 + (3 if experts_in_second else 0)
    w = gguf.GGUFWriter(str(p1), "bailingmoe3")
    w.add_uint32("bailingmoe3.block_count", 4)
    w.add_custom_alignment(32)
    w.add_uint16("split.no", 0)
    w.add_uint16("split.count", 2)
    w.add_int32("split.tensors.count", 1 + 2 * 4 + n2)
    w.add_tensor("token_embd.weight", rng.standard_normal((3, 7)).astype(np.float32))
    for il in (1, 2):
        for p in ("gate", "up", "down"):
            w.add_tensor(f"blk.{il}.ffn_{p}_exps.weight", rng.standard_normal((N_EXPERT, 2, slice_floats // 2)).astype(np.float32))
        w.add_tensor(f"blk.{il}.attn_norm.weight", rng.standard_normal((9,)).astype(np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    w = gguf.GGUFWriter(str(p2), "bailingmoe3")
    w.add_uint16("split.no", 1)
    w.add_uint16("split.count", 2)
    w.add_int32("split.tensors.count", 1 + 2 * 4 + n2)
    w.add_tensor("per_layer_token_embd.weight", rng.standard_normal((11, 5)).astype(np.float32))
    if experts_in_second:
        for p in ("gate", "up", "down"):
            w.add_tensor(f"blk.3.ffn_{p}_exps.weight", rng.standard_normal((N_EXPERT, 2, slice_floats // 2)).astype(np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    return p1, p2


def run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True)


def main():
    failed = 0
    try:
        src = WORK / "m.gguf"
        out = WORK / "m.tw.gguf"
        write_gguf(src, 2048)   # 8192 bytes per expert slice
        seed = WORK / "seed.json"
        seed.write_text(json.dumps({"format": "tokenwing usage v1", "n_expert": N_EXPERT, "layers": {}}))

        cases = []

        def case(fn):
            cases.append(fn)
            return fn

        @case
        def converts_and_checks():
            r = run(src, "-o", out, "--usage-seed", seed, "--jobs", 2)
            assert r.returncode == 0, r.stderr + r.stdout
            assert "all " in r.stdout and "tensor bytes identical" in r.stdout, r.stdout
            assert not Path(str(out) + ".tmp").exists()
            rd = gguf.GGUFReader(str(out))
            assert rd.alignment == 4096 and rd.data_offset % 4096 == 0
            assert rd.fields["tokenwing.format"].contents() == "tokenwing model v1"
            assert len(rd.fields["tokenwing.sums.values"].contents()) == 2 * 9
            off = int(rd.fields["tokenwing.blob.usage_seed.offset"].contents())
            size = int(rd.fields["tokenwing.blob.usage_seed.size"].contents())
            assert off % 4096 == 0 and out.read_bytes()[off:off + size] == seed.read_bytes()
            assert out.stat().st_size == off + size

        @case
        def refuses_a_split_name():
            split = WORK / "x-00001-of-00002.gguf"
            shutil.copy(src, split)
            r = run(split, "-o", WORK / "x.out.gguf")
            assert r.returncode == 1 and "split" in r.stderr, r.stderr
            assert not (WORK / "x.out.gguf").exists()

        @case
        def converts_a_split_model():
            (WORK / "sp").mkdir()
            p1, p2 = write_split(WORK / "sp" / "m", 2048)
            o1 = WORK / "sp" / "o-00001-of-00002.gguf"
            r = run(p1, "-o", o1, "--jobs", 1)
            assert r.returncode == 0, r.stderr + r.stdout
            assert "files 2..2 found next to the output" in r.stdout and "tensor bytes identical" in r.stdout, r.stdout
            o2 = WORK / "sp" / "o-00002-of-00002.gguf"
            assert os.path.samefile(o2, p2)
            rd = gguf.GGUFReader(str(o1))
            assert int(rd.fields["split.count"].contents()) == 2 and int(rd.fields["split.no"].contents()) == 0
            assert all(t.data_offset % 4096 == 0 for t in rd.tensors) and "per_layer_token_embd.weight" not in [t.name for t in rd.tensors]
            r = run(p1, "-o", WORK / "sp" / "o.gguf")
            assert r.returncode == 1 and "-00001-of-00002.gguf" in r.stderr, r.stderr

        @case
        def refuses_experts_in_another_file():
            (WORK / "sx").mkdir()
            p1, _ = write_split(WORK / "sx" / "m", 2048, experts_in_second=True)
            r = run(p1, "-o", WORK / "sx" / "o-00001-of-00002.gguf")
            assert r.returncode == 1 and "holds routed experts too" in r.stderr, r.stderr
            assert not (WORK / "sx" / "o-00001-of-00002.gguf").exists()

        @case
        def refuses_unaligned_slices():
            odd = WORK / "odd.gguf"
            write_gguf(odd, 30)   # 120 bytes per slice
            r = run(odd, "-o", WORK / "odd.out.gguf")
            assert r.returncode == 1 and "not a multiple of 512" in r.stderr and "blk.1.ffn_gate_exps.weight" in r.stderr, r.stderr
            assert not (WORK / "odd.out.gguf").exists() and not (WORK / "odd.out.gguf.tmp").exists()

        @case
        def accepts_sector_multiple_slices():
            s512 = WORK / "s512.gguf"
            write_gguf(s512, 384)   # 1536 bytes per slice: a multiple of 512, not of 4096
            r = run(s512, "-o", WORK / "s512.out.gguf", "--jobs", 1)
            assert r.returncode == 0, r.stderr + r.stdout
            assert "are not multiples of 4096" in r.stdout and "tensor bytes identical" in r.stdout, r.stdout
            rd = gguf.GGUFReader(str(WORK / "s512.out.gguf"))
            assert all(t.data_offset % 4096 == 0 for t in rd.tensors)

        @case
        def refuses_a_wrong_blob():
            bad = WORK / "bad.json"
            bad.write_text(json.dumps({"format": "something else"}))
            r = run(src, "-o", WORK / "b.out.gguf", "--usage-seed", bad)
            assert r.returncode == 1 and "tokenwing usage v1" in r.stderr, r.stderr

        @case
        def refuses_missing_quant_plugin():
            r = run(src, "-o", WORK / "qp.out.gguf", "--quant-plugin", WORK / "missing_plugin.so")
            assert r.returncode == 1 and "--quant-plugin" in r.stderr, r.stderr
            assert not (WORK / "qp.out.gguf").exists()

        @case
        def converts_and_merges_with_quant_plugin():
            if os.name == "nt" or not shutil.which("cc"):
                return
            inc = TOOLS.parent.parent / "ggml" / "include"
            c_src = WORK / "mock_plugin.c"
            so_path = WORK / "libmock_plugin.so"
            c_src.write_text(
                '#include <string.h>\n'
                '#include "ggml-quant-plugin.h"\n'
                'static int32_t mock_init(void) { return 0; }\n'
                'static const char * mock_err(void) { return "mock error"; }\n'
                'static int32_t mock_ti(int32_t i, struct ggml_quant_plugin_type_info * o) {\n'
                '    memset(o, 0, sizeof(*o));\n'
                '    if (i == 0) { o->enabled = 1; o->type_name = "ext0"; o->blck_size = 64; o->type_size = 32; o->vec_dot_type = 8; }\n'
                '    return 0;\n'
                '}\n'
                'static void * mock_cc(void) { static int d = 1; return &d; }\n'
                'static void mock_cd(void * c) { (void)c; }\n'
                'static int32_t mock_probe(void * c, const char * nm, int32_t st, int32_t nd, const uint64_t * ne,\n'
                '        const void * hb, size_t hl, uint64_t ms, struct ggml_quant_plugin_tensor_plan * p) {\n'
                '    (void)c; (void)hb; (void)hl; (void)ms;\n'
                '    memset(p, 0, sizeof(*p)); p->out_type = st; p->out_n_dims = nd;\n'
                '    for (int i = 0; i < nd && i < 4; ++i) p->out_ne[i] = ne[i];\n'
                '    if (strstr(nm, "_exps.weight") && nd == 3) {\n'
                '        uint64_t nel = ne[0] * ne[1] * ne[2];\n'
                '        p->action = GGML_QUANT_PLUGIN_TENSOR_TRANSCODE; p->out_type = 43;\n'
                '        p->src_nbytes = nel * 4; p->out_nbytes = (nel / 64) * 32;\n'
                '    }\n'
                '    return 0;\n'
                '}\n'
                'static uint64_t mock_aux_sz(void * c) { (void)c; return 64; }\n'
                'static int32_t mock_tc(void * c, const char * nm, int32_t st, int32_t nd, const uint64_t * ne,\n'
                '        const void * sb, size_t sl, void * db, size_t dl) {\n'
                '    (void)c; (void)nm; (void)st; (void)nd; (void)ne; (void)sl;\n'
                '    const unsigned char * s = (const unsigned char *)sb; unsigned char * d = (unsigned char *)db;\n'
                '    for (size_t i = 0; i < dl; ++i) d[i] = (unsigned char)(s[i] ^ 0x5Au);\n'
                '    return 0;\n'
                '}\n'
                'static int32_t mock_fin(void * c, void * o, size_t cap, size_t * w) {\n'
                '    (void)c; if (cap < 64) return -1; memset(o, 0xA5, 64); *w = 64; return 0;\n'
                '}\n'
                'static const struct ggml_quant_plugin_api API = {\n'
                '    1, "mock", "1.0", mock_init, NULL, mock_err, mock_ti, NULL, NULL, NULL,\n'
                '    mock_cc, mock_cd, mock_probe, mock_aux_sz, mock_tc, mock_fin\n'
                '};\n'
                'const struct ggml_quant_plugin_api * ggml_quant_plugin_get_api(void) { return &API; }\n'
            )
            cc = subprocess.run(["cc", "-O2", "-shared", "-fPIC", f"-I{inc}", str(c_src), "-o", str(so_path)], capture_output=True, text=True)
            assert cc.returncode == 0, cc.stderr
            (WORK / "qp_sp").mkdir()
            p1, _ = write_split(WORK / "qp_sp" / "m", 2048, experts_in_second=True)
            merged_out = WORK / "qp_sp" / "merged.tw.gguf"
            r = run(p1, "-o", merged_out, "--quant-plugin", so_path, "--jobs", 1)
            assert r.returncode == 0, r.stderr + r.stdout
            gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType.EXT0] = (64, 32)
            try:
                rd = gguf.GGUFReader(str(merged_out))
                assert "split.count" not in rd.fields
                assert int(rd.fields["tokenwing.blob.quant_plugin_aux.size"].contents()) == 64
                exps = [t for t in rd.tensors if t.name.endswith("_exps.weight")]
                assert len(exps) == 9 and all(t.tensor_type == gguf.GGMLQuantizationType.EXT0 for t in exps)
            finally:
                gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType.EXT0] = (1, 0)

        for fn in cases:
            try:
                fn()
                print(f"PASS {fn.__name__}")
            except Exception:
                failed += 1
                print(f"FAIL {fn.__name__}")
                traceback.print_exc()
    finally:
        shutil.rmtree(WORK, ignore_errors=True)
    print(f"{failed} failures")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
