#!/usr/bin/env python3
"""Check a TokenWing build on your own machine, after a change or before you send one.

    python3 check.py --model SMALL.gguf [--build DIR] [--experts FILE.json] [--no-build] [--no-ctest] [-j N] [--words N] [--work DIR]

SMALL.gguf is any model with routed experts that fits in your RAM; a few GB is plenty. The steps:

  1. build   cmake --build DIR
  2. tests   ctest, one test at a time (the vocabulary test needs git-lfs data and is left out)
  3. tools   the tests of the experts-file tools
  4. model   runs of SMALL.gguf, compared with each other on this machine, so no stored answers are needed:
               a plain run on the CPU, and the same run with the experts streamed from the SSD: the text must be identical
               streaming with a pool smaller than a prompt needs (expert waves): must finish with sensible text
               streaming on the GPU, if there is one: must finish with sensible text

The experts file is made in the work directory if you do not give one. Exit status 0 = everything passed.
"""
import argparse
import json
import os
import re
import subprocess
import sys

from runlib import HERE, PROMPTS, bin_dir, generate, numbers, program

fails = []


def result(ok, what, detail=""):
    print("  %-4s %s%s" % ("OK" if ok else "FAIL", what, "  (%s)" % detail if detail else ""), flush=True)
    if not ok:
        fails.append(what)


def sensible(text):
    """Text a model wrote, not a broken run: enough of it, printable, not one character repeated."""
    t = text.strip()
    if len(t) < 20:
        return False
    if sum(ch.isprintable() or ch.isspace() for ch in t) < 0.95 * len(t):
        return False
    return max(t.count(c) for c in set(t)) < 0.6 * len(t)


def gpus(bindir):
    r = subprocess.run([program(bindir, "llama-completion"), "--list-devices"], capture_output=True, text=True, timeout=120)
    lines = (r.stdout + r.stderr).split("Available devices:")[-1].splitlines()
    return [ln.strip() for ln in lines if ln.startswith("  ") and "(none)" not in ln]


def streamed(log):
    m = re.search(r"moe stream: read [\d.]+ MiB/token \([\d.]+ \w+ total, (\d+) reads\)", log)
    return int(m.group(1)) if m else 0


def main():
    ap = argparse.ArgumentParser(description="Check a TokenWing build on this machine.")
    ap.add_argument("--model", required=True, help="a small GGUF with routed experts")
    ap.add_argument("--build", default="build", help="build directory (default: build)")
    ap.add_argument("--experts", help="its experts file (the .json); made in the work directory if not given")
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--no-ctest", action="store_true")
    ap.add_argument("-j", type=int, default=os.cpu_count() or 4, help="build jobs")
    ap.add_argument("--words", type=int, default=64, help="words per model run")
    ap.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 8))
    ap.add_argument("--work", default="tokenwing-check", help="directory for logs and the experts file")
    a = ap.parse_args()
    a.model = os.path.abspath(a.model)
    a.work = os.path.abspath(a.work)
    os.makedirs(a.work, exist_ok=True)
    tools = os.path.join(os.path.dirname(HERE), "experts")

    if not a.no_build:
        print("1. build", flush=True)
        if not os.path.isdir(a.build):
            raise SystemExit("%s does not exist: configure it first, for example cmake -B build -DCMAKE_BUILD_TYPE=Release" % a.build)
        result(subprocess.run(["cmake", "--build", a.build, "--config", "Release", "-j", str(a.j)]).returncode == 0, "the build")
        if fails:
            sys.exit(1)
    if not a.no_ctest:
        print("2. tests", flush=True)
        r = subprocess.run(["ctest", "--test-dir", a.build, "-C", "Release", "-j", "1", "-E", "test-tokenizers-ggml-vocabs"], capture_output=True, text=True)
        tail = [ln for ln in r.stdout.splitlines() if "tests passed" in ln or "tests failed" in ln]
        result(r.returncode == 0, "ctest", tail[-1].strip() if tail else "")
        if r.returncode:
            print(r.stdout[-3000:])
    print("3. tools", flush=True)
    index = os.path.join(a.work, "model.index.json")
    if not os.path.exists(index):
        subprocess.run([sys.executable, os.path.join(tools, "expert_index.py"), a.model, "-o", index], capture_output=True)
    wdir = os.path.abspath(a.work)
    env = dict(os.environ, TMPDIR=wdir, TEMP=wdir, TMP=wdir)   # the repack test writes a file as large as the experts
    for t, targs in (("test_expert_index.py", [a.model, index]), ("test_expert_repack.py", [a.model]), ("test_make_model_file.py", [])):
        r = subprocess.run([sys.executable, os.path.join(tools, t)] + targs, capture_output=True, text=True, cwd=tools, env=env)
        result(r.returncode == 0, t)
        if r.returncode:
            print(r.stdout[-2000:] + r.stderr[-2000:])

    print("4. model", flush=True)
    bindir = bin_dir(a.build)
    exp = a.experts
    if not exp:
        out = os.path.join(a.work, "model.experts")
        exp = out + ".json"
        if not os.path.exists(exp):
            r = subprocess.run([sys.executable, os.path.join(tools, "expert_repack.py"), a.model, "-o", out, "--quiet"])
            if r.returncode:
                raise SystemExit("could not make the experts file: is this a model with routed experts?")
    j = json.load(open(exp))
    k, n = j["n_expert_used"], j["n_expert"]
    slots = max(2 * k, int(0.4 * n))          # small enough that experts really move between the SSD and RAM
    ub = max(1, slots // k)                   # one call never needs more experts of a layer than the pool holds, so no waves
    short = os.path.join(PROMPTS, "01_casual_chat.txt")
    long_ = os.path.join(PROMPTS, "long_prompt.txt")
    stream = ["--moe-stream", exp, "--moe-stream-slots", str(slots)]
    cpu = ["-ngl", "0", "--no-repack"]        # the same kernels with and without streaming, so the same bits
    print("  %d experts, top-%d: pool of %d slots per layer, -ub %d" % (n, k, slots, ub), flush=True)

    rc0, plain, log0 = generate(bindir, a.model, short, a.words, ub, cpu, a.work, "plain_cpu", threads=a.threads, hide_gpu=True)
    result(rc0 == 0 and sensible(plain), "plain run on the CPU", "rc %d, see %s/plain_cpu.log" % (rc0, a.work))
    result("n_ctx" in log0 and re.search(r"n_batch\s*=\s*512", log0) is not None, "the settings reached the program")

    rc, text, log = generate(bindir, a.model, short, a.words, ub, cpu + stream + ["--moe-stream-gpu-layers", "0"], a.work, "stream_cpu",
                             threads=a.threads, hide_gpu=True)
    result(rc == 0 and streamed(log) > 0, "streamed run on the CPU read experts from the SSD", "%d reads" % streamed(log))
    result(rc == 0 and text == plain, "streamed text identical to the plain run", "first difference at character %d" % next(
        (i for i, (x, y) in enumerate(zip(text, plain)) if x != y), min(len(text), len(plain))) if text != plain else "")

    rc, text, log = generate(bindir, a.model, long_, a.words, 512, cpu + stream + ["--moe-stream-gpu-layers", "0"], a.work, "waves_cpu",
                             threads=a.threads, hide_gpu=True)
    result(rc == 0 and sensible(text) and streamed(log) > 0, "streaming with a pool smaller than the prompt needs (waves)", "rc %d" % rc)

    dev = gpus(bindir)
    if dev:
        rc, text, log = generate(bindir, a.model, short, a.words, ub, ["-ngl", "99"] + stream + ["--moe-stream-gpu-layers", "0"], a.work, "stream_gpu", threads=a.threads)
        result(rc == 0 and sensible(text) and streamed(log) > 0, "streamed run on the GPU", dev[0])
        print("  speed there: %s" % numbers(log), flush=True)
    else:
        print("  no GPU found: the GPU run is skipped", flush=True)

    print("\n%s" % ("Everything passed." if not fails else "%d failed: %s" % (len(fails), "; ".join(fails))))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
