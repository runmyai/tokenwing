"""Shared helpers of check.py and compare.py: run a program of a build, read the numbers it prints."""
import os
import re
import subprocess

COMMON = ["--temp", "0", "--top-k", "1", "--ignore-eos", "--load-mode", "dio", "-fit", "off", "--no-warmup",
          "-no-cnv", "-st", "--no-display-prompt", "-v"]
HERE = os.path.dirname(os.path.abspath(__file__))
PROMPTS = os.path.join(HERE, "prompts")


def bin_dir(path):
    """Accepts a build directory or its bin directory."""
    for sub in (("bin", "Release"), ("bin",)):
        b = os.path.join(path, *sub)
        if os.path.isdir(b) and any(f.startswith("llama-") for f in os.listdir(b)):
            return b
    b = os.path.join(path, "bin")
    return b if os.path.isdir(b) else path


def program(bindir, name):
    p = os.path.join(bindir, name + (".exe" if os.name == "nt" else ""))
    if not os.path.exists(p):
        raise SystemExit("%s not found: build it first (cmake --build)" % p)
    return p


def generate(bindir, model, prompt, words, ub, args, work, name, ctx=2048, batch=512, threads=8, hide_gpu=False, timeout=1800):
    """One greedy run. Returns (rc, text, log) and keeps both in the work directory."""
    cmd = [program(bindir, "llama-completion"), "-m", model, "-f", prompt, "-n", str(words), "-c", str(ctx), "-b", str(batch),
           "-ub", str(ub), "-t", str(threads)] + COMMON + list(args)
    env = dict(os.environ)
    if hide_gpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    out, err = os.path.join(work, name + ".txt"), os.path.join(work, name + ".log")
    with open(out, "wb") as fo, open(err, "wb") as fe:
        try:
            rc = subprocess.run(cmd, stdout=fo, stderr=fe, env=env, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            rc = -1
    with open(out, encoding="utf-8", errors="replace") as f:
        text = f.read()
    with open(err, encoding="utf-8", errors="replace") as f:
        log = f.read()
    return rc, text, log


def numbers(log):
    """ms per word of the reply, the reply's word count, and the SSD wait per word (None where not printed)."""
    m = re.findall(r"(?<!prompt )eval time =\s*([\d.]+) ms /\s*(\d+) runs", log)
    total, n = (float(m[-1][0]), int(m[-1][1])) if m else (None, 0)
    s = re.findall(r"stalled ([\d.]+) ms/token", log)
    return {"ms_word": total / n if n else None, "words": n, "ssd_wait": float(s[-1]) if s else None}
