#!/usr/bin/env python3
"""Release check of a packed engine: starts its llama-server from the package folder with a small model, checks that it
serves our web page, and that it writes a reply of 32 tokens. A server that starts but cannot compute (a missing library,
a backend that fails on the first request) fails here, not on a user's machine.

    python3 smoke.py <path to llama-server> <small model .gguf>
"""

import json
import socket
import subprocess
import sys
import time
import urllib.request


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> int:
    server, model = sys.argv[1], sys.argv[2]
    port = free_port()
    log = open("smoke-server.log", "w")
    proc = subprocess.Popen([server, "-m", model, "-c", "512", "-ngl", "0", "--host", "127.0.0.1", "--port", str(port),
                             "--no-stats"], stdout=log, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(240):
            if proc.poll() is not None:
                raise RuntimeError(f"the server ended at start (exit {proc.returncode})")
            try:
                urllib.request.urlopen(base + "/health", timeout=2).read()
                break
            except OSError:
                time.sleep(0.5)
        else:
            raise RuntimeError("the server did not answer /health within 120 s")
        manifest = urllib.request.urlopen(base + "/manifest.webmanifest", timeout=10).read().decode("utf-8", "replace")
        if "TokenWing" not in manifest:
            raise RuntimeError("the server does not carry our web page")
        body = json.dumps({"prompt": "Once upon a time", "n_predict": 32, "temperature": 0, "ignore_eos": True}).encode()
        req = urllib.request.Request(base + "/completion", data=body, headers={"Content-Type": "application/json"})
        reply = json.load(urllib.request.urlopen(req, timeout=300))
        n = reply.get("tokens_predicted", 0)
        if n != 32:
            raise RuntimeError(f"the reply has {n} tokens, not 32")
        print(f"smoke: page OK, reply of {n} tokens: {reply.get('content', '')[:60]!r}")
        return 0
    except Exception as e:  # noqa: BLE001 - every failure is reported with the server's log
        print(f"smoke: FAILED: {e}", file=sys.stderr)
        return 1
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
        with open("smoke-server.log", errors="replace") as f:
            lines = f.read().splitlines()
        # which modules it loaded (the CPU level picked); the whole tail when something went wrong
        print("\n".join([l for l in lines if "load_backend" in l][:8] + lines[-40:]))


if __name__ == "__main__":
    sys.exit(main())
