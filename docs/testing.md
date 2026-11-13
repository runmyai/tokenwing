# Testing

The tools are in `tokenwing-tools/`. They run on your own machine and compare runs with each other, so they need no stored answers.

## Check a build

```bash
python3 tokenwing-tools/testing/check.py --model SMALL.gguf --build build
```

On Windows, `tokenwing-tools\testing\check_windows.cmd SMALL.gguf` finds Visual Studio's x64 C++ environment, configures `build-win` with Ninja if needed, and runs `check.py`.

`SMALL.gguf` is any model with routed experts that fits in your RAM. A few GB is plenty. Logs and the experts file go in `tokenwing-check/` (`--work` changes it), which needs room for about the size of the model, twice over.

| Step | What it proves |
|---|---|
| Build | The tree compiles. Skip with `--no-build`. |
| `ctest` | llama.cpp's and our unit tests pass, run one at a time. Skip with `--no-ctest`. The vocabulary test is left out because it needs git-lfs data. |
| Tool tests | The experts-file tools read and write correctly. |
| Plain run on the CPU | The model runs and the settings reached the program. |
| Streamed run on the CPU | Experts really came from the SSD, and the text is identical to the plain run. |
| Pool smaller than the prompt needs | Expert waves finish and the text is sensible. |
| Streamed run on the GPU | The same on your GPU, if you have one. |

The identical-text check runs both sides with `--no-repack`, at a batch size where no call needs more experts than the pool holds, so both use the same kernels in the same order. Everywhere else the text can differ by rounding between execution paths, and that is not an error. Garbage, NaN or no output is.

## Compare two builds or settings

```bash
python3 tokenwing-tools/testing/compare.py --old build-a --new build-b \
        --model M.gguf --experts M.experts.json --pairs 5 --words 512
```

To compare two settings, give the same build twice:

```bash
python3 tokenwing-tools/testing/compare.py --old build --new build --model M.gguf --experts M.experts.json \
        --old-args "--moe-stream-lookahead 1" --new-args "--moe-stream-lookahead 2"
```

Runs alternate old and new, and each run has the same command otherwise. The result is ms per word, SSD wait per word, how many pairs the new side won, and whether the text matched.

Small gains need care. Between otherwise identical runs, tokens per second can swing by more than 10 % while the SSD wait moves by less than 1 %, so judge prefetch and cache changes by the SSD wait and by ms per word over at least 512 words and 5 pairs. Keep the machine quiet and plugged in. On a laptop, check that the GPU isn't held at a low power limit; a GPU that changes clock speed between runs makes a pair meaningless.

## Measure your hardware

`tokenwing-tools/hardware/` has small programs that measure what limits the engine on your machine. Each file's first lines give its usage and how to compile it.

| Program | What it measures |
|---|---|
| `readbench.py <experts file>` | SSD bandwidth and latency for expert-sized direct reads. Add `latency`, `qd` and other modes for more. |
| `ram_bw.c` | How fast your RAM can be read the way the expert kernels read it. |
| `pcie_h2d.cu` | How fast expert slices move from pinned host memory to the GPU. |
| `vram_hold.cu` | Holds VRAM so the GPU looks like a smaller card, for testing small-GPU behaviour. |

## Test-only switches

Environment variables named `MOE_STREAM_TEST_*` exist for tests. Most of them make a failure happen on purpose, to prove a check really catches it. They are announced on stderr and are listed in [Settings](settings.md). Never leave one set for a real run.
