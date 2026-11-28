# Getting started

This page takes you from nothing to a running server: build TokenWing, get a model, start `llama-server` with a clean command for your PC, and check that it works. The commands are for Linux.

Passing `--moe-stream model` now turns on all fast prefill, decode prediction, CPU, RAM cache, and VRAM cache speed-ups automatically and budgets VRAM whole layers for your GPUs and context size. Run `./build/bin/autotuner --file <MODEL.gguf>` if you want hardware-probed CPU thread counts, affinity masks, I/O queue depth, and multi-GPU device selection in JSON `"recommended_args"`.

## 1. Build

You need CMake, a C++ compiler and Git, and for an NVIDIA card the CUDA Toolkit 13.x.

```bash
git clone https://github.com/runmyai/tokenwing && cd tokenwing
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build build -j
```

For a PC without an NVIDIA card leave out `-DGGML_CUDA=ON`. [Building](building.md) covers AMD and Windows, and the options that shorten the build, such as building for your GPU only. The programs are in `build/bin/`. If you downloaded a release zip instead, use the path of its `llama-server` where the commands below say `./build/bin/llama-server`.

The commands assume a CUDA 13 build. If you build with CUDA 12, change `--moe-stream-async-h2d 1` to `--moe-stream-async-h2d 0` in them.

## 2. Get a model

TokenWing runs two models. Each is one GGUF file: the original model's quantization, laid out for TokenWing. Every tensor starts on a 4 KiB boundary, so the engine can read an expert from the SSD straight into memory, and the file ends with data the engine reads at start, such as which experts are used most in each layer. A plain GGUF of the same model from another source is refused with one line, so use these files.

| Model | File | Size on disk | Download |
|---|---|---|---|
| [Flint 40B](https://huggingface.co/runmyai/Flint-40B-TokenWing) | `flint-UD-Q4_K_XL.tokenwing.gguf` | 25.1 GB | `runmyai/Flint-40B-TokenWing` |
| [Ling 3.0 Flash](https://huggingface.co/runmyai/Ling-3.0-Flash-TokenWing) | `Ling-3.0-flash-MXFP4_MOE.tokenwing.gguf` | 70.2 GB | `runmyai/Ling-3.0-Flash-TokenWing` |

```bash
hf download runmyai/Flint-40B-TokenWing --local-dir $HOME/models/flint
hf download runmyai/Ling-3.0-Flash-TokenWing --local-dir $HOME/models/ling
```

Or download the file from the model's page into a folder of your choice and change the path in the command.

**Put the file on an internal NVMe SSD.** The engine reads experts from it while the model runs, so a slow disk, such as a USB disk, slows the load and every read of an expert that is not in RAM. The disk needs room for the file, 25.1 GB for Flint and 70.2 GB for Ling, and for the files the engine writes next to it, such as the usage table `<model>.gguf.usage.json`. The commands below expect the folders `$HOME/models/flint` and `$HOME/models/ling`.

## 3. Start the server

Before you start:

- **Close other big programs.** The engine fills the RAM that is free when it starts and keeps `--moe-stream-ram-free` of it free for everything else. Raise that value if your PC needs more.
- **Set the memory limit.** The commands start the server with `systemd-run`, which gives it its own memory limit and no swap, so that only the server is stopped if something goes wrong. The engine reads the limit and sizes its RAM cache under it. This line sets it to your RAM minus 1 GiB:

  ```bash
  CAP=$(( $(awk '/MemTotal/{print $2}' /proc/meminfo) - 1048576 ))K
  ```

  On a PC with 32 GB of RAM that is about 30G.

Run the command from the `tokenwing` folder. The server answers on `http://127.0.0.1:8080`: a chat page in the browser, and an OpenAI-compatible API. It listens on this PC only.

Every command is for a context length of 65k (`-c 65536`). The unified whole-layer budget automatically sizes VRAM whole layers and the GPU expert cache from free VRAM for your chosen `-c`, GPU count, and card size (and falls back silently to CPU/RAM on CPU-only hosts).

### An NVIDIA card (8 GB / 12 GB / 24 GB) or multiple GPUs

Flint 40B:

```bash
systemd-run --user --scope -q -p MemoryMax=$CAP -p MemorySwapMax=0 -- \
  ./build/bin/llama-server -m $HOME/models/flint/flint-UD-Q4_K_XL.tokenwing.gguf \
  -c 65536 -ctk q8_0 -ctv q8_0 --moe-stream model --moe-stream-ram-free 0.5GiB --backend-sampling
```

Ling 3.0 Flash:

```bash
systemd-run --user --scope -q -p MemoryMax=$CAP -p MemorySwapMax=0 -- \
  ./build/bin/llama-server -m $HOME/models/ling/Ling-3.0-flash-MXFP4_MOE.tokenwing.gguf \
  -c 65536 -ctk q8_0 -ctv q8_0 --moe-stream model --moe-stream-ram-free 0.5GiB --backend-sampling
```

With multiple GPUs, add `--moe-stream-vram-device CUDA0` (or the fastest device reported in `autotuner`'s `"recommended_vram_device"`) to pin the hot expert pool; remaining GPUs automatically pack whole layers into their free VRAM. On 16 GB RAM hosts (or 6 GB VRAM cards), raise `--moe-stream-ram-free` to `1GiB` if the OS needs more headroom.

### CPU only (or Apple / integrated UMA)

`CUDA_VISIBLE_DEVICES=` (or `--moe-stream-vram-cache 0`) disables the discrete VRAM pool while keeping full CPU, RAM cache, and fast-prefill paths active:

```bash
systemd-run --user --scope -q -p MemoryMax=$CAP -p MemorySwapMax=0 -- env CUDA_VISIBLE_DEVICES= \
  ./build/bin/llama-server -m $HOME/models/flint/flint-UD-Q4_K_XL.tokenwing.gguf \
  -c 65536 -ctk q8_0 -ctv q8_0 --moe-stream model --moe-stream-ram-free 0.5GiB
```

On large-memory integrated GPUs (`>= 24 GiB` shared memory, e.g., GB10 or Strix Halo), add `--moe-stream-uma 1`.

### Hardware tuning with `autotuner`

Measure physical P-cores, NUMA nodes, NVMe queue depth, and per-GPU bandwidth on your host:

```bash
./build/bin/autotuner --file $HOME/models/flint/flint-UD-Q4_K_XL.tokenwing.gguf
```

Append the returned `"recommended_args"` array (or copy `-t`, `-tb`, `--cpu-mask`, `--moe-stream-io-threads`, and `--moe-stream-vram-device`) to your command.

### What the flags do

The full reference, including internal micro-flags hidden under `--help-internal`, is in [Settings](settings.md). In short:

| Flag / Group | What it does |
|---|---|
| `--moe-stream model` | Enables Direct-I/O streaming, activates all 5 feature umbrella groups below (`auto`), sets server `-np 1` / `-cram 0`, and auto-budgets whole GPU layers + VRAM expert pool. |
| `--moe-stream-fast-prefill` | Enables layer-major prompt waves, GPU wave-gather (`512` waves), prompt-split, copy-ahead, CUDA prefill fast kernels, and auto `-ub 2048` (`1024` at `-c >= 262144`). |
| `--moe-stream-fast-decode` | Enables overlapped Direct-I/O + router plan lookahead, prediction correction (`model`), fused routing/norms, and GPU backend sampling (`-bs` on server). |
| `--moe-stream-fast-cpu` | Enables fused SIMD CPU MoE matmuls (`iqk 8`, `x4`, `mmid-merge`, `cpu-topk`, `cpu-fa`, `small-ops`, `weighted-sum`) and the linear-attention state read and written in place (`cpu-state-inplace`). |
| `--moe-stream-vram-cache` | Manages the VRAM hot-expert pool (`vram-free 200MiB`, `vram-follow`, `vram-copyback`, `vram-experts-file model`) and falls back silently on CPU-only runs. |
| `--moe-stream-ram-cache` | Configures RAM decay eviction, prompt-end recency retention, and startup RAM warm-fill (`--moe-stream-idle-load 2`). |
| `--moe-stream-ram-free SIZE` | Keeps `SIZE` of system RAM free outside the expert pools (use `0.5GiB` on dedicated servers or `1GiB`–`2GiB` on desktops). |
| `--moe-stream-timing 1` | Prints per-stage SSD / CPU / GPU / hand-off timing breakdown in stats and logs. |

### Other context lengths

Change `-c` to `4096`, `8192`, `16384`, `32768`, `65536`, `131072`, or `262144`. No manual layer-list edits are needed: `--moe-stream model` automatically recomputes the whole-layer vs. VRAM-pool split for the active KV size and scales `-b 1024 -ub 1024` at `-c 262144`.

## 4. What you see while it loads

The server loads the weights that every token uses, then fills the RAM with experts, and only then does the model count as loaded (`--moe-stream-idle-load 2`). So the load takes longer than plain llama.cpp's, and the first request does not start cold. The progress lines are on the server's output:

```
llama_moe_stream: loading: experts into RAM 0 % (0 of N, 0.0 of X GB, 0.0 s)
llama_moe_stream: loading: experts into RAM 10 % (...)
...
llama_moe_stream: loading: experts into RAM 100 % (...)
llama_moe_stream: loading: experts into RAM done: ... experts read into the RAM pools of ... streamed layers, ... of ... slots filled; ...
```

A line comes at 0 %, at every 10 % and at 100 %. The `done` line says what was read. If every expert fits in RAM it says so, and after the load the SSD is read no more. If not, it says the RAM holds the most-used experts of each layer, and the rest are read from the SSD while the model runs. The server's health check (`GET /health`) passes only after the `done` line.

The engine prints its plan at the start: the RAM slots per layer, and the VRAM it keeps. Every line that starts with `llama_moe_stream:` comes from the engine. When it has to use a slower path for lack of RAM or VRAM, it prints one line saying what it did.

## 5. Check that it works

Send a request, or open `http://127.0.0.1:8080` in a browser:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Why is the sky blue?"}]}'
```

A sensible answer means the model runs. Garbage or no output is an error.

To test a build on your machine, run `tokenwing-tools/testing/check.py` with any small model that has routed experts and fits in your RAM:

```bash
python3 tokenwing-tools/testing/check.py --model SMALL.gguf --build build
```

[Testing](testing.md) lists the steps it runs and what each proves.

### Read the speed lines

After each request the server prints its timings, and one line from the engine:

```
prompt eval time = ... ms / N tokens (... ms per token, ... tokens per second)
       eval time = ... ms / N tokens (... ms per token, ... tokens per second)
      total time = ... ms / N tokens
    picked words = ... on the GPU (backend sampling), ... on the CPU
      moe stream: L layers x S slots, this request P prompt + D decode tokens, hit rate H%, M misses/decode token, X MiB/decode token read, stall Y ms/decode token, Z ms/prompt token
```

| Value | Means |
|---|---|
| `prompt eval time` | How fast the prompt was processed, in tokens per second. For a long prompt its time is your time to first token (TTFT). |
| `eval time` | How fast the reply was written, in tokens per second. |
| `picked words` | Where the reply's tokens were picked: on the GPU with `--backend-sampling`, or on the CPU. |
| `L layers x S slots` | The streamed layers, and the expert slots each holds in RAM. |
| `hit rate` | The share of expert lookups that found the expert in RAM. |
| `misses/decode token`, `MiB/decode token read` | The experts and bytes read from the SSD for each reply token. |
| `stall` | The time the engine waited for the SSD, per reply token and per prompt token. The prediction's job is to keep it low. |

`GET /moe-stream/stats` returns the same figures live, per layer too, and the final response of a request carries them in `timings.moe_stream`. With `--moe-stream-timing 1` they also say where each reply token's time went: waiting for the SSD, the CPU, the GPU, and the hand-off between them. It is off in the commands above.

What to expect:

- Reply speed falls as a chat fills a long context.
- After a very long prompt (16k tokens and more), the first reply tokens are slower while the RAM cache moves from the prompt's experts to the reply's.
- On a PC with an 8 GB NVIDIA card and 16 GiB of free RAM, the 8 GB commands gave Flint 45.8 tokens/s (plain llama.cpp: 25.5) and Ling 22.1 tokens/s (plain llama.cpp: 6.7), at 65k context. Speed depends on the machine: with more RAM, the RAM holds more experts.
- `--moe-stream model` enables the fast defaults automatically; if speed is lower than expected, check `llama_moe_stream:` startup lines for VRAM/RAM fallbacks, and on a laptop ensure AC power and full GPU power limits.

## 6. When something fails

Open an issue at [github.com/runmyai/tokenwing/issues](https://github.com/runmyai/tokenwing/issues) and send:

- the server's whole output, from the start to the error,
- the command you ran,
- your GPU, your RAM and your SSD, and the output of `./build/bin/llama-server --version`.

Before that, look at what the output says:

- A line starting with `llama_moe_stream:` says what the engine chose or why it fell back to a slower path.
- A model file that is refused with one line is not a TokenWing file. Use the file from the table in [Get a model](#2-get-a-model).
- If the GPU runs out of memory at start, use a shorter context, or the values for your length in [Other context lengths](#other-context-lengths).
- If the server stops with no error message, the system probably ran out of memory. Close other programs, or raise `--moe-stream-ram-free` so more RAM stays free, and start again.

[Hardware](hardware.md) says what each part of your PC does for the speed, and [Inside TokenWing](how-it-works.md) covers the whole pipeline.
