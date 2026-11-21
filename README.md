<p align="center">
  <img src="docs/images/tokenwing.png" alt="TokenWing" width="720">
</p>

<p align="center"><b>Run huge AI models on your own PC, even ones bigger than your memory.</b></p>

<p align="center">
  <a href="https://github.com/runmyai/tokenwing/releases">Download</a> ·
  <a href="https://github.com/runmyai/tokenwing">Star</a> ·
  <a href="https://discord.gg/BfpvUFp58">Discord</a> ·
  <a href="https://github.com/sponsors/HarinManiK">❤️ Sponsor</a>
</p>

<p align="center"><b>Download:</b> the build for your GPU from the <a href="https://github.com/runmyai/tokenwing/releases/latest">latest release</a>, or get <a href="https://runmyai.org/download.html">Run My AI</a>, which sets it up for you.</p>

TokenWing is the engine inside [Run My AI](https://github.com/runmyai/run-my-ai). It streams the routed experts of MoE models from NVMe, so models far larger than your memory run at a usable speed. It is built on [llama.cpp](https://github.com/ggml-org/llama.cpp).

## How it works

You download a 50 GB game, yet while you play it uses maybe 8 GB of RAM and 4 GB of VRAM. The engine never holds the whole world in memory: it streams in what's about to appear and evicts what you've left behind.

A Mixture-of-Experts model fits this well. In each MoE layer the router selects the top-k routed experts for the current token: top-6 of 256 for Flint 40B, top-8 of 512 for Ling 3.0 Flash. Only those experts' weights are read, so a token touches a few percent of the model's weights. The problem is latency: which experts a token needs is decided at the last moment, right before they're used, and fetching each one on demand would stall every token on the SSD.

TokenWing hides that latency. It predicts the next layers' experts from their own routers, reads them through a priority I/O queue into a RAM expert cache it manages itself, and overlaps compute with the reads still in flight.

```
model file
├─ at start ──► attention, shared experts, routers, dense layers, embeddings, output head
└─ direct reads (8 threads) ──► RAM expert cache, per layer

RAM expert cache ─PCIe─► VRAM expert cache
GPU (VRAM experts) ‖ CPU (RAM experts) ─► sum
```

**Prediction.** The residual stream changes little from one layer to the next, so a later layer's router, applied to the current hidden state, is a good predictor of what that layer will select. At layer L, the routers of L+1 and L+2 are applied to L's hidden state plus its shared-expert output. Each prediction is the top-k, the same width as the model's own selection. On Ling 3.0 Flash, 75.7 % of the selected experts are in the prediction one layer ahead and 67.2 % two layers ahead. Two layers is the default and the recommended setting, because a prediction only helps if its read lands in time.

**Reads.** Every expert read skips the OS file cache and goes straight into its slot (O_DIRECT on Linux, FILE_FLAG_NO_BUFFERING on Windows). Experts the router has selected always go before predicted ones. Predictions are read in 512 KiB chunks, so a needed read can go between two chunks. A prediction gets a RAM slot only when an I/O thread is ready to read it, so predictions don't evict cached experts early. When a layer starts, its wrong predictions are cancelled and its right ones are promoted.

**One layer of one token.** The GPU holding layer L runs attention, the router, the top-k selection and the predictions for L+1 and L+2. It writes the selected IDs into pinned host memory, so the CPU sees them with no stream sync. The CPU issues the missing reads and starts on the experts already in RAM while the rest land: an expert's gate and up run while its down is still being read. Meanwhile the GPU computes the experts it holds in VRAM, and the two outputs are summed.

**Eviction.** Each expert in the RAM cache is valued by its use count, halved every 64 tokens. The lowest value leaves first.

**Prompts.** A long prompt activates most experts in every layer, and running it batch by batch re-reads them for every batch. So the whole prompt passes through one layer before the next starts, and each layer's experts are read about once per prompt (0.98 to 1.00 reads per distinct expert). If a layer needs more distinct experts than it has slots, they're processed in waves that fit.

[Inside TokenWing](docs/how-it-works.md) covers the whole pipeline.

## What decides the speed

Your SSD's read speed isn't the deciding factor; your memory's size just stops being the limit. TokenWing mainly hides the latency of streaming from the SSD. It doesn't increase your memory bandwidth: your CPU and GPU still read every expert from RAM or VRAM.

You could technically run even a trillion-parameter model, if the math checks out with the active parameters per token, how fast your SSD reads, how much of the model your memory can already hold, and how many layers ahead the prefetcher looks and how accurate it is.

## What you need

| | |
|---|---|
| **SSD** | NVMe, PCIe 3.0 or above. |
| **Memory** | We're targeting 35-40B-class models on 16 GB RAM CPU-only builds, and 120B-class models on at least 24 GB of RAM and 8 GB of VRAM. With less it still runs, only slower. |
| **GPU** | NVIDIA or AMD, one or several, or CPU only. |
| **System** | Windows or Linux. macOS on Apple Silicon is coming soon. |

## Models

| Model | What it is | File |
|---|---|---|
| [Flint 40B](https://huggingface.co/runmyai/Flint-40B-TokenWing) | Coding model by Recursive. 40B total, about 2.5B active per token. | Q4_K_XL, about 25 GB |
| [Ling 3.0 Flash](https://huggingface.co/runmyai/Ling-3.0-Flash-TokenWing) | 124B total, 5.1B active per token, 262k context, by Ant Group | MXFP4, about 70 GB |
| [Qwen 3.8 Flash Next](https://huggingface.co/runmyai/Qwen3.8-Flash-Next-TokenWing) | 180B total, 6B active per token. 262k context, by Alibaba | 3.5 bpw by Grawnch, about 84 GB with its n-gram table |

## Features

Each mechanism is a flag. Most are off by default, so plain TokenWing behaves like llama.cpp; the autotuner sets the values for your machine and model.

| Feature | Flag |
|---|---|
| Whole MoE layers kept on the GPU | `--moe-stream-gpu-layers` |
| A VRAM expert cache, sized from free VRAM and seeded from a usage census | `--moe-stream-vram-experts`, `--moe-stream-vram-free` |
| The VRAM set follows usage: every 8 tokens, up to 64 experts are swapped for hotter ones | `--moe-stream-vram-follow` |
| An optional correction file, trained on a model's recorded routing | `--moe-stream-correction` |
| From 4,096 tokens, the RAM cache works as one window over the layers | `--moe-stream-window` |
| The prompt's compute buffer planned into space borrowed from the VRAM cache | `--moe-stream-vram-lend-prompt` |
| Empty RAM slots filled from a usage table while idle | `--moe-stream-idle-load` |
| Chat states kept on disk | `--cache-disk` |
| An OpenAI-compatible server | `llama-server` |

Every size is computed at startup from what the machine has free. For lack of RAM or VRAM, TokenWing falls back to a slower path and prints one line saying what it did. It stops hard only for wrong data, such as a model file that fails its checksums.

The full list, with what each flag measured, is in [Settings](docs/settings.md).

## Known limits

- After a very long prompt (16k tokens and more), the first reply tokens are slower while the RAM cache moves from the prompt's experts to the reply's.
- One SSD at a time: the model file is read from a single drive, with no striping across drives.

## Get started

1. Download the zip for your machine from [Releases](https://github.com/runmyai/tokenwing/releases).
2. Download a model from the list above.
3. Run it with your model's example command from [Getting started](docs/getting-started.md).

`llama-server` serves an OpenAI-compatible API, so any tool that can use one works with it:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Why is the sky blue?"}]}'
```

## FAQ

**Does it change the model's answers?**
No. Computers round numbers slightly differently depending on the order they add them in, so no two engines give bit-identical results. Even llama.cpp gives slightly different results from itself when you only change its batch size. On Flint 40B, the KL divergence against official llama.cpp is 0.010 to 0.016 across our three test setups, against 0.016 for llama.cpp against itself when only its batch size changes.

**Will it wear out my SSD?**
No. SSDs wear when they're written, not when they're read. Expert streaming only reads.

**Can I add my own model?**
Yes, with an experts file and code for its architecture. See [Adding a model](docs/adding-a-model.md).

**Does it send anything?**
One anonymous record a day from `llama-server`: a random install id, the date, the engine version, the OS, CPU and GPU names, and whether it was started directly or by an app. No prompts, replies, model names, paths or user names. `--no-stats`, `TOKENWING_NO_STATS=1` or `DO_NOT_TRACK=1` turns it off. When you start `llama-server` yourself (not through the Run My AI app, which checks on its own), it also asks runmyai.org once a day for the newest release, sending only the engine version and the system (e.g. `linux-x64`); the site counts these checks. `TOKENWING_NO_UPDATE_CHECK=1` or `DO_NOT_TRACK=1` turns it off. See [Settings](docs/settings.md#usage-statistics).

## Docs

[Getting started](docs/getting-started.md) · [How it works](docs/how-it-works.md) · [Architecture](docs/architecture.md) · [Design decisions](docs/design-decisions.md) · [Settings](docs/settings.md) · [Hardware](docs/hardware.md) · [Building](docs/building.md) · [Testing](docs/testing.md) · [Adding a model](docs/adding-a-model.md)

## Build it yourself

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build build -j
```

That is the NVIDIA build. [Building](docs/building.md) covers the others.

## Contributing

Bug fixes and small changes can go straight to a pull request. For a new feature or a big change, open a short issue first. See [Contributing](CONTRIBUTING.md), the [Code of Conduct](CODE_OF_CONDUCT.md) and [Security](SECURITY.md).

## Thanks

Thanks to these projects. Each does something exceptionally well, and parts of this engine draw on that work.

| What we took | From |
|---|---|
| The base: GGUF loading, the quantization formats, the CPU and GPU kernels, the server, and its later fixes | [llama.cpp](https://github.com/ggml-org/llama.cpp) |
| The first design of expert streaming: a slot pool per layer, the op that remaps expert IDs after top-k, direct reads on I/O threads | [llama.cpp PR #25294](https://github.com/ggml-org/llama.cpp/pull/25294) by freedomljc |
| The GPU computes the experts it holds and the CPU the rest, and the outputs are summed | [llama.cpp RFC #24528](https://github.com/ggml-org/llama.cpp/discussions/24528) by leloch |
| The CPU expert kernels for prompts (code, MIT) | [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) |
| Grouped expert selection as one op, gate, up and activation as one CPU step, one work queue over a layer's experts, linear-attention state updated in place | ik_llama.cpp |
| The most-picked experts kept in spare VRAM and computed alongside the CPU, the idea of a multi-row CPU kernel with prefetch (the AVX-512 kernel is our own), how kernels are validated | [ktransformers](https://github.com/kvcache-ai/ktransformers) |
| The design of a per-layer VRAM tier, the GPU time table by op, copying the next wave while this one computes | [BELLS](https://github.com/DGuckert/llama.cpp-BELLS) |
| Filling the expert cache from routing counted on real text, lending VRAM to prompts, sharing prompt chunks with an idle GPU | [Strata](https://github.com/Niko1221/Strata) |
| The rule for swapping VRAM experts as the text moves, the background uploader, saving chat states on disk | [AI-der](https://github.com/tbro0815/ai-der) |
| Adding the shared expert's output to the guess input | [Colibri](https://github.com/JustVugg/colibri) |
| Remembering which experts are hot across sessions | Colibri, [Pulsar](https://github.com/giannisanni/pulsar) |
| Guessing a layer's experts early, from an earlier layer | [FATE](https://arxiv.org/abs/2502.12224), [Pre-gated MoE](https://arxiv.org/abs/2308.12066) |
| Guesses read at low priority in small pieces so a needed read goes first, right guesses promoted and wrong ones cleared | [ProMoE](https://arxiv.org/abs/2410.22134), [MoE-Infinity](https://arxiv.org/abs/2401.14361) |
| Reading an expert's gate, up and down separately, so work can start before the last part lands | ProMoE, PreScope, BigMoeOnEdge |
| Dropping the guesses a layer did not pick | apus, slotstream, Colibri, [MoE-Direct](https://github.com/tmxkzm1925-max/MoE-Direct) |
| The trained correction of the guess | slotstream |
| Computing an expert on the CPU where it sits, instead of copying it to the GPU | [Fiddler](https://github.com/efeslab/fiddler) |
| Computing the experts already loaded while the others are still being read | nvmoe, qwen-fieldfare |
| Two-segment LRU | nvmoe |
| Eviction by frequency with decay | vllm-expert-cache |
| An unused prefetched expert leaves first, and checking output token by token against a reference | MoE-Direct |
| CPU-GPU hand-off timers, and the weighted sum as one CPU op | [hikarioyama](https://github.com/hikarioyama/llama.cpp-glm52) |
| One timeline trace of the I/O threads and compute | [PowerInfer](https://github.com/SJTU-IPADS/PowerInfer) |
| Correction files for models with BF16 or F16 routers, and VRAM places split between layers by score | [grawnchtoader](https://github.com/grawnchtoader) |

## License

[MIT](LICENSE). The TokenWing and Run My AI names and logos are not part of the licence, so a fork needs its own name.
