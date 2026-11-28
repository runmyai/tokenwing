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

A Mixture-of-Experts model works the same way: for each token only a few of its experts are needed, so only those are read. TokenWing predicts the experts of the next layers and reads them from the SSD into a RAM cache while the current layer computes, so compute rarely waits on the disk. Memory size stops being the limit. It doesn't raise your memory bandwidth, though: your CPU and GPU still read every expert from RAM or VRAM.

[Inside TokenWing](docs/how-it-works.md) covers the whole pipeline.

## Models

| Model | What it is | File |
|---|---|---|
| [Flint 40B](https://huggingface.co/runmyai/Flint-40B-TokenWing) | Coding model by Recursive. 40B total, about 2.5B active per token. | Q4_K_XL, about 25 GB |
| [Ling 3.0 Flash](https://huggingface.co/runmyai/Ling-3.0-Flash-TokenWing) | 124B total, 5.1B active per token, 262k context, by Ant Group | MXFP4, about 70 GB |
| [Qwen 3.8 Flash Next](https://huggingface.co/runmyai/Qwen3.8-Flash-Next-TokenWing) | 180B total, 6B active per token. 262k context, by Alibaba | 3.5 bpw by Grawnch, about 84 GB with its n-gram table |

## What you need

| | |
|---|---|
| **SSD** | NVMe, PCIe 3.0 or above. |
| **Memory** | 16 GB of RAM runs all three models, CPU only. More RAM and a GPU make them faster. |
| **GPU** | NVIDIA or AMD, one or several, or CPU only. |
| **System** | Windows or Linux. macOS on Apple Silicon is coming soon. |

## Get started

1. Download the zip for your machine from [Releases](https://github.com/runmyai/tokenwing/releases).
2. Download a model from the list above.
3. Run `autotuner --file <model>`. It prints one JSON line. Start `llama-server -m <model> -c 32768` followed by the flags in its `"recommended_args"`, then open http://localhost:8080 for the chat page, or use the API.

`llama-server` serves an OpenAI-compatible API, so any tool that can use one works with it:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Why is the sky blue?"}]}'
```

The full commands for each model are in [Getting started](docs/getting-started.md).

## Features

| What you get | Flag |
|---|---|
| Experts streamed from the SSD into a RAM cache, the next layers' experts predicted and read ahead | `--moe-stream model` |
| An autotuner that measures your PC and prints the flags for it | `autotuner --file MODEL` |
| Elastic RAM: uses free RAM for the expert cache and gives it back when other programs need it | `--moe-stream-elastic 1` |
| The hottest experts kept in VRAM, the set following what the text uses | `--moe-stream-vram-free`, `--moe-stream-vram-follow` |
| Whole MoE layers kept on the GPU | `--moe-stream-gpu-layers` |
| Long prompts run layer by layer, so each expert is read about once per prompt | on with `--moe-stream model` |
| Several GPUs in one PC | `--moe-stream-vram-device`, `--moe-stream-gpu-layers-device` |
| Chat states saved on disk, so an old chat continues without re-reading its prompt | `--cache-disk` |
| An OpenAI-compatible server with a built-in web chat page | `llama-server` |

Every size is computed at startup from what the machine has free. For lack of RAM or VRAM, TokenWing falls back to a slower path and prints one line saying what it did. It stops hard only for wrong data, such as a model file that fails its checksums.

The full list, with what each flag measured, is in [Settings](docs/settings.md).

## Known limits

- After a very long prompt (16k tokens and more), the first reply tokens are slower while the RAM cache moves from the prompt's experts to the reply's.
- One SSD at a time: the model file is read from a single drive, with no striping across drives.

## FAQ

**Does it change the model's answers?**
No. Computers round numbers slightly differently depending on the order they add them in, so no two engines give bit-identical results. Even llama.cpp gives slightly different results from itself when you only change its batch size. On Flint 40B, the KL divergence against official llama.cpp is 0.010 to 0.016 across our three test setups, against 0.016 for llama.cpp against itself when only its batch size changes.

**Will it wear out my SSD?**
No. SSDs wear when they're written, not when they're read. Expert streaming only reads.

**Can I add my own model?**
Yes, with an experts file and code for its architecture. See [Adding a model](docs/adding-a-model.md).

**Does it send anything?**
One anonymous record a day from `llama-server`: a random install id, the date, the engine version, the OS, CPU and GPU names, and whether it was started directly or by an app. No prompts, replies, model names, paths or user names. `--no-stats`, `TOKENWING_NO_STATS=1` or `DO_NOT_TRACK=1` turns it off. When you start `llama-server` yourself (not through the Run My AI app, which checks on its own), it also asks runmyai.org once a day for the newest release, sending only the engine version and the system (e.g. `linux-x64`); the site counts these checks. `TOKENWING_NO_UPDATE_CHECK=1` or `DO_NOT_TRACK=1` turns it off. See [Settings](docs/settings.md#usage-statistics).

## Build it yourself

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build build -j
```

That is the NVIDIA build. [Building](docs/building.md) covers the others.

## Docs

[Getting started](docs/getting-started.md) · [How it works](docs/how-it-works.md) · [Architecture](docs/architecture.md) · [Design decisions](docs/design-decisions.md) · [Settings](docs/settings.md) · [Hardware](docs/hardware.md) · [Building](docs/building.md) · [Testing](docs/testing.md) · [Adding a model](docs/adding-a-model.md) · [Credits](CREDITS.md)

## Contributing

Bug fixes and small changes can go straight to a pull request. For a new feature or a big change, open a short issue first. See [Contributing](CONTRIBUTING.md), the [Code of Conduct](CODE_OF_CONDUCT.md) and [Security](SECURITY.md).

## Thanks

Thanks to [llama.cpp](https://github.com/ggml-org/llama.cpp), [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp), [ktransformers](https://github.com/kvcache-ai/ktransformers) and the other projects and papers TokenWing builds on. The full list is in [CREDITS.md](CREDITS.md).

## License

[MIT](LICENSE). The TokenWing and Run My AI names and logos are not part of the licence, so a fork needs its own name.
