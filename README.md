<p align="center">
  <img src="docs/images/tokenwing.png" alt="TokenWing" width="720">
</p>

<p align="center"><b>The model you want. On the machine you have. Fast, private, no limits.</b></p>

<p align="center">
  <a href="https://github.com/runmyai/tokenwing/releases">Download</a> ·
  <a href="https://github.com/runmyai/tokenwing">⭐ Star us on GitHub</a> ·
  <a href="https://github.com/sponsors/HarinManiK">❤️ Sponsor</a>
</p>

<p align="center">
  <a href="https://github.com/runmyai">GitHub</a> ·
  <a href="https://discord.gg/BfpvUFp58">Discord</a> ·
  <a href="https://www.youtube.com/@RunMyAI">YouTube</a> ·
  <a href="https://x.com/runmyai7">X</a> ·
  <a href="https://www.linkedin.com/company/run-my-ai">LinkedIn</a> ·
  <a href="https://huggingface.co/runmyai">Hugging Face</a>
</p>

<p align="center"><b>Download:</b> the build for your GPU from the <a href="https://github.com/runmyai/tokenwing/releases/latest">latest release</a>, or get <a href="https://runmyai.org/download.html">Run My AI</a>, which sets it up for you.</p>

TokenWing is the engine inside [Run My AI](https://github.com/runmyai/run-my-ai). It streams the routed experts of MoE models from NVMe, so models far larger than your memory run at a usable speed. It is built on [llama.cpp](https://github.com/ggml-org/llama.cpp).

## How it works

You download a 50 GB game, yet while you play it uses maybe 8 GB of RAM and 4 GB of VRAM. The engine never holds the whole world in memory: it streams in what's about to appear and evicts what you've left behind.

A Mixture-of-Experts model is built in a way that fits this: it's split into many small experts, and each token uses only a few of them, so most of the model sits unused at any moment. The difference is that a game knows what lies around the next corner, while a model decides which experts it needs at the last moment, right before it uses them. TokenWing works that out two layers early: it predicts the next layers' experts and reads them from the SSD into RAM and VRAM while the current layer computes, so the compute rarely waits on the disk.

Your memory's size stops being the limit, but not its speed: your CPU and GPU still read every expert they use from RAM or VRAM.

### This isn't magic

With TokenWing you could technically run even a trillion-parameter model, if the math checks out with:

- the active parameters per token;
- how fast your SSD reads;
- how much of the model your memory can already hold;
- how many layers ahead the prefetcher looks, and how accurate it is.

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

## Roadmap

- **More hardware.** NVIDIA Blackwell, Ada, Ampere, Turing and Pascal; AMD and Intel GPUs; CUDA, ROCm, HIP and Vulkan; and more multi-GPU setups.
- **Speculative decoding.** The model drafts a few tokens ahead and checks them in one step, so replies come faster.
- **Fewer wasted SSD reads.** Fewer wrong guesses among the experts that are still on the SSD, without losing the right ones. With, say, 200 of a layer's 512 experts in memory, a wrong guess for an expert already in memory costs nothing; one that has to be read from the SSD does.
- **macOS on Apple Silicon.**
- **Image and video input**, starting with Qwen 3.8 Flash Next.
- **More models.** Next: Inkling 2 Mini.
- **Several SSDs at once**, reading a model from two or more drives for more speed.

## FAQ

**Does it change the model's answers or skip experts?**<br>
No. Every expert the model picks is computed, with its full weights in the file's own quantization: no expert skipping, no top-k cut. Computers round numbers slightly differently depending on the order they add them in, so no two engines give bit-identical results. Even llama.cpp gives slightly different results from itself when you only change its batch size. We measured ours against llama.cpp: the difference is no bigger than that.

**What SSD do I need?**<br>
We suggest an NVMe SSD on PCIe 3.0 or above.

**Will it wear out my SSD?**<br>
No. SSDs wear when they're written, not when they're read. That's why every SSD's endurance rating counts terabytes written (TBW) and nothing else. Running a model only reads.

**Can I use it with Hermes Agent, Pi or my own tools?**<br>
Yes. `llama-server` serves an OpenAI-compatible API. Hermes Agent, Pi, or any tool that can use an OpenAI-compatible endpoint can use the models it runs.

**Can I add my own model?**<br>
Yes, with an experts file and code for its architecture. See [Adding a model](docs/adding-a-model.md).

**Does it send anything?**<br>
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
