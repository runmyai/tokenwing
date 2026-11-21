# Autotuner

Most of TokenWing's mechanisms are off-by-default flags, so bare `llama-server` behaves like plain `llama.cpp`. `autotuner` is the single program that measures a machine, inspects a GGUF file without loading it into memory, and chooses the conflict-free flags and environment variables for that machine and model.

```bash
autotuner [--seconds S] [--file MODEL.gguf] [--ctx TOKENS] [--ram-free MIB] [--self-test]
```

It finishes in under 3 seconds and writes a single JSON object to `stdout`.

---

## 1. What it measures on the machine

1. **CPU memory bandwidth and thread counts (`-t`, `-tb`, `--cpu-mask`):**
   - Runs non-compressible `Q4_K` matrix-vector products larger than twice the CPU's last-level cache (`340–680 MiB`) across `1..physical_cores` threads, verifying `(S, 2S, 4S)` working-set scaling (`cache_escaped`).
   - Picks `saturation` (the fewest threads within 5% of peak DRAM bandwidth) and subtracts the OS-measured CPU cores consumed by Direct-I/O readers plus one thread for the GPU driver (`threads` -> `-t`).
   - Sets prompt prefill threads (`threads_batch` -> `-tb`) to all physical cores minus the GPU driver thread.
   - On hybrid CPUs (Intel P/E cores including split Arrow Lake masks, ARM `big.LITTLE`, or multi-socket NUMA), re-measures on primary fast-core hardware threads and emits `--cpu-mask` / `--cpu-range` (with `--cpu-range-batch` across all logical CPUs) only when measured more than 5% faster.
   - Detects `avx2` (`--moe-stream-cpu-iqk 8`) and `avx512_vnni` (`--moe-stream-cpu-x4 1`).
2. **System RAM and container limits (`ram_total_mib`, `ram_avail_mib`):**
   - Reads `MemTotal`, `MemAvailable` and `SwapTotal` on Linux (capped by cgroup v2 `memory.max` and `cpu.max` inside Docker/Kubernetes), `GlobalMemoryStatusEx` on Windows, and `sysctl` / `host_statistics64` on macOS.
3. **GPUs, unified-memory iGPUs, and PCIe transfers (`gpus`, `igpus`, `dxgi_gpus`):**
   - Enumerates every discrete GPU (`gpus`) and integrated unified-memory GPU (`igpus`) visible to the compiled backends (`CUDA`, `ROCm`, `Vulkan`, `Metal`), plus vendor-neutral `DXGI` dedicated/shared memory on Windows.
   - Measures each device's quantized-weight read bandwidth (`gbps`) and, on discrete cards, pinned Host-to-Device `4 KiB` latency (`pcie_lat_us`) and `16 MiB` bandwidth (`pcie_gbps`).
4. **Direct-I/O storage curve (`io_threads`, `io_cores`, `latency_us`, `read_gbps`):**
   - With `--file`, runs three alternating forward/reverse Direct-I/O passes across queue depths `1, 2, 4, 8, 16, 32` at the model's exact 4 KiB-aligned single-projection size (`piece_kib`), capping at `QD <= 4` automatically on slow media (`< 150 MB/s` at `QD = 1`) so spinning drives or USB storage never stall.
   - Chooses `io_threads` (`--moe-stream-io-threads`) as the smallest queue depth within 5% of peak read bandwidth, records actual reader CPU usage (`io_cores`), and records `QD = 1` read latency (`latency_us`) plus peak bandwidth (`read_gbps`).

---

## 2. What it reads from the model file (`model`)

`inspect_gguf` parses GGUF v2/v3 headers and sorted tensor offsets directly — without loading weights and without requiring `--quant-plugin` DLLs at inspection time:

- **Any MoE or dense GGUF:** Reads `arch`, `context_length`, `n_layers`, `n_moe_layers`, `n_experts`, `n_used`, `expert_kib`, and `non_expert_mib` across both single-file GGUFs and multi-shard splits (`-00001-of-NNNNN.gguf`).
- **Projection & tensor layout:** Detects 3-projection (`gate`, `up`, `down`) vs. 2-projection fused (`ffn_gate_up_exps.weight` + `ffn_down_exps.weight`) experts (`n_proj`), per-expert biases, shared experts, hyper-connection stream count (`hc_count`), recurrent `SSM`/`KDA` layers, lazy per-layer embeddings (`has_ple`), external plugin quantization types (`GGML_TYPE_EXT0..EXT5` -> `needs_quant_plugin`), and VRAM-pool compatibility (`vx_compat`).
- **Optional sidecar data:** Automatically picks up adjacent `<model>.correction` / `<model>.vram-experts.json` files or embedded `tokenwing.blob.*` sections when present, and omits those flags cleanly on stock Hugging Face GGUFs.

---

## 3. How slots, the guess gate, and hardware tiers are chosen

### Slot-aware guess gating (`est_slots_per_layer`)

From `ram_avail_mib`, the RAM headroom (`--moe-stream-ram-free`), non-expert weights in RAM, and `expert_kib`, `autotuner` estimates how many experts per MoE layer fit in RAM:

- **Partial-RAM streaming (`est_slots_per_layer < n_experts`):** Turns on `--moe-stream-guess-gate 1` along with `--moe-stream-ssd-latency-us` and `--moe-stream-ssd-gbps`. At bind time (`pl_take`), each non-resident guess's online Empirical-Bayes hit probability (calibrated from normalized top-`k` router shares with zero offline files, persisted in `<model>.gguf.guesser.json`) must beat the candidate victim slot's 64-word reuse probability plus live drive queue congestion — cutting wasted SSD reads and short-horizon evictions while keeping resident RAM/VRAM hits free.
- **Full-RAM fit (`est_slots_per_layer >= n_experts`):** Turns on `--moe-stream-guess-skip-full 1` (and `--moe-stream-guess-skip-full-gpu 1` on CUDA) so layers holding every expert in RAM skip guess work altogether.

### Hardware tiers (`tier`, `recommended_env`, `recommended_flags`)

| Tier | Condition | Key choices |
|---|---|---|
| `multi_gpu` | 2+ discrete GPUs | Ranks cards fastest-first by measured `gbps`; places the VRAM expert pool and RAM/SSD bridge on the slowest card (`--moe-stream-vram-device`) and fills faster cards with whole MoE layers (`--moe-stream-gpu-layers-device FAST=auto`). |
| `discrete_large` | 1 discrete GPU >= 10 GiB | `-ngl 99`, whole-layer auto-fit, `--moe-stream-elastic 1`, `--moe-stream-vram-free 200MiB`, `--moe-stream-vram-follow 1`, `--moe-stream-vram-copyback 1`, `--moe-stream-vram-lend-prompt 1`. |
| `discrete_small` | 1 discrete GPU 4–10 GiB | Streams all MoE layers (`--moe-stream-gpu-layers 0`), keeps attention/routers + a compact `vram-follow` expert pool on the GPU, omits `vram-lend-prompt`. |
| `uma` | No discrete GPU, iGPU >= 16 GiB | `--moe-stream-uma 1` on the unified-memory GPU (64 KiB-aligned zero-copy pools, single RAM budget, no `elastic` or separate `vram-free` pool). |
| `cpu_only` | No GPU >= 4 GiB | `-ngl 0 --device none --moe-stream-gpu-layers 0`, `--moe-stream-cpu-weighted-sum 1`, `--moe-stream-cpu-small-ops 1`, and `--moe-stream-cpu-state-inplace 1` on recurrent models. |

Backend-specific flags follow the active device backend: `CUDA` enables `GGML_CUDA_*` fusion env vars, `--moe-stream-gpu-share-q8 1`, `--moe-stream-vram-parallel 1`, and `--moe-stream-async-h2d` (`1` on Linux, `0` on Windows); `ROCm` enables `--moe-stream-hip-mmvq 1 --moe-stream-hip-sync 1`; `Vulkan` enables `--moe-stream-vram-chunk 1` (`2 GiB` buffer chunks).

---

## 4. Fallbacks and manual overrides

Every setting emitted in `recommended_flags` obeys TokenWing's safety rule: if a user overrides a flag or moves a command to a machine lacking a capability (for example, passing `--moe-stream-vram-experts` on a 2-projection or per-expert-biased model, or running without `O_DIRECT`), the engine logs **one `llama_moe_stream:` diagnostic line on `stderr`**, disables only the incompatible sub-feature, and continues bit-compatibly without hanging or aborting.
