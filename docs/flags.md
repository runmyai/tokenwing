# Flags

One row per changed setting. A new setting is documented here in the same change that adds it.
Numbers come from paired runs (old and new alternating, at least 3 pairs, 5 for a small gain) over
at least 512 reply tokens, judged by tokens/s for both phases and SSD wait, with a fixed-text check
in the same script. A setting that does not help in one case stays off by default.

## Decode and prefill: one hotness ranking across both memory tiers

Off by default. The engine normally keeps some MoE layers whole in VRAM and ranks the VRAM expert
pool with a different count from the one that picks RAM eviction victims. This setting uses one
decayed use count for both and gives all VRAM to the hot pool.

| Name | Default | What it does | What it measured |
|---|---|---|---|
| `--moe-stream-unified-tier` | 0 | 1 = the decayed use count that picks RAM eviction victims also ranks VRAM-follow swaps (the long-term count is the tie-break), and no whole MoE layers are kept in VRAM (`-gpu-layers 0` is forced), so all of VRAM is the hot expert pool. Falls back with one line when no GPU expert pool fits. An explicit `-gpu-layers` given after this flag still wins. | Measured on Qwen 3.8 Flash Next (3.5bpw, 130k context, q8_0 KV, NVMe, RTX 5060 Ti, production Auto preset + autotuner values), 5 alternating pairs of 512 reply tokens: decode 19.75 → **22.09 tok/s (+11.9 %)**, 5/5 pairs, old spread 0.20; prefill 88.29 → **97.67 tok/s (+10.6 %)**, 5/5 pairs. SSD wait rose 4.28 → 4.88 ms/token. The pool grew 2.95 GiB / 1564 places → 7.43 GiB / 3916 places, whole layers 4 → 0, and VRAM picks/word 120 → 305. The engine's own timing break-down shows CPU 30.7 → 17.4 ms/word and GPU busy 39 % → 51 %, with handoff 14.1 → 15.1 ms/word — the gain is expert work moved off the CPU, not fewer SSD reads. Output text differs at rounding level only (placement changes, no expert skipping). Stays off by default; the autotuner decides where it is used. |
