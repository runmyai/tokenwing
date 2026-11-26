# Flags

One row per changed setting. A new or changed setting is documented here in the same change that
adds the code. Numbers come from paired runs (old and new alternating, at least 3 pairs, 5 for a
small gain) over at least 512 reply tokens, judged by SSD wait and total ms/word, with the fixed-text
check in the same script. A setting that does not help in one case stays off by default; the
autotuner decides where it is used.

## Decode: the measured VRAM stack

Off by default. This is a shorthand for a combination of four existing settings, not new engine
logic: it turns on the waiting list, the helper thread that posts the next layer's experts, the
VRAM set that follows the reply, and the GPU pass that runs inside each layer's launch while the
CPU computes the rest. Its value is that the four were measured together, and a pair of them
(`vram-follow` and `vram-parallel`) only pay when the others are on.

| Name | Default | What it does | What it measured |
|---|---|---|---|
| `--moe-stream-vram-stack` | 0 | 1 sets `--moe-stream-plan 1`, `--moe-stream-needed-early 1`, `--moe-stream-vram-follow 1` and `--moe-stream-vram-parallel 1`. Needs a GPU expert pool (`--moe-stream-vram-free SIZE` or `--moe-stream-vram-experts N`); each part still switches itself off with one start-up line when its own prerequisite is missing, so the model runs without it. An explicit setting given after this flag still wins. | Measured on Qwen 3.8 Flash Next (3.5bpw, 130k context, q8_0 KV, NVMe, RTX 5060 Ti), 5 alternating pairs of 512 reply tokens, both sides with `--moe-stream-vram-free 512MiB`. First run with the four flags written out: old 88.93 ms/word, new 79.99 ms/word (**−10.1 %**), new faster in **5 of 5 pairs**, spread 0.86 ms/word. Confirmation run with this flag: old 89.76 ms/word, new 83.10 ms/word (**−7.4 %**), new faster in **5 of 5 pairs**, spread 1.55 ms/word. The new side's logs confirm the stack was live (VRAM follow real mode, 522–525 places; `vram-parallel` in 43/43 streamed layers; ~0.97 GiB pool) against a 0.119 GiB static pool on the old side. The gain is not from fewer SSD reads — in the confirmation run SSD wait was slightly *higher* on the new side — it is the GPU expert pass overlapping the CPU experts. Output text stays coherent but differs beyond a wording tweak (the new answer inserts a three-bullet breakdown), as expected from a changed GPU/CPU expert split. Stays off by default; the autotuner decides where it is used. |
