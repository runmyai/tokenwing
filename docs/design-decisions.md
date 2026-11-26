# Design decisions

Each entry ends with a result. **Kept** means the idea is part of the engine. **Kept as a setting** means the flag exists and is off by default, because the result depends on the machine or has a cost somewhere else. **Not kept** means we tried it and dropped it; where its flag still exists, the entry names it. Results come from paired runs on one test machine, so a different drive, RAM speed or GPU can move them. [Inside TokenWing](how-it-works.md) explains how the kept parts work together.

## Prediction

**Guess with the later layers' own routers.** The residual stream changes little from layer to layer, so a later layer's router applied to the current hidden state names most of the experts that layer will pick. The alternatives below were less accurate or cost more than they gained. Kept; `--moe-stream-lookahead` sets how many layers ahead.

**Add the shared expert's output to the guess input.** The guess then sees the hidden state plus this layer's shared-expert output, which is closer to the next layer's real input. It cut misses by 11 % on Ling. The price is that the guess cannot start before the shared expert has run. Kept as `--moe-stream-pred-shared` (off by default).

**Send early guesses straight to the read queue and correct them a layer later.** Guessing two and three layers ahead this way (`--moe-stream-prefetch-lead`, `--moe-stream-prefetch-m2`) gave no win. The waiting list below replaced it. Not kept; both flags remain, off.

**A waiting list for far guesses.** Guesses for the layers 1 to N ahead go on one list, sorted by target layer and rank, with a vote when several guesses name the same expert. An I/O thread gives an entry a RAM slot only when it has no needed read and no started guess read to work on, and a layer removes the entries it did not pick when it starts. Two layers ahead beat one and three on the test machine. Kept as `--moe-stream-plan`, `--moe-stream-drop-stale` and `--moe-stream-lookahead` (off by default).

**Compute the guesses for all distances in one op.** As separate router products, the two-ahead guess cost more GPU time than it could save, so it was not built at first. One op that does the norm, the gain multiply, every router product and the top-k for all distances gives the same bits as the separate ops and made it cheap enough. Kept as `--moe-stream-guess-combined` (off by default).

**Guess more experts than the model picks.** A wider guess names more experts that the layer then does not use. On real runs it cut the demand reads but barely moved the SSD wait, because the extra reads take drive time, and a guess 64 wide still reaches only 90.8 % recall. The width is the model's top-k unless set. Kept as a setting: `--moe-stream-guess-top`, `--moe-stream-prefetch-m`.

**Read only the guesses with a high chance.** A table gave the chance that a guess at each rank and score is right, and only guesses above a threshold were read. It read 5 to 7 % fewer bytes and left the SSD wait unchanged. Not kept; `--moe-stream-chance-model` and `--moe-stream-read-threshold` remain, off.

**Guess from the previous token, from expert IDs, or from the token ID.** Consecutive tokens share only about a third of a layer's experts (2.7 of 8 on Ling). Tables built from expert IDs alone, or a table from token ID to experts (about 12 % precision), were far below the router guess. Not kept.

**Other changes to the guess's input.** Re-basing the hidden state with the target layer's norm gain was exact but moved recall by under a tenth of a point, because neighbouring layers' gains are almost equal. Extrapolating the score trend across layers made the guess worse. A per-layer score threshold instead of a fixed width cost about twice as many reads per correct guess. Not kept.

**A trained correction.** A file trained offline from recorded routing is folded into the router matrices at load, so a guess costs the same as before. A damaged file, or one for another model or quantisation, falls back to the plain guess with one line. Kept as `--moe-stream-correction FILE` (unset by default; it needs `--moe-stream-pred-shared 1`).

**Guess the first streamed layer of each word from the layers before it.** The first streamed layer follows layers that are not streamed (dense layers, layers kept whole in VRAM), so no streamed layer could guess it. Those layers now guess it from up to D layers back, and one CPU op adds the guesses to the waiting list. It halved that layer's demand reads. Alone, the time per word stayed inside the noise; it paid together with needed-early. Kept as `--moe-stream-wordstart` (off by default).

## What to read and when

**Direct reads, not the file cache or a memory map.** Mapping the whole model file pulled its pages into memory and ran a machine out of memory. The file cache would also hold a second copy of every expert and evict on its own rules. On the small test model, direct loading had the same speed and about half the peak memory. Kept: expert reads use direct I/O, and `--load-mode dio` loads the always-used weights without mapping the file.

**Read each needed expert as one request.** Splitting a read into parallel chunks was slower than one whole read: each extra request adds a fixed cost in the I/O stack, and the drive's own queue already reads in parallel. 8 and 16 I/O threads made no difference. Prefetch reads are not on the critical path, and cutting them into 512 KiB chunks won. Not kept for needed reads (`--moe-stream-chunk`, off); kept for prefetches as `--moe-stream-chunk-prefetch`.

**Priorities for the drive.** Needed reads go first, a needed expert whose prefetch is still queued is promoted, at most a few prefetch chunks run while a needed read is in service, and each chunk wakes its own I/O thread. Together they gave a small gain and no cost on the prompts we measured, with the text unchanged. Kept as settings: `--moe-stream-promote`, `--moe-stream-guess-cap`, `--moe-stream-io-wake chunk`.

**One queue for all decode reads, or a hard cap on guesses in the drive.** Both were slower than two priorities, the single queue by a wide margin. A softer cap won: while no needed read is in service, a reader starts a new guess piece only when fewer than G are running. Kept as `--moe-stream-guess-inflight` (off by default; it needs `--moe-stream-plan 1` and `--moe-stream-promote 1`).

**Hold guesses back while the drive is behind.** The behind rule binds only the best-ranked list entries while enough reads are unfinished. It was neutral on every measure: the drive was behind in many words but at very few reader turns, because entries are bound before a backlog builds. Kept as a setting: `--moe-stream-behind-backlog` with `--moe-stream-behind-rank`.

**Order reads by their deadline.** The waiting list is already sorted by target layer, which is the deadline order within a word, and counted on recordings, reads served ahead of an earlier deadline were rare. Skipping a late guess and reading the expert later as a needed read took longer than letting the late guess finish. Not kept; the sorted list is all there is.

**Start the CPU experts as soon as the router has run.** The router's ids reach the CPU while the GPU is still working. A helper thread turns the guess into list entries in both cases. The ids and the guess come from the GPU, which writes them into pinned host memory inside the layer's launch (the GPU producer), or, where the router runs on the CPU, straight from the CPU graph (the host producer). The guess for the next layer is ready a little later, so the SSD wait grew slightly, but each word got shorter. Kept as `--moe-stream-needed-early` (off by default; it needs `--moe-stream-plan 1`).

**Compute the experts that have landed while others are still being read.** A layer's expert matmul waits per expert instead of once for the whole layer, and with per-projection ready flags an expert's gate and up run while its down is still being read. Each is a small gain on its own. Kept as `--moe-stream-overlap` and `--moe-stream-proj-ready` (off by default; the second needs the first).

## What stays in RAM

**Which expert leaves.** LFU and SLRU gave no gain over LRU. A use count halved every 64 tokens, lowest value out first, won on real runs; it applies to decode only, and prompts keep LRU. Perfect hindsight (Belady) would cut LRU's misses to 0.57 times, and no cheap rule came close. Kept as `--moe-stream-policy decay` (the default is `lru`; `lfu` and `slru` remain as settings).

**Protect the experts a guess named.** An expert that a guess named for a layer not yet run in this word is not the victim of a guess read, at any guess distance or rank. A needed read ignores the rule. Kept as `--moe-stream-protect-named` (off by default; it needs `--moe-stream-plan 1`).

**Structure around the pools.** A side area for guessed experts gave three times as many misses. Slot counts per layer from measured misses, keeping a prompt's experts for the reply, and making unused prefetches the first victims (`--moe-stream-probation`) gave no gain. Not kept; only the probation flag remains, off.

**Order a prompt's last waves by recency.** The last ubatch of a prompt puts the experts the end of the prompt used most into the last wave, with a 128-token half-life, so they stay in RAM for the reply. Kept as `--moe-stream-prompt-end recency`. Counting those uses as decode words in the eviction values (`--moe-stream-prompt-values`) made the first reply words slower: the prompt's experts started with high values and crowded out the reply's own. Not kept; the flag remains, off.

**Size the cache from free RAM.** A larger pool was the one change that paid every time, so every size is computed from what the machine has free and nothing is tuned to one drive. A fixed budget failed when the plan left out the engine's own non-pool memory and when the plan and the run read different amounts of free RAM. Now slots = free RAM minus a headroom minus the non-pool memory, planned before anything loads, and `--moe-stream-plan-only` prints the plan. Kept: `--moe-stream-ram-free` sets the headroom, `--moe-stream-ram` caps the budget, `--moe-stream-slots` pins the count.

**Preload the cache from a census before the first prompt.** Counted on recordings, it removed a fifth of the prompt's reads and almost none of the reply's, because the prompt warms the cache anyway, and reading the whole cache cost about what it saved. Hot sets also differ between prompts. Not kept in that form. What remains is a small usage table next to the model file that fills only empty slots after 500 ms without a request, never evicting anything. Kept as `--moe-stream-idle-load` (off by default).

**Lock the pools, register them with the driver, or use 2 MiB pages.** Where the locked-memory limit is small, registration failed or pinned nothing, and 2 MiB pages gave no gain. Not kept; `--moe-stream-pin` and `--moe-stream-hugepages` remain, off.

## VRAM

**Whole layers or the hottest experts of every layer.** Whole layers in VRAM remove those layers' reads and also remove graph splits. A per-expert pool of the most used experts of all layers did better at equal VRAM when the RAM cache is small, and with plenty of RAM whole layers were as good or better on some prompts. Choosing the whole layers by a usage census is faster than taking the deepest ones, but changes the text more; the change stays inside the difference stock llama.cpp shows between experts on the GPU and on the host, so the census order is the default. Both kept: `--moe-stream-gpu-layers` (on by default, fits as many layers as the free VRAM takes; `--moe-stream-gpu-layers-mode hot|last`), and `--moe-stream-vram-experts` with `--moe-stream-vram-experts-file` (off by default).

**Move the VRAM set with the text.** A set fixed from a census at start misses what this reply uses. Every 8 words, up to 64 of the most-picked experts replace the least-picked ones, and a swap needs the newcomer's recent count to beat 1.25 times the leaver's plus 4. Alone it was close to a tie on speed. It paid in every pair once the GPU computed its experts in parallel with the CPU's. Kept as `--moe-stream-vram-follow` (off by default).

**Compute the VRAM experts while the CPU computes the rest.** The GPU's pass runs inside each layer's launch right after the router is done, with its ids from a table in VRAM. For prompt calls whose waves run on the CPU, the same split exists as prompt split. Kept as settings: `--moe-stream-vram-parallel` (needs needed-early), `--moe-stream-prompt-split`.

**Turn the VRAM decode stack on as one setting.** `vram-follow` alone was close to a tie, and it only paid in every pair once `vram-parallel` was on; `vram-parallel` in turn needs `needed-early`, which needs `plan`. Testing any one of them alone misses the win. Measured together on the target model, 5 alternating pairs of 512 reply tokens went from 88.93 to 79.99 ms/word (−10 %, 5 of 5 pairs), and a second run with the single flag went from 89.76 to 83.10 ms/word (−7 %, 5 of 5 pairs). SSD wait did not fall — the GPU pass runs inside the CPU experts. Kept as `--moe-stream-vram-stack` (off by default; it needs a GPU expert pool).

**Copy an expert that leaves VRAM back into RAM.** Instead of dropping it, copy it into its layer's pool when the pool has an empty slot or the expert's keep value beats the pool's lowest; no SSD read is needed to get it back. Kept as `--moe-stream-vram-copyback` (off by default; it needs `--moe-stream-vram-follow 1`).

**Size the VRAM pool from free VRAM.** A constant reserve for the compute buffers did not scale with the batch size: with whole layers fitted by a 2 GiB reserve, a prompt at a larger batch failed to allocate. The pool now takes the largest expert count that leaves a stated amount of VRAM free after a full prompt, and counts what the CUDA runtime still adds lazily. Kept as `--moe-stream-vram-free`, with `--moe-stream-vram-growth` and `--moe-stream-vram-scratch`.

**Lend the pool's VRAM to prompts.** A prompt's compute buffer is planned by ggml's own allocator into places borrowed from the pool, then returned and refilled, so replies keep more experts in VRAM. It wins on a card with room to spare and for the replies, and loses on long prompts on a small card. Kept as `--moe-stream-vram-lend-prompt` (off by default; the autotuner decides).

**Copy the next wave's experts while this one computes.** In GPU-wave prompts the engine copies into the pool's tail (tail waves) instead of letting the scheduler copy a whole pool per graph. Two sets of copy places overlap copy and compute; one set returns room to the pool. Prompts got faster; the first reply words after a prompt got a little slower. Kept as `--moe-stream-copy-ahead` (off by default).

## The CPU side

**Keep the pools in the repacked CPU layout.** The repacked kernel is faster per matmul, and each read must be converted on its I/O thread. In prompts that run in waves it named about 20 distinct experts per ubatch and layer instead of about 390, and the text after the prompt was garbage: the prompt speed first measured for it was that broken path. Not kept; `--moe-stream-cpu-repack` remains, off, and is for decode only.

**A 4-row kernel for plain MXFP4 pools.** It gives the same bits as the one-row kernel. Without its own software prefetch it was far slower than the one-row kernel; prefetching the next row group made it win. The 256-bit version beat the 512-bit one on the CPU measured. Kept as a setting: `--moe-stream-cpu-x4`, `--moe-stream-cpu-x4-prefetch`.

**ik_llama.cpp's expert work split.** One chunk queue over all of a token's experts, ready experts first, gives the same output and, with the other same-output kernel settings, was a small gain. Kept as `--moe-stream-cpu-dispatch` (off by default). Its interleaved MXFP4 layout with a one-token kernel gained less than we needed and would have required a second full copy of the experts, and its Q8 multiply was no faster than ggml's own in a standalone test on the model's shapes. Both not kept.

**Skip the all-zero slices on the CPU.** In a wave, every pair outside the wave points at an all-zero slice, and the host matmul computed those pairs anyway, several times the real work. Writing zero without computing is exact. Prompts whose waves run on the CPU got much faster. Kept as `--moe-stream-cpu-skip-zero` (off by default; not with the repacked layout).

**Tiled CPU kernels for prompts.** Tiled q8_0 matmul, a flash attention for a q8_0 cache that converts each tile once, wider row windows in the picked-expert matmul, and ik_llama.cpp's kernels for the expert matmul in prompts. On a CPU-only machine prompts got much faster, and the differences are rounding. Kept as settings: `--moe-stream-cpu-q8-tiled`, `--moe-stream-cpu-fa`, `--moe-stream-cpu-mmid-merge`, `--moe-stream-cpu-iqk`.

**Stop copying linear-attention state on the CPU.** The recurrent state is read from its cache slot and written back in place, and small ops (the expert pick, CONCAT) use all threads. The output is the same and CPU-only decode got faster. Most of an apparent gap between two thread layouts was these copies. Kept as `--moe-stream-cpu-state-inplace`, `--moe-stream-cpu-small-ops` (off by default).

**Pin compute threads to cores, change the thread count, combine steps.** Pinning compute threads to some cores and the engine's own threads to others gave no gain, and neither did a compute thread count of 6, 8 or 10 against another. Fusing gate and up into one graph op fell inside the noise of every run, because the host step is limited by memory bandwidth, not by launches. Combining small CPU steps into fewer barriers (`--moe-stream-cpu-fuse-small`) gave the same bits and no change in speed. Not kept as defaults; `-C`, `--cpu-strict` and `--moe-stream-cpu-fuse-small` remain.

**Close the gap to plain memory reads.** The expert kernel reads memory more slowly than a loop that only reads. Deeper software prefetch, 512-bit loads and hand-placed threads did not close it. Not kept.

## The GPU side

**Overlap CPU and GPU work by reordering the graph.** The shared expert is the only GPU work in a layer that does not read the routed experts' output, so the time that can be hidden this way is small, and a second CUDA stream behind the scheduler would be needed to get it. The overlap we kept is different: the GPU computes the experts it holds in VRAM while the CPU computes the rest. Not kept.

**Copy the CPU's output to the GPU asynchronously.** On one GPU, a CPU split's output that lies in the GPU's pinned host buffer goes to the next split as an async copy, and the host no longer waits for the GPU first. Kept as `--moe-stream-async-h2d` (off by default).

**Run small device-to-device copies as kernels.** In a CUDA graph a memcpy node cannot start early, and neither can the kernel after it. A kernel for same-type copies up to a size limit lets the next kernel launch at once. Kept as the environment variable `GGML_CUDA_COPY_KERNEL=<max bytes>` (0 = off).

**Evict CUDA graphs by use, not by time.** A pause of more than 10 s between two replies dropped every reply-word graph, so each reply after a pause captured them again. A graph is now evicted when it was unused in the last 4096 computes of its context. Kept; `GGML_CUDA_TEST_GRAPH_EVICT_TIME=1` restores the time rule for tests.

**Batch the scheduler's per-input waits.** Waiting once per producing backend instead of once per input was slower, and the waits it targeted cost very little. Not kept; `GGML_SCHED_SYNC_BATCH` remains, off.

**Skip zero slices on the GPU, but do not compact the waves.** Writing zero for pairs that point at the all-zero slices saves memory traffic and is exact: kept as `--moe-stream-gpu-skip-zero` (one token) and `--moe-stream-gpu-wave-skip-zero` (prompt waves), both off by default. Compacting each wave's id tensor so the empty rows are never computed saved nothing on the GPU we measured, because a wave matmul's time follows the experts it loads, not its rows. It would need a GPU where wave matmuls are row-bound. Not kept; the code was removed.

## Prompts

**Layer by layer over the whole prompt.** At the decode batch size a prompt re-reads the experts for every batch: 954 GiB for 4k tokens on Ling. Bigger batches read each expert about once per batch and layer, which is still once per batch, and a 4096-token batch touches 442 of Ling's 512 experts in a layer, so trimming to the needed experts cannot cut much. Running every batch through one layer before the next reads each expert about once per prompt. The text differs from batch-major only by rounding. Kept as `--moe-stream-layer-major` (off by default).

**Expert matmuls of waves on the GPU.** Letting the scheduler offload them copied the layer's whole RAM pool once per graph, which cost more than the GPU saved. A view of the pool per wave makes the scheduler copy only the wave's experts when its split runs. Kept as `--moe-stream-gpu-waves N` (runs a wave graph on the GPU from N tokens; the default is 512).

**Read ahead during a layer's compute.** While layer L computes, the most-used experts of the next streamed layer, from the usage table, are read into that layer's empty slots below the needed reads (nearly all of them turn out to be needed). Treating the RAM pools as one window over the layers did better still. Both won in long prompts. Kept as `--moe-stream-read-ahead` and `--moe-stream-window` (off by default; both need `--moe-stream-wave-plan 1`, the window also layer-major).

**Compute the resident experts while a CPU-wave prompt's reads land.** The deferred wait is the same as in replies. It gave a small gain on short messages. Kept as `--moe-stream-prompt-defer` (off by default; it needs `--moe-stream-overlap 1`).

**Plan the waves once, then keep earlier waves' experts as victims.** Building a wave plan once per ubatch (resident experts first, then the misses) is kept as `--moe-stream-wave-plan`. Marking the experts an earlier wave used as the first victims for later waves (`--moe-stream-wave-victim`) changed the reads of an 8k prompt by two. Not kept; the flag remains, off.

## Things that look like wins and are not

**A prompt speed from a path that skipped experts.** The prompt speed first measured for the repacked layout, above, looked far better than any other path. The read-back check, hit rate, stall and speed all looked healthy while the experts were the wrong ones. A change that alters what the expert matmul reads must run a frozen-text identity check in the same script as its benchmark.

**Tokens per second from a few runs.** Between identical runs it swung by double-digit percentages while the SSD stall moved by under a percent. Judge by SSD stall and total time per token over at least 512 decode tokens, in paired runs that alternate the arms, on AC power, with GPU clocks matched. Pairs where the clocks differ measure the clocks. State the token count too: counters once mixed 255 decode tokens with 323 tokens of all kinds.

**Fewer bytes or a better hit rate.** Prefetching does not reduce the bytes read, it hides them: a run with perfect guesses (`--moe-stream-oracle`) read slightly more than a run with no prefetch and stalled far less. A wider guess lowered demand reads without lowering the wait.

**Fewer reader wake-ups.** A change that woke fewer I/O threads cut the empty wake-ups by a sixth to a third. Time per word did not improve, and one prompt was slower in most pairs. Not kept; the code is not in the engine.

**A timing simulator.** A replay of the engine matched its read counts closely and most of its decode speeds, but rated prefetching worse than none, with about three times the real demand misses. A timing model decides nothing here. Real runs decide. Counting on recorded routing, and the oracle mode that runs the real engine with perfect guesses, are measurements and still used.

**Results from the small test model.** The small model never needs waves, so prompt bugs cannot show on it, and a decode gain from a file format measured on it only half held on the target model.

**Healthy counters over wrong data.** A pair with no RAM slot used to get an all-zero slice and add 0, a valid-looking result. Now `--moe-stream-slot-check` stops the run with the layer, expert and token, and `--moe-stream-verify` byte-compares a sample of slots with the model file at the first token. Both are on by default.
