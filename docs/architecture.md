# Architecture

[Inside TokenWing](how-it-works.md) explains what the engine does. This page says where each part is in the code and how the parts connect.

The streaming logic is one class, `llama_moe_stream`, with one instance per loaded model. llama.cpp's own code calls into it at a few fixed points (section 5). The engine reaches the backends only through ggml: the scheduler hooks, ggml-cpu's pool registry and the CUDA backend's proc addresses.

## 1. File map

### Our files

| File | What it holds |
|---|---|
| `src/llama-moe-stream.h` | `llama_moe_stream` (the engine), `llama_moe_stream_layer` (one layer's pools and counters), `llama_moe_stream_p1` (every option that is not a constructor argument), `llama_moe_stream_job` (one read), `llama_moe_stream_wave` (userdata of a remap op), `moe_stream_plan_in`, the op callbacks and the tensor flag bits `LLAMA_MOE_STREAM_TENSOR_*` |
| `src/llama-moe-stream.cpp` | The rest of the engine. Function names carry a prefix per part (table below). Most OS-layer functions are at the top of the file. |
| `src/llama-moe-wave-plan.h` | `moe_wave_plan_build`: which expert goes in which wave of a ubatch. A pure function of the call's ids, the RAM residents and the graph's shape; it has no engine state. |
| `src/llama-moe-correction.{h,cpp}` | At load, builds the corrected router matrices of the trained guess correction (`--moe-stream-correction`). |
| `src/llama-moe-corr-file.{h,cpp}` | Reads and checks the correction file. Knows nothing about the model. |
| `ggml/src/ggml-cuda/moe-mark.cu` | The GPU producers: kernels that copy a layer's guess or router ids into a pinned host slot, write a counter and record an event. |
| `ggml/src/ggml-cuda/moe-group-topk.cu` | The grouped top-k op (`GGML_OP_MOE_GROUP_TOPK`) as one kernel. |
| `ggml/src/ggml-cuda/moe-amax.cu` | One kernel for the down-input scale chain (`GGML_CUDA_FUSE_AMAX=1`). |
| `ggml/src/ggml-cpu/iqk/` | ik_llama.cpp's expert kernels (MIT licence) for prompt-sized `MUL_MAT_ID` (`--moe-stream-cpu-iqk`). |
| `tools/moe-trace/` | `llama-moe-trace`: records which experts every MoE layer selects for every token. Two Python readers sit next to it. |
| `tools/moe-decode-batch/` | `llama-moe-decode-batch`: runs the same words as one call per word and as calls of K words, and prints the engine's counters per call. It can also compare logits between settings. |
| `tokenwing-tools/experts/` | Builds and checks the experts file (`expert_repack.py`, `verify_sidecar.py`), makes the usage seed and the VRAM expert list. |
| `tokenwing-tools/correction/` | Trains and checks the correction file. Ling 3.0 Flash only. |
| `tokenwing-tools/testing/`, `tokenwing-tools/hardware/` | `check.py` and `compare.py`; small programs that measure the SSD, RAM and PCIe. See [Testing](testing.md). |

Function-name prefixes in `llama-moe-stream.cpp`:

| Prefix | Part |
|---|---|
| none | Pools and reads (`create_pool`, `enqueue`, `io_loop_t`, `slot_settle_ready`), eviction (`note_use`, `worse_than`, `pick_victim_core`, `p1_value`, `p1_use`, `drop_slot`), the remap op (`remap`, `remap_needed`, `remap_guess`, `remap_finish`), waves (`n_waves`), `end_call` |
| `ne_`, `rd_` | Needed-early: the guess hand-over to the helper thread (`ne_post`, `ne_mark`, `ne_loop`, `ne_gpu_setup`) and the "router done" input hook (`rd_input_hook`) |
| `la_`, `pl_`, `bl_` | Guesses for the layers 1..N ahead (`la_guesses`, `la_guess`, `la_sort`) and the waiting list (`pl_add_late`, `pl_bind`, `pl_take`) |
| `ws_` | Word-start guesses |
| `vx_`, `vp_`, `vf_` | The VRAM expert pool (`vx_setup`), its id table (`vp_row`), VRAM follow (`vf_tick`, `vf_switch`, `vf_loop`, copy-back `vf_cb_*`) |
| `tail_`, `tailw_`, `lm_`, `lend_` | The pool's tail and tail waves, layer-major on the engine side, VRAM lending |
| `ca_`, `ahead_` | Copy-ahead |
| `psplit_` | Prompt split |
| `win_`, `ra_`, `idle_`, `us_` | The RAM window, read-ahead, idle loading, the usage table |
| `rg_`, `np_` | Slots given back for a prompt's own memory, measured non-pool memory |
| `gsf_`, `g9_` | No guess for target layers that hold every expert in RAM |
| `verify_` | The read-back check |
| `rec_` | The recorder (`--moe-stream-record`) |

### llama.cpp files we changed

| File | Why |
|---|---|
| `src/llama-graph.cpp`, `.h` | `build_moe_ffn` has the streaming block: the guess, the marks, one remap op per wave, the expert matmuls on the pool tensors, the VRAM pass, the join. Shared helpers `moe_guess_routers`, `moe_guess_finish`, `moe_ws_begin`, `moe_ws_layer`, `build_ws_guess`. `allow_reuse` compares `mstream_gen`. `lm_phase` selects the layer-major graphs. |
| `src/models/bailingmoe3.cpp`, `flint.cpp` | Pass the next layers' routers and the shared-expert output to the guess, build word-start sources, take a layer range for layer-major. |
| `src/llama-context.cpp`, `.h` | The engine's call sites: start-up order, `decode` (the decode-or-prompt test, `set_decode`, `set_prompt_end`, the layer-major loop, `end_call`), the scheduler hooks, the per-graph choice of where wave matmuls run, the stats API. |
| `src/llama-model.cpp`, `.h`, `src/llama-model-loader.cpp`, `.h` | Build the engine in `load_tensors` (non-pool memory, VRAM fit, `llama_moe_stream_p1`), turn expert tensors into pools (`create_tensor`, `TENSOR_STREAMED`), place the routers, `llama_moe_stream_get_plan`. |
| `src/llama.cpp` | Retries a failed load with one whole GPU layer fewer. |
| `src/llama-kv-cache*.cpp`, `src/llama-memory*.h` | `rewind()` and `reapply()` so layer-major can replay a call's ubatches per layer. |
| `ggml/src/ggml-backend.cpp` | Scheduler hooks: input, split and copy hooks, async host input, fixed chunks. The profiler and trace timers. |
| `ggml/src/ggml-alloc.c` | `ggml_gallocr_set_fixed_chunks`: plan a graph's buffer inside caller-owned memory (VRAM lending). |
| `ggml/src/ggml.c`, `ggml.h` | `GGML_OP_MUL_MAT_MULTI` and `GGML_OP_MOE_GROUP_TOPK`. |
| `ggml/src/ggml-cpu/ggml-cpu.c` | `mul_mat_id` can wait on the pools' ready flags and compute ready experts first (`--moe-stream-overlap`; `ggml_cpu_moe_slots_register`, `ggml_moe_wait_ready`), and write 0 for zero slices (`--moe-stream-cpu-skip-zero`); one-token paths (`ggml_compute_forward_mul_mat_id_one_token`, `ggml_compute_forward_moe_up_gate_fused`); `ggml_compute_forward_mul_mat_multi`. |
| `ggml/src/ggml-cpu/ops.cpp`, `arch/x86/quants.c`, `repack.cpp`, `tiled/` | `ggml_compute_forward_moe_group_topk`, the in-place delta-net state op, the 4-row MXFP4 kernel `ggml_vec_dot_mxfp4_q8_0_x4`, `ggml_backend_cpu_repack_buffer_from_ptr`, options of the tiled matmul (`ggml_cpu_set_q8_tiled`, `ggml_cpu_set_mmid_merge`). |
| `ggml/src/ggml-cuda/` | `ggml-cuda.cu` runs the new ops, calls the mark kernels and exports the engine's entry points through proc addresses. `mmvq.cu`, `mmid.cu` and `mmq.cu` can skip zero slices (`--moe-stream-gpu-skip-zero`, `--moe-stream-gpu-wave-skip-zero`). `cpy.cu` has a small-copy kernel (`GGML_CUDA_COPY_KERNEL`). |
| `common/arg.cpp`, `common.h`, `common.cpp`, `sampling.cpp` | The `--moe-stream-*` flags, the cross-flag checks, `common_model_params_to_llama`, `--moe-stream-plan-only`, per-request engine stats. |
| `tools/server/` | `--cache-disk*` (`server_cache_disk`), per-request engine counters, the server's snapshot memory in the RAM plan (`moe_stream_srv`), Stop for a request still waiting in line, re-processing a prompt when `llama_decode` returns 3. |

## 2. The main objects

`llama_moe_stream` is built once in `llama_model_base::load_tensors` and reached through `llama_model::moe_stream()`. Every context of the model gets the same pointer in its graph parameters (`mstream`).

| Member | What it owns |
|---|---|
| `layers` | One `llama_moe_stream_layer` per model layer; null where the layer is not streamed (`layer(il)`). |
| `file` | The experts file, or with `--moe-stream model` the model file itself, opened with `O_DIRECT` where the system has it. |
| `q_demand`, `q_prefetch`, `q_idle`, `io_threads` | The read queues and the reader threads, all under `mtx`. |
| `pl_list` | The waiting list of predicted experts. |
| `ne_*` | The guess hand-over: slots, ring, helper thread. |
| `vx_groups`, `vf_*`, `tail_*` | The VRAM expert pool, VRAM follow, the pool's tail. |
| `us_cnt` | The usage table: picks per expert per streamed layer. |
| `mplan` | The plan (see below). |

A layer's pools: `llama_moe_stream_layer::weights` holds three `llama_moe_stream_weight`, one each for gate, up and down. Each is a tensor `[ne0, ne1, slots + n_expert_used]` in a CPU buffer whose usage is `GGML_BACKEND_BUFFER_USAGE_WEIGHTS`, which makes the scheduler keep the expert matmuls on the CPU next to it. A slot is the same index in all three. The last `n_expert_used` slices are all zero. A (token, expert) pair that is outside the current wave points at one of them and adds exactly 0. On Linux the pool is fresh memory, so the zero slices hold no RAM.

Per-slot state, all under `mtx`:

| Array | Meaning |
|---|---|
| `slot_expert`, `expert_slot` | Slot to expert, and expert to slot (-1 = not resident). |
| `slot_state` | `MOE_STREAM_SLOT_EMPTY`, `LOADING` (a read is in flight), `READY`. `enqueue` makes a slot LOADING, `slot_settle_ready` makes it READY, `drop_slot` empties it. |
| `slot_ready`, `proj_ready[3]` | The flags ggml-cpu polls. Written with release stores, read with acquire loads. `proj_ready` is one flag per projection (`--moe-stream-proj-ready`). |
| `keep` | The running call holds the slot; it is never a victim. |
| `slot_gen`, `slot_pending` | Generation and chunk jobs left. A job whose generation is stale belongs to a read that was given up and does nothing. |
| `slot_off` | Switched off (window, give-back): never read into, its pages returned. |
| `vf_mark`, `vx_slot[e]` | The slot is the source of a copy into VRAM. The expert's place in the VRAM pool, or -1. |

### The experts file

Two files: the data (`<name>`) and its metadata (`<name>.json`). `--moe-stream` takes the JSON path; the data path is the same without `.json`. `tokenwing-tools/experts/expert_repack.py` writes both. The format string is `tokenwing expert sidecar v1`, with `io_align` 4096.

Expert `e` of streamed layer `il` is at `base_offset + e * block_bytes`. A block is gate, up and down, one after the other, each starting on a 4096-byte boundary. The JSON has:

- `source`: `path`, `size`, `header_sha256`, `header_bytes_hashed`, `arch`. The constructor checks size (the sum of the shards for a split GGUF), architecture and the hash of the GGUF header (`verify_fingerprint`).
- `sidecar_bytes`, `n_expert`, `n_expert_used`, `expert_layout` (`plain`, `mixed`, or a repacked CPU layout), `block_bytes` or `block_bytes_per_layer`. A block must fit in an `int32` because a job's length is one.
- `blocks.<layer>`: `base_offset`, `block_bytes`, and per projection `offset`, `size`, `source_tensor`, `source_offset`, `source_shard`. The last two let the read-back check read the same bytes from the model file.

Beside the data file the engine also reads or writes `<name>.usage.json` (`tokenwing usage v1`), `<name>.layer-order.json` (`tokenwing layer order v1`) and `<name>.nonpool.txt`. The VRAM expert list is a separate file (`tokenwing vram-experts v1`, `--moe-stream-vram-experts-file`, made by `make_vram_list.py`).

### The model file

`--moe-stream model` needs no experts file: the model's own GGUF carries what the experts file would. `tokenwing-tools/experts/make_model_file.py` writes it (`tokenwing model v1`, key `tokenwing.format`): the same tensors in the same order with the same bytes and metadata, `general.alignment` 4096 so every tensor and every expert slice starts on a 4096-byte boundary, FNV-1a 64-bit checksums of the first, middle (`n_expert / 2`) and last expert's gate, up and down slice of every MoE layer (`tokenwing.sums.layers`, `tokenwing.sums.values`), and optional data appended after the last tensor, 4096-aligned, each named by `tokenwing.blob.<name>.offset` and `.size`: `correction`, `vram_experts`, `usage_seed`, `layer_order`. The appended bytes keep the file a valid GGUF that other programs open as before.

The constructor builds the same description the experts file's JSON gives. A layer's block is virtual (gate, up, down as the slot holds them), and each projection has its own file offset, because the three tensors are in three places; expert `e` of projection `p` is at `file_off[p] + e * size[p]`. A read of an expert is one request per projection, queued like a chunk: a piece never crosses a projection, and it lands at the same place in the slot as before. `--moe-stream-verify` compares the sample experts with the file's checksums and the first filled slot of each layer with the file, once its last piece has landed. The files named after the model (`<model>.gguf.usage.json`, `<model>.gguf.layer-order.json`, `<model>.gguf.nonpool.txt`) take the place of the experts file's. A split GGUF is refused, and so is a file without the key.

### The plan

`moe_stream_plan_in mplan` is filled by the constructor: layers, whole GPU layers, bytes per expert, slots per layer, RAM budget, pool bytes, `MemAvailable`, non-pool bytes. `llama_moe_stream_get_plan` builds an engine with no reader threads; the constructor returns once the plan is made. `--moe-stream-plan-only` prints that, so the plan and a real load come from the same code. Do not confuse it with the waiting list, whose flag is also called `--moe-stream-plan`.

## 3. Threads and locks

| Thread | Started in | Runs | Touches |
|---|---|---|---|
| Graph thread (whoever calls `llama_decode`) | the tool or server | Graph build, `set_decode`, the scheduler loop, CPU splits (the remap op runs on thread 0 of the CPU pool), `end_call`, VRAM follow's tick | Takes `mtx` in the remap op, `end_call` and the call-start hooks. |
| Readers (`--moe-stream-io-threads`, default 8) | the constructor | `io_loop_t<0\|1\|2>`: pick a job, `preadv` into the slot's pool slices (one request for an experts-file block, one per projection from a model file), settle. The loop version is chosen once at start-up. | `mtx` to pick and to settle. The read itself is unlocked. |
| Guess helper (needed-early only) | the constructor | `ne_loop`: waits for a posted guess (`ne_wait_posted`), then `on_guess_ready` adds waiting-list entries (`la_guesses` or `prefetch_next`). | `mtx`. It waits for the GPU event unlocked. |
| VRAM-follow uploader | `vf_setup`, inside `vx_setup` | `vf_loop`: copies experts into VRAM places on its own stream, and copy-backs from VRAM into RAM slots. | `vf_mtx`; takes `mtx` to settle a copy-back. |
| Idle loader (`--moe-stream-idle-load` only) | `idle_arm` | `idle_loop`: after a quiet pause, reads usage-table experts into EMPTY slots at the lowest priority. | `mtx`. |
| GPU streams | ggml-cuda | The GPU splits and the mark kernels. | The pinned guess and router slots and their counters. |

Locks and condition variables:

- `mtx` is the engine lock. It guards the queues, every slot array, `expert_slot`, `pl_list`, the helper's ring and the counters. The remap op holds it from `remap_needed` to the end of `remap_finish`, except inside `cv_done.wait`.
- `cv_work`: a job was queued, an entry was added, or stop. `cv_done`: a read settled or a copy-back landed. `ne_cv`: a slot was posted. `ne_done_cv`: the helper consumed the last posted slot.
- `vf_mtx`, `vf_cv`, `vf_cv_done`: the uploader's queues. `verify_mtx`, `repack_bufs_mtx`: scratch buffers of the read-back check and the repack.
- No lock: `slot_ready` and `proj_ready`, and the pinned counters the GPU writes.

Ordering rules the code depends on:

- `enqueue` clears a slot's ready flag before the read is queued, for every read. Otherwise `mul_mat_id` could compute the slot's previous expert.
- The helper takes a guess only when the slot's counter equals `ne_call_seq`, the number of the current call. A slot left by an aborted call can never pass.
- The waiting list is empty at every word boundary. `end_call` drops the entries of layers that ran; in a complete call, an entry for a layer that did not run stops the run.
- A wait on the GPU happens before `mtx` is taken (see `end_call`). Trace events are written after the unlock.

## 4. Start-up

1. **Flags.** `common_params_parse_ex` (`common/arg.cpp`) parses the `--moe-stream-*` options into `common_params`. A setting whose requirement is not met is switched off with one line (the local `off` helper, repeated in the constructor). `common_model_params_to_llama` copies them into `llama_model_params`. They belong to the model, not the context.
2. **Non-pool memory.** `llama_model_base::load_tensors` sums the weights placed in RAM (not the streamed expert tensors; weights placed on the GPU count for the VRAM fit instead) and adds a fixed reserve (`MOE_STREAM_OTHER_RESERVE`), the server's snapshot terms, the recorder, the helper's slots, the wave-plan tables and the correction matrices. With `--moe-stream-nonpool-measured` the reserve comes from a record next to the experts file.
3. **Whole GPU layers.** With `--moe-stream-gpu-layers` negative, free VRAM minus the GPU weights minus `--moe-stream-vram-reserve` and a per-ubatch margin (`MOE_STREAM_FIT_MIB_PER_TOKEN`) is passed to the constructor, which takes layers in order while each layer's experts (all `n_expert` of them, at that layer's block size) still fit. An explicit list or a count skips the fit.
4. **Constructor.** It opens the experts file, parses the JSON, checks the fingerprint, builds `blocks` and chooses the whole GPU layers: an explicit `--moe-stream-gpu-layers-list` as given, else in the order of the model's `.layer-order.json`, else Ling's built-in order, else shallowest first. Choosing a layer clears its `have[]` entry, and everything after ignores it. Then it computes the slot count (section 7), fills `mplan`, and returns if there are no reader threads. Otherwise it starts the readers and the helper and sets up idle loading.
5. **Pools.** `create_tensor` asks the loader to skip each expert tensor of a streamed layer (`TENSOR_STREAMED`) and returns the tensor that `create_pool` makes. That allocates the memory (4096-aligned; fresh `mmap` memory on Linux, so the zero slices hold no RAM), wraps it as a CPU buffer, tries `mlock` (one warning if it fails) and registers the pool's ready flags with ggml-cpu (`register_pool`). `verify_pools` checks every streamed layer got three.
6. **After the weights are placed.** `ne_set_router_gpu` and `ne_gpu_setup` decide per layer whether its guess comes from the GPU producer or the host producer, and allocate the pinned slots. The correction matrices are built. `pin_pools` runs with `--moe-stream-pin`.
7. **Context.** `llama_context` reserves its graphs. With a VRAM expert pool (`--moe-stream-vram-experts` or `--moe-stream-vram-free`), `vx_prewarm` then creates the backend's lazy cuBLAS memory so the free VRAM read next is real, and `vx_setup` allocates and fills the pool (free VRAM read after the KV cache and compute buffers) and starts the uploader through `vf_setup`. The graphs are reserved again, because they now carry the pool's device pass. `vx_report` prints the placement. Then `llama_moe_stream_verify_gpu_tier` and `idle_arm`.
8. **Read-back check.** `verify_boundaries` compares the first and last expert of the first and last streamed layer with the model file. Then the first single-job needed read of each streamed layer is compared by the reader thread that wrote it (`verify_slot`). A mismatch stops the run. `--moe-stream-verify N` limits it to N layers; 0 turns it off.

## 5. One decode call

`llama_context::decode` calls a batch a decode call when its position is above 0 and it has one token, or up to 8 tokens that continue the last decode position and all ask for an output. Everything else is a prompt call (section 6).

The table describes a GPU run with `--moe-stream-plan`, `--moe-stream-needed-early`, `--moe-stream-overlap` and `--moe-stream-vram-parallel` on. Each is off by default. Without one, its step takes the plain form: the remap op adds the guess to the list itself, waits for its own reads, and so on.

| Step | Where | What |
|---|---|---|
| 1 | `decode` | `idle_call_start` takes queued idle reads back. The call's own host memory (`rg_call_need`) is stored in `rg_need`. |
| 2 | `set_decode` | Call-start hooks, in order: `psplit_call_start`, `ne_call_start` (a new call number goes to the pinned cell the GPU reads), VRAM follow's call start (switches whose copy landed), `win_call_start`, `rg_call_start`, `np_call_start`, `gsf_call_start`. |
| 3 | `process_ubatch` | The previous graph is reused if `allow_reuse` accepts it, else `model.build_graph` builds it. |
| 4 | `build_moe_ffn`, per streamed layer | Router (`build_moe_probs`, `build_moe_topk`; one `ggml_moe_group_topk` with `--moe-stream-fused-route`). The guess for the next layers (`moe_guess_routers`, `moe_guess_finish`; one `ggml_mul_mat_multi` with `--moe-stream-guess-combined`). With the GPU producer the guess is flagged `LLAMA_MOE_STREAM_TENSOR_GUESS_MARK` and the ids `LLAMA_MOE_STREAM_TENSOR_ROUTER_MARK`, with the router input as `src[1]`. With `--moe-stream-vram-parallel` the VRAM pass (`build_vram_chain`) goes in the same GPU launch. Then the remap op: `ggml_map_custom1/2/3` with `llama_moe_stream_remap` and its variants, one task. Then `build_expert_wave` on the pool tensors, and the join: the waves' sum plus the VRAM pass. |
| 5 | scheduler | `ggml_backend_sched_compute_splits` runs the layer's GPU split: attention, router, top-k. After the ids node the mark kernels write ids and router input into the pinned router slot, then the counter, then record "router done". After the guess node they write the guess slot, the counter, "guess done". |
| 6 | input hook | For the CPU split, `llama_moe_stream_sched_input_hook` calls `rd_input_hook`. It reads the layer's counter and waits for the event only if the counter is not this call's number, then copies ids and router input out of the slot. |
| 7 | remap op | `llama_moe_stream::remap` under `mtx` runs `remap_needed`, `remap_guess`, `remap_finish`. |
| 8 | ggml-cpu | With `--moe-stream-overlap`, `mul_mat_id` computes the experts whose flags are set, then waits (`ggml_moe_wait_ready`) for the others. `take_overlap_waits` books the longest wait into the layer's stall counter. |
| 9 | join | The CPU output goes up (an async copy with `--moe-stream-async-h2d`) and is summed with the GPU's. A split hook (`llama_moe_stream_sched_split_hook`) tells the engine when a GPU split with marked word-start sources was issued (`ws_split_issued`). |
| 10 | `end_call` | Called by a destructor in `decode`, so every way out runs it. See below. |

`remap_needed`, in order:

1. Mark the layer as started in this word, so the helper skips a guess that is too late.
2. Clear the previous call's `keep` marks, only on layers that have any (`pin_list`).
3. Collect the distinct experts of the ubatch in first-use order (`uniq`), leaving out VRAM-held ones.
4. Plan the waves when `--moe-stream-wave-plan` is on (once, at wave 0).
5. Pin every expert that is already resident.
6. Waiting list: bind entries of picked experts at the pick, drop the entries of unpicked ones (`--moe-stream-drop-stale`).
7. Fill loop. A READY expert is a hit. A LOADING one is waited for, and `promote_chunks` moves its queued chunks to `q_demand` (`--moe-stream-promote`). An absent one takes a slot from `pick_victim` (`drop_slot`, `enqueue`).

`pick_victim_core` takes an EMPTY slot first. Otherwise it ranks by `worse_than`: with `--moe-stream-policy decay` in decode, the lowest decayed use count, then the oldest use, then the lower expert id. In prompts it is LRU.

`remap_guess` hands the guess on. With the GPU producer the GPU already wrote it, and `ne_mark` queues the layer for the helper. With the host producer `ne_post` copies it into the slot. Without needed-early, `la_guesses` or `prefetch_next` add to the list (or, without the list, issue the reads) directly.

`remap_finish` writes the slot ids the matmul reads: the expert's slot, or a zero slice for an expert outside the wave or held in VRAM. It either waits for the reads itself or leaves the wait to `mul_mat_id` (`defer_wait`: `--moe-stream-overlap` is on, one wave, fewer rows than `GGML_OP_OFFLOAD_MIN_BATCH`, default 32). The slot check (`--moe-stream-slot-check`) stops the run if a pair has neither a slot nor a VRAM place.

The helper does the list work: `ne_loop`, `on_guess_ready`, `la_guess` (add entries, at most `--moe-stream-guess-pace` new ones per far guess), `la_sort`. An idle reader then calls `pl_take`, which binds the first entry to a victim slot and queues its read. The reader order is `q_demand`, `q_prefetch` (under the caps), the list, `q_idle`.

On a CPU-only machine the router runs on the CPU and there is no GPU producer. The remap is split in two ops: `llama_moe_stream_remap_a` queues the needed reads right after the router, `llama_moe_stream_remap_b` does the rest after the shared expert and the guess.

`end_call`, in order: end copy-ahead and prompt split and release their keeps; wait for the GPU producer if the call stopped part-way; `win_end` if a window pass ran; wait for the GPU if a tick is due (before `mtx` is taken); then under `mtx`, wait for the helper to consume what the call posted, check that the list is empty, run `take_overlap_waits` per layer; VRAM follow's word end (`vf_word_end_t` decays the counts, and every 8th reply word `vf_tick` decides swaps and hands them to the uploader); the end fill; idle bookkeeping.

## 6. One prompt call

A prompt call runs the same graph code with more rows. The differences are below.

**Waves.** `n_waves(n_tok)` is the most distinct experts the ubatch can need, `min(n_expert, n_tok * k)`, divided by the wave width and rounded up. The width is the slot count, or the narrower `tailw_wave_w` for tail waves; `--moe-stream-max-waves` caps it. The graph holds one remap op and one set of matmuls per wave. Wave `w` of the remap handles the experts at positions `[w * W, (w + 1) * W)` of `uniq`, or the wave plan's choice. The pairs of other waves get zero slices, and the waves' outputs are summed. A graph with more than one wave carries the flag `LLAMA_MOE_STREAM_TENSOR_WAVE`, which keeps its matmuls on the host unless the ubatch has at least `--moe-stream-gpu-waves` tokens (default 512). `process_ubatch` decides this and `graph_reserve` repeats the same rule. With `--moe-stream-wave-plan`, `moe_wave_plan_build` puts resident experts in the first waves so none is evicted before its wave.

**Layer-major** (`--moe-stream-layer-major`, Ling, Flint and qwen4exp). `lm_layer_major_ok` admits a call of more than one ubatch with one sequence. `llama_context` raises `n_batch` up to the context size (`--moe-stream-layer-major-batch` caps it). The `decode` loop then walks the layers (`lm_group` layers at a time, one by default):

1. Every ubatch runs the layer. For a streamed layer this is a graph with `lm_phase == 1`: attention and router only. The router's ids and weights go into the call's tensors (`lm_hid`, `lm_x`, `lm_ids`, `lm_w`). Before each later layer the memory is rewound with `rewind()` and each ubatch replayed with `reapply()`.
2. Then one expert graph for the whole call (`lm_phase == 2`): one remap op per wave, the matmuls in chunks of `n_ubatch` rows, each chunk added into its rows of `lm_hid`.
3. `lm_signal_done` calls `lm_layer_done` with a per-layer event, so the window knows the layer's slots are free.

The engine side is `lm_call_begin`, `lm_layer_begin`, `lm_layer_done`, `lm_call_end`. If the call's tensors do not fit, or the expert graph holds fewer chunks than the call needs, the call runs in whole-layer groups or ubatch by ubatch, with one line.

**Reading ahead.** In a layer's wave-0 remap op, after its own needed reads are queued, `win_issue` (window) or `ra_issue` (read-ahead) queues guess-priority reads of the next streamed layers' most-used experts from `us_cnt`. `win_layer_start`, `win_finish` and `win_end` move room between layers inside a fixed total.

**GPU waves and tail waves.** The last places of the VRAM pool tensor are its tail. `tail_ubatch` vacates them before a GPU-wave graph, and `llama_moe_stream_sched_copy_hook` gives the scheduler's pool copy a target inside the tail. Tail waves are on with the window on a GPU machine, or with copy-ahead. The remap op's output then carries `LLAMA_MOE_STREAM_TENSOR_TAIL`, and the op calls `tailw_gather` once its slot ids are final. It maps them to tail places and queues "place p from RAM slot s" copies on the compute stream, so a wave's experts may sit in any slot.

**Copy-ahead** (`--moe-stream-copy-ahead`). `tailw_gather` of layer L passes the next layer's READY experts (`expert_ready`) to the VRAM-follow uploader (`ca_run_job`), which copies them into a second set of places, or in one-set mode into the places the current wave has finished with. In one-set mode a node flagged `LLAMA_MOE_STREAM_TENSOR_DONE_MARK` tells the uploader when those places are free.

**Prompt split** (`--moe-stream-prompt-split`). A CPU-wave ubatch of 2 to `ps_rows` rows gets the multi-token form of `--moe-stream-vram-parallel`: the router mark hands the CPU split its inputs through one pinned slot, and the VRAM pass runs in the GPU launch beside it. `psplit_count` counts the next layer's guessed experts, `tail_borrow` takes tail places for the call, and `psplit_after_remap` hands the uploader a job that copies the most-guessed experts into them early.

**VRAM lending** (`--moe-stream-vram-lend-prompt`). A prompt call's GPU compute buffer is planned by ggml in lent places of the VRAM pool (`lend_pieces`, `lend_lend`, `ggml_backend_sched_set_fixed_chunks`). Reply graphs keep a small real buffer. The pool's experts step aside for the call and are refilled after it.

**Handing over to decode:**

1. `set_prompt_end` marks the last ubatch of a call that asks for an output and is shorter than `n_batch` (`MOE_STREAM_PROMPT_END_ALL=1` drops the length test). With `--moe-stream-prompt-end recency` its waves go in recency order, so the experts used last stay in RAM for the reply.
2. `win_end` brings every layer back to its normal slot count and reads back kept experts that were dropped during the call.
3. `end_fill` (`--moe-stream-vram-end-fill`) refills the tail with experts that are already READY in RAM, ranked by the prompt-end weights. Without it, `vf_call_start_t` refills at the first reply word. The same fills refill lent places.
4. `rg_call_start` at the first decode call switches given-back slots on again, EMPTY.
5. After a quiet pause `idle_round` fills EMPTY slots from the usage table.

## 7. How memory is bounded

**RAM slots.** All streamed layers get the same slot count. It is `--moe-stream-slots`, or the budget divided by the bytes of one expert summed over the streamed layers. The budget is the smaller of `--moe-stream-ram` and the room: `MemAvailable` minus `--moe-stream-ram-free`, or the process's memory limit (`os_mem_cap_get`: on Linux the tightest cgroup limit up the tree, cgroup v2 `memory.max` or cgroup v1 `memory.limit_in_bytes`, found through `/proc/self/cgroup` and `/proc/self/mountinfo`, its use less the file cache the kernel can free without swap: the active and inactive file pages less the mapped, dirty and under-writeback ones, so a model another program read earlier does not look like use while a model a running program has mmapped does; tmpfs / shm stay use) less a margin and page tables, whichever is smaller, minus the non-pool bytes. The count is clamped to `n_expert` (a whole layer resident) and raised to at least 3 times the model's top-k, with one line. Pools that would pass the limit or leave less than the keep-free margin shrink to what fits.

**Non-pool memory** is an estimate made before anything is allocated (section 4, step 2). `--moe-stream-nonpool-measured` replaces the fixed reserve by what a previous run with the same settings measured; a start that needs more gives slots back.

**A prompt's own memory.** The call's tensors, the masks a sliding-window cache keeps for replays and pageable compute buffers are not in the RAM plan. `llama_context::rg_call_need` estimates them, and `rg_call_start` compares with the room under the limit. If short, every streamed layer gives back the same number of slots: EMPTY first, then the coldest by the usage table, else by the keep rule. The waves follow the smaller width. A decode call that finds the room takes them back.

**Waves.** A ubatch that needs more distinct experts than slots runs in waves (section 6). With the window on, a pool is reserved for `n_expert` slots as address space; the total of switched-on slots stays fixed and moves between layers.

**VRAM.** Whole GPU layers are fitted in step 3 of section 4, and a load that cannot allocate a weight buffer is tried again with one layer fewer. The VRAM expert pool is sized from free VRAM after the KV cache and compute buffers (`--moe-stream-vram-free`, with `vx_growth_bytes` for what CUDA still allocates lazily) or pinned (`--moe-stream-vram-experts N`); a failed trial reserve shrinks it. If it cannot be made, `vx_flags_off` turns it and everything that needs it off with one line.

**Prompt compute memory.** Layer-major tensors go into the pool's tail or lent places (`lm_lend_place`), else a buffer of their own, else the call runs ubatch by ubatch. Lending that does not fit a long call splits it into calls that do (`lm_lend_max_tokens`). A full card ends a layer-major call with `llama_decode` returning 3, and `llama-server` processes the prompt again.

**What stops the run.** The engine falls back and prints one line for a resource it lacks: no `O_DIRECT` (buffered reads), no pinned memory (pageable), no GPU (host producer), `mlock` refused. `moe_stream_fail` (an abort) is for wrong data: a read that fails after its retries (`MOE_STREAM_READ_RETRIES`), a read-back mismatch, a failed slot check, an entry left on the waiting list, a layer with no free slot after `pick_victim_or_fallback` has waited.

## 8. Conventions for changing this code

- **Every new behaviour is a flag, off by default.** Add the field to `llama_model_params` and its default in `llama_model_default_params`, to `common_params`, the option and its checks in `common/arg.cpp`, the copy in `common_model_params_to_llama`, a field in `llama_moe_stream_p1` filled in `load_tensors`, and a row in [Settings](settings.md). Where the flag needs another flag or hardware it lacks, switch it off with one line; do not refuse to run. Machine-dependent numbers are flags too, never constants for one model or machine.
- **Output must not change when a feature is off.** A feature that changes what the expert matmul reads must be checked for identical text in the same run as its speed measurement (`check.py`, `compare.py` report whether the text matched). Rounding differences from a different execution path are accepted. Garbage, NaN or an empty reply are bugs.
- **System calls go through small named functions** in `llama-moe-stream.cpp`, never inline in the engine's logic: `moe_stream_preadv` (every expert read, from the experts file or the model file), `moe_stream_mem_available`, `moe_aligned_alloc`, `os_mem_cap_get`, `os_proc_rss`, `os_page_size`, `os_write_file_atomic`, `os_read_file_range` (the data appended to a model file), `llama_moe_os_fopen_utf8`. Each system gets its own version of each function, and an extra that only one system has stays in that version. A new system call gets its own function next to these.
- **Test-only code is chosen at start-up.** Switches are `MOE_STREAM_TEST_*` environment variables, read once into a function-local `static const bool` or in the constructor, and most print a `TEST ONLY` line. Where a test changes a hot path, the choice is a template parameter or a member-function pointer fixed at start (`io_loop_t<N>`, `ne_loop<DELAY>`, `vf_fill_fn`), so the normal path has no extra test per word.
- **Lock discipline.** A field written by a reader, the helper or the uploader is guarded by `mtx`. Do not wait on the GPU under it. Add the check that proves a new invariant (the slot check, the empty-list check) rather than a comment.
- **A new architecture** is added to `arch_leads` in the constructor when its graph passes the next layers' routers into `build_moe_ffn`, and to `llama_arch_layer_major` when its builder takes a layer range.
