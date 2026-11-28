# Credits

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
