# Building

TokenWing builds with CMake, the same way as llama.cpp. You need CMake, a C++ compiler and Git. For the web chat page that
`llama-server` shows in the browser you also need Node.js with npm: the build makes the page from `tools/ui` and puts it inside
the server. Without npm the server is built without a page (`-DLLAMA_BUILD_UI=OFF` skips it on purpose).

```bash
git clone https://github.com/runmyai/tokenwing
cd tokenwing
```

Every build below ends with the programs in `build/bin/`. To build only some of them, add `--target`, for example `cmake --build build -j --target llama-server llama-cli`.

## CPU only

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

`GGML_NATIVE` is on by default, so the CPU code is built for the machine you build on. Turn it off (`-DGGML_NATIVE=OFF`) only for a binary that has to run on other CPUs; the expert kernels are much slower without AVX-512 or AVX2.

## NVIDIA (CUDA)

Install the CUDA Toolkit first. TokenWing is built and tested with CUDA 13.4.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build build -j
```

Two options save trouble:

- **Build for your GPU only.** By default the kernels are compiled for several GPU generations, which takes a long time. `nvidia-smi --query-gpu=compute_cap --format=csv` prints your GPU's compute capability; pass it without the dot, for example `-DCMAKE_CUDA_ARCHITECTURES=120` for 12.0.
- **A compiler nvcc accepts.** nvcc rejects GCC versions newer than it knows. If yours is one, point it at an older one: `-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-15`.

After you change the CUDA version, delete the build directory and configure it again. Its cache names the old version's libraries.

## AMD

With ROCm installed (HIP):

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DGPU_TARGETS=gfx1100 \
      -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
cmake --build build
```

Replace `gfx1100` with your GPU's target. Leaving out `GPU_TARGETS` builds for all GPUs in the machine.

With the Vulkan SDK and `glslc` installed:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON
cmake --build build -j
```

## Windows

Use a Visual Studio 2022 "x64 Native Tools" prompt with the Desktop C++ workload and CMake installed. The commands are the same, except that the configuration is chosen at build time:

```bat
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j
```

Programs land in `build\bin\Release\`.

## After you build

- If you change a library, rebuild all targets. A tool still linked against an older `libllama` can crash with no log.
- [Testing](testing.md) checks the result on your machine.
