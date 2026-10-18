# GGUF on the Snapdragon NPU (Hexagon backend, spike)

This branch runs GGUF models on the Hexagon NPU of Snapdragon X laptops through the
ordinary llama-server runner, using llama.cpp's Hexagon backend. It is a spike: the build
below works on a Snapdragon X Elite (X1E78100), but packaging and redistribution are open.

## Why b11644 and signed libraries

The Hexagon backend has two halves: `ggml-hexagon.dll` on the CPU, and HTP ops libraries
(`libggml-htp-v73.so` ... `-v81.so`) that run on the NPU. Windows loads the NPU libraries
only when a trusted catalog signs them; self-built ones need test-signing, which this
project does not enable. GenieX v0.9.1 (Qualcomm, BSD-3) publishes a Microsoft-signed set
built from llama.cpp `e467e5b`, tag `b11644`. The host/NPU protocol changed after the
previous pin `b11521`, so `LLAMA_CPP_VERSION` is `b11644`, and the host side is built from
source at that commit.

## Build

Prerequisites: Visual Studio 2026 with the ARM64 C++ tools and its clang (VC\Tools\Llvm),
CMake and Ninja (both ship with Visual Studio), the trimmed Hexagon SDK 6.6
(`hexagon-sdk-v6.6.0.0-arm64-wos.tar.xz` from the snapdragon-toolchain GitHub releases;
check its SHA256 against the release), and the signed HTP libraries from the GenieX v0.9.1
Windows ARM64 archive (`lib\llama_cpp\libggml-htp-v*.so`, `libggml-htp.cat`,
`libggml-htp.inf`).

From an ARM64 Visual Studio environment (`vcvarsarm64.bat`) with clang first on PATH:

```bat
set HEXAGON_SDK_ROOT=C:\path\to\Hexagon_SDK\6.6.0.0
set HEXAGON_TOOLS_ROOT=%HEXAGON_SDK_ROOT%\tools\HEXAGON_Tools\19.0.07
set OLLAMA_HEXAGON_HTP_DIR=C:\path\to\geniex\lib\llama_cpp
cmake -S llama\server --preset cpu_arm64_clang -B build\cpu
cmake --build build\cpu --target llama-server llama-quantize llama-perplexity
cmake --install build\cpu --prefix dist --component llama-server
cmake -S llama\server --preset hexagon_arm64 -B build\hexagon
cmake --build build\hexagon --target ggml-hexagon
cmake --install build\hexagon --prefix dist --component llama-server
go build -o dist\ollama.exe .
```

Both presets use clang targeting the MSVC ABI, which the Hexagon host code and the SDK's
generated stubs need; the packaging script's `cpu_arm64` preset (LLVM-MinGW) is not used
here. The hexagon preset installs `ggml-hexagon.dll` and the signed libraries into
`lib\ollama\hexagon`; configure fails if any signed file is missing.

## What the runner does

Discovery treats `lib\ollama\hexagon` like any GPU backend directory. When a runner
directory contains `libggml-htp-v*.so`, the runner also sets:

- `ADSP_LIBRARY_PATH` to that directory, so FastRPC finds the NPU libraries;
- `GGML_HEXAGON_MBUF=,,3000M` unless set: the backend otherwise reports 0 MiB, and
  discovery drops zero-memory devices. 3000 MiB is the buffer address space of HTP v73.

Discovery then reports `library=Hexagon name=HTP0`, and models load with all layers on
`HTP0`. Flash attention is allowed on Hexagon (`--flash-attn auto`); turning it off halves
NPU prefill.

## Measured (X1E78100, Qwen2.5-1.5B-Instruct)

Through `/api/generate` (556-token prompt, 128 tokens):

| Model | Prefill | Decode |
|---|---|---|
| GGUF Q4_0 on the NPU | 1,229 tok/s | 28.7 tok/s |
| GGUF Q4_0 on the CPU (`num_gpu 0`) | 834 tok/s | 55.3 tok/s |
| QNN w8a16 ORT bundle (NPU) | 772 tok/s | 20.2 tok/s |

The CPU decodes faster than the NPU; the NPU's gains are prefill, power and a free CPU.
Perplexity on wikitext-2 against BF16: Q4_0 1.09x, Q4_K_M 1.05x, Q8_0 1.00x, with the same
results on the NPU as on the CPU.

## Open

- Redistribution: the signed libraries come from a BSD-3 release; confirm with Qualcomm
  before shipping them in a package.
- Packaging with the clang/MSVC-ABI build, and a memory budget per HTP version.
- Whether GGUF models should default to the NPU or the CPU, given the CPU's faster decode.
