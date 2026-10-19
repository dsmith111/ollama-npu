# GGUF on the Snapdragon NPU (Hexagon backend, experimental)

This branch runs GGUF models on the Hexagon NPU of Snapdragon X laptops through the
ordinary llama-server runner, using llama.cpp's Hexagon backend. It is experimental and
off by default. The build below works on a Snapdragon X Elite (X1E78100); packaging is
open.

## Enabling it

Get Qualcomm's signed NPU libraries once, then start the server with `OLLAMA_HEXAGON=1`.
Upstream first shipped Vulkan the same way, as an opt-in setting (#12931):

```bat
ollama npu setup --provider hexagon --accept-qualcomm-terms
set OLLAMA_HEXAGON=1
ollama serve
```

Setup downloads Qualcomm's GenieX v0.9.1 Windows SDK archive (87 MB) from GitHub, checks
it and the six files it needs against pinned hashes (`x/hexagon`), and installs them with
GenieX's licence into `%LOCALAPPDATA%\Ollama\runtimes\hexagon-htp\win-arm64\b11644`. It
never replaces an existing directory. Without `--accept-qualcomm-terms` it only shows the
terms (see "Redistribution"). `--dir` installs elsewhere (then select that directory with
`OLLAMA_HEXAGON_PATH`), `--package-dir` uses an archive downloaded beforehand, and
`--validate-only` checks the directory runners would use.

Without `OLLAMA_HEXAGON`, discovery skips `lib\ollama\hexagon` and logs `experimental
Hexagon NPU support disabled`, so GGUF models run on the CPU. With it, discovery reports
`library=Hexagon name=HTP0`, and GGUF models load onto the NPU by default; `num_gpu 0`
keeps a model on the CPU. If no verified set is found, discovery logs `Hexagon NPU
libraries unavailable` with the setup command and offers only the other devices.

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
CMake and Ninja (both ship with Visual Studio), and the trimmed Hexagon SDK 6.6
(`hexagon-sdk-v6.6.0.0-arm64-wos.tar.xz` from the snapdragon-toolchain GitHub releases;
check its SHA256 against the release).

From an ARM64 Visual Studio environment (`vcvarsarm64.bat`) with clang first on PATH:

```bat
set HEXAGON_SDK_ROOT=C:\path\to\Hexagon_SDK\6.6.0.0
set HEXAGON_TOOLS_ROOT=%HEXAGON_SDK_ROOT%\tools\HEXAGON_Tools\19.0.07
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
here. The hexagon preset installs `ggml-hexagon.dll` into `lib\ollama\hexagon`. For a local
build you may also bundle the signed set there: set `OLLAMA_HEXAGON_HTP_DIR` to the GenieX
archive's `lib\llama_cpp` folder, and the preset installs the six files and GenieX's
licence (configure fails if a file is missing). Packages leave the set out.

For a portable package, `scripts\build_npu_windows.ps1 -HexagonSDKRoot ...` adds this
backend to the LLVM-MinGW package (see [development.md](development.md)).

## What the runner does

With `OLLAMA_HEXAGON=1`, discovery treats `lib\ollama\hexagon` like any GPU backend
directory. It looks for the signed set in `OLLAMA_HEXAGON_PATH`, then `lib\ollama\hexagon`
(a local build that bundles it), then the setup directory. The first directory holding a
`libggml-htp.cat` must match the pinned hashes; a set that doesn't is refused, with no
fallback to a later directory. The runner then sets:

- `ADSP_LIBRARY_PATH` to that directory, so FastRPC finds the NPU libraries;
- `GGML_HEXAGON_MBUF=,,3000M` unless set: the backend otherwise reports 0 MiB, and
  discovery drops zero-memory devices. 3000 MiB is the buffer address space of HTP v73.

Flash attention is allowed on Hexagon (`--flash-attn auto`); turning it off halves NPU
prefill.

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

The opt-in, checked on the same model: with `OLLAMA_HEXAGON=1` all 29 layers loaded on
HTP0 (898 MiB; `/api/ps` reported `size_vram` equal to `size`). Without it, discovery
offered only the CPU, and the same greedy prompt gave the same text there.

Setup, checked with a payload that leaves the set out: `ollama npu setup --provider hexagon
--accept-qualcomm-terms --dir ...` downloaded and verified the archive in 5 seconds, and the
installed catalog still validates as Microsoft-signed. Selected with
`OLLAMA_HEXAGON_PATH`, the set put all 29 layers on HTP0 again; with no set reachable,
discovery logged the setup hint and the model ran on the CPU.

## Redistribution

- The signed set is a Qualcomm release asset: GenieX v0.9.1 on GitHub
  (`qualcomm/GenieX`), BSD-3-Clause, "Copyright (c) 2024-2026, Qualcomm Technologies, Inc.
  and/or its subsidiaries". GenieX builds it from upstream llama.cpp (MIT; a git
  submodule) at `e467e5b`, and "Microsoft Windows Hardware Compatibility Publisher" signs
  the catalog.
- BSD-3-Clause allows redistribution in binary form when the copyright notice, the
  conditions and the disclaimer go with it, and Qualcomm's name is not used to endorse the
  product. The release archives omit the licence file, so setup (and a local build that
  bundles the set) writes the v0.9.1 `LICENSE`, verbatim, as `GenieX-LICENSE.txt`.
- GenieX's README adds: "Use of this project is also subject to Qualcomm's Terms of Use".
  Those terms (revised May 19, 2026) leave the BSD-3 grant standing: their limits on
  downloaded content (one device, personal and non-commercial use) give way where
  "applicable license agreements with Qualcomm" allow more (section 7.4). They add four
  things:
  - **Responsible AI License** (section 8). AI "features, models, algorithms, or related
    materials" that Qualcomm makes available through its sites and services also come
    under Qualcomm's RAIL (revised February 17, 2026), which itself claims any such
    materials from Qualcomm Technologies. GenieX is Qualcomm's AI runtime, so its
    libraries plausibly do. The RAIL's use
    restrictions apply "notwithstanding any provision of the Applicable Agreement to the
    contrary", and the licensee "will not (and will not permit any other person to)" use
    the materials for any of 20 purposes. These include military use (including any
    project a military organization sponsors), predicting crime, and generating content
    "without expressly and intelligibly disclosing that the text is machine generated".
    Restrictions on fields of use fall outside the Open Source Definition (criterion 6).
  - **Export and sanctions** (section 3): no export of the materials, even inside another
    product, to sanctioned territories or restricted parties, and best efforts down the
    chain for the Russia and Belarus rules.
  - **No patent licence**, express or implied (section 7.5); BSD-3 has no patent grant
    either.
  - **No suggested association** with Qualcomm products or trademarks (section 7.4).
- The signature covers these exact bytes, which setup and runners check. Another
  llama.cpp commit needs a signed set built from that commit.
- `ggml-hexagon.dll` is built from llama.cpp sources. Its Hexagon SDK content is the
  FastRPC stub that `qaic` generates from llama.cpp's `htp_iface.idl`, plus the SDK headers
  the host code includes: `rpcmem.h`, `AEEStdErr.h` and `AEEStdDef.h` are BSD-3-Clause
  (packages carry their notices), and `remote.h` and `dspqueue.h` are marked Qualcomm
  proprietary and contribute declarations only. The DLL links only `ggml-base`
  and Windows libraries; FastRPC (`libcdsprpc.dll`) comes from the device driver. Upstream
  llama.cpp publishes binaries built this way (`bin-android-arm64-snapdragon`,
  `bin-linux-arm64-snapdragon`), though none for Windows.

Bundling the signed set would carry Qualcomm's use restrictions into every fork package,
so packages leave it out, as they leave out Qualcomm's QNN runtime (`ollama npu setup
--provider qnn`). `ollama npu setup --provider hexagon` fetches the GenieX archive below at
setup time, checks it against these hashes, and shows Qualcomm's terms before downloading.
The fork package stays under its own licence, and each user gets Qualcomm's files from
Qualcomm, under Qualcomm's terms.

The files, identical in `geniex-sdk-windows-arm64-v0.9.1.zip` (SHA256 `2ed2bf63f042086c525b442081ed8d7ad4ff76bcce8bad6eeabe1cf38cc0a1b5`)
and `geniex-bench-windows-arm64-v0.9.1.zip` (SHA256 `e465931fbf4ccfbd011632fb1b6e337362963d7550c24d6a808d99504565acdb`),
under `lib\llama_cpp`:

| File | SHA256 |
|---|---|
| `libggml-htp-v73.so` | `e93616d06df4ec812a66ca2b05e3bd1706ef588926383b94091bebf007ed913e` |
| `libggml-htp-v75.so` | `44c064535682df7b71a9f21193aa83c55bc846f3db74f35eb1ac0ab1814d15ed` |
| `libggml-htp-v79.so` | `279187f7a4b0df53deedc18d265268f52ef0f20aa63140f8e150841861442620` |
| `libggml-htp-v81.so` | `3207bd9ce22844c1d9fb4f3ab1c3acd34819de9811f9b4337a7ad1a14965a71e` |
| `libggml-htp.cat` | `bdd10a100e426eb48398a8bc5751c125579f6ec52c15e25106ba57e669444b41` |
| `libggml-htp.inf` | `e19530fffe17a934f0bd5f35a9ef78935305c2349229b9b3552cddf68e42230d` |

## Open

- A memory budget per HTP version (3000 MiB is v73's).
- Moving llama.cpp past b11644 waits for a signed HTP set built from the new commit; the
  `x/hexagon` test fails until `Pinned` names one.
