# GGUF on the Snapdragon NPU (Hexagon backend, experimental)

This branch runs GGUF models on the Hexagon NPU of Snapdragon X laptops through the
ordinary llama-server runner, using llama.cpp's Hexagon backend. It is experimental and
off by default. It works on a Snapdragon X Elite (X1E78100) from a local build (below) or
a portable package (`scripts\build_npu_windows.ps1 -HexagonSDKRoot`).

## Enabling it

Get the portable package from the fork's
[releases](https://github.com/dsmith111/ollama-npu/releases/latest) and extract it into a new
folder. Then get Qualcomm's signed NPU libraries once, and start the server with
`OLLAMA_HEXAGON=1`. Upstream first shipped Vulkan the same way, as an opt-in setting (#12931):

```powershell
.\ollama.exe npu setup --provider hexagon --accept-qualcomm-terms
$env:OLLAMA_HEXAGON = "1"
.\ollama.exe serve
```

In cmd.exe the switch is `set OLLAMA_HEXAGON=1`. Quit the stock Ollama app first if it is
running: it holds port 11434 and has no NPU backend. The fork's client warns when the server
it reaches is not the fork, and when no server answers it says to start one instead of
launching the stock app as upstream's client does.

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

## Models from ollama.com

Nothing needs converting for the models ollama.com already serves. With the switch on, the
fork pulls them as usual and loads them whole on the NPU, and `ollama ps` shows "100% NPU".
Measured 2026-10-20 on the X1E78100 with package `cd2eb656994e`, from an isolated model
store, a 120-token answer at temperature 0:

| Tag | Type | `ollama ps` | Decode |
|---|---|---|---|
| `qwen2.5:1.5b` (default tag) | Q4_K_M | 1.3 GB, 100% NPU | 29.6 tok/s |
| `qwen2.5:1.5b-instruct-q4_0` | Q4_0 | 1.1 GB, 100% NPU | not timed |
| `llama3.2:3b` (default tag) | Q4_K_M | 2.8 GB, 100% NPU | 15.5 tok/s |
| `llama3.2:3b-instruct-q4_0` | Q4_0 | 2.5 GB, 100% NPU | 20.6 tok/s |
| `phi4-mini` (default tag) | Q4_K_M | 3.3 GB, 100% NPU | 12.8 tok/s |

All answered a two-turn capital-cities chat correctly. The backend runs Q4_0, Q8_0, Q4_K,
Q5_K and Q6_K matmuls on HTP, so the default tags (Q4_K_M) work as well as the `-q4_0` ones.
Prefer a `-q4_0` tag when the library has one: Q4_0 is the backend's best-supported type and
decoded about a third faster here, and phi4-mini's default tag processed its short prompt at
79 tok/s against 266-477 for the others. Models up to about 5 GiB load whole (see "Memory
budget"); larger ones are split with the CPU, which is slower. Converting a Hugging Face
model yourself is for models the library does not have, or for a Q4_0 of one that only has
K-quant tags: llama.cpp's `convert_hf_to_gguf.py` and `llama-quantize` produce a GGUF that
`ollama create` imports, and a Llama-3.2-3B converted that way decoded at 21.1 tok/s, the
same as the library's `-q4_0` tag.

Models converted this way and uploaded to ollama.com pull the same way. The first one,
`dsmith111/qwen2.5-0.5b-instruct:npu` (Qwen2.5-0.5B at Q4_0 with its checked chat template),
pulled into a fresh store in 9 s and ran whole on the NPU: 80.8 tok/s decode, 0.16 cores busy
while generating.

```
ollama run dsmith111/qwen2.5-0.5b-instruct:npu
```

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
cmake --build build\cpu --target llama-server llama-quantize llama-perplexity llama-imatrix
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
- `GGML_HEXAGON_MBUF=,,6144M` unless set: the backend otherwise reports 0 MiB, and
  discovery drops zero-memory devices. The total is only reported to the schedulers, not
  enforced. llama-server's fit keeps 1 GiB of it free, so models up to 5 GiB on the NPU
  (Qwen2.5-7B at Q4_0 with up to a 16K context) load whole, and larger ones are split with
  the CPU (see the budget measurements below).

Flash attention is allowed on Hexagon (`--flash-attn auto`); turning it off halves NPU
prefill.

When every selected device is a Hexagon NPU and the model is expected to fit whole (weights
and KV cache plus llama-server's 1 GiB fit margin), the runner also passes `--poll 0`, so
llama-server's CPU threads sleep instead of polling between ops. A model split with the CPU
keeps the default polling: its CPU layers decoded 28% slower without it.

`ollama ps` names the NPU: `100% NPU` for a model whole on it, `25%/75% CPU/NPU` for a split.
`/api/ps` adds `"accelerator": "NPU"` when every device a runner uses is a Hexagon NPU, and
`size_vram` is then the NPU's share. QNN bundles report it too: the ORT runner sizes the
model from its graphs, data and context binaries, and the QNN stages' share is the NPU's
(Qwen2.5-0.5B w8a16: 983 MB, `24%/76% CPU/NPU`, its embeddings and LM head on the CPU).

## Measured (X1E78100, Qwen2.5-1.5B-Instruct)

Through `/api/generate` (556-token prompt, 128 tokens):

| Model | Prefill | Decode |
|---|---|---|
| GGUF Q4_0 on the NPU | 1,229 tok/s | 28.7 tok/s |
| GGUF Q4_0 on the CPU (`num_gpu 0`) | 834 tok/s | 55.3 tok/s |
| QNN w8a16 ORT bundle (NPU) | 772 tok/s | 20.2 tok/s |

The NPU's main gain is leaving the CPU to the user: an agent running on it barely competes
with your own work, where a model on the CPU alone can saturate a weaker machine and slow
you and the agent alike. Power matters mostly for chat use. The CPU decodes faster. The
same model (540-token prompt, 256 tokens, two requests; through the fork, except the
default-polling row, which ran llama-server directly with the fork's arguments):

| Where it runs | llama-server CPU use | Decode | Prefill |
|---|---|---|---|
| NPU, with `--poll 0` | 0.10 cores | 34.0 tok/s | 1,315 tok/s |
| NPU, default polling | 1.42 cores | 30.2 tok/s | 1,223 tok/s |
| CPU (`num_gpu 0`) | 5.32 cores | 73.2 tok/s | 567 tok/s |

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

The portable package (LLVM-MinGW CPU payload plus the MSVC-ABI Hexagon backend), checked
from its extracted ZIP with the set from setup:

- Qwen2.5-1.5B Q4_0 loaded all 29 layers on HTP0 (898 MiB) and decoded 24 tokens at
  about 39 tok/s.
- Qwen2.5-0.5B converted with llama.cpp, quantized with the package's own `llama-quantize`
  and checked with its `llama-server` and `llama-perplexity`: 25/25 layers on HTP0,
  perplexity 1.16x the 16-bit model, long-prompt retrieval correct at 875 and 1,763
  tokens, and the chat template matching transformers'.
- Nine models from 0.36B to 3.8B parameters (Qwen2.5, Qwen3, SmolLM2, TinyLlama,
  DeepSeek-R1-Distill-Qwen, Phi-3.5-mini, Phi-4-mini) converted the same way were
  installed and chatted with through the package.

### CPU backend

The package's CPU backend comes in variants picked at load from the features Windows
reports (`llama/compat/002-windows-arm64-cpu-variants.patch`); the X1E78100 gets dotprod
and i8mm. The previous generic ARMv8 build, measured against it through `/api/generate`
(a prompt of about 540 tokens, 128 tokens, median of three):

| Model (Q4_0, `num_gpu 0`) | Generic ARMv8 | Variants (dotprod, i8mm) |
|---|---|---|
| Qwen2.5-0.5B-Instruct | 377 / 146 tok/s | 1,483 / 157 tok/s |
| Qwen2.5-1.5B-Instruct | 85 / 38 tok/s | 559 / 68 tok/s |
| Qwen3-1.7B | 79 / 40 tok/s | 443 / 50 tok/s |

(prefill / decode). The variants leave out fp16 vector arithmetic: with it, ggml
accumulates F16 dot products in fp16, and DeepSeek-R1-Distill-Qwen-1.5B decoded garbage
on the CPU ("Okay,8888..."), while dotprod and i8mm alone were as fast.

### Memory budget

`GGML_HEXAGON_MBUF`'s total decides, through llama-server's fit (which keeps 1 GiB free),
how much of a model the fork places on the NPU; a model that does not fit is split with
the CPU, which was slower than running whole on the NPU. Through `/api/generate`
(prefill / decode, first token):

| Model, size at 4K context | 3000 MiB (an earlier default) | 5120 MiB or more |
|---|---|---|
| Phi-4-mini-instruct Q4_0, 2.8 GiB | split 20/33 layers: 384 / 9.0 tok/s, 1.42 s | whole: 713 / 12.8 tok/s, 0.78 s |
| Phi-3.5-mini-instruct Q4_0, 3.6 GiB | split: 279 / 8.9 tok/s, 2.26 s | whole: 491 / 9.7 tok/s, 1.32 s |

With an 8K context and a 6000 MiB budget, Phi-3.5-mini put 4.7 GiB on the NPU, answered a
618-token retrieval prompt correctly at 7.5 tok/s, and decoded at 2.3 tok/s after a
5,176-token prompt. Models up to Qwen3-1.7B (1.5 GiB) loaded whole under every budget
tried.

How much fits whole: a session addresses about 3 GiB at a time (3000 MiB on v73, 3200 MiB
on later versions, the backend's defaults) and maps weights in and out beyond that, but
the backend keeps every buffer it has used mapped through FastRPC, so a model's whole NPU
footprint must map at once. Qwen2.5-7B-Instruct on the 16 GB X1E78100, through
llama-server directly with the fork's arguments (540-token prompt, 256 tokens, two
requests):

| Qwen2.5-7B placement | On the NPU | llama-server CPU use | Decode | Prefill |
|---|---|---|---|---|
| Q4_0, one session, 4K context | 4.2 GiB | 0.03 cores | 9.8 tok/s | 376 tok/s |
| Q4_0, one session, 16K context | 4.9 GiB | 0.02 cores | 9.8 tok/s | 376 tok/s |
| Q4_0, one session, 32K context | 5.8 GiB | 0.01 cores | 9.8 tok/s | 378 tok/s |
| Q4_0, 4K, split 27/29 with the CPU (5120 MiB) | 3.9 GiB | 1.42 cores | 6.5 tok/s | 294 tok/s |
| Q4_0, 4K, two sessions (`GGML_HEXAGON_DEVICES=2`) | 4.2 GiB | 0.01 cores | 1.0 tok/s | 160 tok/s |
| Q4_0, 4K, on the CPU | none | 5.81 cores | 20.6 tok/s | 124 tok/s |
| Q8_0, one session, 4K context | 7.4 GiB | did not load (below) | | |

The Q8_0 loaded, then failed to map a buffer on its first run, once with 7.2 GiB already
mapped (3.7 GB of memory was available before the load) and once with 6.2 GiB. The limit
is probably the memory Windows can lock for the NPU, so it would vary with the machine and
its load; 6144 MiB keeps whole models at 5 GiB or less, below every failure.

One of about 19 two-turn chats with Phi-4-mini, split 20/33 under the 3000 MiB budget,
once returned unrelated words for the second turn. It did not recur in 100 more on
2026-10-20, each after a fresh load (50 whole on the NPU, 50 split 20/33): every one
answered "Paris" and "Berlin".

### CPU contention

The point of the NPU on a laptop: a model generating there leaves the CPU to you. Measured
2026-10-21 with `scripts/npu_contention.py` against the fork (package `73264854ae48`; on
Windows the runner starts at above-normal priority, as upstream's does). An agent-style loop
of sequential `/api/chat` requests (a 700-token system prompt with tool definitions, coding
tasks in rotation, 200 new tokens each, temperature 0) runs for two minutes while two probes
stand in for your own work: a fixed single-thread unit every 0.5 s (interactive latency) and
a fixed all-cores unit back to back (a build or a test run). Same machine state as a working
day: browsers and an editor open, 2 GB of RAM free.

| Model | Where | Decode | Runner cores | Interactive p50 / p95 | All-cores unit |
|---|---|---|---|---|---|
| none (baseline) | | | 0 | 296 / 368 ms | 0.62 s |
| Llama-3.1-8B Q4_0 (converted with llama.cpp) | NPU | 9.0 tok/s | 0.02 | 297 / 379 ms | 0.62 s |
| Llama-3.1-8B Q4_0 (converted with llama.cpp) | CPU | 10.2 tok/s | 5.76 | 449 / 1067 ms | 1.67 s |
| `llama3.2:3b-instruct-q4_0` | NPU | 18.4 tok/s | 0.03 | 299 / 363 ms | 0.63 s |
| `llama3.2:3b-instruct-q4_0` | CPU | 21.3 tok/s | 5.47 | 596 / 1783 ms | 1.65 s |

On the NPU the probes match the baseline and the model keeps its idle-machine speed (8.9 and
19.0 tok/s measured alone). On the CPU, interactive work takes 1.5-2x longer at the median
and 3-5x at the 95th percentile, the all-cores unit takes 2.7x longer, 5.5-5.8 cores are busy
in the runner, and the model itself slows too (17.3 and 33.7 tok/s alone, 10.2 and 21.3
beside the probes): both sides lose.

### Context shift

llama-server's context shift drops the oldest tokens when a conversation outgrows `num_ctx` and
re-applies RoPE to the cached keys. The Hexagon backend cannot run that operation on a buffer that
lives on the NPU, and ggml aborts the runner (`pre-allocated tensor (cache_k_l0 (view)) in a buffer
(HTP0) that cannot run the operation (ROPE)`), so the request failed with HTTP 500. Found 2026-10-21
with a 0.23B model that never ends its answers: it filled its 2048-token context in a minute. With
layers on the NPU the fork now starts the runner without context shift: an answer that reaches the
limit ends with `done_reason: "length"`, and a prompt that does not fit is refused with a clear
message instead of being shifted. Raise `num_ctx` for long conversations.

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

- The memory budget was measured on HTP v73 with 16 GB only. What limits a session's
  footprint (failures with 6.2-7.2 GiB mapped) is not confirmed; machines with more memory
  may load larger models whole with a larger `GGML_HEXAGON_MBUF`, and v81 maps weights
  with 64-bit DMA. Neither has been checked.
- CPU prefill: the spike's single CPU backend built with Visual Studio's clang (MSVC ABI,
  armv8.7 with fp16) prefilled about 1.5x faster than the package's LLVM-MinGW variants
  (Qwen2.5-1.5B: 828 vs 556 tok/s) and decoded 10-15% slower. fp16 is not the cause; the
  toolchain or ABI probably is. Building the CPU variants in Visual Studio's environment, as
  the Hexagon backend is, has not been tried.
- Moving llama.cpp past b11644 waits for a signed HTP set built from the new commit; the
  `x/hexagon` test fails until `Pinned` names one. GenieX v0.9.3 (2026-10-19) re-signed the
  set but still built it from `e467e5b`, so the pin stays at v0.9.1 (checked 2026-10-21).
