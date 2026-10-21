# Development

Install prerequisites:

- [Go](https://go.dev/doc/install)
- [CMake](https://cmake.org/download/) 3.24 or newer
- C/C++ compiler: Clang on macOS, Visual Studio 2022 C++ tools on Windows, or GCC/Clang on Linux
- [Ninja](https://github.com/ninja-build/ninja/releases) in `PATH` is recommended, especially on Windows

For pure Go iteration against an existing native payload, run Ollama from the repository root:

```shell
go run . serve
```

> [!NOTE]
> Ollama includes native code compiled with CGO.  From time to time these data structures can change and CGO can get out of sync resulting in unexpected crashes.  You can force a full build of the native code by running `go clean -cache` first. 

## Native build model

For a fresh checkout, or after changing native code, build from the repository root. On macOS arm64, this builds Metal inference. On all other platforms this builds CPU-only inference. It builds the Go binary at the repository root and installs the native runtime payload under `build/lib/ollama`.

```shell
cmake -B build .
cmake --build build --parallel 8
./ollama serve
```

To install into a standard prefix layout:

```shell
cmake --install build --prefix /path/to/install
```

On all platforms except macOS arm64, to build GPU backends select the backends explicitly:

```shell
cmake -B build . -DOLLAMA_LLAMA_BACKENDS="cuda_v13;vulkan"
cmake --build build --parallel 8
```

Supported backend values are `cuda_v12`, `cuda_v13`, `rocm_v7_1`, `rocm_v7_2`, `vulkan`, `cuda_jetpack5`, and `cuda_jetpack6`.

Use standard CMake architecture overrides to narrow GPU builds for local hardware:

```shell
# CUDA
cmake -B build . -DOLLAMA_LLAMA_BACKENDS=cuda_v13 -DCMAKE_CUDA_ARCHITECTURES=native

# ROCm / HIP
cmake -B build . -DOLLAMA_LLAMA_BACKENDS=rocm_v7_2 -DCMAKE_HIP_ARCHITECTURES=gfx1100
```

You can tune GGML build options by setting `GGML_*` values during configure. For example, to disable CUDA flash attention kernels for local debugging:

```shell
cmake -B build . -DOLLAMA_LLAMA_BACKENDS=cuda_v12 -DGGML_CUDA_FA=OFF
```

## macOS (Apple Silicon)

Additional prerequisites:

MLX Metal requires the Metal toolchain. Install [Xcode](https://developer.apple.com/xcode/) first, then:

```shell
xcodebuild -downloadComponent MetalToolchain
```

## Windows

Additional prerequisites:

- [Visual Studio 2022](https://visualstudio.microsoft.com/downloads/) including the Native Desktop Workload
- (Optional) AMD GPU support
    - [ROCm](https://rocm.docs.amd.com/en/latest/)
- (Optional) NVIDIA GPU support
    - [CUDA SDK](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_type=exe_network)
- (Optional) Vulkan GPU support
    - [Vulkan SDK](https://vulkan.lunarg.com/sdk/home) - useful for AMD/Intel GPUs
- (Optional) MLX engine support
    - [CUDA 13+ SDK](https://developer.nvidia.com/cuda-downloads)
    - [cuDNN 9+](https://developer.nvidia.com/cudnn)

For Ninja builds, run CMake from a Developer PowerShell/Command Prompt or another shell where the Visual Studio compiler is available.

> Building for Vulkan requires VULKAN_SDK environment variable:
> 
> PowerShell
> ```powershell
> $env:VULKAN_SDK="C:\VulkanSDK\<version>"
> ```
> CMD
> ```cmd
> set VULKAN_SDK=C:\VulkanSDK\<version>
> ```

## Windows (ARM)

This fork includes an experimental ORT GenAI/QNN runner for compatible models
on Windows ARM64 Snapdragon devices. See [NPU/QNN setup](npu_qnn.md) for the
runtime, model and CGo requirements. It is separate from the ordinary GGUF
CPU/GPU backends; DirectML GPU support does not imply NPU support.

For a native-only MSVC ARM64 development build, use the supported overrides:

```powershell
cmake -B build . -DGGML_NATIVE=ON -DGGML_CPU_ALL_VARIANTS=OFF
cmake --build build --config Release --parallel 2
```

This builds for the local CPU, not a portable all-CPU-variants release. A
Go-only build still needs the matching native payload in an ordinary installed
layout. A nonstandard CMake output directory is not automatically a runtime
search path.

### Portable NPU fork package

From a clean Git checkout, use the fork's CLI/server-only packaging entrypoint.
Prerequisites: native Windows ARM64, Go 1.26.0 or newer, Git, CMake 3.24 or newer,
Ninja, Windows SDK `10.0.26100.0` headers and the official
[LLVM-MinGW UCRT ARM64 toolchain](https://github.com/mstorsjo/llvm-mingw/releases/tag/20261006).
The tested `llvm-mingw-20261006-ucrt-aarch64.zip` asset has SHA256
`9a835d5179c9f3a5a783c11a7f14062a249b4765dd11add64a66786864e82ab2`.
Verify the downloaded archive before extracting or executing its tools. No local
CGo wrapper, Python, converter or prebuilt native payload is required, and Visual
Studio is needed only for the optional Hexagon backend (below). Network access is
needed for pinned source/modules/license tool dependencies on the first build.

```powershell
.\scripts\build_npu_windows.ps1 -DetectOnly   # shows the tools it found, or what to install
.\scripts\build_npu_windows.ps1               # builds the package into .\dist
```

The script finds the tools on the machine: Go on `PATH`; LLVM-MinGW as `LLVM_MINGW_ROOT`
or a `llvm-mingw-*` folder under `C:\tools`, your profile's `tools` or `Downloads`, or
`C:\`; the Windows SDK from the registry; CMake and Ninja on `PATH` or in Visual Studio;
the Hexagon SDK as `HEXAGON_SDK_ROOT`, `C:\Qualcomm\Hexagon_SDK\<version>` or a
`Hexagon_SDK` or `hx` folder in your profile; Visual Studio through vswhere. When one is
missing it stops and says what the tool is, where to get it, where to put it, and the
parameter that names it: `-LLVMMinGWRoot`, `-WindowsSDKInclude`, `-CMake`, `-Ninja`,
`-HexagonSDKRoot`, `-VisualStudioRoot`. `-BuildDirectory` and `-OutputDirectory` choose
where it builds and writes.

The Hexagon SDK (from the
[snapdragon-toolchain releases](https://github.com/snapdragon-toolchain/hexagon-sdk/releases),
`hexagon-sdk-v6.6.0.0-arm64-wos.tar.xz`) builds llama.cpp's Hexagon NPU backend, which
runs GGUF models on the Snapdragon NPU once enabled with `OLLAMA_HEXAGON=1` (see
[npu_hexagon.md](npu_hexagon.md)). That needs Visual Studio's ARM64 C++ tools and its
clang. `-SkipHexagon` packages without the backend (CPU and QNN only). The backend builds in Visual Studio's ARM64 environment with clang targeting
the MSVC ABI, as upstream builds its Windows ARM64 CUDA backend beside the LLVM-MinGW
CPU payload, and reaches `ggml-base` only through its C API. Its build directory is
`<BuildDirectory>-hexagon`. It installs `lib\ollama\hexagon\ggml-hexagon.dll` with its
Visual C++ runtime, plus `licenses\Hexagon-SDK-headers.txt` for the BSD-licensed SDK
headers compiled into it. The package never contains Qualcomm's signed NPU libraries:
`ollama npu setup --provider hexagon --accept-qualcomm-terms` downloads them.

The build directory must be new or empty; cached development builds are refused.
The script scopes/restores compiler variables, disables external Go workspaces
and the local llama.cpp source override. It builds the existing `cpu_arm64`
preset (`GGML_NATIVE=OFF`, `GGML_CPU_ALL_VARIANTS=ON`: CPU backends from plain
ARMv8.0 up to ARMv8.6 with dotprod and i8mm, chosen at load from the features
Windows reports; llama.cpp lists no Windows on Arm variants yet, so
`llama/compat/002-windows-arm64-cpu-variants.patch` adds them) and matching CLI, bundles
runtime peers/notices and audits ARM64/ARM64X binaries and imports without resolving
dependencies through developer `PATH`. Default identity:
`0.0.0-npu.dev.<source-commit>`. `-Version` supplies an explicit fork identity: releases use
`<upstream version>-npu.<n>`, for example `0.40.0-npu.1` for the first package on upstream
0.40.0, and are published on GitHub as the zip with its `.sha256` (the fork's client reads
the `-npu` marker to tell its own server from stock Ollama). Dirty checkouts fail unless
`-AllowDirty` is supplied; development
outputs then receive `.dirty`. Existing package outputs are never overwritten.
On Windows, the pinned supplier clone enables `core.longpaths` only in that owned
checkout, so nested builds can fetch its unused UI files. User/system Git settings
are not changed.

The versioned ZIP contains `ollama.exe`, its peer DLLs, `lib\ollama` native
payload/notices, setup/development documentation, `build-info.json` and
`files.json` (sizes/SHA256 for every other file). An adjacent `.zip.sha256` records
the archive hash. Compare it with `Get-FileHash -Algorithm SHA256` before extraction.
These checksums establish integrity relative to trusted metadata, not publisher
authentication. The package is **unsigned**, excludes the desktop app, models,
ORT/QNN runtime and drivers, and does not declare itself hardware-qualified.
Reproducible here means an attributable clean-source build, not byte-identical
compiler or ZIP output.
Qualification instructions are bundled in `integration\README.md`; running the
Go suite still requires the source checkout at the recorded `source_commit`.

For focused Go tests with the same standard compiler:

```powershell
$env:PATH = 'C:\tools\llvm-mingw-20261006-ucrt-aarch64\bin;' + $env:PATH
$env:CGO_ENABLED = '1'
$env:CC = 'aarch64-w64-mingw32-gcc.exe'
$env:CXX = 'aarch64-w64-mingw32-g++.exe'
$env:CGO_CXXFLAGS = '-idirafter "C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\um" -idirafter "C:\Program Files (x86)\Windows Kits\10\Include\10.0.26100.0\shared"'
Remove-Item Env:\CGO_LDFLAGS -ErrorAction SilentlyContinue
go test .\server .\parser .\cmd .\discover\dxcore .\x\ortrunner\...
go vet .\server .\parser .\cmd .\discover\dxcore .\x\ortrunner\...
```

SDK headers supply DXCore/DirectML interfaces absent from MinGW; `-idirafter`
retains the toolchain's compatible Windows headers. Hardware qualification uses
the explicit opt-in suite documented in [integration tests](../integration/README.md).
A local package build does not close fresh-user/clean-machine, signing, hosted
model distribution or multi-device release gates.
If Windows security policy blocks an unsigned executable, retain the security
event and use an administrator-approved/signing workflow or an approved test
machine. Do not disable protection, add exclusions or relocate/rename the binary
to evade the policy.

## Linux

Additional prerequisites:

- (Optional) AMD GPU support
    - [ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html)
- (Optional) NVIDIA GPU support
    - [CUDA SDK](https://developer.nvidia.com/cuda-downloads)
- (Optional) Vulkan GPU support
    - [Vulkan SDK](https://vulkan.lunarg.com/sdk/home) - useful for AMD/Intel GPUs
    - Or install via package manager: `sudo apt install vulkan-sdk` (Ubuntu/Debian) or `sudo dnf install vulkan-sdk` (Fedora/CentOS)
- (Optional) MLX engine support
    - [CUDA 13+ SDK](https://developer.nvidia.com/cuda-downloads)
    - [cuDNN 9+](https://developer.nvidia.com/cudnn)
    - OpenBLAS/LAPACK: `sudo apt install libopenblas-dev liblapack-dev liblapacke-dev` (Ubuntu/Debian)
> [!IMPORTANT]
> Ensure prerequisites are in `PATH` before running CMake.

## MLX Engine (Optional)

The MLX engine enables running safetensor based models. On macOS arm64, MLX is enabled by default. On other platforms, MLX backends are selected with `OLLAMA_MLX_BACKENDS`.

### CUDA

Requires CUDA 13+ and [cuDNN](https://developer.nvidia.com/cudnn) 9+.

```shell
cmake -B build . -DOLLAMA_MLX_BACKENDS=cuda_v13
cmake --build build --parallel 8
```

### Local MLX source overrides

To build against a local checkout of MLX and/or MLX-C (useful for development), set environment variables before running CMake:

```shell
export OLLAMA_MLX_SOURCE=/path/to/mlx
export OLLAMA_MLX_C_SOURCE=/path/to/mlx-c
```

On macOS arm64:

```shell
OLLAMA_MLX_SOURCE=../mlx OLLAMA_MLX_C_SOURCE=../mlx-c cmake -B build .
cmake --build build --parallel 8
```

For CUDA:

```powershell
$env:OLLAMA_MLX_SOURCE="../mlx"
$env:OLLAMA_MLX_C_SOURCE="../mlx-c"
cmake -B build . -DOLLAMA_MLX_BACKENDS=cuda_v13
cmake --build build --parallel 8
```

## Docker

```shell
docker build .
```

### ROCm

```shell
docker build --build-arg FLAVOR=rocm .
```

## Running tests

To run tests, use `go test`:

```shell
go test ./...
```

## Library detection

Ollama looks for native helper binaries and acceleration libraries in installed and local development layouts:

* `../lib/ollama` for standard installs where `ollama` is under `bin/`
* `./lib/ollama` for Windows release-style payloads and local dist output
* `.` for macOS release artifacts that colocate helpers with `ollama`
* `build/lib/ollama` and `dist/<platform>/lib/ollama` for local development builds

If the libraries are not found, Ollama will not run with any acceleration libraries.
