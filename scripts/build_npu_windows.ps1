[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$LLVMMinGWRoot,
    [Parameter(Mandatory = $true)][string]$WindowsSDKInclude,
    # Optional: also build llama.cpp's Hexagon NPU backend with this Hexagon SDK and
    # Visual Studio's ARM64 clang (default: the newest Visual Studio with ARM64 tools).
    [string]$HexagonSDKRoot,
    [string]$VisualStudioRoot,
    [string]$CMake = "cmake",
    [string]$Ninja = "ninja",
    [string]$BuildDirectory = "build\npu-arm64",
    [string]$OutputDirectory = "dist",
    [ValidatePattern('^\d+\.\d+\.\d+-npu\.[A-Za-z0-9.-]+$')][string]$Version,
    [ValidateRange(1, 64)][int]$Jobs = 2,
    [switch]$AllowDirty
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-Checked {
    param([string]$Tool, [string[]]$Arguments)
    # Native stderr can contain successful compiler warnings in Windows PowerShell.
    $ErrorActionPreference = "Continue"
    $global:LASTEXITCODE = $null
    & $Tool @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Tool failed with exit code $LASTEXITCODE"
    }
}

function Resolve-Tool {
    param([string]$Tool)
    return (Get-Command $Tool -CommandType Application -ErrorAction Stop).Source
}

function Write-UTF8 {
    param([string]$Path, [string]$Text)
    [IO.File]::WriteAllText($Path, $Text, [Text.UTF8Encoding]::new($false))
}

function Save-Environment {
    $snapshot = @{}
    Get-ChildItem Env: | ForEach-Object { $snapshot[$_.Name] = $_.Value }
    return $snapshot
}

function Restore-Environment {
    param($Snapshot)
    foreach ($item in @(Get-ChildItem Env:)) {
        if (-not $Snapshot.ContainsKey($item.Name)) { Remove-Item "Env:$($item.Name)" -ErrorAction SilentlyContinue }
    }
    foreach ($name in $Snapshot.Keys) { Set-Item "Env:$name" $Snapshot[$name] }
}

# The copyright, conditions and disclaimer of a BSD-licensed SDK header.
function Get-BSDNotice {
    param([string]$Path)
    $lines = @(Get-Content -LiteralPath $Path)
    $start = @($lines | Select-String -Pattern 'Copyright \(c\)' | Select-Object -First 1)
    if ($start.Count -eq 0) { throw "No copyright notice in $Path" }
    $end = @($lines | Select-String -SimpleMatch '*/' | Where-Object { $_.LineNumber -gt $start[0].LineNumber } |
        Select-Object -First 1)
    if ($end.Count -eq 0) { throw "Unterminated notice in $Path" }
    $block = @($lines[($start[0].LineNumber - 1)..($end[0].LineNumber - 2)] | ForEach-Object { $_ -replace '^\s*\*?\s?', '' })
    if (-not (($block -join "`n") -match 'Redistribution and use in source and binary forms')) {
        throw "$Path is not under a BSD licence"
    }
    return ($block -join "`n").Trim()
}

function Get-Imports {
    param([string]$Path)
    $headers = @(& $readobj --file-headers --coff-imports $Path)
    if ($LASTEXITCODE -ne 0) { throw "Cannot inspect $Path" }
    # New LLVM-MinGW runtimes use ARM64X: native ARM64 plus an ARM64EC view.
    if (-not ($headers -match 'Machine: IMAGE_FILE_MACHINE_(ARM64 \(0xAA64\)|ARM64X \(0xA64E\))')) {
        throw "Non-ARM64 binary in payload: $Path"
    }
    return @($headers | ForEach-Object {
        if ($_ -match '^\s+Name: ([A-Za-z0-9._-]+\.dll)\s*$') { $Matches[1].ToLowerInvariant() }
    } | Sort-Object -Unique)
}

$source = Split-Path -Parent $PSScriptRoot
$originalEnvironment = @{}
foreach ($name in @("PATH", "GOOS", "GOARCH", "CGO_ENABLED", "CC", "CXX", "CGO_CFLAGS",
    "CGO_CXXFLAGS", "CGO_CPPFLAGS", "CGO_LDFLAGS", "CFLAGS", "CXXFLAGS", "CPPFLAGS",
    "LDFLAGS", "GOWORK", "GOFLAGS", "CMAKE_GENERATOR", "CMAKE_GENERATOR_PLATFORM",
    "CMAKE_GENERATOR_TOOLSET", "OLLAMA_LLAMA_CPP_SOURCE", "OLLAMA_HEXAGON_HTP_DIR",
    "HEXAGON_SDK_ROOT", "HEXAGON_TOOLS_ROOT")) {
    $originalEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
}
Push-Location $source
try {
    if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) {
        throw "This packaging entrypoint requires Windows"
    }
    $go = Resolve-Tool "go"
    $cmake = Resolve-Tool $CMake
    $ninja = Resolve-Tool $Ninja
    $llvm = (Resolve-Path -LiteralPath $LLVMMinGWRoot).Path
    $sdk = (Resolve-Path -LiteralPath $WindowsSDKInclude).Path
    $bin = Join-Path $llvm "bin"
    $cc = Join-Path $bin "aarch64-w64-mingw32-gcc.exe"
    $cxx = Join-Path $bin "aarch64-w64-mingw32-g++.exe"
    $readobj = Join-Path $bin "llvm-readobj.exe"
    foreach ($path in @($cc, $cxx, $readobj, (Join-Path $llvm "LICENSE.TXT"),
        (Join-Path $sdk "um\dxcore_interface.h"), (Join-Path $sdk "um\DirectML.h"),
        (Join-Path $sdk "shared\sdkddkver.h"))) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Missing build prerequisite: $path"
        }
    }
    # No developer source overrides or compiler wrappers may enter this package.
    $env:PATH = "$bin;$(Split-Path -Parent $ninja);$env:PATH"
    $env:GOOS = "windows"
    $env:GOARCH = "arm64"
    $env:CGO_ENABLED = "1"
    $env:CC = "`"$cc`""
    $env:CXX = "`"$cxx`""
    $env:CGO_CFLAGS = "-O2"
    $env:CGO_CXXFLAGS = "-O2 -idirafter `"$sdk\um`" -idirafter `"$sdk\shared`""
    $env:CGO_LDFLAGS = ""
    $env:CGO_CPPFLAGS = ""
    $env:GOWORK = "off"
    $env:GOFLAGS = "-mod=readonly"
    $env:CMAKE_GENERATOR = "Ninja"
    foreach ($name in @("CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS",
        "CMAKE_GENERATOR_PLATFORM", "CMAKE_GENERATOR_TOOLSET", "OLLAMA_LLAMA_CPP_SOURCE",
        "OLLAMA_HEXAGON_HTP_DIR", "HEXAGON_SDK_ROOT", "HEXAGON_TOOLS_ROOT")) {
        [Environment]::SetEnvironmentVariable($name, $null, "Process")
    }

    # The Hexagon backend's prerequisites are checked before anything is built.
    $hexagon = $null
    if ($HexagonSDKRoot) {
        $hexagonSdk = (Resolve-Path -LiteralPath $HexagonSDKRoot).Path
        $sdkInfo = Get-Content -LiteralPath (Join-Path $hexagonSdk "hexagon_sdk.json") -Raw | ConvertFrom-Json
        $toolsInfo = @($sdkInfo.root.tools.info | Where-Object { $_.name -eq "Hexagon Tools" })
        if ($toolsInfo.Count -ne 1) { throw "hexagon_sdk.json in $hexagonSdk does not name its Hexagon Tools" }
        $hexagonTools = Join-Path $hexagonSdk $toolsInfo[0].path
        $vsInstaller = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer"
        if (-not $VisualStudioRoot) {
            $vswhere = Join-Path $vsInstaller "vswhere.exe"
            if (-not (Test-Path -LiteralPath $vswhere -PathType Leaf)) { throw "Missing build prerequisite: $vswhere (or pass -VisualStudioRoot)" }
            $VisualStudioRoot = @(& $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.ARM64 -property installationPath)[0]
            if (-not $VisualStudioRoot) { throw "No Visual Studio with the ARM64 C++ tools was found" }
        }
        $vs = (Resolve-Path -LiteralPath $VisualStudioRoot).Path
        $vsClang = Join-Path $vs "VC\Tools\Llvm\ARM64\bin\clang.exe"
        $vsClangxx = Join-Path $vs "VC\Tools\Llvm\ARM64\bin\clang++.exe"
        $devShell = Join-Path $vs "Common7\Tools\Microsoft.VisualStudio.DevShell.dll"
        # SDK headers compiled into ggml-hexagon.dll under a BSD licence; their notices ship with it.
        $bsdHeaders = @("incs\rpcmem.h", "incs\stddef\AEEStdErr.h", "incs\stddef\AEEStdDef.h")
        foreach ($path in @($vsClang, $vsClangxx, $devShell) + @($bsdHeaders | ForEach-Object { Join-Path $hexagonSdk $_ })) {
            if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
                throw "Missing build prerequisite: $path"
            }
        }
        if (-not (Test-Path -LiteralPath $hexagonTools -PathType Container)) { throw "Missing build prerequisite: $hexagonTools" }
        $hexagon = [ordered]@{
            backend = "lib/ollama/hexagon/ggml-hexagon.dll"
            hexagon_sdk = $sdkInfo.root.id.version
            hexagon_tools = $toolsInfo[0].build
            compiler = $null
            signed_htp_libraries = "not included; 'ollama npu setup --provider hexagon' downloads Qualcomm's set"
        }
    }

    $revision = @(& git rev-parse HEAD)
    if ($LASTEXITCODE -ne 0) { throw "Source must be a Git checkout" }
    $revision = $revision[0].Trim()
    $status = @(& git status --porcelain --untracked-files=normal)
    if ($LASTEXITCODE -ne 0) { throw "Cannot inspect source worktree" }
    $dirty = $status.Count -gt 0
    if ($dirty -and -not $AllowDirty) { throw "Refusing to package a dirty checkout; commit changes or use -AllowDirty for development only" }
    if (-not $Version) { $Version = "0.0.0-npu.dev.$($revision.Substring(0, 12))" }
    if ($dirty) { $Version += ".dirty" }
    $name = "ollama-npu-windows-arm64-$Version"
    $build = [IO.Path]::GetFullPath($BuildDirectory)
    $hexagonBuild = "$build-hexagon"
    $output = [IO.Path]::GetFullPath($OutputDirectory)
    foreach ($directory in @($build) + @(if ($hexagon) { $hexagonBuild })) {
        if ((Test-Path -LiteralPath $directory) -and
            (-not (Test-Path -LiteralPath $directory -PathType Container) -or
            @(Get-ChildItem -LiteralPath $directory -Force).Count -gt 0)) {
            throw "$directory must be new or empty; cached compiler/source overrides are not permitted"
        }
    }
    $stage = Join-Path $output $name
    $archive = Join-Path $output "$name.zip"
    foreach ($path in @($stage, $archive, "$archive.sha256")) {
        if (Test-Path -LiteralPath $path) { throw "Refusing to overwrite package output: $path" }
    }
    New-Item -ItemType Directory -Path $build, $stage -Force | Out-Null

    Invoke-Checked $cmake @("-S", "llama\server", "--preset", "cpu_arm64", "-B", $build,
        "-DCMAKE_C_COMPILER=$cc", "-DCMAKE_CXX_COMPILER=$cxx", "-DCMAKE_MAKE_PROGRAM=$ninja",
        "--install-prefix", $stage)
    # llama-perplexity lets converters measure a GGUF's quality with the packaged runtime.
    Invoke-Checked $cmake @("--build", $build, "--target", "llama-server", "llama-quantize", "llama-perplexity",
        "--parallel", "$Jobs")
    Invoke-Checked $cmake @("--install", $build, "--component", "llama-server")
    if ($hexagon) {
        # llama.cpp's Windows Hexagon backend builds with clang targeting the MSVC ABI, like
        # upstream's MSVC-built Windows ARM64 CUDA backend beside the LLVM-MinGW CPU payload;
        # it reaches ggml-base only through its C API. Qualcomm's signed NPU libraries are
        # not packaged, because their terms restrict use: `ollama npu setup --provider
        # hexagon` downloads them.
        $saved = Save-Environment
        try {
            # Visual Studio's own environment scripts call vswhere.exe through PATH.
            if (Test-Path -LiteralPath $vsInstaller -PathType Container) { $env:PATH = "$vsInstaller;$env:PATH" }
            Import-Module $devShell
            Enter-VsDevShell -VsInstallPath $vs -SkipAutomaticLocation -DevCmdArguments "-arch=arm64 -host_arch=arm64 -no_logo" | Out-Null
            $env:PATH = (@($env:PATH -split ';') | Where-Object { $_ -and $_ -ne $bin }) -join ';'
            foreach ($name in @("CC", "CXX", "OLLAMA_HEXAGON_HTP_DIR")) { Remove-Item "Env:$name" -ErrorAction SilentlyContinue }
            $env:HEXAGON_SDK_ROOT = $hexagonSdk
            $env:HEXAGON_TOOLS_ROOT = $hexagonTools
            Invoke-Checked $cmake @("-S", "llama\server", "--preset", "hexagon_arm64", "-B", $hexagonBuild,
                "-DCMAKE_C_COMPILER=$vsClang", "-DCMAKE_CXX_COMPILER=$vsClangxx", "-DCMAKE_MAKE_PROGRAM=$ninja",
                "--install-prefix", $stage)
            Invoke-Checked $cmake @("--build", $hexagonBuild, "--target", "ggml-hexagon", "--parallel", "$Jobs")
            Invoke-Checked $cmake @("--install", $hexagonBuild, "--component", "llama-server")
            $hexagon.compiler = @(& $vsClang --version)[0]
        } finally {
            Restore-Environment $saved
        }
        # The ARM64 Visual C++ redist also ships vcruntime140_1.dll as ARM64EC, for emulated
        # x64 code. Native ARM64 binaries never import it and llama-server cannot load it;
        # the import audit below still fails the package if anything needs it.
        foreach ($file in @(Get-ChildItem -LiteralPath (Join-Path $stage "lib\ollama\hexagon") -File -Filter "*.dll")) {
            # The file header's machine: ARM64X files also list their ARM64EC view further down.
            $machine = @(& $readobj --file-headers $file.FullName | Select-String -Pattern '^\s*Machine:' | Select-Object -First 1)
            if ($machine.Count -eq 1 -and $machine[0].Line -match 'IMAGE_FILE_MACHINE_ARM64EC') {
                Remove-Item -LiteralPath $file.FullName
            }
        }
        if (@(Get-ChildItem -LiteralPath $stage -Recurse -File -Filter "libggml-htp*").Count -gt 0) {
            throw "Qualcomm's signed HTP libraries must not be packaged; ollama npu setup --provider hexagon downloads them"
        }
    }
    Invoke-Checked $go @("build", "-trimpath", "-ldflags",
        "-s -w -X=github.com/ollama/ollama/version.Version=$Version -X=github.com/ollama/ollama/server.mode=release",
        "-o", (Join-Path $stage "ollama.exe"), ".")
    Invoke-Checked $cmake @("-DGO_EXECUTABLE=$go", "-DSOURCE_DIR=$source", "-DBINARY_DIR=$build",
        "-DOUTPUT_DIR=$stage\lib\ollama", "-DTARGETS=windows/arm64", "-DPACKAGES=.",
        "-P", (Join-Path $source "cmake\generate_go_license.cmake"))

    # The CGo diagnostics link C++ runtime DLLs; they must also be peers of the CLI.
    $targetRuntime = Join-Path $llvm "aarch64-w64-mingw32\bin"
    foreach ($dll in @("libc++.dll", "libunwind.dll", "libwinpthread-1.dll")) {
        Copy-Item -LiteralPath (Join-Path $targetRuntime $dll) -Destination $stage
    }
    New-Item -ItemType Directory -Path (Join-Path $stage "licenses"), (Join-Path $stage "docs"),
        (Join-Path $stage "integration") | Out-Null
    Copy-Item -LiteralPath (Join-Path $llvm "LICENSE.TXT") -Destination (Join-Path $stage "licenses\LLVM-MinGW.txt")
    $runtimeNotices = Join-Path $llvm "aarch64-w64-mingw32\share\mingw32"
    if (-not (Test-Path -LiteralPath $runtimeNotices -PathType Container)) {
        throw "LLVM-MinGW runtime license inventory is missing: $runtimeNotices"
    }
    New-Item -ItemType Directory -Path (Join-Path $stage "licenses\mingw-runtime") | Out-Null
    foreach ($notice in @(Get-ChildItem -LiteralPath $runtimeNotices -File -Filter "COPYING*")) {
        Copy-Item -LiteralPath $notice.FullName -Destination (Join-Path $stage "licenses\mingw-runtime")
    }
    if (-not (Test-Path -LiteralPath (Join-Path $stage "licenses\mingw-runtime\COPYING.MinGW-w64-runtime.txt"))) {
        throw "Missing MinGW runtime redistribution notice"
    }
    Copy-Item -LiteralPath (Join-Path $source "LICENSE") -Destination $stage
    foreach ($doc in @("npu_qnn.md", "development.md") + @(if ($hexagon) { "npu_hexagon.md" })) {
        Copy-Item -LiteralPath (Join-Path $source "docs\$doc") -Destination (Join-Path $stage "docs")
    }
    if ($hexagon) {
        $notices = @($bsdHeaders | ForEach-Object {
            "== Hexagon SDK $($hexagon.hexagon_sdk), $_ (compiled into lib\ollama\hexagon\ggml-hexagon.dll) ==`n`n" +
            (Get-BSDNotice (Join-Path $hexagonSdk $_)) + "`n"
        })
        Write-UTF8 (Join-Path $stage "licenses\Hexagon-SDK-headers.txt") (($notices -join "`n") + "`n")
    }
    Copy-Item -LiteralPath (Join-Path $source "integration\README.md") -Destination (Join-Path $stage "integration")

    $systemDLLs = @("advapi32.dll", "bcrypt.dll", "cfgmgr32.dll", "combase.dll",
        "comctl32.dll", "comdlg32.dll", "crypt32.dll", "d3d12.dll", "dbghelp.dll",
        "dxcore.dll", "dxgi.dll", "gdi32.dll", "gdi32full.dll", "imm32.dll",
        "iphlpapi.dll", "kernel32.dll", "mpr.dll", "msasn1.dll", "msvcrt.dll",
        "ncrypt.dll", "normaliz.dll", "ntdll.dll", "ole32.dll", "oleaut32.dll",
        "powrprof.dll", "propsys.dll", "rpcrt4.dll", "sechost.dll", "secur32.dll",
        "setupapi.dll", "shell32.dll", "shlwapi.dll", "ucrtbase.dll", "user32.dll",
        "userenv.dll", "version.dll", "winhttp.dll", "winmm.dll", "ws2_32.dll")
    $dependencies = @()
    $libOllama = Join-Path $stage "lib\ollama"
    foreach ($binary in @(Get-ChildItem -LiteralPath $stage -Recurse -File | Where-Object { $_.Extension -in @(".dll", ".exe") })) {
        $imports = @(Get-Imports $binary.FullName)
        $peers = @($binary.DirectoryName)
        # Backends in lib\ollama\<runner> load into llama-server, whose directory runners put first on PATH.
        if ((Split-Path -Parent $binary.DirectoryName) -eq $libOllama) { $peers += $libOllama }
        foreach ($dll in $imports) {
            # Resolve only the loader's peer directories and explicitly OS-owned APIs, not PATH.
            if (@($peers | Where-Object { Test-Path -LiteralPath (Join-Path $_ $dll) -PathType Leaf }).Count -gt 0) { continue }
            if ($systemDLLs -contains $dll -or $dll -like "api-ms-win-*.dll" -or $dll -like "ext-ms-*.dll") { continue }
            throw "Unbundled dependency: $($binary.Name) -> $dll"
        }
        $dependencies += [ordered]@{ file = $binary.FullName.Substring($stage.Length + 1); imports = $imports }
    }
    foreach ($required in @("ollama.exe", "lib\ollama\llama-server.exe", "lib\ollama\llama-quantize.exe",
        "lib\ollama\llama-perplexity.exe", "lib\ollama\ggml-cpu.dll", "lib\ollama\GO_LICENSE",
        "lib\ollama\LLAMA_CPP_VENDORS_LICENSE") + @(if ($hexagon) {
            "lib\ollama\hexagon\ggml-hexagon.dll", "licenses\Hexagon-SDK-headers.txt", "docs\npu_hexagon.md" })) {
        if (-not (Test-Path -LiteralPath (Join-Path $stage $required) -PathType Leaf)) { throw "Missing package file: $required" }
    }
    $compiler = @(& $cc --version)
    if ($LASTEXITCODE -ne 0) { throw "Cannot record compiler identity" }
    $goVersion = & $go version
    if ($LASTEXITCODE -ne 0) { throw "Cannot record Go identity" }
    $cmakeVersion = @(& $cmake --version)
    if ($LASTEXITCODE -ne 0) { throw "Cannot record CMake identity" }
    $finalRevision = @(& git rev-parse HEAD)
    if ($LASTEXITCODE -ne 0 -or $finalRevision[0].Trim() -ne $revision) {
        throw "Source revision changed during packaging"
    }
    $finalStatus = @(& git status --porcelain --untracked-files=normal)
    if ($LASTEXITCODE -ne 0 -or (-not $dirty -and $finalStatus.Count -gt 0)) {
        throw "Source worktree changed during packaging"
    }
    $metadata = [ordered]@{
        schema_version = 1
        product = "ollama-npu"
        version = $Version
        source_commit = $revision
        source_dirty = $dirty
        platform = "windows/arm64"
        go = $goVersion
        compiler = $compiler[0]
        cmake = $cmakeVersion[0]
        windows_sdk = Split-Path -Leaf $sdk
        llama_cpp_pin = (Get-Content -LiteralPath (Join-Path $source "LLAMA_CPP_VERSION") -Raw).Trim()
        native_profile = "cpu_arm64; GGML_NATIVE=OFF; GGML_CPU_ALL_VARIANTS=OFF"
        hexagon = $hexagon
        package_scope = "unsigned portable CLI/server; no desktop app, model, ORT/QNN runtime, Qualcomm signed HTP libraries or OEM driver"
        hardware_qualified = $false
        dependencies = $dependencies
    }
    Write-UTF8 (Join-Path $stage "build-info.json") (($metadata | ConvertTo-Json -Depth 8) + "`n")
    $inventory = @(Get-ChildItem -LiteralPath $stage -Recurse -File | Sort-Object FullName | ForEach-Object {
        [ordered]@{
            path = $_.FullName.Substring($stage.Length + 1)
            size = $_.Length
            sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    })
    Write-UTF8 (Join-Path $stage "files.json") (([ordered]@{schema_version = 1; files = $inventory} | ConvertTo-Json -Depth 5) + "`n")
    Compress-Archive -Path (Join-Path $stage "*") -DestinationPath $archive -CompressionLevel Optimal
    $hash = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
    Write-UTF8 "$archive.sha256" "$hash  $name.zip`n"
    Write-Output "Package: $archive"
    Write-Output "SHA256: $hash"
    Write-Output "Unsigned development package; qualify on hardware before release. ORT/QNN acquired separately with npu setup."
    if ($hexagon) {
        Write-Output "Hexagon NPU backend included (opt-in: OLLAMA_HEXAGON=1); Qualcomm's signed NPU libraries come from 'ollama npu setup --provider hexagon --accept-qualcomm-terms'."
    }
} finally {
    Pop-Location
    foreach ($name in $originalEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($name, $originalEnvironment[$name], "Process")
    }
}
