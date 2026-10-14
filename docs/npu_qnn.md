# Snapdragon NPU models in this Ollama fork

This fork adds an experimental ONNX Runtime GenAI runner for model bundles
prepared for Qualcomm QNN/HTP on Windows ARM64. It is not a universal ONNX
importer, a model converter, or a claim that every NPU supports DirectML.

The normal Ollama HTTP endpoint and model commands remain the application
interface. The ORT runner is a private child process, not a second user-facing
server.

## Support boundary

- Target: Windows 11 ARM64 on Snapdragon, with a compatible Qualcomm NPU driver.
- Runtime: the pinned ORT GenAI QNN and ORT QNN packages selected by
  `ollama npu setup`. Do not mix unrelated runtime DLL versions.
- Model: a complete, compatible ORT GenAI bundle, including
  `genai_config.json`, every referenced graph/context/weight file, and tokenizer
  assets.
- Model architecture, quantization, compiled context dimensions and target
  compatibility are properties of the bundle. Renaming a model to `qnn` does not
  make it NPU-compatible.
- Text inference is the initial scope. Tool use, structured output, vision and
  embeddings must not be assumed supported merely because a model imports.

Hardware detection and DLL loading are prerequisites, not proof that an
inference request ran on the NPU. Likewise, a CPU/GPU completion is not QNN
validation.

## Runtime setup

Use the fork executable consistently for setup, serving and model commands.
Check that a stock Ollama service is not already occupying the chosen port.
Do not overwrite or terminate an existing installation silently.

```powershell
.\ollama.exe npu setup --provider qnn
.\ollama.exe npu doctor
.\ollama.exe debug ortgenai
.\ollama.exe debug npu
```

For offline acquisition, use `npu setup --package-dir C:\approved-packages`.
The directory must contain the exact approved, lowercase package-ID/version
`.nupkg` filenames; hashes are checked before extraction. `--json` emits a
versioned report for setup, doctor and debug commands.

An old manually assembled runtime may have correct DLLs but lack the new
integrity manifest or license inventory. It is not silently replaced, including
with `--force`. Select a fresh directory and use it consistently:

```powershell
.\ollama.exe npu setup --provider qnn --dir C:\runtimes\ollama-qnn
$env:OLLAMA_ORT_PATH = 'C:\runtimes\ollama-qnn'
.\ollama.exe npu doctor --json
```

`--validate-only` makes no installation changes. `--force` revalidates an
immutable installation; it does not overwrite native libraries in use.

The supported consumer runtime path does not require Python, Olive or the QNN
SDK. Model production is a separate workflow.

For a separately acquired compatible runtime directory, `OLLAMA_ORT_PATH` can
select the native assets. It must contain matching ARM64 libraries and their
required dependencies. Do not copy random DLLs from Edge, Office, another
machine, or an unrelated vendor package.

## Importing a model

Acquire a model that is explicitly prepared for the desired execution provider
and target. Review its source/model license and redistribution restrictions.
Neither an arbitrary HF repository nor an arbitrary GGUF can be loaded by this
ORT runner.

Create a Modelfile next to the bundle:

```text
FROM .\model
```

Add the model's correct Ollama `TEMPLATE`, parameters and stop strings when
needed. A `chat_template.jinja` bundled as a model asset is not automatically an
Ollama Go template. Do not guess role tokens.

```powershell
.\ollama.exe create my-model:qnn -f .\Modelfile
.\ollama.exe run my-model:qnn
```

QNN decoder pipelines may include portable embedding/head graphs alongside
compiled context graphs. The actual config references determine the required
assets; a nonzero `.bin` beside an invalid ONNX wrapper is not a usable model.
Compiled target assets must be validated on compatible hardware.

### Text generation controls

The runner forwards temperature, top-p/top-k, seed, stop strings, repetition
penalty/window, context and prediction limit. Explicit zero values are preserved.
`num_predict=-1` generates up to the available context; it does not enlarge a
compiled context. A zero prediction limit performs no native generation.
Automatic context is bounded by the bundle; explicitly oversized context fails.

Media, embeddings, structured output, logprobs, min-p, nondefault typical-p,
presence/frequency penalties and unsupported context-shifting controls are not
silently approximated. Text chat requires the correct Ollama template.

Requests are serialized within each child because the native API is not thread
safe. With no explicit loaded-model limit, ORT requests use a conservative
one-resident-model limit; ordinary GGUF keeps its upstream automatic policy.
An explicit `OLLAMA_MAX_LOADED_MODELS` overrides that conservative limit.
Cancellation retires the child when needed, so the next request may require a
cold reload.

## Publishing and pulling

The fork stores ORT assets as named layers in Ollama's content-addressed model
store. Users of a compatible fork should be able to pull a qualified published
model and use it through their usual Ollama application.

Publishing is an explicit operation requiring an authorized registry namespace
and redistribution rights:

```powershell
.\ollama.exe cp my-model:qnn myuser/my-model:qnn
.\ollama.exe push myuser/my-model:qnn
```

Do not treat a local create or a successful upload alone as distribution
certification. Verify a clean pull, intact assets, runtime compatibility and
generation on another compatible profile before promoting a model.

Stock Ollama does not gain the ORT runner by pulling an ORT model tag. Public
registry acceptance and target compatibility are separate from local packaging.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Unsupported platform or runtime architecture | Run native Windows ARM64 Ollama and install matching assets |
| Missing/corrupt runtime | Run setup/doctor; verify the selected package and dependency set |
| Provider or backend fails to load | Check runtime versions, QNN/HTP dependencies and OEM driver |
| Model load fails | Check graph/config references, context binaries, tokenizer and compiled target |
| Requested setting is rejected | Use the supported option set; do not assume all llama.cpp options apply |
| Prompt exceeds context | Use the bundle's qualified context, not a larger guessed `num_ctx` |
| Chat exposes role tokens or produces poor output | Verify tokenizer, template and source/quantization quality |
| GPU supports DML but NPU does not | This is not unusual on Snapdragon; QNN/HTP is a different path |

Do not use a benchmark label or a Task Manager graph as sole placement proof.
Record the requested provider, loaded model/runtime and actual provider/backend
execution evidence.

## Development and qualification

See [development](development.md#windows-arm) for the standard LLVM-MinGW ARM64
build, focused tests and `scripts\build_npu_windows.ps1` portable packaging.
Extract the complete archive; retain the CLI's peer DLLs and `lib\ollama` layout.
Do not replace a stock installation with this unsigned preview. Runtime setup is
still a separate hash-verified acquisition; no model or ORT/QNN runtime is bundled.
Private compiler wrappers from earlier development are not build prerequisites.

Hardware qualification must include correct output, context/options, streaming,
client disconnect/cancellation, unload/reload, provider failure and ordinary
CPU/GPU regression checks. Retain exact source, binary, model, runtime and driver
identities. Hosted unit tests do not exercise an NPU.
The versioned [opt-in qualification suite](../integration/README.md#opt-in-f0-npu-qualification-existing-local-server-only)
uses an existing dedicated loopback server and existing models. It never downloads,
publishes or repairs models. Retain its JSON results alongside independent native
provider logs; HTTP success alone cannot prove QNN placement.

### October 13 hardware check

The reconciled fork ran a freshly acquired, hash-verified
GenAI QNN `0.12.2` / ORT QNN `1.24.1` runtime on Snapdragon X Elite X1E78100.
The existing compiled Phi-3 bundle passed 20 deterministic requests across
three simple prompts, native and OpenAI chat, stop/length/repetition controls,
explicit unsupported requests, disconnect/reload and CPU GGUF switching.
These are 42 local checks, not broad model-quality or multi-device certification.
The compiled decoder uses QNN; CPU embedding/head stages are intentional.

In the final short-prompt run, warm HTTP time had a median of about 0.39 seconds
(19 samples, 0.37-0.42 seconds); cold load plus generation was about 20 seconds.
An earlier run had a 0.34-second warm median and a 23-second cold request.
This is not an
equivalent-weights CPU/NPU performance comparison or an energy measurement.
The warm path avoids rehashing multi-gigabyte immutable assets on every lookup:
verified large files are memoized in-process by file identity and metadata.
Changed files and copied assets are SHA256-checked; this is not protection
against a privileged local process deliberately rewriting bytes and timestamps.

Focused tests/build/vet pass and the desktop UI has 220 passing tests. Full
Windows ARM64 Go tests/vet still encounter upstream desktop/tree-sitter compiler
limitations, Windows symlink privileges and permission-mode tests. Do not
interpret this qualification as full-suite-green or a signed consumer installer.
Full vet also reports existing unsafe-pointer warnings outside the scoped
packages. Public registry round-trips, clean-machine installation and other
devices/backends remain release gates.

## GGUF/Hexagon is a separate route

Current upstream llama.cpp has a Snapdragon Hexagon backend that can execute
compatible GGUF without an ORT/QNN model conversion. That does not make a QAIRT
bundle, an ORT model and a GGUF interchangeable.

The Windows developer build instructions require Hexagon SDK components and
trusted signatures for DSP operation libraries. This project must not require
ordinary users to disable Secure Boot or enable test-signing as its supported
installation workflow. A signed, compatible, hardware-qualified backend is
required before advertising seamless GGUF/NPU support.

## Historical results

Local March 2026 runs on Snapdragon X Elite demonstrate successful QNN
completions; selected warm samples reported roughly 10-12 tokens/s and
1.8-1.9 seconds TTFT. They used different artifacts/settings from later source
changes. They are historical evidence, not current performance guarantees or
an accelerator-only comparison with equivalent weights.
