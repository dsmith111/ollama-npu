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

See [development](development.md) for the current upstream native build. The
fork's ORT bindings additionally require a working CGo compiler. On a developer
machine that already has the local ARM64 compiler wrappers:

```powershell
$env:CGO_ENABLED = '1'
$env:CC = (Resolve-Path '.\_tools\cgo-clang.exe').Path
$env:CXX = (Resolve-Path '.\_tools\cgo-clang++.exe').Path
$env:CGO_LDFLAGS = '-ladvapi32'
go build .
go test .\parser .\server .\cmd .\x\ortrunner\...
```

Those local binaries are not a portable installer or a substitute for a
documented clean build environment.

Hardware qualification must include correct output, context/options, streaming,
client disconnect/cancellation, unload/reload, provider failure and ordinary
CPU/GPU regression checks. Retain exact source, binary, model, runtime and driver
identities. Hosted unit tests do not exercise an NPU.

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
