# NPU support in this fork

The current integration runs compatible ORT GenAI models through Qualcomm
QNN/HTP on Windows ARM64 Snapdragon systems. See the
[canonical setup and model guide](npu_qnn.md).

Ordinary Ollama model storage, HTTP endpoints and CPU/GPU workflows remain the
application interface. Pulling a model tag does not add a missing execution
backend to stock Ollama.

## Different accelerators are not interchangeable

| Path | Scope |
| --- | --- |
| ORT GenAI + QNN/HTP | Compatible prepared Qualcomm model bundles |
| llama.cpp + Hexagon | Separate GGUF path requiring a qualified native backend |
| DirectML | Device/driver-dependent; GPU availability does not prove NPU support |
| Intel or AMD NPU | Requires a separate implemented and qualified adapter/runtime |

The earlier experimental ggml-DirectML and hybrid hooks are not the supported
Snapdragon NPU path. On the development device, DXCore detects Hexagon but
does not report D3D12/DirectML support for that NPU. An Adreno GPU and a Hexagon
NPU are different devices.

## Diagnostics and evidence

Use the fork executable:

```powershell
.\ollama.exe npu doctor
.\ollama.exe debug ortgenai
.\ollama.exe debug npu
```

Hardware detection, runtime integrity, model compatibility, successful loading,
actual execution and output correctness are separate checks. No single DLL
check or benchmark label proves NPU execution.

There is no universal "any model on any NPU" guarantee. Unsupported targets,
formats and options must fail explicitly rather than count CPU/GPU fallback as
successful NPU operation.
