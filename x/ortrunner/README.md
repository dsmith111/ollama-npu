# ORT GenAI runner

This fork's ORT runner executes compatible ORT GenAI bundles in a child process
behind Ollama's normal model/server interfaces. The supported development target
is Windows ARM64 with Qualcomm QNN/HTP; broad Intel/AMD, DirectML-NPU and generic
ONNX support must not be inferred from provider names.

See [the canonical QNN guide](../../docs/npu_qnn.md) for setup, import, publishing,
support boundaries and troubleshooting.

## Model assets

The bundle must include a valid `genai_config.json`, tokenizer assets, and every
graph, external weight and compiled context referenced by its configuration.
Graph splits, context sizes and target compatibility are model-specific.
Standalone GenieX/QAIRT bundles are not automatically ORT GenAI bundles.

Ollama stores model assets as named content-addressed layers. Import/push/pull
must preserve relative asset paths, and the model cache is a derived runtime
directory rather than the authoritative copy of the model.

A bundled Jinja template is not automatically translated into an Ollama
template. Supply the correct template/parameters when creating the model.

## Evidence and capabilities

Distinguish hardware detection, verified native dependencies, model loading and
actual inference execution. A successful dependency doctor does not establish
NPU placement or output quality.

The runner must reject unsupported request features explicitly. Text completion
alone does not qualify tools, vision, structured output, embedding or arbitrary
context expansion.

Unit tests cover protocol and failure behavior without proprietary models.
Hardware qualification requires a compatible model/runtime/driver tuple and
recorded execution evidence; it is separate from compilation and packaging
tests.
