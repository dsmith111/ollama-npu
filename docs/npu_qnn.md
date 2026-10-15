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
- The pinned Windows QNN file mapper mishandles non-ASCII directory names.
  The fork uses a Windows-provided ASCII short-path alias only after verifying
  that it identifies the same model directory. It does not move model files,
  rewrite the bundle or change filesystem settings. Without a usable alias it
  rejects the load before native initialization; this is not universal Unicode
  filename support.
- Long runtime paths also encounter the pinned ORT provider-discovery
  `MAX_PATH` limit. The same directory-identity checks select a Windows ASCII
  alias for native loading, without changing the installation or verified bytes.
  If none fits, `NPU_RUNTIME_PATH_UNSUPPORTED` requests the default setup location
  or a separately qualified runtime; do not change filesystem policy.

Hardware detection and DLL loading are prerequisites, not proof that an
inference request ran on the NPU. Likewise, a CPU/GPU completion is not QNN
validation.

## Runtime setup

Review the runtime's Microsoft and Qualcomm terms before acquisition; see
[distribution evidence](#distribution-evidence-and-publication-gates). Microsoft
MIT notices do not replace the separately included Qualcomm AI Stack license.

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

After an abruptly terminated setup, retry at the same destination. A destination
is published only after complete validation. Abrupt termination may leave an
unused private `.ortgenai-install-*` staging directory; automatic orphan recovery
remains a release follow-up. Do not delete unknown stages while installers run.

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

### Versioned compatibility declaration

New publishers can include an optional named artifact, **`ollama_npu.json`**,
alongside `genai_config.json`. Schema 1 currently supports only Windows ARM64 QNN,
text completion and decoder graphs/pipelines. This is a publisher's **declaration**,
not a certification, conversion result, license grant or NPU-placement measurement.
Do not add it to an old bundle by guessing its provenance, precision or targets.

Illustrative schema, **not an approved downloadable/qualified model**:

```json
{
  "schema_version": 1,
  "platform": "windows/arm64",
  "provider": "qnn",
  "genai_version": "0.12.2",
  "ort_version": "1.24.1",
  "processor_names": [
    "Snapdragon(R) X Elite - X1E78100 - Qualcomm(R) Oryon(TM) CPU"
  ],
  "context_length": 4096,
  "precision": "publisher-declared-mixed",
  "capabilities": ["completion"],
  "stages": [
    {"filename": "embeddings.onnx", "provider": "cpu"},
    {"filename": "context_ctx.onnx", "provider": "qnn"},
    {"filename": "iterator_ctx.onnx", "provider": "qnn"},
    {"filename": "lm_head.onnx", "provider": "cpu"}
  ]
}
```

Use the artifact's actual precision and exact qualified processor descriptors,
not this example as permission to claim support. The CPU descriptor is read from
Windows' `ProcessorNameString`; matching is case-insensitive, otherwise exact.
This conservative allowlist is not QNN device binding, driver qualification or
proof that every operation runs on the NPU.

Contract:

- UTF-8 JSON object, at most 64 KiB; unknown/duplicate fields, nulls, trailing
  data, unsupported schema/platform/provider and malformed declarations fail.
- All example fields are required. Exact runtime versions are `major.minor.patch`;
  no version ranges. Other runtime tuples may be stored/imported but cannot run
  on this fork's approved tuple without separate migration/qualification.
- `context_length` is a positive int32 and matches the model config.
  `precision` is publisher-declared text of 1..64 bytes, not inferred.
  `capabilities` must be exactly `["completion"]`: chat also needs the correct
  Ollama template; this does not promise tools, vision or structured output.
- `processor_names` has 1..32 unique names of at most 256 bytes.
  `stages` has 1..64 unique, canonical relative `.onnx` filenames, with `cpu` or
  `qnn` providers and at least one QNN stage. Every referenced graph must be a
  described decoder stage. Non-decoder graph configurations are outside schema 1.
  Providers follow the config's actual session-option inheritance; CPU
  embedding/head stages must be described as CPU.
  Pipelines require `model.type="decoder-pipeline"` and no `decoder.filename`;
  decoder-only model types require just `decoder.filename`, with no pipeline.
  Inactive graphs cannot be counted as declared NPU execution stages.
  Declared ORT bundles cannot be mixed with native tensor layers to bypass checks
  through the native transfer path.
- QNN stages must use HTP: optional `backend_type` must be `htp`, and optional
  `backend_path` must be exactly `QnnHtp.dll`, not an arbitrary library path.
- Optional `source` is `{"repository":"https://host/owner/repo",
  "revision":"full-lowercase-Git-commit"}`. The revision is 40 or 64 hex digits.
  Credentials, URL query/fragment and mutable revision names are rejected.
  This records a claim, not independently verified source-to-artifact provenance.
- The enclosing Ollama manifest already binds hashes for profile/config, graphs,
  weights, tokenizer and separate template/parameter/license layers. Do not
  duplicate that inventory or embed native runtime libraries in the model.

Import, `FROM`, copy, Show and download validate structure without requiring the
current machine to match the target. A producer can package/archive another
target's model. A declared profile is checked against config, selected provider,
host and fork runtime pins **before native loading**, in both server and child.
`OLLAMA_ONNX_MODEL` cannot replace a stored declared model with another directory.
QNN backend/HTP/SoC environment overrides are rejected for declared profiles.

Declared models also require a complete, hash-verified runtime installation.
The first runtime search candidate containing GenAI must validate; a corrupt or
legacy candidate fails instead of falling back to PATH or a different DLL.
Use `npu setup` in a fresh directory and select it with `OLLAMA_ORT_PATH`.
Native initialization validates this once per child, not on each HTTP request.

Show's `model_info` reports `npu.compatibility_status` as `declared`, with the
profile in `npu.compatibility`, or `legacy-unverified` when absent. Legacy models
retain their previous loader behavior and emit an explicit compatibility warning.
An absent profile is allowed; a malformed or case-aliased declaration is not.
The October 13 local hardware assets remain unchanged, without newly invented
metadata. The declaration/verified-load route now has separately identified local
engineering-fixture evidence below; publisher-profile and clean-consumer
qualification remain open.

Affected Go suites/scoped vet, qualification helpers and CLI compilation pass.
The full Go suite was rerun: 74 packages pass, three fail, 31 skip/have no tests.
Failures remain the previously reproduced upstream Windows symlink/Codex-mode
issues (`manifest`, `compatmigrate`, `cmd/launch`), not a full-suite-green claim.
At this F1 source checkpoint, hardware qualification was not opted in; no new
executable/NPU run occurred. The older package's subsequent result is recorded below.

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
store. Seamless hosted pull/run is the intended outcome, **not yet a verified
hosted service guarantee**. Synthetic loopback push/empty-store pull tests preserve
names/digests, compatibility declaration, template, parameters and license.
They reject corrupted and well-hashed contradictory declarations before writing
the local model tag; create/FROM/copy/Show tests also preserve the profile.
Manifest-list push preflights every child's declaration before registry access.

Publishing is an explicit operation requiring an authorized registry namespace
and redistribution rights. The following is only for an approved artifact and
destination, after the service-policy/visibility decisions below:

```powershell
.\ollama.exe cp my-model:qnn myuser/my-model:qnn
.\ollama.exe push myuser/my-model:qnn
```

Do not treat a local create or a successful upload alone as distribution
certification. Verify a clean pull, intact assets, runtime compatibility and
generation on another compatible profile before promoting a model.

Stock Ollama does not gain the ORT runner by pulling an ORT model tag. Public
registry acceptance and target compatibility are separate from local packaging.

### Distribution evidence and publication gates

Primary-source research checked October 14, 2026:

| Item | Evidence | Consequence |
| --- | --- | --- |
| Microsoft runtime code | [GenAI 0.12.2 MIT](https://raw.githubusercontent.com/microsoft/onnxruntime-genai/v0.12.2/LICENSE) and [ORT 1.24.1 MIT](https://raw.githubusercontent.com/microsoft/onnxruntime/v1.24.1/LICENSE) | Preserve notices; package version/dependency minima do not certify compiled contexts |
| Qualcomm runtime components | ORT's [QNN build copies the SDK license](https://raw.githubusercontent.com/microsoft/onnxruntime/v1.24.1/cmake/onnxruntime_providers_qnn.cmake); exact installed `licenses\microsoft.ml.onnxruntime.qnn\Qualcomm_LICENSE.pdf` SHA256 `ec1dccfdcba5c6e64126e84199b8362bf4999107bfa567ebe831dbb4c461692b` | AI Stack license section 1 allows object-code distribution incorporated in an application, not standalone redistribution; sections 2, 3 and 10 add notice/use/rights/export obligations. This is not blanket permission for runtime/model mirrors |
| Original Phi-3 weights | [Microsoft source MIT license at fixed revision](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/raw/f39ac1d28e925b323eae81227eaba4464caced4e/LICENSE) | Does not establish the provenance/terms of this existing compiled derivative |
| Existing local derivative | Preserved manifest has no license layer or immutable source revision; `_publish_manifest.json` is an asset hash inventory, not provenance evidence | Do not upload it based only on successful inference or the original weights' license |
| Other catalog artifacts | [Microsoft ONNX catalog](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx) lists other execution targets; [Qualcomm Phi-3.5](https://huggingface.co/qualcomm/Phi-3.5-mini-instruct) offers GENIE/QAIRT artifacts | Different model/runtime/artifact route, not an established drop-in ORT bundle or license for the local derivative |
| Hosted model protocol | [Documented imports](https://docs.ollama.com/import) and [push API](https://docs.ollama.com/api/push) do not promise ORT named-layer support or a private/unlisted push selector | Client fixture success cannot establish hosted acceptance, descriptor preservation or privacy; a staging tag must not be assumed private |
| Hosted service rights | [Ollama terms](https://ollama.com/terms), sections 4 and 7 | Competing-product/automated-access and branding provisions need clarification for fork registry use. The source MIT license is not a hosted-service authorization |

This is engineering evidence, not a legal opinion. Obtain artifact-specific
rights/provenance review for source weights, generated contexts and any included
vendor software; preserve the applicable notices and review runtime terms before
acquisition. No definitive compiled-output grant was established for the local
bundle. The portable fork archive contains no ORT/QNN DLLs or model weights.

Before any hosted test:

1. Clarify permitted fork/registry use and custom ORT layer handling with the
   service owner; `hello@ollama.com` is the published terms contact. No contact or
   approval is implied by this research.
2. Select a model with recorded, immutable source and producer/SDK provenance,
   reviewed redistribution rights and separately qualified template/profile.
3. Explicitly approve the account/namespace, visibility and exact upload contents.
   Require genuine service controls for private testing, not a private-looking tag.
4. On approved compatible hardware, push and pull into a separate empty store,
   check all identities and run generation through an existing Ollama client.
   Retain host/runtime/driver and independent placement evidence.

Until then, F2 hosted distribution stays open. A reviewed HF/offline artifact can
support a narrower preview, but neither bypasses artifact rights nor fulfills the
hosted `ollama pull` goal. No model/vendor material was uploaded in this work.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Unsupported platform or runtime architecture | Run native Windows ARM64 Ollama and install matching assets |
| Missing/corrupt runtime | Run setup/doctor; verify the selected package and dependency set |
| Provider or backend fails to load | Check runtime versions, QNN/HTP dependencies and OEM driver |
| Model load fails | Check graph/config references, context binaries, tokenizer and compiled target |
| `NPU_MODEL_INCOMPATIBLE` / `NPU_TARGET_INCOMPATIBLE` | Acquire an artifact declared for this platform/provider/processor; do not guess or edit target metadata |
| `NPU_RUNTIME_INCOMPATIBLE` | Use a separately qualified model/fork tuple; do not mix DLL versions |
| Declared model rejects a legacy runtime or QNN override | Select a fresh setup-verified directory and remove backend/SoC/HTP overrides |
| `NPU_PROFILE_LOAD_FAILED` | A declaration is not hardware proof; preserve config/runtime and check the publisher's target and OEM driver |
| `NPU_MODEL_PATH_UNSUPPORTED` | Windows supplied no usable ASCII alias for the QNN model directory; select an ASCII `OLLAMA_MODELS` directory or a separately qualified Unicode-capable runtime, without changing filesystem policy |
| `NPU_RUNTIME_PATH_UNSUPPORTED` | The pinned runtime cannot discover providers beyond `MAX_PATH` and no safe short alias fits; use the default setup location or a separately qualified runtime |
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

### October 14 packaging checkpoint

The portable builder produced a clean-checkout package from source
`4e18dc1de9a283dd201ff3be02fac0736f3a861e`, using Go `1.26.1`,
LLVM-MinGW UCRT ARM64 `20261006` (Clang `23.1.3`), Windows SDK `10.0.26100.0`
and native pin `b11521` (llama.cpp commit `b42b7e6d3`). The archive is
`ollama-npu-windows-arm64-0.0.0-npu.dev.4e18dc1de9a2.zip`, SHA256:
`e28aaec6091ce608ee2f5540ce35914211096d8215eeb6f8fdc661888253b774`.
It is a retained local development artifact, **not a published release**.

- Clean build, native ARM64/ARM64X/import audit, runtime/license inventory,
  extracted archive hashes and all 32 inventoried files passed verification.
  Private CGo wrappers and a prebuilt native payload were not used.
- Clean-checkout focused Go tests/scoped vet, qualification helper tests and UI
  build/220 tests pass. No-opt-in hardware qualification correctly skips; that
  skip is not hardware evidence.
- Standard-toolchain full Go tests: 74 packages pass, three fail and 31 skip/have
  no tests. All three failing packages reproduce the same failures on official
  `8718c2bd`: Windows symlink privileges (`manifest`, `compatmigrate`) and Codex
  auth mode (`cmd/launch`, 666 versus 600). The earlier MSVC-wrapper compiler
  blockers do not reproduce with this toolchain.
- Full vet still reports existing unsafe-pointer/SliceHeader warnings. The
  unchanged UI lint baseline has 112 errors/14 warnings; `npm ci` reports 61
  dependency audit findings, not independently assessed here. Desktop/UI assets
  are excluded from this portable package.
- **Initial execution blocker:** launching the development package's CLI
  and native server was denied by Microsoft Defender ASR:
  event `1121`, rule `C1DB55AB-C21A-4637-BB3F-A12568109D35`. No protection,
  exclusion, trust, boot or driver settings were changed. No server was started,
  and the new Go hardware suite has not run against the packaged binary.

That was the build-time result; the subsequent retry below is separate evidence.
The immutable archive's `build-info.json` retains `hardware_qualified=false`.

### October 14 approved package retry

The device owner independently configured an ASR-only exclusion for the exact
retained CLI and requested a retry. The agent changed no security settings.
The unchanged CLI's `--help`, minimal-system-PATH `npu doctor --json`, and exact
packaged native server's `--version` now exit successfully. Doctor alone still
does not verify inference. No separate native exemption was requested by the agent;
the reason its launch behavior changed is not established.

Two complete runs of the versioned hardware suite each pass all 37 subtests:
20 exact normal completions, controls, Ollama/OpenAI chat, unsupported requests,
stream termination, disconnect recovery, unload/reload and CPU GGUF/QNN switching.
The server used the existing isolated store/runtime on a dedicated loopback port,
with no developer PATH, downloads, publication or original model changes.
The owned server was stopped after testing.

The first actual harness execution exposed ambiguous color casing and an invalid
assumption that active repetition penalties preserve the ordinary answer. The
prompt now explicitly requests `BLUE`; ordinary outputs remain exact/case-sensitive.
Repetition checks require bounded deterministic output and exact restoration when
the window is zero. The penalty-mode arithmetic answer was incorrect (`10`):
its functional check is **not a model-quality approval**. Failed runs are retained.
Helpers, file-scoped harness vet and ORT regression suites pass; whole integration
vet reports existing non-test-goroutine `Fatalf` calls in unrelated tests.

Evidence binds the immutable package hashes above to harness SHA256
`09588c1ed7427937307cb2d9af6ae9e04c2d0a0db8b59bbd1dc0f703a2edbcaf`.
Native logs and a live owned-child module inventory confirm loaded verified ORT/QNN,
`QnnHtp.dll`, `QnnHtpV73Stub.dll` and `libcdsprpc.dll` during compiled-context inference.
This is not per-node profiling, accelerator-only placement or energy evidence.
Native context-load (`0x1392`) and shutdown warnings also occur in the earlier
successful baseline; they are preserved, not silently classified as harmless.
All 32 extracted inventory entries and the original archive remain unchanged.

This closes F0's local packaged execution check, **not F1 or a consumer release**.
Package source `4e18dc1d` predates the new compatibility declaration/verified-loader
code. The subsequent local route check below is separate; fresh-user/clean-machine
setup, broader quality, signing and authorized hosted distribution remain open.

### October 14 declared-loader check and startup fix

A separately built current-source CLI launched normally without another exception.
Its first real declared-model load exposed a startup bug: `Execute` initialized
the unchecked legacy loader before `Runner.Load` could select verified loading.
The child refused the declaration and the parent reported readiness EOF. This
was a functional failure, not a fresh Defender denial.

Startup now leaves initialization to model validation/loader selection. Shutdown
only cleans up a successfully loaded runtime; a pre-native validation failure must
not initialize a legacy runtime during cleanup. Both subprocess regressions fail
before the fix and pass afterward. Affected ORT/server/CLI suites and scoped vet
pass. Full Go regression retains the same 74 pass/three baseline failures/31
skip-or-no-tests outcome.

The fixed Go-only test CLI, SHA256
`ac381468998e43dc7dc265e76146a003f7b1b73d0b1a3b7df96608861ae83aa7`,
uses source `9d954419` plus the recorded startup patch and the unchanged F0 native
payload. Its source is explicitly dirty; this is not a newly clean-built release
archive. Two complete declared-model suites each pass 67 subtests, including 50
exact normal requests: **100 exact requests total**, plus controls, chat,
streaming/recovery and CPU/QNN switching. A separate legacy-model suite passes
all 37 subtests. Live child modules confirm the exact approved GenAI DLL and QNN HTP.
Wrong-target generation rejects before creating an ORT child; an actual direct
runner also rejects an incomplete first runtime candidate without native-load/
readiness announcements or fallback to the intact installed runtime.

The two temporary declarations were **local engineering fixtures**, not publisher
or license claims. Stage/provider/context values came from the unchanged config,
the processor from the registry and runtime versions from verified installation.
Precision was explicitly producer-undeclared and source provenance omitted.
Original tags were not assigned declarations. Both owned tags/profile blobs/caches
and the incomplete runtime fixture were removed, owned servers stopped, and all
29 unique blobs referenced by the three original models were hash-verified intact.
The original runtime/package remain unchanged.

This validates the local verified-loader mechanism, not completion of F1's licensed
publisher-profile, fresh-user/clean-machine and consumer recovery gates. The 100
short requests do not replace longer-workload quality, reboot, memory or usability
qualification; the penalty-mode quality issue and native warnings above remain.

### October 14 isolated consumer-path fixes

A new clean `3f034e72` package, including rebuilt native peers, launched normally
and passed ten actual offline setup/doctor/recovery checks in an isolated profile
with spaces and a non-ASCII name. Its CLI imported the local engineering fixture,
but QNN context loading failed on that directory. The pinned supplier
[Windows file mapper](https://github.com/microsoft/onnxruntime/blob/v1.24.1/onnxruntime/core/providers/qnn/builder/qnn_windows_file_mapper.cc)
widens narrow path bytes directly before `CreateFile2`. The model-directory alias
bridge above preserves the original files and runtime identity.

Consumer testing also exposed a terminal callback deadlock in ordinary
`/api/generate`: the callback tokenized the conversation while the client held
its operation lock. The client now validates the stream through EOF, rejects
trailing data and releases the lock before delivering terminal success. Startup
errors retain a bounded, fully drained child-error tail instead of only readiness
EOF. Reproducing unit tests and affected suites/scoped vet pass.

The separately identified dirty developer CLI with the first two fixes passed
all 39 hardware subtests: 20 exact normal prompts plus controls, templated
completion/stream conversation tokens, chat, cancellation, unload/reload and
ordinary CPU GGUF/QNN switching in the unchanged Unicode store. Actual CLI run,
new-directory online acquisition, same-destination retry after forced installer
termination, and refusal to take over an owned occupied port also passed.
The original archive and user assets were not overwritten.

This is process-environment isolation on the same developer Windows account,
not a clean machine or new account. No stock installation was found; an owned
port-holder test is not stock-service coexistence. At that developer checkpoint, clean-package
qualification remained open; the final artifact retry below supplies local
package evidence. Licensed publisher provenance, no-alias Unicode support,
reboot/longer-workload quality and usability remain release gates.

The next clean package, source `78764493`, passed two complete 69-subtest
hardware runs (50 exact normal prompts each), all ten isolated offline CLI
checks, actual CLI import/run, fresh online setup, abrupt-install retry and
owned-port refusal. Its archive SHA256 is
`d85a5d4f79b57b6eff7546ecca9c7f68796a6655adddcce8c459e91efbe3872d`.
This immutable local artifact predates the following long-runtime fix.

The 307-character model store worked through its verified alias, but a
308-character runtime directory exposed two limits: direct loading returned
Windows error 206, and extended-length spelling alone then exposed ORT's
truncated provider-discovery path. Do not treat a standalone DLL fixture that
loads as proof that the actual pinned runtime works. Sharing the validated
directory-alias helper between model and runtime loading fixes actual templated
inference on those unchanged paths. The separately identified dirty CLI returned
exactly `4`; eight live modules were hash-recorded, and file identity confirms
their alias directory is the verified long runtime. A final clean package of
this last fix was subsequently built; its execution gate is recorded below.

### Final October 14 artifact and approved qualification

Final clean source: `c8b363a0acaf7601ee9df9f5441d090bb1007ae1`. The path guard
also includes the longest pinned provider DLL filename, not just the shorter
GenAI filename. Focused suites/scoped vet pass; unsafe or missing aliases fail
explicitly before native loading.
The final-source full Go rerun remains 74 passing packages, three previously
reproduced upstream Windows-permission failures (`manifest`, `compatmigrate`,
`cmd/launch`) and 31 skipped/no-test packages; no full-suite-green claim.

Archive: `ollama-npu-windows-arm64-0.0.0-npu.dev.c8b363a0acaf.zip`, SHA256
`cd135f6d7ce2810bb0c131bc5ab519a243cdf7c4f8190be058d59f37e980459b`.
All 32 extracted inventory entries and clean-source/native build identities pass.
The CLI's normal `--help` succeeds. Initially, the normal extracted native
`lib\ollama\llama-server.exe --version` launch was denied by fresh Defender
ASR event `1121`, rule `C1DB55AB-C21A-4637-BB3F-A12568109D35`, at
`2026-10-15T00:44:19Z` (October 14 EDT). Exact native SHA256:
`60885a10d787eb38ddc961d52276070765fe9823382d7444617b85efff702953`.

The owner approved this exact executable at 21:08 EDT. Normal retry of the
unchanged file then succeeded; no agent security change, rebuild, rename or
alternate executable path was used. The exact final archive now has its own
local consumer and hardware evidence:

- All 32 extracted inventory entries, minimal-PATH CLI help and native version
  pass. The source checkout remained clean at `c8b363a0`.
- Ten actual offline setup/doctor/recovery outcomes, fresh-directory official
  online acquisition, abrupt installer termination/same-destination offline retry
  and owned occupied-port refusal pass.
- Actual CLI import/run and declared compatibility reporting pass in the new
  space/non-ASCII environment profile. Its strict LF template was retained.
- Two full hardware suites each pass 69 subtests, including 50 exact normal
  prompts: **138 subtests and 100 exact prompts total**, with no failed or skipped
  cases. Controls, templated completion/stream conversation tokens, chat,
  disconnect/reload, unload/reload and ordinary CPU GGUF/QNN switching pass.
- The same packaged executable returns exactly `4` with conversation tokens
  from independently hashed copies in a 307-character Unicode model store and
  308-character runtime directory. Both normal and long-path runs record eight
  live native modules; seven match the approved runtime file hashes and directory
  identity, with the OEM DSP RPC module recorded separately. This is not a
  per-node or accelerator-only placement claim.

These are local engineering-fixture results, not licensed publisher certification
or a clean-machine test. The immutable archive's `hardware_qualified=false`
build metadata is unchanged; hash-bound post-build qualification evidence is
separate. Existing penalty-mode quality limitations and native context/shutdown
warnings remain visible, not waived by short-prompt success. Abrupt termination
can leave an unpromoted private stage; retry succeeds, but ownership-safe automatic
orphan cleanup remains a release follow-up.

Owned test servers, model/runtime trial stores and build scratch were stopped or
removed. The exact approved extraction/source, approved offline package cache and
hash-bound evidence are retained. Original model/runtime assets and converter
changes remain untouched. A real fresh account/clean OS, stock-service
coexistence, licensed publisher profile, no-alias vendor fixes and authorized
hosted distribution remain additional release gates.

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
