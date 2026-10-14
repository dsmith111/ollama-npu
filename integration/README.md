# Integration Tests

This directory contains integration tests to exercise Ollama end-to-end to verify behavior

By default, these tests are disabled so `go test ./...` will exercise only unit tests. To run integration tests, pass the `integration` tag and one of the scoped tags:

```bash
go test -tags=integration,fast -v -count 1 ./integration/
go test -tags=integration,release -v -count 1 -timeout 30m ./integration/
go test -tags=integration,library -v -count 1 -timeout 120m ./integration/
```

Tags:

- `fast`: quick runner/model smoke coverage.
- `release`: release regression coverage.
- `library`: broad library coverage requiring about 2.5 TiB of disk space.

Scope wiring and model selections live in `integration/reg_fast_test.go`, `integration/reg_release_test.go`, and `integration/reg_library_test.go`.

The integration tests have 2 modes of operating.

1. By default, on Unix systems, they will start the server on a random port, run the tests, and then shutdown the server.  On Windows you must ALWAYS run the server on OLLAMA_HOST for the tests to work.
2. If `OLLAMA_TEST_EXISTING` is set to a non-empty string, the tests will run against an existing running server, which can be remote based on your `OLLAMA_HOST` environment variable

Set `OLLAMA_TEST_LOG_SERVER=1` to print the managed server log after each test
run, even when the tests pass. This only applies when the integration test
harness starts the server.

> [!IMPORTANT]
> Before running the tests locally without the "test existing" setting, compile ollama from the top of the source tree  `go build .` in addition to GPU support with cmake if applicable on your platform.  The integration tests expect to find an ollama binary at the top of the tree.


## Testing a New Model

When implementing new model architecture, use `OLLAMA_TEST_MODEL` to run the
integration suite against your model with either the `fast` or `release` coverage.

```bash
# Build the binary first
go build .

# Run integration tests against it
OLLAMA_TEST_MODEL=mymodel go test -tags=integration,fast -v -count 1 ./integration/
```

## Opt-in F0 NPU qualification (existing local server only)

`TestNPUQualification` is a separate, profile-specific stdlib Go test, not part
of the generic model/capability matrices. Its locally qualified profile is the
text-only compiled Phi-3 QNN bundle with GenAI QNN **0.12.2** / ORT QNN
**1.24.1**. The compiled decoder uses QNN; CPU embedding/head stages are
intentional. This is not qualification of arbitrary models or accelerators.

Prepare the server, runtime and both models **separately** before running.
The test never starts/stops a server or downloads, creates, pushes or deletes
models. It preserves stored templates/model assets, but intentionally changes
residency through cancellation, CPU/QNN switching and explicit unload/reload.
Use a dedicated, otherwise idle server; concurrent requests can invalidate
latency/residency observations. Do not run the broad integration suite alongside
it.

Required settings:

- `OLLAMA_HOST`: explicit HTTP(S) origin on `localhost`, `127.0.0.1` or `[::1]`,
  with an explicit port other than `11434`. Remote/unspecified addresses,
  credentials, URL paths, queries, redirects and environment proxies are rejected
  or disabled. `localhost` is pinned to IPv4 loopback; use `[::1]` for IPv6.
- `OLLAMA_TEST_NPU_MODEL`: existing explicit `model:tag` for this compiled bundle.
- `OLLAMA_TEST_CPU_MODEL`: existing ordinary `phi3:mini` GGUF (a fully qualified
  spelling of that name is also accepted), exercised with `num_gpu=0`.
- Optional `OLLAMA_TEST_NPU_REQUESTS`: integer **20..1000**, default **20**,
  controlling the deterministic raw-request loop for later soak runs.

With none of the three `OLLAMA_TEST_*` settings above present, the hardware test
skips (`OLLAMA_HOST` alone does not opt in). Any partial/empty/invalid opt-in,
missing model or failed prerequisite is a failure, not a skip. `Show` checks both
existing local models before inference; `/api/tags` supplies actual model digests.
There is no pull-if-missing or model-store/template repair path.

The NPU model must already have the correct Ollama template attached, without
a default system message, preset conversation or alternate renderer. For each
single-user prompt it must render **exactly** (including final newline):

```text
<|user|>
{text}<|end|>
<|assistant|>
```

Equivalently: `"<|user|>\n{text}<|end|>\n<|assistant|>\n"`. Raw requests use that
same profile-specific framing. The test renders the attached template locally
to validate it; it never overrides or replaces it.

Coverage includes:

- 20 deterministic raw requests across arithmetic/capital/sky prompts, expecting
  exactly `4`, `Paris`, `Blue` respectively (only surrounding whitespace ignored).
  Options: context 512, prediction limit 32, temperature 0, seed 42.
- One-token `length`, stop `Paris`, repetition penalty 1.1/window 64, explicit
  top-k 0/top-p 1/seed 0, automatic context, Ollama chat and OpenAI chat.
- Explicit rejection of JSON format, min-p 0.1, presence penalty 0.1,
  context 8192 and CPU fallback (`num_gpu=0`) for the compiled QNN model.
- Streaming terminal validation, client disconnect after the first content
  chunk followed by a bounded recovery request, explicit unload/reload, ordinary
  Phi-3 CPU inference and switching back to QNN.
- Final unload of both test models and bounded `/api/ps` verification that
  **those models only** are absent; unrelated resident models are ignored.

All required cases fail on errors, never capability-skip after opt-in. Each
request is bounded to two minutes; residency polling is bounded to 15 seconds.
A one-token response may legitimately have zero `eval_duration` because OGA
prefill can supply the first token. No arbitrary historical check tally is
asserted.

From the repository root in PowerShell (substitute your existing bundle tag):

```powershell
$env:OLLAMA_HOST = 'http://127.0.0.1:11435'
$env:OLLAMA_TEST_NPU_MODEL = 'your-existing-phi3:qnn'
$env:OLLAMA_TEST_CPU_MODEL = 'phi3:mini'
go test -tags integration -run '^TestNPUQualification$' -count=1 -timeout=30m -json .\integration |
    Tee-Object -FilePath .\npu-qualification.jsonl
```

Increase the suite timeout for longer soak runs. Use the standard LLVM-MinGW
ARM64 CGo environment documented in [development](../docs/development.md#windows-arm)
before either command. Private local compiler wrappers are not prerequisites.
Retain the test process exit code as well as the JSON stream; a missing opt-in
skip is not a successful hardware qualification.

The pure helper tests use `httptest`, do not invoke hardware and need no model
or server configuration:

```powershell
go test -tags integration -run '^TestNPUQualificationHelpers$' -count=1 .\integration
```

Retain the `go test -json` evidence: `t.Logf` records the source Git revision and
dirty-worktree flag, client/server versions, reported model digests, request
outputs/options cases, and wall/API metrics for cold-after-unload and warm
requests. Also retain the source diff for a dirty tree, server binary identity,
runtime/driver identities and independent provider/native execution logs.
Runtime versions above are the **declared profile**, not discovered via HTTP.
Tags, successful completions and `/api/ps` cannot prove NPU placement:
**independent provider/native log evidence is required**. Latencies are
observations, not equivalent-weights CPU/NPU throughput or energy claims.
