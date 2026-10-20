package server

import (
	"context"
	"encoding/binary"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/types/model"
	"github.com/ollama/ollama/x/ortrunner"
)

func ortTestFiles(t *testing.T) map[string]string {
	t.Helper()
	files := make(map[string]string)
	for name, content := range map[string]string{
		"model.onnx":          string(ortIdentityGraph()),
		"model.onnx.data":     "external weights",
		"model.onnx.data_01":  "more external weights",
		"context.bin":         "\x00\x01context-cache-sidecar",
		"genai_config.json":   `{"model":{"type":"llama","context_length":128,"vocab_size":8,"eos_token_id":2,"decoder":{"filename":"model.onnx"},"encoder":{"filename":"nested/model.onnx"}}}`,
		"tokenizer.json":      `{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"[UNK]":0,"hello":1,"[EOS]":2},"unk_token":"[UNK]"}}`,
		"chat_template.jinja": "{{ messages }}",
		"nested/model.onnx":   string(ortIdentityGraph()),
		"nested/weights.bin":  "nested external data",
		"optional-empty.txt":  "",
	} {
		layer, err := manifest.NewLayer(strings.NewReader(content), MediaTypeORTGenAI)
		if err != nil {
			t.Fatal(err)
		}

		files[name] = layer.Digest
	}
	return files
}

// A serialized ONNX ModelProto containing a float Identity graph, not text
// masquerading as ONNX. These tests verify packaging, not native inference.
func ortIdentityGraph() []byte {
	bytesField := func(field int, data []byte) []byte {
		result := binary.AppendUvarint(nil, uint64(field<<3|2))
		result = binary.AppendUvarint(result, uint64(len(data)))
		return append(result, data...)
	}
	valueInfo := func(name string) []byte {
		dimension := []byte{8, 1}
		shape := bytesField(1, dimension)
		tensor := append([]byte{8, 1}, bytesField(2, shape)...)
		return append(bytesField(1, []byte(name)), bytesField(2, bytesField(1, tensor))...)
	}
	node := bytesField(1, []byte("input"))
	node = append(node, bytesField(2, []byte("output"))...)
	node = append(node, bytesField(4, []byte("Identity"))...)
	graph := bytesField(1, node)
	graph = append(graph, bytesField(2, []byte("identity"))...)
	graph = append(graph, bytesField(11, valueInfo("input"))...)
	graph = append(graph, bytesField(12, valueInfo("output"))...)
	model := []byte{8, 8} // ir_version = 8
	model = append(model, bytesField(7, graph)...)
	return append(model, bytesField(8, []byte{16, 13})...) // opset version = 13
}

func TestORTGenAIPackageRoundTrip(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	files := ortTestFiles(t)
	kind, err := detectModelTypeFromFiles(files)
	if err != nil || kind != "ortgenai" {
		t.Fatalf("detect = %q, %v", kind, err)
	}
	progress := func(api.ProgressResponse) {}
	layers, err := convertModelFromFiles(t.Context(), files, progress)
	if err != nil {
		t.Fatal(err)
	}
	if len(layers) != len(files) {
		t.Fatalf("layers = %d, want %d", len(layers), len(files))
	}
	var names []string
	for _, layer := range layers {
		names = append(names, layer.Name)
		if layer.MediaType != MediaTypeORTGenAI || files[layer.Name] != layer.Digest {
			t.Fatalf("invalid packaged layer: %+v", layer)
		}
	}
	if !slices.IsSorted(names) {
		t.Fatal("layer order must be deterministic")
	}

	name := model.ParseName("ort-round-trip")
	config := new(model.ConfigV2)
	if err := createModel(t.Context(), api.CreateRequest{}, name, layers, config, progress); err != nil {
		t.Fatal(err)
	}
	mf, err := manifest.ParseNamedManifest(name)
	if err != nil {
		t.Fatal(err)
	}
	if mf.Runner != manifest.RunnerORTGenAI || mf.Format != "ortgenai" {
		t.Fatalf("manifest metadata = %s/%s", mf.Runner, mf.Format)
	}
	m, err := GetModelForRunner(name.String(), manifest.RunnerORTGenAI)
	if err != nil {
		t.Fatal(err)
	}
	if !m.IsORTGenAI() || !slices.Contains(m.Capabilities(), model.CapabilityCompletion) {
		t.Fatalf("model not recognized as ORT completion: %+v", m.Config)
	}
	for filename := range files {
		if _, err := os.Stat(filepath.Join(m.ModelPath, filepath.FromSlash(filename))); err != nil {
			t.Fatal(err)
		}
	}
	again, err := GetModel(name.String())
	if err != nil || again.ModelPath != m.ModelPath {
		t.Fatalf("idempotent materialization failed: %v", err)
	}
	resp, err := GetModelInfo(api.ShowRequest{Model: name.String()})
	if err != nil || resp.ModelInfo["general.type"] != "ortgenai" {
		t.Fatalf("show = %+v, %v", resp, err)
	}
	fromLayers, fromConfig, err := parseFromModel(t.Context(), name, progress)
	if err != nil {
		t.Fatal(err)
	}
	derived := model.ParseName("ort-derived")
	if err := createModel(t.Context(), api.CreateRequest{System: "test system"}, derived, fromLayers, &fromConfig, progress); err != nil {
		t.Fatal(err)
	}
	copied := model.ParseName("ort-copied")
	if _, err := CopyModel(derived, copied); err != nil {
		t.Fatal(err)
	}
	if _, err := GetModel(copied.String()); err != nil {
		t.Fatal(err)
	}
	if !m.HasChatTemplate || chatModeForModel(m) != chatExecutionModeNative {
		t.Fatal("an ORT model with its own chat template and no Ollama template must be rendered by its runner")
	}
	m.HasGoTemplate = true
	if chatModeForModel(m) != chatExecutionModeRendered {
		t.Fatal("an ORT model with an Ollama template must use the rendered completion path")
	}
	m.HasGoTemplate, m.HasChatTemplate = false, false
	if chatModeForModel(m) != chatExecutionModeRendered {
		t.Fatal("an ORT model without a chat template must use the rendered completion path")
	}
}

func TestORTGenAIPackageValidation(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	files := ortTestFiles(t)
	for _, required := range []string{"genai_config.json", "tokenizer.json", "model.onnx"} {
		t.Run(required, func(t *testing.T) {
			bad := make(map[string]string)
			for name, digest := range files {
				if name != required {
					bad[name] = digest
				}
			}
			_, err := convertFromORTGenAI(t.Context(), bad, func(api.ProgressResponse) {})
			if !errors.Is(err, errInvalidORTGenAI) {
				t.Fatalf("missing %s: %v", required, err)
			}
		})
	}
	mixed := make(map[string]string)
	for name, digest := range files {
		mixed[name] = digest
	}
	mixed["other.gguf"] = files["model.onnx"]
	if _, err := detectModelTypeFromFiles(mixed); !errors.Is(err, errMixedModelTypes) {
		t.Fatalf("mixed files: %v", err)
	}
	for _, path := range []string{"", "../escape.onnx", "C:/escape.onnx", `..\escape.onnx`} {
		mf := &manifest.Manifest{Layers: []manifest.Layer{{MediaType: MediaTypeORTGenAI, Name: path}}}
		if _, err := materializeORTGenAIDir(mf); err == nil {
			t.Fatalf("accepted unsafe layer name %q", path)
		}
	}
	if err := validateCreateOptions(api.CreateRequest{Quantize: "int4"}, "ortgenai"); err == nil {
		t.Fatal("ORT create-time quantization must be rejected")
	}
}

func TestORTSchedulerLoadFailureIsTerminal(t *testing.T) {
	s := InitScheduler(t.Context())
	calls := 0
	s.newORTServerFn = func(ctx context.Context, opts ortrunner.ClientOptions) (llm.LlamaServer, error) {
		calls++
		if opts.Provider != "qnn" {
			t.Fatalf("unexpected fallback provider %s", opts.Provider)
		}
		return nil, errors.New("QNN unavailable")
	}
	req := &LlmRequest{
		ctx: t.Context(), model: &Model{ModelPath: "ORT bundle", Config: model.ConfigV2{ModelFormat: "ortgenai"}},
		opts: api.DefaultOptions(), errCh: make(chan error, 1), successCh: make(chan *runnerRef, 1),
		ortOptions: &ortrunner.ClientOptions{ModelDir: "ORT bundle", Provider: "qnn"},
	}
	if retry := s.load(req, ml.SystemInfo{}, nil, false); retry {
		t.Fatal("provider load failure must not request retry/fallback")
	}
	if calls != 1 || len(req.errCh) != 1 || len(req.successCh) != 0 || len(s.loaded) != 0 {
		t.Fatalf("unexpected load outcome: calls=%d errors=%d success=%d loaded=%d", calls, len(req.errCh), len(req.successCh), len(s.loaded))
	}
}

func TestORTSchedulerUsesNormalLifecycle(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	s := InitScheduler(ctx)
	mock := &mockLlm{modelPath: "ORT bundle"}
	s.newORTServerFn = func(requestCtx context.Context, _ ortrunner.ClientOptions) (llm.LlamaServer, error) {
		if requestCtx.Done() == nil {
			t.Fatal("request cancellation was not forwarded")
		}

		return mock, nil
	}
	req := &LlmRequest{
		ctx: ctx, model: &Model{ModelPath: "ORT bundle", Config: model.ConfigV2{ModelFormat: "ortgenai"}},
		opts: api.DefaultOptions(), errCh: make(chan error, 1), successCh: make(chan *runnerRef, 1),
		ortOptions: &ortrunner.ClientOptions{ModelDir: "ORT bundle", Provider: "cpu"},
	}
	if retry := s.load(req, ml.SystemInfo{}, nil, false); retry {
		t.Fatal("successful ORT load must not retry")
	}
	select {
	case runner := <-req.successCh:
		if runner.runner != manifest.RunnerORTGenAI || runner.refCount != 1 || runner.loading {
			t.Fatalf("invalid runner: %+v", runner)
		}
		runner.refMu.Lock()
		if !reflect.DeepEqual(runner.ortOptions, req.ortOptions) {
			t.Fatal("runner lost provider policy")
		}
		runner.refMu.Unlock()
		cancel()
		select {
		case <-s.finishedReqCh:
		case <-time.After(time.Second):
			t.Fatal("missing request completion event")
		}
	case err := <-req.errCh:
		t.Fatal(err)
	case <-time.After(time.Second):
		t.Fatal("ORT load did not complete")
	}
	s.unloadAllRunners()
	if !mock.closeCalled {
		t.Fatal("ORT runner was not closed")
	}
}
