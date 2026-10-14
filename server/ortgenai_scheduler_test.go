package server

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/types/model"
	"github.com/ollama/ollama/x/ortrunner"
)

func TestORTAutomaticContextUsesBundleCapacity(t *testing.T) {
	t.Setenv("OLLAMA_CONTEXT_LENGTH", "")
	dir := t.TempDir()
	config := `{"model":{"type":"llama","context_length":2048,"vocab_size":8,"decoder":{"filename":"model.onnx"}}}`
	if err := os.WriteFile(filepath.Join(dir, "genai_config.json"), []byte(config), 0o644); err != nil {
		t.Fatal(err)
	}
	m := &Model{ModelPath: dir, Config: model.ConfigV2{ModelFormat: "ortgenai"}}
	server := &Server{defaultNumCtx: 4096}
	opts, err := server.modelOptions(m, nil)
	if err != nil || opts.NumCtx != 2048 {
		t.Fatalf("automatic context = %d, %v", opts.NumCtx, err)
	}
	opts, err = server.modelOptions(m, map[string]any{"num_ctx": float64(8192)})
	if err != nil || opts.NumCtx != 8192 {
		t.Fatalf("explicit context was silently clamped: %d, %v", opts.NumCtx, err)
	}
	m.Options = map[string]any{"num_ctx": float64(1024)}
	opts, err = server.modelOptions(m, nil)
	if err != nil || opts.NumCtx != 1024 {
		t.Fatalf("model context = %d, %v", opts.NumCtx, err)
	}
}

func TestORTStartupDoesNotHoldLoadedMutex(t *testing.T) {
	s := InitScheduler(t.Context())
	started := make(chan struct{})
	s.newORTServerFn = func(ctx context.Context, _ ortrunner.ClientOptions) (llm.LlamaServer, error) {
		close(started)
		<-ctx.Done()
		return nil, ctx.Err()
	}
	req := &LlmRequest{
		ctx: t.Context(), model: &Model{ModelPath: "ORT bundle"},
		opts: api.DefaultOptions(), errCh: make(chan error, 1), successCh: make(chan *runnerRef, 1),
		ortOptions: &ortrunner.ClientOptions{ModelDir: "ORT bundle", Provider: "cpu"},
	}
	done := make(chan struct{})
	go func() {
		s.load(req, ml.SystemInfo{}, nil, false)
		close(done)
	}()
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("startup did not begin")
	}
	shutdown := make(chan struct{})
	go func() {
		s.unloadAllRunners()
		close(shutdown)
	}()
	select {
	case <-shutdown:
	case <-time.After(time.Second):
		t.Fatal("startup blocked loaded-runner access or shutdown")
	}
	select {
	case <-done:
		if !errors.Is(<-req.errCh, context.Canceled) {
			t.Fatal("shutdown did not cancel startup")
		}
	case <-time.After(time.Second):
		t.Fatal("startup did not stop on scheduler shutdown")
	}
}

func TestORTAutomaticResidencyLimit(t *testing.T) {
	t.Run("ORT first", func(t *testing.T) { testORTResidencyOrdering(t, false, "") })
	t.Run("native first", func(t *testing.T) { testORTResidencyOrdering(t, true, "") })
	t.Run("explicit maximum", func(t *testing.T) { testORTResidencyOrdering(t, false, "2") })
}

func testORTResidencyOrdering(t *testing.T, nativeFirst bool, maximum string) {
	t.Setenv("OLLAMA_MAX_LOADED_MODELS", maximum)
	t.Setenv("OLLAMA_ONNX_MODEL", "")
	t.Setenv("OLLAMA_ONNX_PROVIDER", "")
	t.Setenv("OLLAMA_ACCEL_MODE", "")
	t.Setenv("OLLAMA_ONNX_NPU", "")
	dir := t.TempDir()
	config := `{"model":{"type":"llama","context_length":128,"vocab_size":8,"decoder":{"filename":"model.onnx"}}}`
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	s := InitScheduler(ctx)
	s.waitForRecovery = 0
	loads := 0
	s.newORTServerFn = func(_ context.Context, opts ortrunner.ClientOptions) (llm.LlamaServer, error) {
		loads++
		return &mockLlm{modelPath: opts.ModelDir}, nil
	}
	s.getGpuFn = func(context.Context, []ml.FilteredRunnerDiscovery) []ml.DeviceInfo { return nil }
	s.getSystemInfoFn = func() ml.SystemInfo { return ml.SystemInfo{} }
	s.loadFn = func(req *LlmRequest, info ml.SystemInfo, gpus []ml.DeviceInfo, requireFull bool) bool {
		if req.ortOptions != nil {
			return s.load(req, info, gpus, requireFull)
		}
		req.successCh <- &runnerRef{}
		return false
	}
	s.Run(ctx)
	loadNative := func() {
		t.Helper()
		success, failure := s.getRunner(ctx, &Model{ModelPath: "native.gguf"}, api.DefaultOptions(), nil, true, true, nil)
		select {
		case <-success:
		case err := <-failure:
			t.Fatal(err)
		case <-time.After(2 * time.Second):
			t.Fatal("native load was not scheduled")
		}
	}
	if nativeFirst {
		loadNative()
	}
	for i := 0; i < 2; i++ {
		reqCtx, finish := context.WithCancel(ctx)
		m := &Model{ModelPath: filepath.Join(dir, strings.Repeat("b", i+1)), Config: model.ConfigV2{ModelFormat: "ortgenai"}}
		if err := os.Mkdir(m.ModelPath, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(m.ModelPath, "genai_config.json"), []byte(config), 0o644); err != nil {
			t.Fatal(err)
		}
		success, failure := s.getRunner(reqCtx, m, api.DefaultOptions(), &api.Duration{Duration: time.Hour}, true, true, nil)
		select {
		case <-success:
		case err := <-failure:
			t.Fatal(err)
		case <-time.After(2 * time.Second):
			t.Fatal("model did not load")
		}
		finish()
	}
	expectedResidents := 2
	if maximum == "" {
		loadNative()
		expectedResidents = 1
	}
	s.loadedMu.Lock()
	defer s.loadedMu.Unlock()
	if loads != 2 || len(s.loaded) != expectedResidents {
		t.Fatalf("loads=%d resident models=%d", loads, len(s.loaded))
	}
}
