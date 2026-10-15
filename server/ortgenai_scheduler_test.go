package server

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
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

type queuedORTServer struct {
	mockLlm
	pid         int
	stopped     atomic.Bool
	idle        chan struct{}
	waitStarted chan struct{}
}

func (s *queuedORTServer) Pid() int { return s.pid }

func (s *queuedORTServer) Ping(context.Context) error {
	if s.stopped.Load() {
		return errors.New("retired ORT child")
	}
	return nil
}

func (s *queuedORTServer) Close() error {
	s.stopped.Store(true)
	return nil
}

func (s *queuedORTServer) WaitUntilIdle(ctx context.Context) error {
	select {
	case s.waitStarted <- struct{}{}:
	default:
	}
	select {
	case <-s.idle:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func TestORTQueuedRequestsWaitForOperationAndRecover(t *testing.T) {
	for _, retire := range []bool{false, true} {
		t.Run(fmt.Sprintf("retired=%t", retire), func(t *testing.T) {
			t.Setenv("OLLAMA_ONNX_MODEL", "")
			t.Setenv("OLLAMA_ONNX_PROVIDER", "")
			t.Setenv("OLLAMA_ACCEL_MODE", "")
			t.Setenv("OLLAMA_ONNX_NPU", "")
			dir := t.TempDir()
			if err := os.WriteFile(filepath.Join(dir, "genai_config.json"), []byte(`{"model":{"type":"llama","context_length":128,"vocab_size":8,"decoder":{"filename":"model.onnx"}}}`), 0o644); err != nil {
				t.Fatal(err)
			}
			m := &Model{ModelPath: dir, Config: model.ConfigV2{ModelFormat: "ortgenai"}}
			ctx, stop := context.WithTimeout(t.Context(), 5*time.Second)
			defer stop()
			s := InitScheduler(ctx)
			var loads atomic.Int32
			old := &queuedORTServer{pid: 100, idle: make(chan struct{}), waitStarted: make(chan struct{}, 1)}
			next := &queuedORTServer{pid: 101, idle: make(chan struct{}), waitStarted: make(chan struct{}, 1)}
			close(next.idle)
			s.newORTServerFn = func(context.Context, ortrunner.ClientOptions) (llm.LlamaServer, error) {
				if loads.Add(1) == 1 {
					return old, nil
				}
				return next, nil
			}
			s.Run(ctx)
			options := api.DefaultOptions()
			options.NumCtx = 128
			duration := &api.Duration{Duration: time.Hour}
			active, finish := context.WithCancel(ctx)
			defer finish()
			success, failure := s.getRunner(active, m, options, duration, false, false, nil)
			select {
			case <-success:
			case err := <-failure:
				t.Fatal(err)
			case <-ctx.Done():
				t.Fatal(ctx.Err())
			}
			type waiting struct {
				success chan *runnerRef
				failure chan error
				finish  context.CancelFunc
			}
			var requests []waiting
			for range 3 {
				reqCtx, done := context.WithCancel(ctx)
				defer done()
				success, failure := s.getRunner(reqCtx, m, options, duration, false, false, nil)
				select {
				case <-success:
					t.Fatal("waiting request was assigned to a busy child")
				default:
				}
				requests = append(requests, waiting{success, failure, done})
			}
			finish()
			select {
			case <-old.waitStarted:
			case <-ctx.Done():
				t.Fatal("finished reference did not wait for the active operation")
			}
			for _, req := range requests {
				select {
				case <-req.success:
					t.Fatal("waiting request was assigned before the cancelled operation drained")
				default:
				}
			}
			old.stopped.Store(retire)
			close(old.idle)
			expected := llm.LlamaServer(old)
			if retire {
				expected = next
			}
			for _, req := range requests {
				select {
				case runner := <-req.success:
					if runner.llama != expected {
						t.Fatal("queued request used the wrong runner generation")
					}
					req.finish()
				case err := <-req.failure:
					t.Fatal(err)
				case <-ctx.Done():
					t.Fatal(ctx.Err())
				}
			}
			expectedLoads := int32(1)
			if retire {
				expectedLoads++
			}
			if loads.Load() != expectedLoads {
				t.Fatalf("loads = %d, want %d", loads.Load(), expectedLoads)
			}
		})
	}
}
func TestORTQueuedRequestDeadlineAndCapacity(t *testing.T) {
	t.Setenv("OLLAMA_MAX_QUEUE", "1")
	t.Setenv("OLLAMA_ONNX_MODEL", "")
	t.Setenv("OLLAMA_ONNX_PROVIDER", "")
	t.Setenv("OLLAMA_ACCEL_MODE", "")
	t.Setenv("OLLAMA_ONNX_NPU", "")
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "genai_config.json"), []byte(`{"model":{"type":"llama","context_length":128,"vocab_size":8,"decoder":{"filename":"model.onnx"}}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	m := &Model{ModelPath: dir, Config: model.ConfigV2{ModelFormat: "ortgenai"}}
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	s := InitScheduler(ctx)
	opts := api.DefaultOptions()
	opts.NumCtx = 128
	ortOptions, err := resolveORTOptions(m, opts)
	if err != nil {
		t.Fatal(err)
	}
	ref := &runnerRef{
		model: m, modelKey: schedulerModelKey(m), Options: &opts, ortOptions: ortOptions,
		llama: &mockLlm{}, refCount: 1, idleCh: make(chan struct{}), contextShift: true,
	}
	s.loaded[ref.modelKey] = ref
	// Before the worker runs, a busy model consumes the normal bounded queue.
	waitCtx, stopWait := context.WithCancel(ctx)
	defer stopWait()
	success, failure := s.getRunner(waitCtx, m, opts, nil, false, false, nil)
	if len(success) != 0 || len(failure) != 0 || len(s.pendingReqCh) != 1 {
		t.Fatal("busy ORT request bypassed the bounded pending queue")
	}
	_, failure = s.getRunner(ctx, m, opts, nil, false, false, nil)
	select {
	case err := <-failure:
		if !errors.Is(err, ErrMaxQueue) {
			t.Fatal(err)
		}
	default:
		t.Fatal("overflow request was not rejected")
	}
	stopWait()
	s.Run(ctx)
	until := time.After(time.Second)
	for len(s.pendingReqCh) != 0 {
		select {
		case <-until:
			t.Fatal("cancelled queued request was not removed")
		case <-time.After(time.Millisecond):
		}
	}
	deadlineCtx, stopDeadline := context.WithTimeout(ctx, 50*time.Millisecond)
	defer stopDeadline()
	server := &Server{sched: s, defaultNumCtx: 128}
	done := make(chan error, 1)
	go func() {
		_, _, _, err := server.scheduleRunnerForModel(deadlineCtx, m, nil, nil, nil, nil)
		done <- err
	}()
	select {
	case err := <-done:
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Fatal(err)
		}
	case <-time.After(time.Second):
		t.Fatal("cancelled queued route failed to return")
	}
}

func TestSchedulerFinishRetainsAssignedGeneration(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	s := InitScheduler(ctx)
	opts := api.DefaultOptions()
	m := &Model{ModelPath: "same model"}
	old := &runnerRef{model: m, modelKey: schedulerModelKey(m), Options: &opts, refCount: 1, sessionDuration: time.Hour}
	replacement := &runnerRef{model: m, modelKey: old.modelKey, Options: &opts, refCount: 1}
	s.loaded[old.modelKey] = replacement
	req := &LlmRequest{model: m, assigned: old}
	go s.processCompleted(ctx)
	s.finishedReqCh <- req
	deadline := time.After(time.Second)
	for {
		old.refMu.Lock()
		finished := req.released
		old.refMu.Unlock()
		if finished {
			break
		}
		select {
		case <-deadline:
			t.Fatal("old generation was not released")
		case <-time.After(time.Millisecond):
		}
	}
	s.finishedReqCh <- req
	replacement.refMu.Lock()
	defer replacement.refMu.Unlock()
	if replacement.refCount != 1 {
		t.Fatal("finished old request decremented the replacement reference")
	}
	old.refMu.Lock()
	defer old.refMu.Unlock()
	if old.refCount != 0 {
		t.Fatal("old generation reference was not released exactly once")
	}
	if old.expireTimer != nil {
		old.expireTimer.Stop()
	}
}

func TestSchedulerExpirationRetainsReplacementWithSamePID(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	s := InitScheduler(ctx)
	m := &Model{ModelPath: "same model"}
	previous := &queuedORTServer{pid: 100}
	current := &queuedORTServer{pid: 100}
	old := &runnerRef{model: m, modelKey: schedulerModelKey(m), llama: previous, pid: 100}
	replacement := &runnerRef{model: m, modelKey: old.modelKey, llama: current, pid: 100}
	s.loaded[old.modelKey] = replacement
	go s.processCompleted(ctx)
	s.expiredCh <- old
	deadline := time.After(time.Second)
	for !previous.stopped.Load() {
		select {
		case <-deadline:
			t.Fatal("stale expiration did not close the orphaned generation")
		case <-time.After(time.Millisecond):
		}
	}
	s.loadedMu.Lock()
	defer s.loadedMu.Unlock()
	if s.loaded[replacement.modelKey] != replacement || current.stopped.Load() {
		t.Fatal("stale expiration unloaded a replacement with the same process ID")
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
