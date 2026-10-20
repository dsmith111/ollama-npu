package ortrunner

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

type fakeServerBackend struct {
	generate   func(context.Context, Request) error
	tokenize   func(context.Context, string) ([]int32, error)
	detokenize func(context.Context, []int32) (string, error)
}

func (b *fakeServerBackend) Generate(ctx context.Context, req Request) error {
	if b.generate != nil {
		return b.generate(ctx, req)
	}
	return sendResponse(ctx, req.Responses, CompletionResponse{Content: "ok", Done: true, DoneReason: DoneReasonStop})
}
func (b *fakeServerBackend) Tokenize(ctx context.Context, content string) ([]int32, error) {
	if b.tokenize != nil {
		return b.tokenize(ctx, content)
	}
	return []int32{1, 2}, nil
}
func (b *fakeServerBackend) Detokenize(ctx context.Context, tokens []int32) (string, error) {
	if b.detokenize != nil {
		return b.detokenize(ctx, tokens)
	}
	return "decoded", nil
}
func (b *fakeServerBackend) ApplyChatTemplate(_ context.Context, messages string) (string, error) {
	return "rendered:" + messages, nil
}
func (b *fakeServerBackend) Status() StatusResponse {
	return StatusResponse{Status: 0, Progress: 100, ContextLength: 4096, Provider: "cpu", ProviderState: "portable-cpu", WatchdogRequired: true}
}

func testService(t *testing.T, backend *fakeServerBackend, capacity int) *runnerService {
	t.Helper()
	s := newRunnerService(context.Background(), backend, capacity)
	t.Cleanup(func() {
		s.stop()
		select {
		case <-s.done:
		case <-time.After(time.Second):
			t.Error("worker failed to stop")
		}
	})
	return s
}

func requestHandler(handler http.Handler, method, path, body, token string) *httptest.ResponseRecorder {
	req := httptest.NewRequest(method, path, strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	return rec
}

func TestHTTPStrictRequests(t *testing.T) {
	handler := testService(t, &fakeServerBackend{}, 4).handler("")
	for _, test := range []struct {
		path, body string
		status     int
	}{
		{"/completion", `{"prompt":"ok","options":{"temperature":0}}`, 200},
		{"/completion", `{"prompt":"ok","options":{"min_p":0.1}}`, 400},
		{"/completion", `{"prompt":"ok","options":{"repeat_last_n":64,"repeat_penalty":1.1}}`, 200},
		{"/completion", `{"prompt":"ok","options":{"repeat_last_n":-2}}`, 400},
		{"/completion", `{"prompt":"ok","request_id":"two words"}`, 400},
		{"/completion", `{"prompt":"ok","grammar":"json"}`, 400},
		{"/completion", `{"prompt":"ok","tools":[]}`, 400},
		{"/completion", `{"prompt":"ok","format":"json"}`, 400},
		{"/completion", `{"prompt":"ok","images":[]}`, 400},
		{"/completion", `{"prompt":"ok","options":{"temperature":-1}}`, 400},
		{"/completion", `{"prompt":"ok","options":{"temperature":null}}`, 400},
		{"/completion", `{"prompt":"ok","prompt":"other"}`, 400},
		{"/completion", `{"prompt":"ok","Prompt":"other"}`, 400},
		{"/completion", `{"prompt":"ok"} {}`, 400},
		{"/completion", `null`, 400},
		{"/completion", `[]`, 400},
		{"/completion", `{}`, 400},
		{"/completion", `{"prompt":"a\u0000b"}`, 400},
		{"/v1/tokenize", `{}`, 400},
		{"/v1/tokenize", `{"content":"ok","extra":1}`, 400},
		{"/v1/detokenize", `{"tokens":[-1]}`, 400},
		{"/v1/detokenize", `{"tokens":[2147483648]}`, 400},
		{"/v1/detokenize", `{"tokens":[1.1]}`, 400},
		{"/v1/detokenize", `{"tokens":null}`, 400},
		{"/v1/detokenize", `{}`, 400},
		{"/v1/detokenize", `{"tokens":[]}`, 200},
		{"/v1/chat_template", `{"messages":[{"role":"user","content":"hi"}]}`, 200},
		{"/v1/chat_template", `{"messages":[]}`, 400},
		{"/v1/chat_template", `{}`, 400},
		{"/v1/chat_template", `{"messages":[{"role":"tool","content":"hi"}]}`, 400},
		{"/v1/chat_template", `{"messages":[{"role":"user","content":"a\u0000b"}]}`, 400},
		{"/v1/chat_template", `{"messages":[{"role":"user","content":"hi"}],"tools":[]}`, 400},
		{"/load", `{"model":"ignored"}`, 400},
		{"/load", `{}`, 200},
		{"/completion", `{"prompt":"` + strings.Repeat("x", maxRequestBytes) + `"}`, 413},
	} {
		rec := requestHandler(handler, "POST", test.path, test.body, "")
		if rec.Code != test.status {
			t.Errorf("%s body=%s: status=%d want %d; response=%s", test.path, test.body[:min(len(test.body), 100)], rec.Code, test.status, rec.Body.String())
		}
	}
	rec := requestHandler(handler, "GET", "/completion", "", "")
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("GET completion status=%d", rec.Code)
	}
	req := httptest.NewRequest("POST", "/completion", strings.NewReader(`{"prompt":"ok"}`))
	req.Header.Set("Content-Type", "text/plain")
	rec = httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	if rec.Code != http.StatusUnsupportedMediaType {
		t.Fatal(rec.Code)
	}
}

func TestHTTPChatTemplateRendersMessages(t *testing.T) {
	handler := testService(t, &fakeServerBackend{}, 4).handler("")
	rec := requestHandler(handler, "POST", "/v1/chat_template",
		`{"messages":[{"role":"system","content":"be \"brief\""},{"role":"user","content":"hi"}]}`, "")
	var body struct {
		Prompt string `json:"prompt"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil || rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s err=%v", rec.Code, rec.Body.String(), err)
	}
	// The backend receives the messages re-encoded as a JSON array.
	if want := `rendered:[{"role":"system","content":"be \"brief\""},{"role":"user","content":"hi"}]`; body.Prompt != want {
		t.Fatalf("prompt = %s, want %s", body.Prompt, want)
	}
}

func TestModelMemorySplitsQNNFromCPUStages(t *testing.T) {
	dir := t.TempDir()
	for name, size := range map[string]int{
		"embeddings.onnx": 10, "lm_head.onnx": 20, "lm_head.onnx.data": 7,
		"context_ctx.onnx": 5, "context_ctx_qnn.bin": 100, "tokenizer.json": 1000,
	} {
		if err := os.WriteFile(filepath.Join(dir, name), make([]byte, size), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	qnn := map[string]map[string]json.RawMessage{"qnn": {}}
	meta := modelMetadata{stages: []modelStage{
		{name: "decoder", primary: true},
		{name: "embeddings", filename: "embeddings.onnx"},
		{name: "context", filename: "context_ctx.onnx", providers: qnn},
		{name: "lm_head", filename: "lm_head.onnx"},
	}}
	if total, npu := modelMemory(dir, meta, "qnn"); total != 142 || npu != 105 {
		t.Fatalf("qnn model memory = %d, %d; want 142 total with 105 on the NPU", total, npu)
	}
	if total, npu := modelMemory(dir, meta, "cpu"); total != 142 || npu != 0 {
		t.Fatalf("cpu model memory = %d, %d; want 142 total and none on the NPU", total, npu)
	}
}

func TestReadChatTemplate(t *testing.T) {
	dir := t.TempDir()
	if template, err := readChatTemplate(dir); err != nil || template != "" {
		t.Fatalf("missing chat_template.jinja = %q, %v; want empty", template, err)
	}
	path := filepath.Join(dir, "chat_template.jinja")
	if err := os.WriteFile(path, []byte("{{ messages }}"), 0o644); err != nil {
		t.Fatal(err)
	}
	if template, err := readChatTemplate(dir); err != nil || template != "{{ messages }}" {
		t.Fatalf("chat_template.jinja = %q, %v", template, err)
	}
	for _, bad := range [][]byte{[]byte("a\x00b"), []byte("\xff"), make([]byte, maxChatTemplateBytes+1)} {
		if err := os.WriteFile(path, bad, 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := readChatTemplate(dir); err == nil {
			t.Fatalf("accepted an invalid chat template of %d bytes", len(bad))
		}
	}
}

func TestHTTPBearerAuthAndCompatibility(t *testing.T) {
	handler := testService(t, &fakeServerBackend{}, 4).handler("secret")
	for _, token := range []string{"", "wrong"} {
		rec := requestHandler(handler, "GET", "/v1/status", "", token)
		if rec.Code != 401 {
			t.Fatalf("token=%q status=%d", token, rec.Code)
		}
	}
	for _, path := range []string{"/v1/status", "/health"} {
		rec := requestHandler(handler, "GET", path, "", "secret")
		var status StatusResponse
		if rec.Code != 200 || json.Unmarshal(rec.Body.Bytes(), &status) != nil || status.ContextLength != 4096 || status.Provider != "cpu" {
			t.Fatalf("status response=%s", rec.Body.String())
		}
	}
	req := httptest.NewRequest("GET", "/health", nil)
	req.Header.Add("Authorization", "Bearer secret")
	req.Header.Add("Authorization", "Bearer secret")
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)
	if rec.Code != 401 {
		t.Fatal("accepted duplicate authorization headers")
	}
	rec = requestHandler(handler, "POST", "/detokenize", `{"tokens":[1,2]}`, "secret")
	if rec.Code != 200 || rec.Body.String() != "{\"content\":\"decoded\"}\n" {
		t.Fatalf("detokenize=%s", rec.Body.String())
	}
	rec = requestHandler(handler, "POST", "/tokenize", `{"content":"ok"}`, "secret")
	if rec.Code != 200 || rec.Body.String() != "[1,2]\n" {
		t.Fatalf("tokenize=%s", rec.Body.String())
	}
}

func TestHTTPBoundedSerialQueue(t *testing.T) {
	entered := make(chan struct{})
	release := make(chan struct{})
	backend := &fakeServerBackend{generate: func(ctx context.Context, req Request) error {
		close(entered)
		<-release
		return sendResponse(ctx, req.Responses, CompletionResponse{Done: true})
	}}
	s := testService(t, backend, 1)
	handler := s.handler("")
	done := make(chan struct{})
	go func() { requestHandler(handler, "POST", "/completion", `{"prompt":"ok"}`, ""); close(done) }()
	<-entered
	if !s.enqueue(httptest.NewRecorder(), context.Background(), func() {}) {
		t.Fatal("failed to enqueue waiting job")
	}
	rec := requestHandler(handler, "POST", "/v1/tokenize", `{"content":"ok"}`, "")
	if rec.Code != 429 || rec.Header().Get("Retry-After") != "1" {
		t.Fatalf("backpressure status=%d", rec.Code)
	}
	close(release)
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("request did not finish")
	}
}

func TestHTTPQueuedCanceledWorkIsSkipped(t *testing.T) {
	s := testService(t, &fakeServerBackend{}, 1)
	entered, release := make(chan struct{}), make(chan struct{})
	s.enqueue(httptest.NewRecorder(), context.Background(), func() { close(entered); <-release })
	<-entered
	var called atomic.Bool
	s.backend.(*fakeServerBackend).tokenize = func(context.Context, string) ([]int32, error) {
		called.Store(true)
		return nil, nil
	}
	ctx, cancel := context.WithCancel(context.Background())
	req := httptest.NewRequest("POST", "/v1/tokenize", strings.NewReader(`{"content":"ok"}`)).WithContext(ctx)
	done := make(chan struct{})
	go func() { s.handler("").ServeHTTP(httptest.NewRecorder(), req); close(done) }()
	deadline := time.After(time.Second)
	for len(s.jobs) != 1 {
		select {
		case <-deadline:
			t.Fatal("request not queued")
		default:
			time.Sleep(time.Millisecond)
		}
	}
	cancel()
	<-done
	close(release)
	barrier := make(chan struct{})
	for !s.enqueue(httptest.NewRecorder(), context.Background(), func() { close(barrier) }) {
		time.Sleep(time.Millisecond)
	}
	<-barrier
	if called.Load() {
		t.Fatal("executed canceled queued native work")
	}
}

type failedResponseWriter struct{ header http.Header }

func (w failedResponseWriter) Header() http.Header       { return w.header }
func (w failedResponseWriter) WriteHeader(int)           {}
func (w failedResponseWriter) Write([]byte) (int, error) { return 0, errors.New("client disconnected") }
func (w failedResponseWriter) Flush()                    {}

func TestHTTPWriteFailureCancelsGeneration(t *testing.T) {
	canceled := make(chan struct{})
	s := testService(t, &fakeServerBackend{generate: func(ctx context.Context, req Request) error {
		if err := sendResponse(ctx, req.Responses, CompletionResponse{Content: "first"}); err != nil {
			return err
		}
		<-ctx.Done()
		close(canceled)
		return ctx.Err()
	}}, 1)
	req := httptest.NewRequest("POST", "/completion", strings.NewReader(`{"prompt":"ok"}`))
	s.handler("").ServeHTTP(failedResponseWriter{header: make(http.Header)}, req)
	select {
	case <-canceled:
	case <-time.After(time.Second):
		t.Fatal("writer failure did not cancel generation")
	}
}

func TestHTTPShutdownDoesNotFreeBlockedNativeWork(t *testing.T) {
	entered, release := make(chan struct{}), make(chan struct{})
	s := testService(t, &fakeServerBackend{}, 1)
	s.enqueue(httptest.NewRecorder(), context.Background(), func() { close(entered); <-release })
	<-entered
	rec := requestHandler(s.handler("secret"), "POST", "/v1/shutdown", "{}", "secret")
	if rec.Code != 200 {
		t.Fatalf("shutdown=%d", rec.Code)
	}
	select {
	case <-s.done:
		t.Fatal("worker reported completion before native return")
	default:
	}
	if s.enqueue(httptest.NewRecorder(), context.Background(), func() {}) {
		t.Fatal("accepted work after shutdown")
	}
	close(release)
	<-s.done
}

func TestHTTPServerLimits(t *testing.T) {
	server := newHTTPServer(context.Background(), http.NotFoundHandler())
	if server.MaxHeaderBytes != maxHeaderBytes || server.ReadHeaderTimeout <= 0 || server.ReadTimeout <= 0 || server.WriteTimeout <= 0 || server.IdleTimeout <= 0 {
		t.Fatalf("server is missing limits: %+v", server)
	}
}
