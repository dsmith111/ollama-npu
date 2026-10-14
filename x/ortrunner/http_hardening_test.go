package ortrunner

import (
	"context"
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestHTTPActiveRequestGenerationIdentity(t *testing.T) {
	entered := make(chan string, 2)
	release := make(chan struct{}, 2)
	backend := &fakeServerBackend{generate: func(ctx context.Context, req Request) error {
		entered <- req.RequestID
		<-release
		return sendResponse(ctx, req.Responses, CompletionResponse{Done: true})
	}}
	s := testService(t, backend, 2)
	handler := s.handler("secret")
	t.Cleanup(func() { release <- struct{}{} })
	var previous uint64
	for _, id := range []string{"canceled-old-request", "unrelated-future-request"} {
		ctx, cancel := context.WithCancel(context.Background())
		req := httptest.NewRequest("POST", "/v1/completions", strings.NewReader(`{"prompt":"ok","request_id":"`+id+`"}`)).WithContext(ctx)
		req.Header.Set("Authorization", "Bearer secret")
		done := make(chan struct{})
		go func() { handler.ServeHTTP(httptest.NewRecorder(), req); close(done) }()
		if got := <-entered; got != id {
			t.Fatal(got)
		}
		rec := requestHandler(handler, "GET", "/v1/status", "", "secret")
		var status StatusResponse
		if err := json.Unmarshal(rec.Body.Bytes(), &status); err != nil {
			t.Fatal(err)
		}
		if status.ProtocolVersion != ProtocolVersion || !status.Busy || status.ActiveRequestID != id || status.ActiveOperation != "completion" || status.ActiveGeneration <= previous {
			t.Fatalf("incorrect active identity: %+v", status)
		}
		previous = status.ActiveGeneration
		cancel()
		select {
		case <-done:
		case <-time.After(time.Second):
			t.Fatal("HTTP request failed to cancel while native work was blocked")
		}
		if status := s.status(); !status.Busy || status.ActiveRequestID != id {
			t.Fatalf("blocked canceled native work lost its identity: %+v", status)
		}
		release <- struct{}{}
		barrier := make(chan struct{})
		if !s.enqueue(httptest.NewRecorder(), context.Background(), func() { close(barrier) }) {
			t.Fatal("could not enqueue completion barrier")
		}
		<-barrier
	}
}

func TestReadyResponseWireVersion(t *testing.T) {
	s := testService(t, &fakeServerBackend{}, 1)
	data, err := json.Marshal(ReadyResponse{Event: "ready", Port: 49123, StatusResponse: s.status()})
	if err != nil {
		t.Fatal(err)
	}
	var ready ReadyResponse
	if err := json.Unmarshal(data, &ready); err != nil || ready.Event != "ready" || ready.Port != 49123 || ready.ProtocolVersion != ProtocolVersion {
		t.Fatalf("ready=%+v err=%v", ready, err)
	}
}
