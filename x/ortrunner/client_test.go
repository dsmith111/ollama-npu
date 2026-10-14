package ortrunner

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
)

func testHTTPClient(t *testing.T, handler http.HandlerFunc) *Client {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	_, port, err := net.SplitHostPort(strings.TrimPrefix(server.URL, "http://"))
	if err != nil {
		t.Fatal(err)
	}
	n, err := strconv.Atoi(port)
	if err != nil {
		t.Fatal(err)
	}
	c := &Client{
		port: n, token: "secret", client: server.Client(), done: make(chan struct{}),
		closeDone: make(chan struct{}), operations: make(chan struct{}, 1),
	}
	c.contextLength.Store(512)
	return c
}

func TestClientCompletionForwardsOptions(t *testing.T) {
	client := testHTTPClient(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer secret" {
			t.Error("missing child authentication")
		}
		var request TextCompletionsRequest
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatal(err)
		}
		if request.Prompt != "hello" || request.Options.Temperature == nil || *request.Options.Temperature != 0 ||
			request.Options.Seed == nil || *request.Options.Seed != 42 ||
			request.Options.NumPredict == nil || *request.Options.NumPredict != -1 ||
			len(request.Options.Stop) != 1 || request.Options.Stop[0] != "STOP" {
			t.Errorf("lost options: %+v", request)
		}
		fmt.Fprintln(w, `{"content":"hi","done":false}`)
		fmt.Fprintln(w, `{"done":true,"done_reason":1,"eval_count":2}`)
	})
	options := api.DefaultOptions()
	options.NumCtx = 512
	options.Temperature = 0
	options.Seed = 42
	options.Stop = []string{"STOP"}
	var responses []llm.CompletionResponse
	err := client.Completion(context.Background(), llm.CompletionRequest{Prompt: "hello", Options: &options},
		func(response llm.CompletionResponse) { responses = append(responses, response) })
	if err != nil || len(responses) != 2 || responses[1].DoneReason != llm.DoneReasonLength {
		t.Fatalf("responses=%+v err=%v", responses, err)
	}
}

func TestClientRejectsUnsupportedRequests(t *testing.T) {
	client := testHTTPClient(t, func(w http.ResponseWriter, r *http.Request) {
		t.Error("unsupported request reached the child")
	})
	for _, request := range []llm.CompletionRequest{
		{Media: []llm.MediaData{{Data: []byte{1}}}},
		{Format: json.RawMessage(`"json"`)},
		{Logprobs: true},
		{TopLogprobs: 1},
	} {
		if err := client.Completion(context.Background(), request, func(llm.CompletionResponse) {}); err == nil {
			t.Errorf("accepted %+v", request)
		}
	}
}

func TestClientTerminalCallbackCanTokenize(t *testing.T) {
	client := testHTTPClient(t, func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/completion":
			fmt.Fprintln(w, `{"content":"hi","done":false}`)
			fmt.Fprintln(w, `{"done":true,"done_reason":0}`)
		case "/v1/tokenize":
			fmt.Fprintln(w, `[1,2,3]`)
		default:
			t.Errorf("unexpected request %s", r.URL.Path)
		}
	})
	ctx, cancel := context.WithTimeout(t.Context(), time.Second)
	defer cancel()
	var tokenErr error
	var tokens []int
	err := client.Completion(ctx, llm.CompletionRequest{Prompt: "hello"}, func(response llm.CompletionResponse) {
		if response.Done {
			tokens, tokenErr = client.Tokenize(ctx, "hello"+response.Content)
		}
	})
	if err != nil || tokenErr != nil || len(tokens) != 3 {
		t.Fatalf("terminal callback blocked tokenization: completion=%v tokenization=%v tokens=%v", err, tokenErr, tokens)
	}
	if len(client.operations) != 0 {
		t.Fatal("operation lock retained after terminal callback")
	}
}

func TestClientRequiresTerminalStream(t *testing.T) {
	for _, test := range []struct {
		name string
		body string
		ok   bool
	}{
		{"empty", "", false},
		{"truncated", `{"content":"partial","done":false}` + "\n", false},
		{"malformed", "not json\n", false},
		{"negative metrics", `{"done":true,"eval_count":-1}`, false},
		{"invalid reason", `{"done":true,"done_reason":99}`, false},
		{"backend error", `{"error":{"StatusCode":500,"error":"failed"}}`, false},
		{"terminal", `{"done":true,"done_reason":0}`, true},
		{"duplicate terminal", "{\"done\":true}\n{\"done\":true}\n", false},
		{"data after terminal", "{\"done\":true}\n{\"content\":\"extra\"}\n", false},
		{"invalid data after terminal", "{\"done\":true}\ninvalid\n", false},
		{"large fragment", `{"content":"` + strings.Repeat("a", 128<<10) + `","done":true}`, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			terminalCalls := 0
			err := readCompletionStream(strings.NewReader(test.body), func(response llm.CompletionResponse) {
				if response.Done {
					terminalCalls++
				}
			})
			if (err == nil) != test.ok {
				t.Errorf("err=%v", err)
			}
			if !test.ok && terminalCalls != 0 {
				t.Error("corrupt stream published a terminal success")
			}
		})
	}
}

func TestClientTokenOperations(t *testing.T) {
	client := testHTTPClient(t, func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/tokenize":
			fmt.Fprintln(w, `[1,2,3]`)
		case "/v1/detokenize":
			fmt.Fprintln(w, `{"content":"hello"}`)
		}
	})
	tokens, err := client.Tokenize(context.Background(), "hello")
	if err != nil || len(tokens) != 3 {
		t.Fatalf("tokens=%v err=%v", tokens, err)
	}
	content, err := client.Detokenize(context.Background(), tokens)
	if err != nil || content != "hello" {
		t.Fatalf("content=%q err=%v", content, err)
	}
	for _, token := range []int{-1, int(math.MaxInt32) + 1} {
		if _, err := client.Detokenize(context.Background(), []int{token}); err == nil {
			t.Errorf("accepted token %d", token)
		}
	}
}

func TestClientTokenizeChecksHTTPStatus(t *testing.T) {
	client := testHTTPClient(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprintln(w, `{"error":"invalid tokenizer input"}`)
	})
	_, err := client.Tokenize(context.Background(), "hello")
	var status api.StatusError
	if !errors.As(err, &status) || status.StatusCode != 400 {
		t.Fatalf("expected status error, got %v", err)
	}
}

func TestClientExitIsPersistent(t *testing.T) {
	c := &Client{done: make(chan struct{})}
	close(c.done)
	for i := 0; i < 4; i++ {
		if !c.HasExited() {
			t.Fatal("lost child exit state")
		}
	}
}

func TestRunnerStderrKeepsBoundedTail(t *testing.T) {
	client := &Client{}
	writer := runnerStderr{client}
	for _, text := range []string{strings.Repeat("x", 2*maxRunnerErrorBytes), "error split ", "across writes\n"} {
		n, err := writer.Write([]byte(text))
		if err != nil || n != len(text) {
			t.Fatalf("stderr write=%d, error=%v", n, err)
		}
	}
	if len(client.lastErr) != maxRunnerErrorBytes || !strings.HasSuffix(client.lastErr, "error split across writes\n") {
		t.Fatalf("lost bounded error tail: length=%d", len(client.lastErr))
	}
}

func TestClientReadinessValidation(t *testing.T) {
	base := clientReadiness{
		Event: "ready", Port: 1234, ProtocolVersion: clientProtocolVersion,
		StatusResponse: StatusResponse{ContextLength: 512, Provider: "qnn"},
	}
	if err := validateReadiness(base); err != nil {
		t.Fatal(err)
	}
	for _, mutate := range []func(*clientReadiness){
		func(e *clientReadiness) { e.ProtocolVersion = 99 },
		func(e *clientReadiness) { e.Port = 0 },
		func(e *clientReadiness) { e.ContextLength = 0 },
		func(e *clientReadiness) { e.Provider = "unknown" },
		func(e *clientReadiness) { e.Status = 5 },
	} {
		changed := base
		mutate(&changed)
		if err := validateReadiness(changed); err == nil {
			t.Errorf("accepted %+v", changed)
		}
	}
}

func TestChildEnvironmentReplacesOverrides(t *testing.T) {
	env := []string{"A=one", "OLLAMA_ONNX_PROVIDER=cpu", "OLLAMA_ONNX_PROVIDER=dml"}
	env = setChildEnv(env, "OLLAMA_ONNX_PROVIDER", "qnn")
	count := 0
	for _, entry := range env {
		if strings.HasPrefix(entry, "OLLAMA_ONNX_PROVIDER=") {
			count++
			if entry != "OLLAMA_ONNX_PROVIDER=qnn" {
				t.Error(entry)
			}
		}
	}
	if count != 1 {
		t.Fatalf("duplicate provider overrides: %v", env)
	}
}

func TestClientChildLifecycle(t *testing.T) {
	cmd := exec.Command(os.Args[0], "-test.run=^TestORTClientHelper$")
	cmd.Env = append(os.Environ(), "OLLAMA_TEST_ORT_HELPER=ready")
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	c, err := startClient(ctx, ClientOptions{ModelDir: t.TempDir(), Provider: "qnn"}, cmd)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { c.Close() })
	if c.ContextLength() != 512 || c.Pid() <= 0 {
		t.Fatalf("invalid child metadata context=%d pid=%d", c.ContextLength(), c.Pid())
	}
	if err := c.Completion(ctx, llm.CompletionRequest{Prompt: "test"}, func(llm.CompletionResponse) {}); err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	for i := 0; i < 3; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := c.Close(); err != nil {
				t.Error(err)
			}
		}()
	}
	wg.Wait()
	if !c.HasExited() || !c.HasExited() {
		t.Fatal("child remains alive")
	}
}

func TestClientStartupCancellation(t *testing.T) {
	cmd := exec.Command(os.Args[0], "-test.run=^TestORTClientHelper$")
	cmd.Env = append(os.Environ(), "OLLAMA_TEST_ORT_HELPER=hang")
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	start := time.Now()
	_, err := startClient(ctx, ClientOptions{ModelDir: t.TempDir()}, cmd)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("expected startup cancellation, got %v", err)
	}
	if time.Since(start) > 2*cancellationGrace {
		t.Fatal("startup cancellation was not bounded")
	}
}

func TestClientStartupReportsLoadFailure(t *testing.T) {
	cmd := exec.Command(os.Args[0], "-test.run=^TestORTClientHelper$")
	cmd.Env = append(os.Environ(), "OLLAMA_TEST_ORT_HELPER=load-error")
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancel()
	client, err := startClient(ctx, ClientOptions{ModelDir: t.TempDir()}, cmd)
	if client != nil || err == nil || !strings.Contains(err.Error(), "NPU_MODEL_PATH_UNSUPPORTED") {
		t.Fatalf("startup hid the actionable load failure: client=%v error=%v", client, err)
	}
}

func TestClientCancellationRetiresChild(t *testing.T) {
	cmd := exec.Command(os.Args[0], "-test.run=^TestORTClientHelper$")
	cmd.Env = append(os.Environ(), "OLLAMA_TEST_ORT_HELPER=blocked")
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	c, err := startClient(ctx, ClientOptions{ModelDir: t.TempDir()}, cmd)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { c.Close() })
	reqCtx, reqCancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer reqCancel()
	err = c.Completion(reqCtx, llm.CompletionRequest{Prompt: "test"}, func(llm.CompletionResponse) {})
	if !errors.Is(err, context.DeadlineExceeded) || !c.HasExited() {
		t.Fatalf("err=%v exited=%v", err, c.HasExited())
	}
}

func TestORTClientHelper(t *testing.T) {
	mode := os.Getenv("OLLAMA_TEST_ORT_HELPER")
	if mode == "" {
		return
	}
	if mode == "load-error" {
		fmt.Fprintln(os.Stderr, "Error: NPU_MODEL_PATH_UNSUPPORTED: no usable ASCII alias")
		os.Stdout.Close()
		time.Sleep(100 * time.Millisecond)
		os.Exit(1)
	}
	if mode == "hang" {
		for {
			time.Sleep(time.Hour)
		}
	}
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		os.Exit(2)
	}
	status := StatusResponse{ContextLength: 512, Provider: "qnn"}
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/status", func(w http.ResponseWriter, r *http.Request) { json.NewEncoder(w).Encode(status) })
	mux.HandleFunc("/completion", func(w http.ResponseWriter, r *http.Request) {
		if mode == "blocked" {
			<-r.Context().Done()
			return
		}
		fmt.Fprintln(w, `{"done":true,"eval_count":1}`)
	})
	mux.HandleFunc("/v1/shutdown", func(w http.ResponseWriter, r *http.Request) {
		io.WriteString(w, "{}")
		http.NewResponseController(w).Flush()
		go func() {
			time.Sleep(10 * time.Millisecond)
			os.Exit(0)
		}()
	})
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer "+os.Getenv("OLLAMA_ORT_RUNNER_TOKEN") {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		mux.ServeHTTP(w, r)
	})
	fmt.Fprintln(os.Stdout, strings.Repeat("native log ", 10<<10))
	json.NewEncoder(os.Stdout).Encode(clientReadiness{
		Event: "ready", Port: listener.Addr().(*net.TCPAddr).Port,
		ProtocolVersion: clientProtocolVersion, StatusResponse: status,
	})
	server := &http.Server{Handler: handler}
	server.Serve(listener)
	os.Exit(0)
}

func TestProcessGuardTerminatesChild(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("kill-on-close job behavior is Windows-specific")
	}
	cmd := exec.Command(os.Args[0], "-test.run=^TestORTClientHelper$")
	cmd.Env = append(os.Environ(), "OLLAMA_TEST_ORT_HELPER=hang")
	guard, err := newProcessGuard(cmd)
	if err != nil {
		t.Fatal(err)
	}
	if err := cmd.Start(); err != nil {
		guard.Close()
		t.Fatal(err)
	}
	t.Cleanup(func() { cmd.Process.Kill() })
	if err := guard.Attach(cmd.Process); err != nil {
		cmd.Process.Kill()
		cmd.Wait()
		guard.Close()
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	if err := guard.Close(); err != nil {
		t.Fatal(err)
	}
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("job close left child alive")
	}
}
