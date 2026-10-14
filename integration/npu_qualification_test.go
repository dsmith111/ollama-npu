//go:build integration

package integration

import (
	"bytes"
	"context"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/internal/modelref"
	"github.com/ollama/ollama/template"
	"github.com/ollama/ollama/types/model"
	"github.com/ollama/ollama/version"
)

const (
	npuRequestTimeout = 2 * time.Minute
	npuResponseLimit  = 1 << 20
)

var npuPrompts = []struct {
	name, text, want string
}{
	{"arithmetic", "What is 2 + 2? Reply with only the number.", "4"},
	{"capital", "What is the capital of France? Reply with only the city name.", "Paris"},
	{"color", "What color is a clear daytime sky? Reply with exactly one word in uppercase.", "BLUE"},
}

type npuQualificationConfig struct {
	endpoint           *url.URL
	npuModel, cpuModel string
	requests           int
}

func npuEndpoint(s string) (*url.URL, error) {
	if !strings.Contains(s, "://") {
		s = "http://" + s
	}
	u, err := url.Parse(s)
	if err != nil {
		return nil, fmt.Errorf("invalid OLLAMA_HOST: %w", err)
	}
	if (u.Scheme != "http" && u.Scheme != "https") || u.User != nil ||
		(u.Path != "" && u.Path != "/") || u.RawQuery != "" || u.ForceQuery || u.Fragment != "" || u.Opaque != "" {
		return nil, fmt.Errorf("OLLAMA_HOST must be an http(s) loopback origin without credentials, path, query or fragment")
	}
	switch strings.ToLower(u.Hostname()) {
	case "localhost", "127.0.0.1", "::1":
	default:
		return nil, fmt.Errorf("OLLAMA_HOST must use localhost, 127.0.0.1 or [::1], not %q", u.Hostname())
	}
	port, err := strconv.Atoi(u.Port())
	if err != nil || port < 1 || port > 65535 || port == 11434 {
		return nil, fmt.Errorf("OLLAMA_HOST requires an explicit nondefault port (1..65535, not 11434)")
	}
	u.Path = ""
	return u, nil
}

func npuConfig(lookup func(string) (string, bool)) (npuQualificationConfig, bool, error) {
	var c npuQualificationConfig
	enabled := false
	for _, key := range []string{"OLLAMA_TEST_NPU_MODEL", "OLLAMA_TEST_CPU_MODEL", "OLLAMA_TEST_NPU_REQUESTS"} {
		if _, ok := lookup(key); ok {
			enabled = true
		}
	}
	if !enabled {
		return c, false, nil
	}
	values := make(map[string]string)
	for _, key := range []string{"OLLAMA_HOST", "OLLAMA_TEST_NPU_MODEL", "OLLAMA_TEST_CPU_MODEL"} {
		v, ok := lookup(key)
		if !ok || v == "" || strings.TrimSpace(v) != v {
			return c, true, fmt.Errorf("%s must be explicitly set and nonempty, without surrounding whitespace", key)
		}
		values[key] = v
	}
	var err error
	c.endpoint, err = npuEndpoint(values["OLLAMA_HOST"])
	if err != nil {
		return c, true, err
	}
	c.npuModel, c.cpuModel = values["OLLAMA_TEST_NPU_MODEL"], values["OLLAMA_TEST_CPU_MODEL"]
	for _, tag := range []string{c.npuModel, c.cpuModel} {
		ref, err := modelref.ParseRef(tag)
		if err != nil || ref.Source != modelref.ModelSourceUnspecified ||
			!model.ParseName(tag).IsValid() || model.ParseNameBare(tag).Tag == "" || strings.Contains(tag, "://") {
			return c, true, fmt.Errorf("model %q must be an explicit valid existing model:tag", tag)
		}
	}
	if !npuSameModel(c.cpuModel, "phi3:mini") {
		return c, true, fmt.Errorf("OLLAMA_TEST_CPU_MODEL must select the ordinary phi3:mini GGUF for this profile")
	}
	if npuSameModel(c.npuModel, c.cpuModel) {
		return c, true, fmt.Errorf("NPU and CPU model tags must differ")
	}
	c.requests = 20
	if s, ok := lookup("OLLAMA_TEST_NPU_REQUESTS"); ok {
		c.requests, err = strconv.Atoi(s)
		if err != nil || c.requests < 20 || c.requests > 1000 {
			return c, true, fmt.Errorf("OLLAMA_TEST_NPU_REQUESTS must be an integer from 20 through 1000")
		}
	}
	return c, true, nil
}

func npuSameModel(a, b string) bool {
	x, y := model.ParseName(a), model.ParseName(b)
	return x.IsValid() && y.IsValid() && x.EqualFold(y)
}

func npuRaw(text string) string {
	return "<|user|>\n" + text + "<|end|>\n<|assistant|>\n"
}

func npuValidateTemplate(show api.ShowResponse) error {
	if show.Template == "" || show.System != "" || len(show.Messages) != 0 || show.Renderer != "" {
		return fmt.Errorf("profile requires an attached Ollama template, no default system/messages and no alternate renderer")
	}
	tmpl, err := template.Parse(show.Template)
	if err != nil {
		return fmt.Errorf("parse attached template: %w", err)
	}
	for _, p := range npuPrompts {
		var rendered bytes.Buffer
		if err := tmpl.Execute(&rendered, template.Values{Messages: []api.Message{{Role: "user", Content: p.text}}}); err != nil {
			return fmt.Errorf("render attached template: %w", err)
		}
		if rendered.String() != npuRaw(p.text) {
			return fmt.Errorf("attached template renders %q, want exact profile prompt %q; fix it separately, not in this test", rendered.String(), npuRaw(p.text))
		}
	}
	return nil
}

type npuHTTP struct {
	base   *url.URL
	client *http.Client
}

func npuNewHTTP(u *url.URL) *npuHTTP {
	dialer := &net.Dialer{Timeout: 5 * time.Second}
	transport := &http.Transport{
		// No environment proxy; localhost is pinned rather than resolved by DNS.
		DialContext: func(ctx context.Context, network, address string) (net.Conn, error) {
			host, port, err := net.SplitHostPort(address)
			if err != nil {
				return nil, err
			}
			if strings.EqualFold(host, "localhost") {
				host = "127.0.0.1"
			}
			if host != "127.0.0.1" && host != "::1" {
				return nil, fmt.Errorf("refusing non-loopback dial %q", address)
			}
			return dialer.DialContext(ctx, network, net.JoinHostPort(host, port))
		},
		TLSHandshakeTimeout:   5 * time.Second,
		ResponseHeaderTimeout: npuRequestTimeout,
	}
	return &npuHTTP{base: u, client: &http.Client{
		Transport: transport,
		Timeout:   npuRequestTimeout,
		CheckRedirect: func(*http.Request, []*http.Request) error {
			return errors.New("qualification forbids redirects")
		},
	}}
}

func (h *npuHTTP) open(ctx context.Context, path string, payload any) (*http.Response, error) {
	method := http.MethodGet
	var body io.Reader
	if payload != nil {
		data, err := json.Marshal(payload)
		if err != nil {
			return nil, err
		}
		body, method = bytes.NewReader(data), http.MethodPost
	}
	req, err := http.NewRequestWithContext(ctx, method, h.base.JoinPath(path).String(), body)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := h.client.Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		data, err := io.ReadAll(io.LimitReader(resp.Body, npuResponseLimit+1))
		if err != nil {
			return nil, err
		}
		var rejection struct{ Error string }
		if len(data) > npuResponseLimit || json.Unmarshal(data, &rejection) != nil || rejection.Error == "" {
			return nil, fmt.Errorf("%s: HTTP %d without a bounded JSON error", path, resp.StatusCode)
		}
		return nil, api.StatusError{StatusCode: resp.StatusCode, ErrorMessage: rejection.Error}
	}
	return resp, nil
}

func (h *npuHTTP) json(ctx context.Context, path string, payload, out any) error {
	resp, err := h.open(ctx, path, payload)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(io.LimitReader(resp.Body, npuResponseLimit+1))
	if err != nil {
		return err
	}
	if len(data) > npuResponseLimit {
		return fmt.Errorf("%s: response exceeds %d bytes", path, npuResponseLimit)
	}
	var failure struct{ Error string }
	if err := json.Unmarshal(data, &failure); err != nil {
		return fmt.Errorf("%s: invalid JSON: %w", path, err)
	}
	if failure.Error != "" {
		return fmt.Errorf("%s: unexpected success-status error: %s", path, failure.Error)
	}
	return json.Unmarshal(data, out)
}

func npuOptions() map[string]any {
	return map[string]any{"num_ctx": 512, "num_predict": 32, "temperature": 0, "seed": 42}
}

func npuGenerateRequest(tag, text string) api.GenerateRequest {
	return api.GenerateRequest{
		Model: tag, Prompt: npuRaw(text), Raw: true, Stream: new(false),
		Options: npuOptions(), KeepAlive: &api.Duration{Duration: 5 * time.Minute},
	}
}

func npuValidateGenerationResult(r api.GenerateResponse, tag string, limit int) error {
	if !r.Done || (r.DoneReason != "stop" && r.DoneReason != "length") || !npuSameModel(r.Model, tag) ||
		r.RemoteHost != "" || r.RemoteModel != "" || r.Thinking != "" || len(r.ToolCalls) != 0 {
		return fmt.Errorf("invalid completion: model=%q done=%v reason=%q remote=%q/%q thinking=%q tools=%d",
			r.Model, r.Done, r.DoneReason, r.RemoteHost, r.RemoteModel, r.Thinking, len(r.ToolCalls))
	}
	if r.PromptEvalCount <= 0 || r.EvalCount < 1 || r.EvalCount > limit ||
		(r.DoneReason == "length" && r.EvalCount != limit) ||
		r.TotalDuration <= 0 || r.LoadDuration < 0 || r.PromptEvalDuration < 0 || r.EvalDuration < 0 {
		return fmt.Errorf("invalid generation metrics: %+v", r.Metrics)
	}
	// OGA prefill can provide the only output token without any decode time.
	return nil
}

func npuValidateGeneration(r api.GenerateResponse, tag, want, reason string, limit int) error {
	if err := npuValidateGenerationResult(r, tag, limit); err != nil {
		return err
	}
	if r.DoneReason != reason || strings.TrimSpace(r.Response) != want {
		return fmt.Errorf("unexpected completion: reason=%q text=%q (want %q/%q)", r.DoneReason, r.Response, want, reason)
	}
	return nil
}

func (h *npuHTTP) generateResponse(t *testing.T, req api.GenerateRequest) (api.GenerateResponse, time.Duration) {
	t.Helper()
	ctx, cancel := context.WithTimeout(t.Context(), npuRequestTimeout)
	defer cancel()
	start := time.Now()
	var r api.GenerateResponse
	if err := h.json(ctx, "/api/generate", req, &r); err != nil {
		t.Fatalf("generate %s: %v", req.Model, err)
	}
	elapsed := time.Since(start)
	t.Logf("model=%q raw=%v options=%v wall=%s text=%q done_reason=%q metrics=%+v",
		req.Model, req.Raw, req.Options, elapsed, r.Response, r.DoneReason, r.Metrics)
	return r, elapsed
}

func (h *npuHTTP) generate(t *testing.T, req api.GenerateRequest, want, reason string, limit int) time.Duration {
	t.Helper()
	r, elapsed := h.generateResponse(t, req)
	if err := npuValidateGeneration(r, req.Model, want, reason, limit); err != nil {
		t.Fatal(err)
	}
	return elapsed
}

func npuReadStream(reader io.Reader, firstOnly bool) (api.GenerateResponse, int, error) {
	var result api.GenerateResponse
	limited := &io.LimitedReader{R: reader, N: npuResponseLimit + 1}
	decoder := json.NewDecoder(limited)
	chunks := 0
	for {
		var chunk struct {
			api.GenerateResponse
			Error string `json:"error"`
		}
		err := decoder.Decode(&chunk)
		if limited.N <= 1 {
			return result, chunks, fmt.Errorf("stream exceeds response limit")
		}
		if err == io.EOF {
			if !result.Done {
				return result, chunks, fmt.Errorf("stream ended without a terminal done chunk")
			}
			return result, chunks, nil
		}
		if err != nil {
			return result, chunks, fmt.Errorf("decode stream: %w", err)
		}
		if chunk.Error != "" || result.Done || chunks >= 4096 ||
			chunk.RemoteHost != "" || chunk.RemoteModel != "" || chunk.Thinking != "" || len(chunk.ToolCalls) != 0 {
			return result, chunks, fmt.Errorf("invalid stream chunk: %+v", chunk)
		}
		chunks++
		text := result.Response + chunk.Response
		result = chunk.GenerateResponse
		result.Response = text
		if firstOnly && chunk.Response != "" {
			if chunk.Done {
				return result, chunks, fmt.Errorf("disconnect requires a content chunk before terminal done")
			}
			return result, chunks, nil
		}
		if firstOnly && chunk.Done {
			return result, chunks, fmt.Errorf("stream completed before disconnect")
		}
	}
}

func (h *npuHTTP) streaming(t *testing.T, req api.GenerateRequest, disconnect bool) api.GenerateResponse {
	t.Helper()
	ctx, cancel := context.WithTimeout(t.Context(), npuRequestTimeout)
	defer cancel()
	req.Stream = new(true)
	start := time.Now()
	resp, err := h.open(ctx, "/api/generate", req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	r, chunks, err := npuReadStream(resp.Body, disconnect)
	if disconnect {
		cancel()
	}
	if err != nil {
		t.Fatal(err)
	}
	if !npuSameModel(r.Model, req.Model) {
		t.Fatalf("stream returned model %q, want %q", r.Model, req.Model)
	}
	t.Logf("stream model=%q disconnect=%v chunks=%d wall=%s text=%q done=%v metrics=%+v",
		req.Model, disconnect, chunks, time.Since(start), r.Response, r.Done, r.Metrics)
	return r
}

func npuDigest(list api.ListResponse, tag string) (string, error) {
	for _, m := range list.Models {
		if !npuSameModel(m.Name, tag) && !npuSameModel(m.Model, tag) {
			continue
		}
		digest := strings.TrimPrefix(m.Digest, "sha256:")
		raw, err := hex.DecodeString(digest)
		if err != nil || len(raw) != 32 || m.RemoteHost != "" || m.RemoteModel != "" {
			return "", fmt.Errorf("model %q must have an actual SHA256 digest and be local", tag)
		}
		return m.Digest, nil
	}
	return "", fmt.Errorf("existing model %q not found in /api/tags; no model will be downloaded", tag)
}

func (h *npuHTTP) unload(t *testing.T, tag string) error {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), npuRequestTimeout)
	defer cancel()
	var r api.GenerateResponse
	err := h.json(ctx, "/api/generate", api.GenerateRequest{
		Model: tag, Stream: new(false), KeepAlive: &api.Duration{Duration: 0},
	}, &r)
	if err != nil {
		return err
	}
	if !r.Done || r.DoneReason != "unload" || !npuSameModel(r.Model, tag) || r.Response != "" {
		return fmt.Errorf("unexpected unload response: %+v", r)
	}
	t.Logf("unloaded model=%q", tag)
	return nil
}

func (h *npuHTTP) waitUnloaded(t *testing.T, tags ...string) error {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	for {
		var ps api.ProcessResponse
		if err := h.json(ctx, "/api/ps", nil, &ps); err != nil {
			return err
		}
		var loaded []string
		for _, m := range ps.Models {
			for _, tag := range tags {
				if npuSameModel(m.Name, tag) || npuSameModel(m.Model, tag) {
					loaded = append(loaded, tag)
				}
			}
		}
		if len(loaded) == 0 {
			t.Logf("ps empty for test models only: %q (unrelated models ignored)", tags)
			return nil
		}
		select {
		case <-ctx.Done():
			return fmt.Errorf("test models still loaded: %v: %w", loaded, ctx.Err())
		case <-time.After(100 * time.Millisecond):
		}
	}
}

func npuLogSource(t *testing.T) {
	t.Helper()
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
	defer cancel()
	revision, err := exec.CommandContext(ctx, "git", "rev-parse", "HEAD").Output()
	if err != nil {
		t.Fatalf("cannot record source revision: %v", err)
	}
	status, err := exec.CommandContext(ctx, "git", "status", "--porcelain", "--untracked-files=normal").Output()
	if err != nil {
		t.Fatalf("cannot record source worktree state: %v", err)
	}
	t.Logf("source_commit=%s source_dirty=%v client_version=%q go=%s platform=%s/%s",
		strings.TrimSpace(string(revision)), len(status) != 0, version.Version, runtime.Version(), runtime.GOOS, runtime.GOARCH)
}

// TestNPUQualification is intentionally separate from generic capability/model matrices.
func TestNPUQualification(t *testing.T) {
	c, enabled, err := npuConfig(os.LookupEnv)
	if !enabled {
		t.Skip("opt in with OLLAMA_TEST_NPU_MODEL, OLLAMA_TEST_CPU_MODEL and an explicit local nondefault OLLAMA_HOST")
	}
	if err != nil {
		t.Fatal(err)
	}
	npuLogSource(t)
	t.Log("declared profile: Phi-3 compiled QNN text-only, GenAI QNN 0.12.2 / ORT QNN 1.24.1; CPU embedding/head intentional")
	t.Log("HTTP/tag/status cannot prove NPU placement or runtime identity; independent provider/native log evidence is required")
	h := npuNewHTTP(c.endpoint)
	defer h.client.CloseIdleConnections()
	ctx, cancel := context.WithTimeout(t.Context(), npuRequestTimeout)
	defer cancel()
	var server struct{ Version string }
	if err := h.json(ctx, "/api/version", nil, &server); err != nil || server.Version == "" {
		t.Fatalf("server version prerequisite: version=%q error=%v", server.Version, err)
	}
	t.Logf("endpoint=%s server_version=%q requests=%d", c.endpoint, server.Version, c.requests)
	// Check the local inventory before Show, which can proxy remote model references.
	var list api.ListResponse
	if err := h.json(ctx, "/api/tags", nil, &list); err != nil {
		t.Fatal(err)
	}
	for _, tag := range []string{c.npuModel, c.cpuModel} {
		digest, err := npuDigest(list, tag)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("model=%q digest=%q (reported by /api/tags, not inferred from tag)", tag, digest)
	}
	for _, tag := range []string{c.npuModel, c.cpuModel} {
		var show api.ShowResponse
		if err := h.json(ctx, "/api/show", api.ShowRequest{Model: tag}, &show); err != nil {
			t.Fatalf("existing model %q Show prerequisite (no pull/create): %v", tag, err)
		}
		if show.RemoteHost != "" || show.RemoteModel != "" {
			t.Fatalf("model %q must be local, not a remote model", tag)
		}
		if tag == c.npuModel {
			switch show.Details.Format {
			case "ortgenai", "onnx", "genai-qnn":
			default:
				t.Fatalf("NPU profile needs an existing ORT GenAI bundle, got format %q", show.Details.Format)
			}
			if err := npuValidateTemplate(show); err != nil {
				t.Fatalf("model %q: %v", tag, err)
			}
		} else if show.Details.Format != "gguf" || show.Details.Family != "phi3" {
			t.Fatalf("CPU prerequisite: ordinary phi3:mini GGUF required, got %+v", show.Details)
		}
		t.Logf("Show model=%q details=%+v template=%q parameters=%q", tag, show.Details, show.Template, show.Parameters)
	}
	// Registered only after both existing, local test models have been checked.
	t.Cleanup(func() {
		defer h.client.CloseIdleConnections()
		for _, tag := range []string{c.npuModel, c.cpuModel} {
			if err := h.unload(t, tag); err != nil {
				t.Errorf("final unload %q: %v", tag, err)
			}
		}
		if err := h.waitUnloaded(t, c.npuModel, c.cpuModel); err != nil {
			t.Errorf("final ps: %v", err)
		}
	})
	for _, tag := range []string{c.npuModel, c.cpuModel} {
		if err := h.unload(t, tag); err != nil {
			t.Fatalf("initial cold-state unload: %v", err)
		}
	}
	if err := h.waitUnloaded(t, c.npuModel, c.cpuModel); err != nil {
		t.Fatal(err)
	}
	for i := range c.requests {
		p := npuPrompts[i%len(npuPrompts)]
		t.Run(fmt.Sprintf("raw_%03d_%s", i+1, p.name), func(t *testing.T) {
			elapsed := h.generate(t, npuGenerateRequest(c.npuModel, p.text), p.want, "stop", 32)
			state := "warm"
			if i == 0 {
				state = "cold-after-explicit-unload"
			}
			t.Logf("latency_state=%s sample=%d wall=%s (not a throughput/energy measurement)", state, i+1, elapsed)
		})
	}
	t.Run("token_limit_one", func(t *testing.T) {
		req := npuGenerateRequest(c.npuModel, npuPrompts[1].text)
		req.Options["num_predict"] = 1
		h.generate(t, req, "Paris", "length", 1)
	})
	t.Run("stop_Paris", func(t *testing.T) {
		req := npuGenerateRequest(c.npuModel, npuPrompts[1].text)
		req.Options["stop"] = []string{"Paris"}
		h.generate(t, req, "", "stop", 32)
	})
	t.Run("repetition_window", func(t *testing.T) {
		req := npuGenerateRequest(c.npuModel, npuPrompts[0].text)
		req.Options["repeat_penalty"], req.Options["repeat_last_n"] = 1.1, 64
		// Penalties can change the answer; test bounded, deterministic completion instead.
		first, _ := h.generateResponse(t, req)
		if err := npuValidateGenerationResult(first, c.npuModel, 32); err != nil {
			t.Fatal(err)
		}
		if strings.TrimSpace(first.Response) == "" {
			t.Fatal("repetition controls produced an empty completion")
		}
		second, _ := h.generateResponse(t, req)
		if err := npuValidateGenerationResult(second, c.npuModel, 32); err != nil {
			t.Fatal(err)
		}
		if second.Response != first.Response || second.DoneReason != first.DoneReason || second.EvalCount != first.EvalCount {
			t.Fatal("greedy repetition-controlled completion changed across identical requests")
		}
		req.Options["repeat_last_n"] = 0
		h.generate(t, req, npuPrompts[0].want, "stop", 32)
	})
	t.Run("explicit_zero_sampling", func(t *testing.T) {
		req := npuGenerateRequest(c.npuModel, npuPrompts[0].text)
		req.Options["top_k"], req.Options["top_p"], req.Options["seed"] = 0, 1, 0
		h.generate(t, req, "4", "stop", 32)
	})
	t.Run("automatic_context", func(t *testing.T) {
		req := npuGenerateRequest(c.npuModel, npuPrompts[0].text)
		delete(req.Options, "num_ctx")
		h.generate(t, req, "4", "stop", 32)
	})
	t.Run("ollama_chat", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(t.Context(), npuRequestTimeout)
		defer cancel()
		var r api.ChatResponse
		start := time.Now()
		err := h.json(ctx, "/api/chat", api.ChatRequest{
			Model: c.npuModel, Messages: []api.Message{{Role: "user", Content: npuPrompts[0].text}},
			Stream: new(false), Options: npuOptions(),
		}, &r)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("Ollama chat wall=%s text=%q metrics=%+v", time.Since(start), r.Message.Content, r.Metrics)
		if r.Message.Role != "assistant" || r.Message.Thinking != "" || len(r.Message.ToolCalls) != 0 {
			t.Fatalf("unexpected chat message: %+v", r.Message)
		}
		if err := npuValidateGeneration(api.GenerateResponse{
			Model: r.Model, Done: r.Done, DoneReason: r.DoneReason, Response: r.Message.Content,
			RemoteModel: r.RemoteModel, RemoteHost: r.RemoteHost, Metrics: r.Metrics,
		}, c.npuModel, "4", "stop", 32); err != nil {
			t.Fatal(err)
		}
	})
	t.Run("openai_chat", func(t *testing.T) {
		ctx, cancel := context.WithTimeout(t.Context(), npuRequestTimeout)
		defer cancel()
		var r struct {
			Model, Object string
			Choices       []struct {
				Message      api.Message
				FinishReason string `json:"finish_reason"`
			}
			Usage struct {
				PromptTokens     int `json:"prompt_tokens"`
				CompletionTokens int `json:"completion_tokens"`
			}
		}
		start := time.Now()
		if err := h.json(ctx, "/v1/chat/completions", map[string]any{
			"model": c.npuModel, "messages": []api.Message{{Role: "user", Content: npuPrompts[0].text}},
			"stream": false, "max_tokens": 32, "temperature": 0, "seed": 42, "top_p": 1,
		}, &r); err != nil {
			t.Fatal(err)
		}
		t.Logf("OpenAI chat wall=%s response=%+v", time.Since(start), r)
		if !npuSameModel(r.Model, c.npuModel) || r.Object != "chat.completion" || len(r.Choices) != 1 ||
			r.Choices[0].Message.Role != "assistant" || strings.TrimSpace(r.Choices[0].Message.Content) != "4" ||
			r.Choices[0].FinishReason != "stop" || r.Usage.PromptTokens <= 0 ||
			r.Usage.CompletionTokens < 1 || r.Usage.CompletionTokens > 32 {
			t.Fatalf("unexpected OpenAI completion: %+v", r)
		}
	})
	for _, rejected := range []struct {
		name, option string
		value        any
		reasons      []string
	}{
		{"JSON_format", "", nil, []string{"ORT GenAI does not support structured output"}},
		{"min_p", "min_p", 0.1, []string{"ORT GenAI does not support min_p"}},
		{"presence_penalty", "presence_penalty", 0.1, []string{"ORT GenAI does not support presence_penalty"}},
		{"oversized_context", "num_ctx", 8192, []string{"num_ctx must be between 1 and model context length "}},
		{"QNN_CPU_fallback", "num_gpu", 0, []string{
			"QNN-compiled models require provider=qnn; CPU/DML fallback is not supported",
			"CPU-only request conflicts with ORT provider/device selection",
		}},
	} {
		t.Run("reject_"+rejected.name, func(t *testing.T) {
			req := npuGenerateRequest(c.npuModel, npuPrompts[0].text)
			if rejected.option == "" {
				req.Format = json.RawMessage(`"json"`)
			} else {
				req.Options[rejected.option] = rejected.value
			}
			ctx, cancel := context.WithTimeout(t.Context(), npuRequestTimeout)
			defer cancel()
			var r api.GenerateResponse
			err := h.json(ctx, "/api/generate", req, &r)
			var status api.StatusError
			matched := false
			if errors.As(err, &status) {
				for _, reason := range rejected.reasons {
					matched = matched || strings.Contains(status.ErrorMessage, reason)
				}
			}
			// Some existing scheduling/runner validation paths report 500, not 400.
			if !matched || (status.StatusCode != 400 && status.StatusCode != 500) {
				t.Fatalf("expected explicit %q rejection, got response=%+v error=%v", rejected.reasons, r, err)
			}
			t.Logf("expected rejection HTTP=%d reason=%q", status.StatusCode, status.ErrorMessage)
		})
	}
	t.Run("stream_terminal", func(t *testing.T) {
		r := h.streaming(t, npuGenerateRequest(c.npuModel, npuPrompts[2].text), false)
		if err := npuValidateGeneration(r, c.npuModel, npuPrompts[2].want, "stop", 32); err != nil {
			t.Fatal(err)
		}
	})
	t.Run("disconnect_then_bounded_reload", func(t *testing.T) {
		req := npuGenerateRequest(c.npuModel, "Count from 1 to 100, separated by commas. Do not explain.")
		h.streaming(t, req, true)
		h.generate(t, npuGenerateRequest(c.npuModel, npuPrompts[0].text), "4", "stop", 32)
	})
	t.Run("explicit_unload_reload", func(t *testing.T) {
		if err := h.unload(t, c.npuModel); err != nil {
			t.Fatal(err)
		}
		if err := h.waitUnloaded(t, c.npuModel); err != nil {
			t.Fatal(err)
		}
		elapsed := h.generate(t, npuGenerateRequest(c.npuModel, npuPrompts[0].text), "4", "stop", 32)
		t.Logf("latency_state=cold-after-explicit-unload wall=%s", elapsed)
	})
	t.Run("ordinary_phi3_CPU", func(t *testing.T) {
		req := npuGenerateRequest(c.cpuModel, npuPrompts[0].text)
		req.Raw, req.Prompt = false, npuPrompts[0].text
		req.Options["num_gpu"] = 0
		h.generate(t, req, "4", "stop", 32)
	})
	t.Run("switch_back_QNN", func(t *testing.T) {
		h.generate(t, npuGenerateRequest(c.npuModel, npuPrompts[0].text), "4", "stop", 32)
	})
}

func TestNPUQualificationHelpers(t *testing.T) {
	t.Run("endpoint", func(t *testing.T) {
		for _, s := range []string{"localhost:11435", "http://127.0.0.1:11435/", "https://[::1]:443"} {
			if _, err := npuEndpoint(s); err != nil {
				t.Errorf("%q: %v", s, err)
			}
		}
		for _, s := range []string{"", "localhost", "localhost:11434", "127.0.0.1:0", "[::1]:65536",
			"http://0.0.0.0:11435", "http://127.0.0.2:11435", "http://[::]:11435", "remote:11435",
			"http://localhost.evil:11435", "http://localhost:11435/path", "http://user@localhost:11435",
			"ftp://localhost:11435", "http://localhost:11435?x=1", "http://localhost:11435#x"} {
			if _, err := npuEndpoint(s); err == nil {
				t.Errorf("accepted unsafe endpoint %q", s)
			}
		}
	})
	t.Run("config", func(t *testing.T) {
		lookup := func(values map[string]string) func(string) (string, bool) {
			return func(key string) (string, bool) { s, ok := values[key]; return s, ok }
		}
		if _, enabled, err := npuConfig(lookup(nil)); enabled || err != nil {
			t.Fatalf("unset environment: enabled=%v error=%v", enabled, err)
		}
		if _, enabled, err := npuConfig(lookup(map[string]string{"OLLAMA_HOST": "remote:11434"})); enabled || err != nil {
			t.Fatalf("host alone must not opt in: enabled=%v error=%v", enabled, err)
		}
		valid := map[string]string{"OLLAMA_HOST": "127.0.0.1:11435", "OLLAMA_TEST_NPU_MODEL": "local-phi3:qnn", "OLLAMA_TEST_CPU_MODEL": "phi3:mini"}
		for _, count := range []string{"20", "1000", "", "19", "1001", "bad"} {
			values := make(map[string]string)
			for k, v := range valid {
				values[k] = v
			}
			values["OLLAMA_TEST_NPU_REQUESTS"] = count
			_, enabled, err := npuConfig(lookup(values))
			wantValid := count == "20" || count == "1000"
			if !enabled || (err == nil) != wantValid {
				t.Errorf("requests=%q enabled=%v error=%v", count, enabled, err)
			}
		}
		if c, enabled, err := npuConfig(lookup(valid)); !enabled || err != nil || c.requests != 20 {
			t.Fatalf("default requests: config=%+v enabled=%v error=%v", c, enabled, err)
		}
		for _, bad := range []struct{ key, value string }{
			{"OLLAMA_TEST_NPU_MODEL", "untagged"},
			{"OLLAMA_TEST_NPU_MODEL", "test:cloud"},
			{"OLLAMA_TEST_NPU_MODEL", "test:qnn-cloud"},
			{"OLLAMA_TEST_NPU_MODEL", " test:qnn"},
			{"OLLAMA_TEST_NPU_MODEL", "phi3:mini"},
			{"OLLAMA_TEST_CPU_MODEL", "other:cpu"},
			{"OLLAMA_HOST", "remote:11435"},
		} {
			values := make(map[string]string)
			for k, v := range valid {
				values[k] = v
			}
			values[bad.key] = bad.value
			if _, enabled, err := npuConfig(lookup(values)); !enabled || err == nil {
				t.Errorf("accepted %s=%q", bad.key, bad.value)
			}
		}
		for _, key := range []string{"OLLAMA_HOST", "OLLAMA_TEST_NPU_MODEL", "OLLAMA_TEST_CPU_MODEL"} {
			values := make(map[string]string)
			for k, v := range valid {
				if k != key {
					values[k] = v
				}
			}
			if _, enabled, err := npuConfig(lookup(values)); !enabled || err == nil {
				t.Errorf("missing %s did not fail: enabled=%v error=%v", key, enabled, err)
			}
		}
		for _, values := range []map[string]string{
			{"OLLAMA_TEST_NPU_MODEL": ""},
			{"OLLAMA_TEST_CPU_MODEL": ""},
			{"OLLAMA_TEST_NPU_REQUESTS": ""},
		} {
			if _, enabled, err := npuConfig(lookup(values)); !enabled || err == nil {
				t.Errorf("partial opt-in accepted: %v", values)
			}
		}
	})
	t.Run("template_and_metrics", func(t *testing.T) {
		show := api.ShowResponse{Template: "{{ range .Messages }}<|{{ .Role }}|>\n{{ .Content }}<|end|>\n{{ end }}<|assistant|>\n"}
		if err := npuValidateTemplate(show); err != nil {
			t.Fatal(err)
		}
		for _, bad := range []string{"", "{{ .Prompt }}", "{{", show.Template + " "} {
			if err := npuValidateTemplate(api.ShowResponse{Template: bad}); err == nil {
				t.Errorf("accepted incorrect template %q", bad)
			}
		}
		r := api.GenerateResponse{Model: "test:qnn", Done: true, DoneReason: "length", Response: "Paris",
			Metrics: api.Metrics{PromptEvalCount: 10, EvalCount: 1, TotalDuration: time.Second}}
		if err := npuValidateGeneration(r, "test:qnn", "Paris", "length", 1); err != nil {
			t.Fatalf("prefill-only single token must allow zero eval_duration: %v", err)
		}
		r.Done = false
		if err := npuValidateGeneration(r, "test:qnn", "Paris", "length", 1); err == nil {
			t.Fatal("accepted incomplete generation")
		}
		r.Done, r.Response = true, " BLUE "
		if err := npuValidateGeneration(r, "test:qnn", "BLUE", "length", 1); err != nil {
			t.Fatalf("surrounding whitespace must be ignored: %v", err)
		}
		r.Response = "Blue"
		if err := npuValidateGeneration(r, "test:qnn", "BLUE", "length", 1); err == nil {
			t.Fatal("accepted incorrect completion letter case")
		}
		for _, change := range []func(*api.GenerateResponse){
			func(r *api.GenerateResponse) { r.DoneReason = "unknown" },
			func(r *api.GenerateResponse) { r.Model = "other:qnn" },
			func(r *api.GenerateResponse) { r.RemoteHost = "https://remote.invalid" },
			func(r *api.GenerateResponse) { r.Thinking = "unexpected" },
			func(r *api.GenerateResponse) { r.PromptEvalCount = 0 },
			func(r *api.GenerateResponse) { r.EvalCount = 0 },
			func(r *api.GenerateResponse) { r.EvalCount = 2 },
			func(r *api.GenerateResponse) { r.TotalDuration = 0 },
		} {
			bad := r
			change(&bad)
			if err := npuValidateGenerationResult(bad, "test:qnn", 1); err == nil {
				t.Fatalf("accepted invalid controlled completion: %+v", bad)
			}
		}
		if err := npuValidateGenerationResult(r, "test:qnn", 2); err == nil {
			t.Fatal("accepted length termination before the token limit")
		}
	})
	t.Run("digest", func(t *testing.T) {
		list := api.ListResponse{Models: []api.ListModelResponse{{Name: "local:qnn", Digest: strings.Repeat("a", 64)}}}
		if _, err := npuDigest(list, "registry.ollama.ai/library/local:qnn"); err != nil {
			t.Fatal(err)
		}
		list.Models[0].Digest = "local:qnn"
		if _, err := npuDigest(list, "local:qnn"); err == nil {
			t.Fatal("accepted tag as digest")
		}
		list.Models[0].Digest, list.Models[0].RemoteHost = strings.Repeat("a", 64), "https://remote.invalid"
		if _, err := npuDigest(list, "local:qnn"); err == nil {
			t.Fatal("accepted remote model in local inventory")
		}
		if _, err := npuDigest(api.ListResponse{}, "missing:qnn"); err == nil {
			t.Fatal("accepted absent model")
		}
	})
	t.Run("stream_parser", func(t *testing.T) {
		for _, tc := range []struct {
			body      string
			firstOnly bool
			valid     bool
		}{
			{`{"response":"Blue","done":false}` + "\n" + `{"done":true,"done_reason":"stop"}`, false, true},
			{`{"response":"1","done":false}`, true, true},
			{`{"response":"Blue","done":false}`, false, false},
			{`{"done":true}` + "\n" + `{"done":true}`, false, false},
			{`{"done":true,"response":"Blue"}`, true, false},
			{`{"error":"native failure"}`, false, false},
			{`not-json`, false, false},
		} {
			if _, _, err := npuReadStream(strings.NewReader(tc.body), tc.firstOnly); (err == nil) != tc.valid {
				t.Errorf("stream %q firstOnly=%v: %v", tc.body, tc.firstOnly, err)
			}
		}
	})
	t.Run("HTTP_validation", func(t *testing.T) {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			switch r.URL.Path {
			case "/ok":
				fmt.Fprint(w, `{"version":"test"}`)
			case "/reject":
				w.WriteHeader(400)
				fmt.Fprint(w, `{"error":"ORT GenAI does not support min_p"}`)
			case "/redirect":
				http.Redirect(w, r, "/ok", http.StatusFound)
			case "/error":
				fmt.Fprint(w, `{"error":"native failure"}`)
			case "/trailing":
				fmt.Fprint(w, `{} {}`)
			case "/large":
				fmt.Fprint(w, strings.Repeat(" ", npuResponseLimit+1))
			default:
				fmt.Fprint(w, `not-json`)
			}
		}))
		defer server.Close()
		u, err := npuEndpoint(server.URL)
		if err != nil {
			t.Fatal(err)
		}
		h := npuNewHTTP(u)
		defer h.client.CloseIdleConnections()
		ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
		defer cancel()
		var out struct{ Version string }
		if err := h.json(ctx, "/ok", nil, &out); err != nil || out.Version != "test" {
			t.Fatalf("HTTP decode: %+v, %v", out, err)
		}
		for _, path := range []string{"/reject", "/redirect", "/error", "/trailing", "/large", "/malformed"} {
			if err := h.json(ctx, path, nil, &out); err == nil {
				t.Errorf("accepted invalid response at %s", path)
			}
		}
		var status api.StatusError
		if err := h.json(ctx, "/reject", nil, &out); !errors.As(err, &status) || status.StatusCode != 400 {
			t.Fatalf("typed rejection: %v", err)
		}
		cancelled, stop := context.WithCancel(t.Context())
		stop()
		if err := h.json(cancelled, "/ok", nil, &out); !errors.Is(err, context.Canceled) {
			t.Fatalf("canceled context: %v", err)
		}
	})
	t.Run("HTTP_disconnect_and_residency", func(t *testing.T) {
		disconnected := make(chan struct{})
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			switch r.URL.Path {
			case "/api/generate":
				var req api.GenerateRequest
				if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
					t.Errorf("decode mock request: %v", err)
					w.WriteHeader(400)
					return
				}
				if req.Stream != nil && *req.Stream {
					if req.Prompt != npuRaw("disconnect") || !req.Raw {
						t.Errorf("incorrect raw streaming request: %+v", req)
					}
					fmt.Fprintln(w, `{"model":"test:qnn","response":"1","done":false}`)
					w.(http.Flusher).Flush()
					select {
					case <-r.Context().Done():
						close(disconnected)
					case <-time.After(5 * time.Second):
						t.Error("client did not disconnect after first content chunk")
					}
					return
				}
				fmt.Fprint(w, `{"model":"test:qnn","response":"4","done":true,"done_reason":"stop","prompt_eval_count":10,"eval_count":1,"total_duration":1}`)
			case "/api/ps":
				// Residency checks must not require unrelated models to be absent.
				fmt.Fprint(w, `{"models":[{"name":"unrelated:latest","model":"unrelated:latest"}]}`)
			default:
				t.Errorf("unexpected mock route: %s", r.URL.Path)
				w.WriteHeader(404)
			}
		}))
		defer server.Close()
		u, err := npuEndpoint(server.URL)
		if err != nil {
			t.Fatal(err)
		}
		h := npuNewHTTP(u)
		defer h.client.CloseIdleConnections()
		h.streaming(t, npuGenerateRequest("test:qnn", "disconnect"), true)
		select {
		case <-disconnected:
		case <-time.After(5 * time.Second):
			t.Fatal("mock server did not observe request cancellation")
		}
		h.generate(t, npuGenerateRequest("test:qnn", "recover"), "4", "stop", 32)
		if err := h.waitUnloaded(t, "test:qnn", "phi3:mini"); err != nil {
			t.Fatal(err)
		}
	})
}
