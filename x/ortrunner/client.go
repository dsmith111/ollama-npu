package ortrunner

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/ml"
	ortruntime "github.com/ollama/ollama/x/ortrunner/runtime"
)

const (
	clientProtocolVersion = 1
	maxRunnerLineBytes    = 8 << 20
	maxRunnerErrorBytes   = 8 << 10
)

// Client owns a private runner. A cancelled or corrupt stream retires the child
// instead of reusing native state whose completion cannot be established.
type Client struct {
	port          int
	modelDir      string
	token         string
	contextLength atomic.Int64
	memory        atomic.Uint64
	done          chan struct{}
	doneErr       error // written before done is closed
	client        *http.Client
	lastErr       string
	lastErrLk     sync.Mutex
	mu            sync.Mutex
	cmd           *exec.Cmd
	guard         *processGuard
	closed        bool
	closeDone     chan struct{}
	closeErr      error
	operations    chan struct{}
}

type ClientOptions struct {
	ModelDir   string
	DeviceType string
	DeviceID   string
	Provider   string
}

func NewClient(modelDir string) (*Client, error) {
	return NewClientWithOpts(ClientOptions{ModelDir: modelDir})
}

func NewClientWithOpts(opts ClientOptions) (*Client, error) {
	return NewClientWithContext(context.Background(), opts)
}

func NewClientWithContext(ctx context.Context, opts ClientOptions) (*Client, error) {
	if opts.ModelDir == "" {
		return nil, errors.New("ORT model directory is required")
	}
	exe, err := os.Executable()
	if err != nil {
		return nil, fmt.Errorf("resolve runner executable: %w", err)
	}
	if resolved, err := filepath.EvalSymlinks(exe); err == nil {
		exe = resolved
	}
	cmd := exec.Command(exe, "runner", "--ortgenai-engine", "--model", opts.ModelDir, "--port", "0")
	cmd.Env = os.Environ()
	return startClient(ctx, opts, cmd)
}

type clientReadiness struct {
	Event           string `json:"event"`
	Port            int    `json:"port"`
	ProtocolVersion int    `json:"protocol_version"`
	StatusResponse
}

func startClient(ctx context.Context, opts ClientOptions, cmd *exec.Cmd) (*Client, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	switch opts.Provider {
	case "", "auto", "qnn", "dml", "cpu":
	default:
		return nil, fmt.Errorf("unsupported ORT provider %q", opts.Provider)
	}
	switch opts.DeviceType {
	case "", "npu", "gpu":
	default:
		return nil, fmt.Errorf("unsupported ORT device type %q", opts.DeviceType)
	}
	var nonce [32]byte
	if _, err := rand.Read(nonce[:]); err != nil {
		return nil, fmt.Errorf("generate runner credentials: %w", err)
	}
	c := &Client{
		modelDir: opts.ModelDir, token: hex.EncodeToString(nonce[:]),
		done: make(chan struct{}), closeDone: make(chan struct{}),
		operations: make(chan struct{}, 1), cmd: cmd,
		client: &http.Client{Transport: &http.Transport{
			Proxy: nil, MaxIdleConnsPerHost: 2, IdleConnTimeout: 30 * time.Second,
		}},
	}
	cmd.Env = setChildEnv(cmd.Env, "OLLAMA_ORT_RUNNER_TOKEN", c.token)
	for _, option := range []struct{ key, value string }{
		{"OLLAMA_ORT_DEVICE_TYPE", opts.DeviceType},
		{"OLLAMA_ORT_DEVICE_ID", opts.DeviceID},
		{"OLLAMA_ONNX_PROVIDER", opts.Provider},
	} {
		if option.value != "" {
			cmd.Env = setChildEnv(cmd.Env, option.key, option.value)
		}
	}
	pathKey := "LD_LIBRARY_PATH"
	if runtime.GOOS == "windows" {
		pathKey = "PATH"
	}
	paths := ortruntime.SearchDirs()
	if ml.LibOllamaPath != "" {
		paths = append(paths, ml.LibOllamaPath)
		subdirs, err := filepath.Glob(filepath.Join(ml.LibOllamaPath, "ortgenai*"))
		if err != nil {
			return nil, fmt.Errorf("discover ORT libraries: %w", err)
		}
		paths = append(paths, subdirs...)
	}
	paths = append(paths, filepath.SplitList(os.Getenv(pathKey))...)
	cmd.Env = setChildEnv(cmd.Env, pathKey, strings.Join(paths, string(filepath.ListSeparator)))

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("open runner stdout: %w", err)
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		stdout.Close()
		return nil, fmt.Errorf("open runner stderr: %w", err)
	}
	c.guard, err = newProcessGuard(cmd)
	if err != nil {
		stdout.Close()
		stderr.Close()
		return nil, err
	}
	if err := cmd.Start(); err != nil {
		c.guard.Close()
		stdout.Close()
		stderr.Close()
		return nil, fmt.Errorf("start ORT runner: %w", err)
	}
	if err := c.guard.Attach(cmd.Process); err != nil {
		killErr := cmd.Process.Kill()
		waitErr := cmd.Wait()
		guardErr := c.guard.Close()
		return nil, errors.Join(fmt.Errorf("isolate ORT runner: %w", err), killErr, waitErr, guardErr)
	}
	readiness := make(chan struct {
		value clientReadiness
		err   error
	}, 1)
	go func() {
		scanner := bufio.NewScanner(stdout)
		scanner.Buffer(make([]byte, 64<<10), maxRunnerLineBytes)
		announced := false
		for scanner.Scan() {
			var event clientReadiness
			if err := json.Unmarshal(scanner.Bytes(), &event); err != nil || event.Event != "ready" {
				continue // Native libraries may write diagnostics to stdout.
			}
			if !announced {
				readiness <- struct {
					value clientReadiness
					err   error
				}{value: event}
				announced = true
			}
		}
		if !announced {
			err := scanner.Err()
			if err == nil {
				err = io.ErrUnexpectedEOF
			}
			readiness <- struct {
				value clientReadiness
				err   error
			}{err: fmt.Errorf("read runner readiness: %w", err)}
		}
	}()
	go func() {
		scanner := bufio.NewScanner(stderr)
		scanner.Buffer(make([]byte, 64<<10), maxRunnerLineBytes)
		for scanner.Scan() {
			line := scanner.Text()
			fmt.Fprintln(os.Stderr, line)
			c.setLastErr(line)
		}
		if err := scanner.Err(); err != nil && !errors.Is(err, os.ErrClosed) {
			c.setLastErr(fmt.Sprintf("read runner stderr: %v", err))
		}
	}()
	go func() {
		c.doneErr = cmd.Wait()
		close(c.done)
	}()
	startCtx, cancel := context.WithTimeout(ctx, envconfig.LoadTimeout())
	defer cancel()
	var event clientReadiness
	select {
	case result := <-readiness:
		err, event = result.err, result.value
	case <-c.done:
		err = c.exitError()
	case <-startCtx.Done():
		err = fmt.Errorf("wait for ORT runner readiness: %w", startCtx.Err())
	}
	if err == nil {
		err = validateReadiness(event)
	}
	if err == nil {
		c.port = event.Port
		c.contextLength.Store(int64(event.ContextLength))
		c.memory.Store(event.Memory)
		err = c.Ping(startCtx)
	}
	if err != nil {
		return nil, errors.Join(err, c.Close())
	}
	slog.Info("ORT runner ready", "pid", cmd.Process.Pid, "port", c.port, "provider", event.Provider, "context", event.ContextLength)
	return c, nil
}

func validateReadiness(event clientReadiness) error {
	if event.Event != "ready" || event.ProtocolVersion != clientProtocolVersion ||
		event.Port <= 0 || event.Port > 65535 || event.ContextLength <= 0 || event.Status != 0 {
		return errors.New("invalid or incompatible ORT runner readiness handshake")
	}
	switch event.Provider {
	case "qnn", "dml", "cpu":
		return nil
	default:
		return fmt.Errorf("unknown runner provider %q", event.Provider)
	}
}

func setChildEnv(env []string, key, value string) []string {
	out := make([]string, 0, len(env)+1)
	for _, entry := range env {
		name, _, ok := strings.Cut(entry, "=")
		if ok && (name == key || runtime.GOOS == "windows" && strings.EqualFold(name, key)) {
			continue
		}
		out = append(out, entry)
	}
	return append(out, key+"="+value)
}

func (c *Client) setLastErr(message string) {
	if len(message) > maxRunnerErrorBytes {
		message = message[len(message)-maxRunnerErrorBytes:]
	}
	c.lastErrLk.Lock()
	c.lastErr = message
	c.lastErrLk.Unlock()
}

func (c *Client) exitError() error {
	c.lastErrLk.Lock()
	message := c.lastErr
	c.lastErrLk.Unlock()
	return fmt.Errorf("ORT runner exited (exit: %v): %s", c.doneErr, message)
}

func (c *Client) Close() error {
	c.mu.Lock()
	if c.closed {
		done := c.closeDone
		c.mu.Unlock()
		<-done
		return c.closeErr
	}
	c.closed = true
	process := c.cmd
	c.mu.Unlock()
	var closeErr error
	defer func() {
		c.mu.Lock()
		c.closeErr = closeErr
		close(c.closeDone)
		c.mu.Unlock()
		c.client.CloseIdleConnections()
	}()
	if process == nil || process.Process == nil {
		return nil
	}
	if !c.HasExited() && c.port != 0 {
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		resp, err := c.do(ctx, http.MethodPost, "/v1/shutdown", struct{}{})
		if err == nil {
			resp.Body.Close()
		}
		cancel()
	}
	select {
	case <-c.done:
	case <-time.After(cancellationGrace):
		slog.Warn("ORT runner exceeded shutdown grace; terminating", "pid", process.Process.Pid)
		if err := process.Process.Kill(); err != nil && !errors.Is(err, os.ErrProcessDone) {
			closeErr = fmt.Errorf("terminate ORT runner: %w", err)
		}
	}
	if c.guard != nil {
		closeErr = errors.Join(closeErr, c.guard.Close())
	}
	select {
	case <-c.done:
	case <-time.After(cancellationGrace):
		closeErr = errors.Join(closeErr, errors.New("ORT runner did not exit after termination"))
	}
	return closeErr
}

func (c *Client) acquire(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	select {
	case c.operations <- struct{}{}:
	case <-ctx.Done():
		return ctx.Err()
	case <-c.done:
		return c.exitError()
	}
	c.mu.Lock()
	closed := c.closed
	c.mu.Unlock()
	if closed || c.HasExited() {
		<-c.operations
		return errors.New("ORT runner is closed")
	}
	return nil
}

func (c *Client) do(ctx context.Context, method, path string, value any) (*http.Response, error) {
	var body io.Reader
	if value != nil {
		data, err := json.Marshal(value)
		if err != nil {
			return nil, err
		}
		if len(data) > maxRequestBytes {
			return nil, api.StatusError{StatusCode: http.StatusRequestEntityTooLarge, ErrorMessage: "ORT runner request exceeds limit"}
		}
		body = bytes.NewReader(data)
	}
	req, err := http.NewRequestWithContext(ctx, method, "http://127.0.0.1:"+strconv.Itoa(c.port)+path, body)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+c.token)
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := c.client.Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		data, readErr := io.ReadAll(io.LimitReader(resp.Body, maxRunnerErrorBytes))
		if readErr != nil {
			return nil, fmt.Errorf("read ORT error response: %w", readErr)
		}
		var status api.StatusError
		if json.Unmarshal(data, &status) != nil || status.ErrorMessage == "" {
			status.ErrorMessage = strings.TrimSpace(string(data))
		}
		status.StatusCode = resp.StatusCode
		status.Status = resp.Status
		return nil, status
	}
	return resp, nil
}

func (c *Client) Completion(ctx context.Context, req llm.CompletionRequest, fn func(llm.CompletionResponse)) (err error) {
	switch {
	case len(req.Media) != 0:
		return api.StatusError{StatusCode: 400, ErrorMessage: "ORT GenAI does not support media input"}
	case req.Logprobs || req.TopLogprobs != 0:
		return api.StatusError{StatusCode: 400, ErrorMessage: "ORT GenAI does not support logprobs"}
	case len(req.Format) != 0 && string(req.Format) != "null" && string(req.Format) != `""`:
		return api.StatusError{StatusCode: 400, ErrorMessage: "ORT GenAI does not support structured output"}
	}
	options := api.DefaultOptions()
	options.NumCtx = c.ContextLength()
	if req.Options != nil {
		options = *req.Options
	}
	converted, err := CompletionOptionsFromAPI(options)
	if err != nil {
		return api.StatusError{StatusCode: 400, ErrorMessage: err.Error()}
	}
	if err := c.acquire(ctx); err != nil {
		return err
	}
	defer func() { <-c.operations }()
	ctx, cancel := context.WithTimeout(ctx, requestTimeout)
	defer cancel()
	// Closing the child also bounds a blocked native call after the HTTP client
	// disconnects. The scheduler observes HasExited and creates a fresh runner.
	defer func() {
		if err != nil {
			var status api.StatusError
			if ctx.Err() != nil || !errors.As(err, &status) || status.StatusCode >= 500 {
				err = errors.Join(err, c.Close())
			}
		}
	}()
	resp, err := c.do(ctx, http.MethodPost, "/completion", TextCompletionsRequest{Prompt: req.Prompt, Options: converted})
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	return readCompletionStream(resp.Body, fn)
}

func readCompletionStream(reader io.Reader, fn func(llm.CompletionResponse)) error {
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 64<<10), maxRunnerLineBytes)
	for scanner.Scan() {
		var raw CompletionResponse
		if err := json.Unmarshal(scanner.Bytes(), &raw); err != nil {
			return fmt.Errorf("invalid ORT response: %w", err)
		}
		if raw.Error != nil {
			return api.StatusError{StatusCode: raw.Error.StatusCode, Status: raw.Error.Status, ErrorMessage: raw.Error.ErrorMessage}
		}
		if raw.DoneReason < DoneReasonStop || raw.DoneReason > DoneReasonConnectionClosed ||
			raw.PromptEvalCount < 0 || raw.EvalCount < 0 || raw.PromptEvalDuration < 0 || raw.EvalDuration < 0 {
			return errors.New("invalid ORT response metrics or done reason")
		}
		fn(llm.CompletionResponse{
			Content: raw.Content, Done: raw.Done, DoneReason: llm.DoneReason(raw.DoneReason),
			PromptEvalCount: raw.PromptEvalCount, PromptEvalDuration: raw.PromptEvalDuration,
			EvalCount: raw.EvalCount, EvalDuration: raw.EvalDuration,
		})
		if raw.Done {
			return nil
		}
	}
	if err := scanner.Err(); err != nil {
		return fmt.Errorf("read ORT response: %w", err)
	}
	return fmt.Errorf("ORT stream ended without a terminal response: %w", io.ErrUnexpectedEOF)
}

func (c *Client) Chat(context.Context, llm.ChatRequest, func(llm.ChatResponse)) error {
	return errors.New("ORT GenAI requires Ollama-rendered chat")
}

func (c *Client) ApplyChatTemplate(context.Context, llm.ChatRequest) (string, error) {
	return "", errors.New("ORT GenAI requires an Ollama chat template")
}

func (c *Client) Ping(ctx context.Context) error {
	ctx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	resp, err := c.do(ctx, http.MethodGet, "/v1/status", nil)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	var status StatusResponse
	if err := json.NewDecoder(io.LimitReader(resp.Body, maxRunnerErrorBytes)).Decode(&status); err != nil {
		return fmt.Errorf("decode ORT status: %w", err)
	}
	if status.Status != 0 || status.ContextLength != c.ContextLength() {
		return errors.New("ORT runner status is not ready or has a different context")
	}
	return nil
}

func (c *Client) operation(ctx context.Context, path string, input, output any) (err error) {
	if err := c.acquire(ctx); err != nil {
		return err
	}
	defer func() { <-c.operations }()
	ctx, cancel := context.WithTimeout(ctx, requestTimeout)
	defer cancel()
	defer func() {
		if ctx.Err() != nil {
			err = errors.Join(ctx.Err(), err, c.Close())
		}
	}()
	resp, err := c.do(ctx, http.MethodPost, path, input)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	decoder := json.NewDecoder(io.LimitReader(resp.Body, maxRunnerLineBytes))
	if err := decoder.Decode(output); err != nil {
		return fmt.Errorf("decode ORT operation response: %w", err)
	}
	if err := decoder.Decode(new(any)); err != io.EOF {
		return errors.New("ORT operation returned extra or invalid data")
	}
	return nil
}

func (c *Client) Tokenize(ctx context.Context, content string) ([]int, error) {
	var tokens []int32
	err := c.operation(ctx, "/v1/tokenize", map[string]string{"content": content}, &tokens)
	if err != nil {
		return nil, err
	}
	result := make([]int, len(tokens))
	for i, token := range tokens {
		if token < 0 {
			return nil, errors.New("ORT tokenizer returned a negative token ID")
		}
		result[i] = int(token)
	}
	return result, nil
}

func (c *Client) Detokenize(ctx context.Context, tokens []int) (string, error) {
	input := make([]int32, len(tokens))
	for i, token := range tokens {
		if token < 0 || token > math.MaxInt32 {
			return "", fmt.Errorf("invalid token ID %d", token)
		}
		input[i] = int32(token)
	}
	var output struct {
		Content string `json:"content"`
	}
	err := c.operation(ctx, "/v1/detokenize", map[string][]int32{"tokens": input}, &output)
	return output.Content, err
}

func (c *Client) Embedding(context.Context, string) ([]float32, int, error) {
	return nil, 0, errors.New("ORT GenAI does not support embeddings")
}

func (c *Client) GetDeviceInfos(context.Context) []ml.DeviceInfo { return nil }
func (c *Client) GetPort() int                                   { return c.port }
func (c *Client) ModelPath() string                              { return c.modelDir }
func (c *Client) ContextLength() int                             { return int(c.contextLength.Load()) }
func (c *Client) VRAMByGPU(ml.DeviceID) uint64                   { return 0 }
func (c *Client) MemorySize() (uint64, uint64)                   { return c.memory.Load(), 0 }

func (c *Client) HasExited() bool {
	select {
	case <-c.done:
		return true
	default:
		return false
	}
}

func (c *Client) Load(ctx context.Context, _ ml.SystemInfo, _ []ml.DeviceInfo, _ bool) ([]ml.DeviceID, error) {
	return nil, c.WaitUntilRunning(ctx)
}

func (c *Client) Pid() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed || c.HasExited() || c.cmd == nil || c.cmd.Process == nil {
		return -1
	}
	return c.cmd.Process.Pid
}

func (c *Client) WaitUntilRunning(ctx context.Context) error {
	if c.HasExited() {
		return c.exitError()
	}
	return c.Ping(ctx)
}

var _ llm.LlamaServer = (*Client)(nil)
