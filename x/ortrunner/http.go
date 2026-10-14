package ortrunner

import (
	"bytes"
	"context"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"net"
	"net/http"
	"strings"
	"sync"
	"time"
	"unicode/utf8"
)

const (
	requestQueueCapacity = 4
	maxRequestBytes      = 1 << 20
	maxHeaderBytes       = 16 << 10
	cancellationGrace    = 5 * time.Second
	streamWriteTimeout   = 30 * time.Second
	requestTimeout       = 15 * time.Minute
)

type serverBackend interface {
	Generate(context.Context, Request) error
	Tokenize(context.Context, string) ([]int32, error)
	Detokenize(context.Context, []int32) (string, error)
	Status() StatusResponse
}

type runnerService struct {
	ctx              context.Context
	cancel           context.CancelFunc
	backend          serverBackend
	jobs             chan func()
	done             chan struct{}
	mu               sync.Mutex
	stopped          bool
	busy             bool
	activeGeneration uint64
	activeRequestID  string
	activeOperation  string
}

func newRunnerService(ctx context.Context, backend serverBackend, capacity int) *runnerService {
	ctx, cancel := context.WithCancel(ctx)
	s := &runnerService{ctx: ctx, cancel: cancel, backend: backend, jobs: make(chan func(), capacity), done: make(chan struct{})}
	go func() {
		defer close(s.done)
		for {
			select {
			case <-ctx.Done():
				return
			case job := <-s.jobs:
				if ctx.Err() != nil {
					return
				}
				job()
			}
		}
	}()
	return s
}

func (s *runnerService) stop() {
	s.mu.Lock()
	s.stopped = true
	s.cancel()
	s.mu.Unlock()
}

func (s *runnerService) enqueue(w http.ResponseWriter, ctx context.Context, job func()) bool {
	return s.enqueueOperation(w, ctx, "", "", job)
}

func (s *runnerService) enqueueOperation(w http.ResponseWriter, ctx context.Context, requestID, operation string, job func()) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.stopped || s.ctx.Err() != nil {
		writeHTTPError(w, http.StatusServiceUnavailable, "runner is shutting down")
		return false
	}
	if ctx.Err() != nil {
		return false
	}
	select {
	case s.jobs <- func() {
		s.mu.Lock()
		if ctx.Err() == nil {
			s.busy = true
			s.activeGeneration++
			s.activeRequestID, s.activeOperation = requestID, operation
		}
		s.mu.Unlock()
		defer func() {
			s.mu.Lock()
			s.busy = false
			s.activeRequestID, s.activeOperation = "", ""
			s.mu.Unlock()
		}()
		job()
	}:
		return true
	default:
		w.Header().Set("Retry-After", "1")
		writeHTTPError(w, http.StatusTooManyRequests, "runner request queue is full")
		return false
	}
}

func (s *runnerService) status() StatusResponse {
	response := s.backend.Status()
	response.ProtocolVersion = ProtocolVersion
	s.mu.Lock()
	response.QueueDepth = len(s.jobs)
	response.Busy, response.ActiveGeneration = s.busy, s.activeGeneration
	response.ActiveRequestID, response.ActiveOperation = s.activeRequestID, s.activeOperation
	s.mu.Unlock()
	return response
}

func (s *runnerService) requestContext(r *http.Request) (context.Context, context.CancelFunc) {
	ctx, cancel := context.WithTimeout(r.Context(), requestTimeout)
	stop := context.AfterFunc(s.ctx, cancel)
	return ctx, func() {
		stop()
		cancel()
	}
}

func (s *runnerService) handler(token string) http.Handler {
	mux := http.NewServeMux()
	status := func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, s.status())
	}
	mux.HandleFunc("GET /v1/status", status)
	mux.HandleFunc("GET /health", status)
	models := func(w http.ResponseWriter, r *http.Request) {
		var body struct{}
		if !decodeRequest(w, r, &body, true) {
			return
		}
		writeJSON(w, http.StatusOK, map[string]bool{"Success": true})
	}
	mux.HandleFunc("POST /v1/models", models)
	mux.HandleFunc("POST /load", models)
	mux.HandleFunc("GET /v1/models", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]bool{"Success": true})
	})
	mux.HandleFunc("POST /v1/completions", s.completion)
	mux.HandleFunc("POST /completion", s.completion)
	tokenize := func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Content string `json:"content"`
		}
		if !decodeRequest(w, r, &body, false, "content") {
			return
		}
		if strings.ContainsRune(body.Content, 0) || !utf8.ValidString(body.Content) {
			writeHTTPError(w, http.StatusBadRequest, "content must be UTF-8 without NUL")
			return
		}
		s.operation(w, r, "tokenize", func(ctx context.Context) (any, error) {
			tokens, err := s.backend.Tokenize(ctx, body.Content)
			if tokens == nil && err == nil {
				tokens = []int32{}
			}
			return tokens, err
		})
	}
	mux.HandleFunc("POST /v1/tokenize", tokenize)
	mux.HandleFunc("POST /tokenize", tokenize)
	detokenize := func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Tokens []int32 `json:"tokens"`
		}
		if !decodeRequest(w, r, &body, false, "tokens") {
			return
		}
		for _, token := range body.Tokens {
			if token < 0 {
				writeHTTPError(w, http.StatusBadRequest, "token IDs must be nonnegative int32 values")
				return
			}
		}
		s.operation(w, r, "detokenize", func(ctx context.Context) (any, error) {
			content, err := s.backend.Detokenize(ctx, body.Tokens)
			return map[string]string{"content": content}, err
		})
	}
	mux.HandleFunc("POST /v1/detokenize", detokenize)
	mux.HandleFunc("POST /detokenize", detokenize)
	mux.HandleFunc("POST /v1/shutdown", func(w http.ResponseWriter, r *http.Request) {
		var body struct{}
		if !decodeRequest(w, r, &body, true) {
			return
		}
		writeJSON(w, http.StatusOK, map[string]bool{"Success": true})
		_ = http.NewResponseController(w).Flush()
		s.stop()
	})
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-Content-Type-Options", "nosniff")
		if token != "" {
			authorization := r.Header.Values("Authorization")
			if len(authorization) != 1 || subtle.ConstantTimeCompare([]byte(authorization[0]), []byte("Bearer "+token)) != 1 {
				w.Header().Set("WWW-Authenticate", "Bearer")
				writeHTTPError(w, http.StatusUnauthorized, "invalid runner credentials")
				return
			}
		}
		if s.ctx.Err() != nil {
			writeHTTPError(w, http.StatusServiceUnavailable, "runner is shutting down")
			return
		}
		mux.ServeHTTP(w, r)
	})
}

func (s *runnerService) completion(w http.ResponseWriter, r *http.Request) {
	var body TextCompletionsRequest
	if !decodeRequest(w, r, &body, false, "prompt") {
		return
	}
	if strings.ContainsRune(body.Prompt, 0) || !utf8.ValidString(body.Prompt) {
		writeHTTPError(w, http.StatusBadRequest, "prompt must be UTF-8 without NUL")
		return
	}
	ids := r.Header.Values("X-Ollama-Request-ID")
	if len(ids) > 1 || len(ids) == 1 && body.RequestID != "" && body.RequestID != ids[0] {
		writeHTTPError(w, http.StatusBadRequest, "conflicting request identities")
		return
	}
	if body.RequestID == "" && len(ids) == 1 {
		body.RequestID = ids[0]
	}
	if len(body.RequestID) > 128 || strings.ContainsAny(body.RequestID, "\x00\r\n \t") {
		writeHTTPError(w, http.StatusBadRequest, "request_id must be at most 128 bytes without whitespace or NUL")
		return
	}
	if _, err := planGeneration(s.backend.Status().ContextLength, 0, body.Options); err != nil {
		writeHTTPError(w, http.StatusBadRequest, err.Error())
		return
	}
	ctx, cancel := s.requestContext(r)
	defer cancel()
	request := Request{TextCompletionsRequest: body, Ctx: ctx, Responses: make(chan CompletionResponse, 1)}
	if !s.enqueueOperation(w, ctx, body.RequestID, "completion", func() {
		defer close(request.Responses)
		if err := ctx.Err(); err != nil {
			return
		}
		if err := s.backend.Generate(ctx, request); err != nil {
			var responseError ResponseError
			if !errors.As(err, &responseError) {
				responseError = ResponseError{StatusCode: http.StatusInternalServerError, ErrorMessage: err.Error()}
			}
			_ = sendResponse(ctx, request.Responses, CompletionResponse{Error: &responseError})
		}
	}) {
		return
	}
	controller := http.NewResponseController(w)
	// Long prompt evaluation is not a stalled client write.
	_ = controller.SetWriteDeadline(time.Time{})
	encoder := json.NewEncoder(w)
	started := false
	for {
		select {
		case <-ctx.Done():
			if !started && s.ctx.Err() == nil && r.Context().Err() == nil {
				writeHTTPError(w, http.StatusGatewayTimeout, ctx.Err().Error())
			}
			return
		case response, ok := <-request.Responses:
			if !ok {
				return
			}
			if !started {
				if response.Error != nil {
					writeHTTPError(w, response.Error.StatusCode, response.Error.ErrorMessage)
					return
				}
				w.Header().Set("Content-Type", "application/jsonl")
				started = true
			}
			_ = controller.SetWriteDeadline(time.Now().Add(streamWriteTimeout))
			if err := encoder.Encode(response); err != nil {
				return
			}
			if err := controller.Flush(); err != nil {
				return
			}
		}
	}
}

func (s *runnerService) operation(w http.ResponseWriter, r *http.Request, name string, operation func(context.Context) (any, error)) {
	ctx, cancel := s.requestContext(r)
	defer cancel()
	type result struct {
		value any
		err   error
	}
	resultCh := make(chan result, 1)
	if !s.enqueueOperation(w, ctx, r.Header.Get("X-Ollama-Request-ID"), name, func() {
		if err := ctx.Err(); err != nil {
			resultCh <- result{err: err}
			return
		}
		value, err := operation(ctx)
		resultCh <- result{value: value, err: err}
	}) {
		return
	}
	select {
	case <-ctx.Done():
		if r.Context().Err() == nil && s.ctx.Err() == nil {
			writeHTTPError(w, http.StatusGatewayTimeout, ctx.Err().Error())
		}
	case result := <-resultCh:
		if result.err != nil {
			var responseError ResponseError
			if errors.As(result.err, &responseError) {
				writeHTTPError(w, responseError.StatusCode, responseError.ErrorMessage)
			} else {
				writeHTTPError(w, http.StatusInternalServerError, result.err.Error())
			}
			return
		}
		writeJSON(w, http.StatusOK, result.value)
	}
}

func newHTTPServer(ctx context.Context, handler http.Handler) *http.Server {
	return &http.Server{
		Handler: handler, MaxHeaderBytes: maxHeaderBytes,
		ReadHeaderTimeout: 5 * time.Second, ReadTimeout: 30 * time.Second,
		WriteTimeout: streamWriteTimeout, IdleTimeout: 30 * time.Second,
		BaseContext: func(net.Listener) context.Context { return ctx },
	}
}

func decodeRequest(w http.ResponseWriter, r *http.Request, target any, allowEmpty bool, required ...string) bool {
	defer r.Body.Close()
	if contentType := r.Header.Get("Content-Type"); contentType != "" {
		mediaType, _, err := mime.ParseMediaType(contentType)
		if err != nil || mediaType != "application/json" {
			writeHTTPError(w, http.StatusUnsupportedMediaType, "Content-Type must be application/json")
			return false
		}
	}
	reader := http.MaxBytesReader(w, r.Body, maxRequestBytes)
	body, err := io.ReadAll(reader)
	if err != nil {
		var tooLarge *http.MaxBytesError
		if errors.As(err, &tooLarge) {
			writeHTTPError(w, http.StatusRequestEntityTooLarge, "request body exceeds limit")
		} else {
			writeHTTPError(w, http.StatusBadRequest, "failed to read request body")
		}
		return false
	}
	if allowEmpty && len(bytes.TrimSpace(body)) == 0 {
		return true
	}
	if !utf8.Valid(body) || len(bytes.TrimSpace(body)) == 0 || bytes.TrimSpace(body)[0] != '{' {
		writeHTTPError(w, http.StatusBadRequest, "request body must be a UTF-8 JSON object")
		return false
	}
	if err := validateJSON(json.NewDecoder(bytes.NewReader(body)), 0); err != nil {
		writeHTTPError(w, http.StatusBadRequest, err.Error())
		return false
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil {
		writeHTTPError(w, http.StatusBadRequest, err.Error())
		return false
	}
	for _, field := range required {
		if fields[field] == nil {
			writeHTTPError(w, http.StatusBadRequest, "missing required field "+field)
			return false
		}
	}
	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		writeHTTPError(w, http.StatusBadRequest, err.Error())
		return false
	}
	if err := decoder.Decode(new(any)); err != io.EOF {
		writeHTTPError(w, http.StatusBadRequest, "request must contain exactly one JSON object")
		return false
	}
	return true
}

func validateJSON(decoder *json.Decoder, depth int) error {
	if depth > 32 {
		return fmt.Errorf("JSON nesting exceeds limit")
	}
	token, err := decoder.Token()
	if err != nil {
		return err
	}
	if token == nil {
		return fmt.Errorf("null request fields are not supported")
	}
	delim, ok := token.(json.Delim)
	if !ok {
		return nil
	}
	if delim != '{' && delim != '[' {
		return fmt.Errorf("invalid JSON delimiter")
	}
	keys := make(map[string]bool)
	for decoder.More() {
		if delim == '{' {
			key, err := decoder.Token()
			if err != nil {
				return err
			}
			name, ok := key.(string)
			name = strings.ToLower(name)
			if !ok || keys[name] {
				return fmt.Errorf("duplicate or invalid JSON field %q", key)
			}
			keys[name] = true
		}
		if err := validateJSON(decoder, depth+1); err != nil {
			return err
		}
	}
	_, err = decoder.Token()
	return err
}

func writeHTTPError(w http.ResponseWriter, status int, message string) {
	if status < 400 || status > 599 {
		status = http.StatusInternalServerError
	}
	writeJSON(w, status, map[string]string{"error": message})
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}
