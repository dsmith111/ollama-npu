package ortrunner

import (
	"context"
	"time"
)

// CompletionOptions contains only options implemented by this runner.
// Pointers preserve explicit zero values, including greedy sampling and no prediction.
type CompletionOptions struct {
	Temperature       *float32 `json:"temperature,omitempty"`
	TopP              *float32 `json:"top_p,omitempty"`
	TopK              *int     `json:"top_k,omitempty"`
	MaxTokens         *int     `json:"max_tokens,omitempty"`
	NumPredict        *int     `json:"num_predict,omitempty"`
	NumCtx            *int     `json:"num_ctx,omitempty"`
	RepeatPenalty     *float32 `json:"repeat_penalty,omitempty"`
	RepetitionPenalty *float32 `json:"repetition_penalty,omitempty"`
	RepeatLastN       *int     `json:"repeat_last_n,omitempty"`
	Seed              *int     `json:"seed,omitempty"`
	RandomSeed        *int     `json:"random_seed,omitempty"`
	Stop              []string `json:"stop,omitempty"`
}

// ChatTemplateMessage is one message for the runner to render with the model's
// own chat template (POST /v1/chat_template).
type ChatTemplateMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type TextCompletionsRequest struct {
	Prompt    string            `json:"prompt"`
	Options   CompletionOptions `json:"options,omitempty"`
	RequestID string            `json:"request_id,omitempty"`
}

const ProtocolVersion = 1

// ReadyResponse is the single JSON readiness record written to child stdout.
type ReadyResponse struct {
	Event string `json:"event"`
	Port  int    `json:"port"`
	StatusResponse
}

type Request struct {
	TextCompletionsRequest
	Responses chan CompletionResponse
	Ctx       context.Context
}

const (
	DoneReasonStop = iota
	DoneReasonLength
	DoneReasonConnectionClosed
)

// ResponseError retains the api.StatusError wire representation.
type ResponseError struct {
	StatusCode   int
	Status       string
	ErrorMessage string `json:"error"`
}

func (e ResponseError) Error() string { return e.ErrorMessage }

type CompletionResponse struct {
	Content    string         `json:"content,omitempty"`
	Done       bool           `json:"done"`
	DoneReason int            `json:"done_reason"`
	Error      *ResponseError `json:"error,omitempty"`

	PromptEvalCount    int           `json:"prompt_eval_count,omitempty"`
	PromptEvalDuration time.Duration `json:"prompt_eval_duration,omitempty"`
	EvalCount          int           `json:"eval_count,omitempty"`
	EvalDuration       time.Duration `json:"eval_duration,omitempty"`
}

type StatusResponse struct {
	ProtocolVersion       int    `json:"protocol_version"`
	Status                int    `json:"status"`
	Progress              int    `json:"progress"`
	Memory                uint64 `json:"memory,omitempty"`
	NPUMemory             uint64 `json:"npu_memory,omitempty"` // the part of Memory the QNN stages hold
	ContextLength         int    `json:"context_length"`
	PromptPaddingMultiple int    `json:"prompt_padding_multiple,omitempty"`
	Provider              string `json:"provider"`
	ProviderState         string `json:"provider_state"`
	NativeCancelAvailable bool   `json:"native_cancel_available"`
	CancellationMode      string `json:"cancellation_mode"`
	CancellationGraceMS   int    `json:"cancellation_grace_ms"`
	WatchdogRequired      bool   `json:"watchdog_required"`
	QueueDepth            int    `json:"queue_depth,omitempty"`
	Busy                  bool   `json:"busy"`
	ActiveGeneration      uint64 `json:"active_generation"`
	ActiveRequestID       string `json:"active_request_id,omitempty"`
	ActiveOperation       string `json:"active_operation,omitempty"`
	ProviderConfigured    bool   `json:"provider_configured"`
	ProviderLoaded        bool   `json:"provider_loaded"`
	OperationVerified     bool   `json:"operation_verified"`
	NPUVerified           bool   `json:"npu_verified"`
	Hybrid                bool   `json:"hybrid"`
	StrictQNN             bool   `json:"strict_qnn"`
}
