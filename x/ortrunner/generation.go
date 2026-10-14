package ortrunner

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"
	"unicode/utf8"
)

type generatorParams interface {
	SetNumber(string, float64) error
	SetBool(string, bool) error
	Close()
}

type tokenGenerator interface {
	AppendTokens([]int32) error
	GenerateNextToken() error
	GetNextTokens() ([]int32, error)
	IsDone() bool
	Terminate() error
	ApplyRepetitionPenalty([]int32, float32) error
	Close()
}

type tokenDecoder interface {
	Decode(int32) (string, error)
	Close()
}

type generationBackend interface {
	Encode(string) ([]int32, error)
	NewParams() (generatorParams, error)
	NewGenerator(generatorParams) (tokenGenerator, error)
	NewStream() (tokenDecoder, error)
}

type promptLengthBackend interface {
	PromptLength(int) (int, error)
}

func sendResponse(ctx context.Context, responses chan CompletionResponse, response CompletionResponse) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	select {
	case responses <- response:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func generate(ctx context.Context, backend generationBackend, modelContext int, eosTokens []int32, req Request) (err error) {
	if err := ctx.Err(); err != nil {
		return err
	}
	if !utf8.ValidString(req.Prompt) || strings.ContainsRune(req.Prompt, 0) {
		return ResponseError{StatusCode: 400, ErrorMessage: "prompt must be UTF-8 without NUL"}
	}
	promptStart := time.Now()
	tokens, err := backend.Encode(req.Prompt)
	if err != nil {
		return fmt.Errorf("tokenize prompt: %w", err)
	}
	promptLength := len(tokens)
	if padded, ok := backend.(promptLengthBackend); ok {
		promptLength, err = padded.PromptLength(promptLength)
		if err != nil {
			return ResponseError{StatusCode: 400, ErrorMessage: err.Error()}
		}
	}
	plan, err := planGeneration(modelContext, promptLength, req.Options)
	if err != nil {
		return ResponseError{StatusCode: 400, ErrorMessage: err.Error()}
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if plan.maxNewTokens == 0 {
		return sendResponse(ctx, req.Responses, CompletionResponse{
			Done: true, DoneReason: DoneReasonLength,
			// Encoding is not prompt evaluation; no native inference ran.
		})
	}
	if len(tokens) == 0 {
		return ResponseError{StatusCode: 400, ErrorMessage: "prompt tokenization is empty"}
	}
	params, err := backend.NewParams()
	if err != nil {
		return err
	}
	defer params.Close()
	if err := params.SetBool("do_sample", plan.doSample); err != nil {
		return fmt.Errorf("set do_sample: %w", err)
	}
	for _, option := range plan.numbers {
		if err := params.SetNumber(option.name, option.value); err != nil {
			return fmt.Errorf("set %s: %w", option.name, err)
		}
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	gen, err := backend.NewGenerator(params)
	if err != nil {
		return err
	}
	defer gen.Close()
	defer func() {
		if ctx.Err() != nil {
			// OGA's C API is not thread safe. Terminate only after the native call returns.
			err = errors.Join(ctx.Err(), gen.Terminate(), err)
		}
	}()
	if err := ctx.Err(); err != nil {
		return err
	}
	if err := gen.AppendTokens(tokens); err != nil {
		return err
	}
	promptDuration := time.Since(promptStart)
	if err := ctx.Err(); err != nil {
		return err
	}
	stream, err := backend.NewStream()
	if err != nil {
		return err
	}
	defer stream.Close()
	filter := stopFilter{stops: plan.stops}
	evalCount := 0
	doneReason := DoneReasonStop
	var evalDuration time.Duration
	var history []int32
	if plan.repeatLastN > 0 && plan.repeatPenalty != 1 {
		history = append([]int32(nil), tokens[max(0, len(tokens)-plan.repeatLastN):]...)
	}
	stopped := false
	for evalCount < plan.maxNewTokens {
		if err := ctx.Err(); err != nil {
			return err
		}
		if gen.IsDone() {
			break
		}
		tokenStart := time.Now()
		if len(history) > 0 {
			if err := gen.ApplyRepetitionPenalty(history, plan.repeatPenalty); err != nil {
				return fmt.Errorf("apply repeat_last_n: %w", err)
			}
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := gen.GenerateNextToken(); err != nil {
			return err
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		nextTokens, err := gen.GetNextTokens()
		if err != nil {
			return err
		}
		if len(nextTokens) != 1 {
			return fmt.Errorf("single-sequence generator returned %d token IDs, expected 1", len(nextTokens))
		}
		evalDuration += time.Since(tokenStart)
		evalCount++
		token := nextTokens[0]
		if len(history) > 0 {
			if len(history) == plan.repeatLastN {
				copy(history, history[1:])
				history[len(history)-1] = token
			} else {
				history = append(history, token)
			}
		}
		for _, eos := range eosTokens {
			if token == eos {
				stopped = true
				break
			}
		}
		if stopped {
			break
		}
		text, err := stream.Decode(token)
		if err != nil {
			return err
		}
		content, stop := filter.push(text)
		if content != "" {
			if !utf8.ValidString(content) {
				return fmt.Errorf("tokenizer stream returned invalid UTF-8")
			}
			if err := sendResponse(ctx, req.Responses, CompletionResponse{Content: content}); err != nil {
				return err
			}
		}
		if stop {
			stopped = true
			break
		}
	}
	if !stopped && evalCount >= plan.maxNewTokens {
		doneReason = DoneReasonLength
	}
	if content := filter.finish(); content != "" {
		if !utf8.ValidString(content) {
			return fmt.Errorf("tokenizer stream ended with incomplete or invalid UTF-8")
		}
		if err := sendResponse(ctx, req.Responses, CompletionResponse{Content: content}); err != nil {
			return err
		}
	}
	return sendResponse(ctx, req.Responses, CompletionResponse{
		Done: true, DoneReason: doneReason,
		PromptEvalCount: len(tokens), PromptEvalDuration: promptDuration,
		EvalCount: evalCount, EvalDuration: evalDuration,
	})
}
