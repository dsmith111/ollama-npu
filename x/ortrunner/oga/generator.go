package oga

// #include "oga.h"
// #include <stdlib.h>
import "C"

import (
	"fmt"
	"math"
	"strings"
	"unsafe"
)

// GeneratorParams wraps OgaGeneratorParams.
type GeneratorParams struct {
	ctx *C.OgaGeneratorParams
}

// NewGeneratorParams creates generator parameters from a model.
func NewGeneratorParams(model *Model) (*GeneratorParams, error) {
	if err := CheckInit(); err != nil {
		return nil, err
	}
	if model == nil || model.ctx == nil {
		return nil, fmt.Errorf("ort genai: model is closed")
	}
	var params *C.OgaGeneratorParams
	if err := ogaError(C.OgaCreateGeneratorParams(model.ctx, &params)); err != nil {
		return nil, err
	}
	return &GeneratorParams{ctx: params}, nil
}

// SetNumber sets a numeric search parameter (e.g., "temperature", "top_p", "max_length").
func (p *GeneratorParams) SetNumber(name string, value float64) error {
	if p.ctx == nil || strings.ContainsRune(name, 0) {
		return fmt.Errorf("ort genai: parameters are closed or name contains NUL")
	}
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return ogaError(C.OgaGeneratorParamsSetSearchNumber(p.ctx, cName, C.double(value)))
}

// SetBool sets a boolean search parameter (e.g., "do_sample").
func (p *GeneratorParams) SetBool(name string, value bool) error {
	if p.ctx == nil || strings.ContainsRune(name, 0) {
		return fmt.Errorf("ort genai: parameters are closed or name contains NUL")
	}
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return ogaError(C.OgaGeneratorParamsSetSearchBool(p.ctx, cName, C.bool(value)))
}

// Close frees the generator params.
func (p *GeneratorParams) Close() {
	if p.ctx != nil {
		C.OgaDestroyGeneratorParams(p.ctx)
		p.ctx = nil
	}
}

// Generator wraps OgaGenerator for token-by-token generation.
type Generator struct {
	ctx *C.OgaGenerator
}

// NewGenerator creates a generator from a model and parameters.
func NewGenerator(model *Model, params *GeneratorParams) (*Generator, error) {
	if err := CheckInit(); err != nil {
		return nil, err
	}
	if model == nil || model.ctx == nil || params == nil || params.ctx == nil {
		return nil, fmt.Errorf("ort genai: model or generator parameters are closed")
	}
	var gen *C.OgaGenerator
	if err := ogaError(C.OgaCreateGenerator(model.ctx, params.ctx, &gen)); err != nil {
		return nil, err
	}
	return &Generator{ctx: gen}, nil
}

// AppendTokens appends input token sequences to the generator.
func (g *Generator) AppendTokens(tokens []int32) error {
	if g.ctx == nil || len(tokens) == 0 {
		return fmt.Errorf("ort genai: generator is closed or input tokens are empty")
	}
	for _, token := range tokens {
		if token < 0 {
			return fmt.Errorf("ort genai: token IDs must be nonnegative")
		}
	}
	return ogaError(C.OgaGenerator_AppendTokens(g.ctx, (*C.int32_t)(unsafe.Pointer(&tokens[0])), C.size_t(len(tokens))))
}

// AppendTokenSequencesFromEncoding encodes once and evaluates the prompt.
func (g *Generator) AppendTokenSequencesFromEncoding(tok *Tokenizer, text string) error {
	tokens, err := tok.Encode(text)
	if err != nil {
		return err
	}
	return g.AppendTokens(tokens)
}

// GenerateNextToken generates the next token. Call repeatedly until IsDone.
func (g *Generator) GenerateNextToken() error {
	if g.ctx == nil {
		return fmt.Errorf("ort genai: generator is closed")
	}
	return ogaError(C.OgaGenerator_GenerateNextToken(g.ctx))
}

// IsDone returns true when generation is complete (EOS or max tokens reached).
func (g *Generator) IsDone() bool {
	if g.ctx == nil {
		return true
	}
	return bool(C.OgaGenerator_IsDone(g.ctx))
}

// GetNextTokens returns the most recently generated tokens.
func (g *Generator) GetNextTokens() ([]int32, error) {
	if g.ctx == nil {
		return nil, fmt.Errorf("ort genai: generator is closed")
	}
	var data *C.int32_t
	var count C.size_t
	if err := ogaError(C.OgaGenerator_GetNextTokens(g.ctx, &data, &count)); err != nil {
		return nil, err
	}
	if count == 0 {
		return nil, nil
	}
	if data == nil {
		return nil, fmt.Errorf("ort genai: next token data is nil")
	}
	tokens := make([]int32, count)
	copy(tokens, unsafe.Slice((*int32)(unsafe.Pointer(data)), count))
	return tokens, nil
}

// ApplyRepetitionPenalty uses the pinned CPU float32 logits copy, owned by OgaTensor.
// GetLogits computes the next decode step if necessary; SetLogits copies it back.
func (g *Generator) ApplyRepetitionPenalty(history []int32, penalty float32) error {
	if g.ctx == nil || !bool(C.OgaHasLogits()) {
		return fmt.Errorf("ort genai: finite repeat_last_n requires the GenAI logits APIs")
	}
	if penalty <= 0 || math.IsNaN(float64(penalty)) || math.IsInf(float64(penalty), 0) {
		return fmt.Errorf("ort genai: repetition penalty must be finite and positive")
	}
	var tensor *C.OgaTensor
	if err := ogaError(C.OgaGenerator_GetLogits(g.ctx, &tensor)); err != nil {
		return err
	}
	defer C.OgaDestroyTensor(tensor)
	var kind C.OgaElementType
	if err := ogaError(C.OgaTensorGetType(tensor, &kind)); err != nil {
		return err
	}
	var rank C.size_t
	if err := ogaError(C.OgaTensorGetShapeRank(tensor, &rank)); err != nil {
		return err
	}
	if kind != C.OgaElementType_float32 || rank != 3 {
		return fmt.Errorf("ort genai: expected rank-3 float32 logits")
	}
	var shape [3]C.int64_t
	if err := ogaError(C.OgaTensorGetShape(tensor, &shape[0], 3)); err != nil {
		return err
	}
	if shape[0] != 1 || shape[1] != 1 || shape[2] < 1 || shape[2] > math.MaxInt32 {
		return fmt.Errorf("ort genai: expected single-sequence logits with a positive int32 vocabulary")
	}
	var data unsafe.Pointer
	if err := ogaError(C.OgaTensorGetData(tensor, &data)); err != nil {
		return err
	}
	if data == nil {
		return fmt.Errorf("ort genai: logits data is nil")
	}
	if err := applyRepetitionPenalty(unsafe.Slice((*float32)(data), int(shape[2])), history, penalty); err != nil {
		return err
	}
	return ogaError(C.OgaGenerator_SetLogits(g.ctx, tensor))
}

func applyRepetitionPenalty(logits []float32, history []int32, penalty float32) error {
	seen := make(map[int32]bool, len(history))
	for _, token := range history {
		if token < 0 || int(token) >= len(logits) {
			return fmt.Errorf("ort genai: token %d is outside logits vocabulary", token)
		}
		if seen[token] {
			continue
		}
		seen[token] = true
		if logits[token] < 0 {
			logits[token] *= penalty
		} else {
			logits[token] /= penalty
		}
	}
	return nil
}

// Close frees the generator.
func (g *Generator) Close() {
	if g.ctx != nil {
		C.OgaDestroyGenerator(g.ctx)
		g.ctx = nil
	}
}

func NativeCancellationAvailable() bool {
	return CheckInit() == nil && bool(C.OgaHasRuntimeOption())
}

// Terminate must not run concurrently with another OGA call or Close.
// The pinned C API is not thread safe, including its session_terminated_ state.
func (g *Generator) Terminate() error {
	if g.ctx == nil {
		return fmt.Errorf("ort genai: generator is closed")
	}
	if !NativeCancellationAvailable() {
		return fmt.Errorf("ort genai: native session termination API is unavailable")
	}
	key := C.CString("terminate_session")
	defer C.free(unsafe.Pointer(key))
	value := C.CString("1")
	defer C.free(unsafe.Pointer(value))
	return ogaError(C.OgaGenerator_SetRuntimeOption(g.ctx, key, value))
}

// Shutdown cleans up global ORT GenAI state. Call once at program exit.
func Shutdown() {
	if CheckInit() == nil {
		C.OgaShutdown()
	}
}
