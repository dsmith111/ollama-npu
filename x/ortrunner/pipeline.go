package ortrunner

import (
	"context"
	"fmt"

	"github.com/ollama/ollama/x/ortrunner/oga"
)

func (r *Runner) Generate(ctx context.Context, req Request) error {
	r.nativeMu.Lock()
	defer r.nativeMu.Unlock()
	if r.model == nil || r.tokenizer == nil {
		return fmt.Errorf("model is not loaded")
	}
	return generate(ctx, nativeGenerationBackend{
		model: r.model, tokenizer: r.tokenizer, modelType: r.modelType,
		verified: r.markOperationVerified, singleRopeFactor: r.Status().Provider == "dml",
		promptPaddingMultiple: r.promptPaddingMultiple,
	}, r.contextLength, r.eosTokens, req)
}

type nativeGenerationBackend struct {
	model            *oga.Model
	tokenizer        *oga.Tokenizer
	modelType        string
	verified         func()
	singleRopeFactor bool
	promptPaddingMultiple int
}

func (b nativeGenerationBackend) Encode(text string) ([]int32, error) {
	return b.tokenizer.Encode(text)
}

func (b nativeGenerationBackend) NewParams() (generatorParams, error) {
	return oga.NewGeneratorParams(b.model)
}

func (b nativeGenerationBackend) NewGenerator(params generatorParams) (tokenGenerator, error) {
	p, ok := params.(*oga.GeneratorParams)
	if !ok {
		return nil, fmt.Errorf("invalid native generator parameters")
	}
	gen, err := oga.NewGenerator(b.model, p)
	if err != nil {
		return nil, err
	}
	return &nativeTokenGenerator{Generator: gen, modelType: b.modelType, verified: b.verified,
		singleRopeFactor: b.singleRopeFactor, promptPaddingMultiple: b.promptPaddingMultiple}, nil
}

type nativeTokenGenerator struct {
	*oga.Generator
	modelType        string
	length           int
	verified         func()
	singleRopeFactor bool
	promptPaddingMultiple int
}

func (g *nativeTokenGenerator) AppendTokens(tokens []int32) error {
	if err := g.Generator.AppendTokens(tokens); err != nil {
		return err
	}
	length, err := paddedPromptLength(len(tokens), g.promptPaddingMultiple)
	if err != nil {
		return err
	}
	g.length += length
	g.verified()
	return nil
}

func (g *nativeTokenGenerator) GenerateNextToken() error {
	if err := g.Generator.GenerateNextToken(); err != nil {
		return err
	}
	g.length++
	return nil
}

func (g *nativeTokenGenerator) ApplyRepetitionPenalty(history []int32, penalty float32) error {
	// Pinned Phi RoPE switching recomputes logits inside GenerateNextToken,
	// discarding externally corrected logits. Never silently lose the window penalty.
	if !g.singleRopeFactor && ((g.length == 4097 && (g.modelType == "phi3" || g.modelType == "phimoe")) ||
		(g.length == 8193 && g.modelType == "phi3small")) {
		return fmt.Errorf("finite repeat_last_n is unsupported at the pinned Phi RoPE context switch; use repeat_last_n=-1")
	}
	return g.Generator.ApplyRepetitionPenalty(history, penalty)
}

func (b nativeGenerationBackend) NewStream() (tokenDecoder, error) {
	return oga.NewTokenStream(b.tokenizer)
}

func (b nativeGenerationBackend) PromptLength(count int) (int, error) {
	return paddedPromptLength(count, b.promptPaddingMultiple)
}

func paddedPromptLength(count, multiple int) (int, error) {
	if count < 0 || count > 1<<31-1 || multiple < 0 || multiple > 1<<31-1 {
		return 0, fmt.Errorf("invalid prompt token count or sliding-window padding multiple")
	}
	if multiple <= 1 || count == 0 {
		return count, nil
	}
	// GenAI 0.12.2 appends padding to its search sequence before evaluating
	// sliding-window prompts. Reserve that space before allocating max_length.
	length := ((int64(count) + int64(multiple) - 1) / int64(multiple)) * int64(multiple)
	if length > 1<<31-1 {
		return 0, fmt.Errorf("sliding-window padded prompt exceeds int32 context capacity")
	}
	return int(length), nil
}
