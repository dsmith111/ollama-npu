package ortrunner

import (
	"fmt"
	"math"
	"strings"
	"unicode/utf8"
)

type searchNumber struct {
	name  string
	value float64
}

type generationPlan struct {
	contextLength int
	maxLength     int
	maxNewTokens  int
	doSample      bool
	numbers       []searchNumber
	stops         []string
	repeatLastN   int
	repeatPenalty float32
}

func optionValue[T any](p *T, fallback T) T {
	if p != nil {
		return *p
	}
	return fallback
}

func planGeneration(modelContext, promptTokens int, o CompletionOptions) (generationPlan, error) {
	var p generationPlan
	if modelContext <= 0 || modelContext > math.MaxInt32 {
		return p, fmt.Errorf("invalid model context length %d", modelContext)
	}
	p.contextLength = optionValue(o.NumCtx, modelContext)
	if p.contextLength == 0 {
		p.contextLength = modelContext
	}
	if p.contextLength < 1 || p.contextLength > modelContext {
		return p, fmt.Errorf("num_ctx must be between 1 and model context length %d", modelContext)
	}
	if promptTokens < 0 || promptTokens > p.contextLength {
		return p, fmt.Errorf("prompt has %d tokens but context length is %d", promptTokens, p.contextLength)
	}
	predict := optionValue(o.NumPredict, -1)
	if o.MaxTokens != nil {
		if o.NumPredict != nil && *o.NumPredict != *o.MaxTokens {
			return p, fmt.Errorf("max_tokens and num_predict disagree")
		}
		predict = *o.MaxTokens
	}
	if predict < -2 {
		return p, fmt.Errorf("num_predict must be -2, -1, zero, or positive")
	}
	remaining := p.contextLength - promptTokens
	switch predict {
	case -2, -1:
		p.maxNewTokens = remaining
	default:
		p.maxNewTokens = min(predict, remaining)
	}
	p.maxLength = promptTokens + p.maxNewTokens
	temperature := float64(optionValue(o.Temperature, float32(0.8)))
	topP := float64(optionValue(o.TopP, float32(0.9)))
	topK := optionValue(o.TopK, 40)
	if !finite(temperature) || temperature < 0 {
		return p, fmt.Errorf("temperature must be finite and nonnegative")
	}
	if !finite(topP) || topP < 0 || topP > 1 {
		return p, fmt.Errorf("top_p must be in [0, 1]")
	}
	if topK < 0 || topK > math.MaxInt32 {
		return p, fmt.Errorf("top_k must be a nonnegative int32")
	}
	if o.RepeatPenalty != nil && o.RepetitionPenalty != nil && *o.RepeatPenalty != *o.RepetitionPenalty {
		return p, fmt.Errorf("repeat_penalty and repetition_penalty disagree")
	}
	penalty := optionValue(o.RepetitionPenalty, optionValue(o.RepeatPenalty, float32(1)))
	if !finite(float64(penalty)) || penalty <= 0 {
		return p, fmt.Errorf("repetition_penalty must be finite and positive")
	}
	// The GenAI spelling without repeat_last_n retains its whole-history semantics.
	p.repeatLastN = optionValue(o.RepeatLastN, -1)
	if p.repeatLastN < -1 || p.repeatLastN > math.MaxInt32 {
		return p, fmt.Errorf("repeat_last_n must be -1 or a nonnegative int32")
	}
	p.repeatPenalty = penalty
	if p.repeatLastN >= 0 {
		// Finite windows are applied to logits, never approximated as whole-history penalties.
		penalty = 1
	}
	if o.Seed != nil && o.RandomSeed != nil && *o.Seed != *o.RandomSeed {
		return p, fmt.Errorf("seed and random_seed disagree")
	}
	seed := optionValue(o.RandomSeed, optionValue(o.Seed, -1))
	if seed < -1 || seed > math.MaxInt32 {
		return p, fmt.Errorf("random_seed must be -1 or a nonnegative int32")
	}
	if len(o.Stop) > 64 {
		return p, fmt.Errorf("at most 64 stop sequences are supported")
	}
	for _, stop := range o.Stop {
		if stop == "" || len(stop) > 4096 || !utf8.ValidString(stop) || strings.ContainsRune(stop, 0) {
			return p, fmt.Errorf("stop sequences must be nonempty UTF-8 strings without NUL of at most 4096 bytes")
		}
	}
	p.stops = append([]string(nil), o.Stop...)
	p.doSample = temperature > 0
	p.numbers = []searchNumber{
		{"max_length", float64(p.maxLength)},
		{"batch_size", 1}, {"num_beams", 1}, {"num_return_sequences", 1},
		{"min_length", 0}, {"no_repeat_ngram_size", 0},
		{"temperature", temperature}, {"top_p", topP}, {"top_k", float64(topK)},
		{"repetition_penalty", float64(penalty)}, {"random_seed", float64(seed)},
	}
	return p, nil
}

func finite(f float64) bool { return !math.IsNaN(f) && !math.IsInf(f, 0) }

type stopFilter struct {
	stops   []string
	pending string
	stopped bool
}

func (f *stopFilter) push(fragment string) (string, bool) {
	if f.stopped {
		return "", true
	}
	f.pending += fragment
	cut := -1
	for _, stop := range f.stops {
		if index := strings.Index(f.pending, stop); index >= 0 && (cut < 0 || index < cut) {
			cut = index
		}
	}
	if cut >= 0 {
		text := f.pending[:cut]
		f.pending = ""
		f.stopped = true
		return text, true
	}
	hold := 0
	for _, stop := range f.stops {
		for n := min(len(stop)-1, len(f.pending)); n > hold; n-- {
			if strings.HasSuffix(f.pending, stop[:n]) {
				hold = n
				break
			}
		}
	}
	cut = len(f.pending) - hold
	// Tokenizer streams may return only the leading bytes of a rune.
	for cut > 0 && !utf8.ValidString(f.pending[:cut]) {
		cut--
	}
	// Never split a UTF-8 rune at the withheld stop prefix.
	for cut > 0 && cut < len(f.pending) && !utf8.RuneStart(f.pending[cut]) {
		cut--
	}
	text := f.pending[:cut]
	f.pending = f.pending[cut:]
	return text, false
}

func (f *stopFilter) finish() string {
	text := f.pending
	f.pending = ""
	return text
}
