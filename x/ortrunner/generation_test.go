package ortrunner

import (
	"context"
	"errors"
	"math"
	"reflect"
	"strings"
	"testing"
	"time"
)

func ptr[T any](v T) *T { return &v }

func TestGenerationPlanContextAndPrediction(t *testing.T) {
	for _, test := range []struct {
		name    string
		model   int
		prompt  int
		options CompletionOptions
		want    int
		bad     bool
	}{
		{"model-derived", 8192, 10, CompletionOptions{}, 8182, false},
		{"unlimited", 4096, 5, CompletionOptions{NumPredict: ptr(-1)}, 4091, false},
		{"fill-context", 4096, 5, CompletionOptions{NumPredict: ptr(-2)}, 4091, false},
		{"zero", 4096, 5, CompletionOptions{NumPredict: ptr(0)}, 0, false},
		{"positive", 4096, 5, CompletionOptions{NumPredict: ptr(3)}, 3, false},
		{"capped", 16, 14, CompletionOptions{NumPredict: ptr(math.MaxInt)}, 2, false},
		{"max-tokens", 16, 1, CompletionOptions{MaxTokens: ptr(3)}, 3, false},
		{"context-limit", 4096, 5, CompletionOptions{NumCtx: ptr(512)}, 507, false},
		{"full-context", 16, 16, CompletionOptions{}, 0, false},
		{"model-zero", 0, 1, CompletionOptions{}, 0, true},
		{"prompt-overflow", 16, 17, CompletionOptions{}, 0, true},
		{"context-overflow", 16, 1, CompletionOptions{NumCtx: ptr(17)}, 0, true},
		{"negative-context", 16, 1, CompletionOptions{NumCtx: ptr(-1)}, 0, true},
		{"invalid-predict", 16, 1, CompletionOptions{NumPredict: ptr(-3)}, 0, true},
		{"conflicting-alias", 16, 1, CompletionOptions{NumPredict: ptr(2), MaxTokens: ptr(3)}, 0, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			p, err := planGeneration(test.model, test.prompt, test.options)
			if (err != nil) != test.bad {
				t.Fatalf("error=%v, want invalid=%v", err, test.bad)
			}
			if err == nil && (p.maxNewTokens != test.want || p.maxLength > p.contextLength) {
				t.Fatalf("plan=%+v, want %d new tokens", p, test.want)
			}
		})
	}
}

func TestGenerationPlanSamplingValidation(t *testing.T) {
	p, err := planGeneration(100, 1, CompletionOptions{Temperature: ptr(float32(0)), Seed: ptr(0), RepeatPenalty: ptr(float32(1.2))})
	if err != nil || p.doSample {
		t.Fatalf("greedy plan: %+v, %v", p, err)
	}
	numbers := make(map[string]float64)
	for _, n := range p.numbers {
		numbers[n.name] = n.value
	}
	if numbers["temperature"] != 0 || numbers["random_seed"] != 0 || numbers["repetition_penalty"] < 1.1 {
		t.Fatalf("parameters=%v", numbers)
	}
	for _, o := range []CompletionOptions{
		{Temperature: ptr(float32(math.NaN()))},
		{Temperature: ptr(float32(-1))},
		{TopP: ptr(float32(-0.1))}, {TopP: ptr(float32(2))},
		{TopK: ptr(-1)}, {RepeatPenalty: ptr(float32(0))},
		{Seed: ptr(-2)}, {Seed: ptr(math.MaxInt32 + 1)},
		{Seed: ptr(1), RandomSeed: ptr(2)},
		{RepeatPenalty: ptr(float32(1)), RepetitionPenalty: ptr(float32(2))},
		{Stop: []string{""}}, {Stop: []string{strings.Repeat("x", 4097)}},
		{RepeatLastN: ptr(-2)}, {RepeatLastN: ptr(math.MaxInt32 + 1)},
	} {
		if _, err := planGeneration(100, 1, o); err == nil {
			t.Errorf("accepted invalid options %+v", o)
		}
	}
}

func TestStopFilter(t *testing.T) {
	for _, test := range []struct {
		stops     []string
		fragments []string
		want      string
		stopped   bool
	}{
		{[]string{"STOP"}, []string{"hello S", "", "T", "OP leaked"}, "hello ", true},
		{[]string{"abc", "bc"}, []string{"x", "a", "bc trailing"}, "x", true},
		{[]string{"abc"}, []string{"a", "b", "x"}, "abx", false},
		{[]string{"abc"}, []string{"text ab"}, "text ab", false},
		{[]string{"😀end"}, []string{"hello 😀", "end"}, "hello ", true},
		{[]string{"end"}, []string{"end"}, "", true},
		{nil, []string{"hello", " world"}, "hello world", false},
		{[]string{"😀end"}, []string{"hello \xf0", "\x9f\x98", "\x80", "end"}, "hello ", true},
		{nil, []string{"\xf0\x9f", "\x98\x80"}, "😀", false},
	} {
		f := stopFilter{stops: test.stops}
		var output string
		var stop bool
		for _, fragment := range test.fragments {
			text, stopped := f.push(fragment)
			output += text
			stop = stop || stopped
		}
		output += f.finish()
		if output != test.want || stop != test.stopped {
			t.Errorf("stops=%v output=%q stopped=%v; want %q/%v", test.stops, output, stop, test.want, test.stopped)
		}
	}
}

type fakeParams struct {
	numbers map[string]float64
	sample  bool
	fail    string
	closed  bool
}

func (p *fakeParams) SetNumber(name string, value float64) error {
	if p.fail == name {
		return errors.New("unsupported search setting")
	}
	p.numbers[name] = value
	return nil
}
func (p *fakeParams) SetBool(name string, value bool) error {
	if p.fail == name {
		return errors.New("unsupported search setting")
	}
	p.sample = value
	return nil
}
func (p *fakeParams) Close() { p.closed = true }

type fakeGenerator struct {
	tokens      []int32
	index       int
	appended    []int32
	appendErr   error
	generateErr error
	getErr      error
	closed      bool
	terminated  bool
	appendFn    func()
	generateFn  func()
	penaltyErr  error
	histories   [][]int32
	penalties   []float32
}

func (g *fakeGenerator) AppendTokens(tokens []int32) error {
	g.appended = append([]int32(nil), tokens...)
	if g.appendFn != nil {
		g.appendFn()
	}
	return g.appendErr
}
func (g *fakeGenerator) GenerateNextToken() error {
	g.index++
	if g.generateFn != nil {
		g.generateFn()
	}
	return g.generateErr
}
func (g *fakeGenerator) GetNextTokens() ([]int32, error) {
	if g.getErr != nil {
		return nil, g.getErr
	}
	return []int32{g.tokens[g.index-1]}, nil
}
func (g *fakeGenerator) IsDone() bool     { return g.index >= len(g.tokens) }
func (g *fakeGenerator) Terminate() error { g.terminated = true; return nil }
func (g *fakeGenerator) Close()           { g.closed = true }
func (g *fakeGenerator) ApplyRepetitionPenalty(history []int32, penalty float32) error {
	g.histories = append(g.histories, append([]int32(nil), history...))
	g.penalties = append(g.penalties, penalty)
	return g.penaltyErr
}

type fakeDecoder struct {
	text   map[int32]string
	closed bool
	fail   bool
}

func (d *fakeDecoder) Decode(token int32) (string, error) {
	if d.fail {
		return "", errors.New("decode failed")
	}
	return d.text[token], nil
}
func (d *fakeDecoder) Close() { d.closed = true }

type fakeGenerationBackend struct {
	prompt       []int32
	encodeErr    error
	streamErr    error
	paramsErr    error
	generatorErr error
	params       *fakeParams
	gen          *fakeGenerator
	decoder      *fakeDecoder
}

func newFakeGeneration() *fakeGenerationBackend {
	return &fakeGenerationBackend{
		prompt:  []int32{10, 11},
		params:  &fakeParams{numbers: make(map[string]float64)},
		gen:     &fakeGenerator{tokens: []int32{1, 2, 3}},
		decoder: &fakeDecoder{text: map[int32]string{1: "", 2: "hello ", 3: "world"}},
	}
}
func (b *fakeGenerationBackend) Encode(string) ([]int32, error)      { return b.prompt, b.encodeErr }
func (b *fakeGenerationBackend) NewParams() (generatorParams, error) { return b.params, b.paramsErr }
func (b *fakeGenerationBackend) NewGenerator(generatorParams) (tokenGenerator, error) {
	return b.gen, b.generatorErr
}
func (b *fakeGenerationBackend) NewStream() (tokenDecoder, error) { return b.decoder, b.streamErr }

func runFakeGeneration(t *testing.T, b *fakeGenerationBackend, options CompletionOptions, eos []int32) ([]CompletionResponse, error) {
	t.Helper()
	responses := make(chan CompletionResponse, 16)
	err := generate(context.Background(), b, 100, eos, Request{
		TextCompletionsRequest: TextCompletionsRequest{Prompt: "prompt", Options: options},
		Responses:              responses,
	})
	close(responses)
	var result []CompletionResponse
	for response := range responses {
		result = append(result, response)
	}
	return result, err
}

func TestGenerationCountsEmptyFragmentsAndStats(t *testing.T) {
	b := newFakeGeneration()
	responses, err := runFakeGeneration(t, b, CompletionOptions{Temperature: ptr(float32(0))}, nil)
	if err != nil {
		t.Fatal(err)
	}
	last := responses[len(responses)-1]
	if !last.Done || last.DoneReason != DoneReasonStop || last.EvalCount != 3 || last.PromptEvalCount != 2 {
		t.Fatalf("final response=%+v", last)
	}
	if last.PromptEvalDuration < 0 || last.EvalDuration < 0 {
		t.Fatalf("negative stats: %+v", last)
	}
	if b.params.sample || !b.params.closed || !b.gen.closed || !b.decoder.closed || len(b.gen.appended) != 2 {
		t.Fatalf("sampling or cleanup incorrect: %+v", b)
	}
}

func TestGenerationStopAndLengthReasons(t *testing.T) {
	for _, test := range []struct {
		name    string
		options CompletionOptions
		eos     []int32
		count   int
		reason  int
		content string
	}{
		{"length", CompletionOptions{NumPredict: ptr(2)}, nil, 2, DoneReasonLength, "hello "},
		{"eos-at-limit", CompletionOptions{NumPredict: ptr(2)}, []int32{2}, 2, DoneReasonStop, ""},
		{"stop-across-tokens", CompletionOptions{Stop: []string{"hello world"}}, nil, 3, DoneReasonStop, ""},
		{"zero", CompletionOptions{NumPredict: ptr(0)}, nil, 0, DoneReasonLength, ""},
	} {
		t.Run(test.name, func(t *testing.T) {
			responses, err := runFakeGeneration(t, newFakeGeneration(), test.options, test.eos)
			if err != nil {
				t.Fatal(err)
			}
			last := responses[len(responses)-1]
			var content string
			for _, response := range responses {
				content += response.Content
			}
			if last.EvalCount != test.count || last.DoneReason != test.reason || content != test.content {
				t.Fatalf("final=%+v content=%q", last, content)
			}
		})
	}
}

func TestGenerationFailuresAndCleanup(t *testing.T) {
	for _, stage := range []string{"encode", "params", "setting", "bool", "generator", "append", "stream", "generate", "tokens", "decode", "penalty"} {
		t.Run(stage, func(t *testing.T) {
			b := newFakeGeneration()
			fail := errors.New("test failure")
			switch stage {
			case "encode":
				b.encodeErr = fail
			case "params":
				b.paramsErr = fail
			case "setting":
				b.params.fail = "max_length"
			case "bool":
				b.params.fail = "do_sample"
			case "generator":
				b.generatorErr = fail
			case "append":
				b.gen.appendErr = fail
			case "stream":
				b.streamErr = fail
			case "generate":
				b.gen.generateErr = fail
			case "tokens":
				b.gen.getErr = fail
			case "decode":
				b.decoder.fail = true
			case "penalty":
				b.gen.penaltyErr = fail
			}
			options := CompletionOptions{}
			if stage == "penalty" {
				options = CompletionOptions{RepeatLastN: ptr(64), RepeatPenalty: ptr(float32(1.1))}
			}
			responses, err := runFakeGeneration(t, b, options, nil)
			if err == nil {
				t.Fatal("failure suppressed")
			}
			for _, response := range responses {
				if response.Done {
					t.Fatal("sent success after failure")
				}
			}
			if stage != "encode" && stage != "params" && !b.params.closed {
				t.Fatal("params leaked")
			}
			if b.gen.appended != nil && !b.gen.closed {
				t.Fatal("generator leaked")
			}
		})
	}
}

func TestGenerationCancellationDuringBlockedResponse(t *testing.T) {
	b := newFakeGeneration()
	ctx, cancel := context.WithCancel(context.Background())
	responses := make(chan CompletionResponse)
	done := make(chan error, 1)
	go func() {
		done <- generate(ctx, b, 100, nil, Request{TextCompletionsRequest: TextCompletionsRequest{Prompt: "prompt"}, Responses: responses})
	}()
	cancel()
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("error=%v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("cancellation blocked")
	}
}

func TestSendResponseCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := sendResponse(ctx, make(chan CompletionResponse), CompletionResponse{Done: true}); !errors.Is(err, context.Canceled) {
		t.Fatalf("send error=%v", err)
	}
}

func TestGenerationFiniteRepetitionWindow(t *testing.T) {
	b := newFakeGeneration()
	b.prompt = []int32{9, 10, 11}
	responses, err := runFakeGeneration(t, b, CompletionOptions{RepeatLastN: ptr(2), RepeatPenalty: ptr(float32(1.1))}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(responses) == 0 || b.params.numbers["repetition_penalty"] != 1 {
		t.Fatal("native whole-history penalty was not disabled")
	}
	want := [][]int32{{10, 11}, {11, 1}, {1, 2}}
	if !reflect.DeepEqual(b.gen.histories, want) {
		t.Fatalf("penalty history=%v want=%v", b.gen.histories, want)
	}
	for _, window := range []int{-1, 0} {
		b := newFakeGeneration()
		if _, err := runFakeGeneration(t, b, CompletionOptions{RepeatLastN: ptr(window), RepeatPenalty: ptr(float32(1.1))}, nil); err != nil {
			t.Fatal(err)
		}
		if len(b.gen.histories) != 0 || window == 0 && b.params.numbers["repetition_penalty"] != 1 || window == -1 && b.params.numbers["repetition_penalty"] < 1.09 {
			t.Fatalf("incorrect repetition semantics for window %d", window)
		}
	}
}

func TestGenerationSeparatesPrefillAndDecodeTiming(t *testing.T) {
	b := newFakeGeneration()
	b.gen.appendFn = func() { time.Sleep(20 * time.Millisecond) }
	b.gen.generateFn = func() { time.Sleep(10 * time.Millisecond) }
	responses, err := runFakeGeneration(t, b, CompletionOptions{NumPredict: ptr(1)}, nil)
	if err != nil {
		t.Fatal(err)
	}
	last := responses[len(responses)-1]
	if last.PromptEvalDuration < 20*time.Millisecond || last.EvalDuration < 10*time.Millisecond || last.EvalCount != 1 {
		t.Fatalf("prefill/decode times=%+v", last)
	}
}

func TestCancellationDuringAppendDoesNotTerminateConcurrently(t *testing.T) {
	b := newFakeGeneration()
	entered, release := make(chan struct{}), make(chan struct{})
	b.gen.appendFn = func() {
		close(entered)
		<-release
		if b.gen.terminated || b.gen.closed {
			t.Error("terminated or freed a generator while AppendTokens was running")
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		done <- generate(ctx, b, 100, nil, Request{TextCompletionsRequest: TextCompletionsRequest{Prompt: "prompt"}, Responses: make(chan CompletionResponse)})
	}()
	<-entered
	cancel()
	select {
	case <-done:
		t.Fatal("generation returned while native prompt evaluation was blocked")
	default:
	}
	close(release)
	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) || !b.gen.terminated || !b.gen.closed || !b.params.closed || b.gen.index != 0 {
			t.Fatalf("unsafe cancellation or cleanup: err=%v, gen=%+v", err, b.gen)
		}
	case <-time.After(time.Second):
		t.Fatal("generation failed to return after native prompt evaluation")
	}
}

func TestNativePhiWindowSwitchIsExplicitlyRejected(t *testing.T) {
	for _, test := range []struct {
		model  string
		length int
	}{{"phi3", 4097}, {"phimoe", 4097}, {"phi3small", 8193}} {
		g := nativeTokenGenerator{modelType: test.model, length: test.length}
		if err := g.ApplyRepetitionPenalty([]int32{1}, 1.1); err == nil || !strings.Contains(err.Error(), "RoPE") {
			t.Fatalf("accepted a discarded finite window correction: %v", err)
		}
	}
}
