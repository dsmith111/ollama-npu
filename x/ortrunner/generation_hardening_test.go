package ortrunner

import (
	"context"
	"math"
	"testing"
)

type paddedGenerationBackend struct {
	*fakeGenerationBackend
	multiple int
}

func (b paddedGenerationBackend) PromptLength(count int) (int, error) {
	return paddedPromptLength(count, b.multiple)
}

func TestPinnedSlidingWindowPromptCapacity(t *testing.T) {
	for _, test := range []struct {
		count, multiple, want int
		bad bool
	}{
		{0, 128, 0, false},
		{3, 128, 128, false},
		{128, 128, 128, false},
		{129, 128, 256, false},
		{3, 0, 3, false},
		{math.MaxInt32, 128, 0, true},
		{math.MaxInt, 128, 0, true},
	} {
		got, err := paddedPromptLength(test.count, test.multiple)
		if (err != nil) != test.bad || err == nil && got != test.want {
			t.Fatalf("count=%d multiple=%d: padded=%d error=%v", test.count, test.multiple, got, err)
		}
	}
	b := paddedGenerationBackend{newFakeGeneration(), 16}
	responses := make(chan CompletionResponse, 16)
	err := generate(context.Background(), b, 32, nil, Request{
		TextCompletionsRequest: TextCompletionsRequest{Prompt: "prompt", Options: CompletionOptions{NumPredict: ptr(3)}},
		Responses: responses,
	})
	if err != nil {
		t.Fatal(err)
	}
	if b.params.numbers["max_length"] != 19 {
		t.Fatalf("native search buffer did not reserve prompt padding: %v", b.params.numbers)
	}
	close(responses)
	var last CompletionResponse
	for response := range responses {
		last = response
	}
	if last.PromptEvalCount != 2 || last.EvalCount != 3 {
		t.Fatalf("padding counted as user prompt or decode tokens: %+v", last)
	}
	b = paddedGenerationBackend{newFakeGeneration(), 16}
	if err := generate(context.Background(), b, 15, nil, Request{
		TextCompletionsRequest: TextCompletionsRequest{Prompt: "prompt"},
		Responses: make(chan CompletionResponse, 16),
	}); err == nil || b.params.closed {
		t.Fatalf("unsafe padded prompt reached native allocation: err=%v", err)
	}
}

func TestSlidingWindowMetadataValidation(t *testing.T) {
	for _, test := range []struct {
		window string
		multiple int
		bad bool
	}{
		{`{"window_size":128}`, 128, false},
		{`{"window_size":128,"alignment":"left","slide_inputs":false}`, 0, false},
		{`{"window_size":0}`, 0, true},
		{`{"window_size":4097}`, 0, true},
		{`{"window_size":128,"alignment":"other"}`, 0, true},
		{`{"window_size":128,"slide_inputs":"false"}`, 0, true},
	} {
		meta, err := parseModelMetadata(testModelJSON(`{"filename":"model.onnx","sliding_window":`+test.window+`}`, "4096"))
		if (err != nil) != test.bad || err == nil && meta.promptPaddingMultiple != test.multiple {
			t.Fatalf("window=%s metadata=%+v err=%v", test.window, meta, err)
		}
	}
}
