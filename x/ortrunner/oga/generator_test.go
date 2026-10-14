package oga

import (
	"reflect"
	"testing"
)

func TestFiniteRepetitionPenaltyLogits(t *testing.T) {
	logits := []float32{8, -4, 2, 0}
	if err := applyRepetitionPenalty(logits, []int32{0, 1, 1, 3}, 2); err != nil {
		t.Fatal(err)
	}
	if want := []float32{4, -8, 2, 0}; !reflect.DeepEqual(logits, want) {
		t.Fatalf("logits=%v want=%v", logits, want)
	}
	for _, token := range []int32{-1, 4} {
		if err := applyRepetitionPenalty(logits, []int32{token}, 2); err == nil {
			t.Fatalf("accepted token %d outside vocabulary", token)
		}
	}
}

func TestClosedNativeHandlesFailWithoutCallingDLL(t *testing.T) {
	var g Generator
	if g.GenerateNextToken() == nil || g.AppendTokens([]int32{1}) == nil || g.Terminate() == nil || g.ApplyRepetitionPenalty([]int32{1}, 1.1) == nil {
		t.Fatal("closed generator accepted a native operation")
	}
	if _, err := g.GetNextTokens(); err == nil || !g.IsDone() {
		t.Fatal("closed generator returned tokens or remained active")
	}
	var p GeneratorParams
	if p.SetNumber("max_length", 4) == nil || p.SetBool("do_sample", true) == nil {
		t.Fatal("closed params accepted a native operation")
	}
	var tok Tokenizer
	if _, err := tok.Encode("text"); err == nil {
		t.Fatal("closed tokenizer encoded text")
	}
	var stream TokenStream
	if _, err := stream.Decode(1); err == nil {
		t.Fatal("closed stream decoded text")
	}
	var config Config
	if config.ClearProviders() == nil || config.AppendProvider("QNN") == nil || config.SetProviderOption("QNN", "soc_model", "60") == nil {
		t.Fatal("closed config accepted a native operation")
	}
}
