package ortrunner

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestCompletionOptionsFromAPI(t *testing.T) {
	options := api.DefaultOptions()
	options.Temperature = 0
	options.NumPredict = 0
	options.Seed = 0
	options.Stop = []string{"stop"}
	converted, err := CompletionOptionsFromAPI(options)
	if err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(TextCompletionsRequest{Prompt: "prompt", Options: converted})
	if err != nil {
		t.Fatal(err)
	}
	for _, field := range []string{`"temperature":0`, `"num_predict":0`, `"seed":0`, `"stop":["stop"]`} {
		if !strings.Contains(string(data), field) {
			t.Errorf("missing explicit zero or stop field %s in %s", field, data)
		}
	}
	options.Stop[0] = "changed"
	if converted.Stop[0] != "stop" {
		t.Fatal("stop options alias the caller's slice")
	}
}

func TestCompletionOptionsFromAPIRejectsUnsupportedSettings(t *testing.T) {
	for _, mutate := range []func(*api.Options){
		func(o *api.Options) { o.MinP = 0.1 },
		func(o *api.Options) { o.TypicalP = 0.5 },
		func(o *api.Options) { o.PresencePenalty = 0.1 },
		func(o *api.Options) { o.FrequencyPenalty = 0.1 },
		func(o *api.Options) { o.NumKeep = 2 },
	} {
		o := api.DefaultOptions()
		mutate(&o)
		if _, err := CompletionOptionsFromAPI(o); err == nil {
			t.Errorf("accepted unsupported options %+v", o)
		}
	}
	o := api.DefaultOptions()
	o.RepeatPenalty, o.RepeatLastN = 1.2, -1
	if _, err := CompletionOptionsFromAPI(o); err != nil {
		t.Fatal(err)
	}
	o.RepeatPenalty, o.RepeatLastN = 1.1, 64
	converted, err := CompletionOptionsFromAPI(o)
	if err != nil || converted.RepeatLastN == nil || *converted.RepeatLastN != 64 || *converted.RepeatPenalty != 1.1 {
		t.Fatalf("historical defaults were not preserved: %+v, %v", converted, err)
	}
}
