package ortrunner

import (
	"fmt"

	"github.com/ollama/ollama/api"
)

// CompletionOptionsFromAPI rejects generation controls which OGA cannot honor.
// Load-time device settings must be applied when constructing the child, not here.
func CompletionOptionsFromAPI(options api.Options) (CompletionOptions, error) {
	switch {
	case options.MinP != 0:
		return CompletionOptions{}, fmt.Errorf("ORT GenAI does not support min_p")
	case options.TypicalP != 0 && options.TypicalP != 1:
		return CompletionOptions{}, fmt.Errorf("ORT GenAI does not support typical_p")
	case options.PresencePenalty != 0:
		return CompletionOptions{}, fmt.Errorf("ORT GenAI does not support presence_penalty")
	case options.FrequencyPenalty != 0:
		return CompletionOptions{}, fmt.Errorf("ORT GenAI does not support frequency_penalty")
	case options.NumKeep != 0 && options.NumKeep != 4:
		return CompletionOptions{}, fmt.Errorf("ORT GenAI does not support num_keep or context shifting")
	}
	return CompletionOptions{
		Temperature: &options.Temperature, TopP: &options.TopP, TopK: &options.TopK,
		NumPredict: &options.NumPredict, NumCtx: &options.NumCtx,
		RepeatPenalty: &options.RepeatPenalty, Seed: &options.Seed,
		RepeatLastN: &options.RepeatLastN,
		Stop:        append([]string(nil), options.Stop...),
	}, nil
}
