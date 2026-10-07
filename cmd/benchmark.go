//go:build windows

package cmd

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/spf13/cobra"
)

// BenchmarkNPUHandler runs benchmark comparison across one or more models.
// Use comma-separated model names to compare, e.g.:
//
//	ollama benchmark npu --model phi3:mini,dsmith111/phi3:mini-qnn
func BenchmarkNPUHandler(cmd *cobra.Command, args []string) error {
	modelFlag, _ := cmd.Flags().GetString("model")
	numCtx, _ := cmd.Flags().GetInt("ctx")
	numPredict, _ := cmd.Flags().GetInt("predict")
	epochs, _ := cmd.Flags().GetInt("epochs")
	warmup, _ := cmd.Flags().GetInt("warmup")
	jsonFile, _ := cmd.Flags().GetString("json")

	models := strings.Split(modelFlag, ",")
	for i := range models {
		models[i] = strings.TrimSpace(models[i])
	}

	fmt.Println("=== Ollama NPU Benchmark ===")
	fmt.Printf("Models:  %s\n", strings.Join(models, ", "))
	fmt.Printf("Context: %d\n", numCtx)
	fmt.Printf("Predict: %d\n", numPredict)
	fmt.Printf("Epochs:  %d (warmup: %d)\n", epochs, warmup)
	fmt.Println()

	client, err := api.ClientFromEnvironment()
	if err != nil {
		return fmt.Errorf("could not create client: %w", err)
	}

	var results []benchResult

	for _, modelName := range models {
		fmt.Printf("--- %s ---\n", modelName)

		metrics, err := runBenchEpochs(client, modelName, numCtx, numPredict, epochs, warmup)
		if err != nil {
			fmt.Printf("  ERROR: %v\n\n", err)
			continue
		}

		br := benchResult{Provider: modelName, Epochs: metrics}
		results = append(results, br)

		printProfileResults(modelName, metrics)
		fmt.Println()

		// Unload model before next one
		unloadBenchModel(client, modelName)
	}

	// Print comparison table if multiple models
	if len(results) >= 2 {
		printComparisonTable(results)
	}

	// Write JSON output
	if jsonFile != "" {
		data, err := json.MarshalIndent(map[string]any{
			"models":  models,
			"ctx":     numCtx,
			"predict": numPredict,
			"epochs":  epochs,
			"results": results,
		}, "", "  ")
		if err != nil {
			return fmt.Errorf("marshal JSON: %w", err)
		}
		if err := os.WriteFile(jsonFile, data, 0o644); err != nil {
			return fmt.Errorf("write JSON: %w", err)
		}
		fmt.Printf("Results written to %s\n", jsonFile)
	}

	return nil
}

type epochMetrics struct {
	Epoch              int           `json:"epoch"`
	TTFT               time.Duration `json:"ttft_ns"`
	PrefillCount       int           `json:"prefill_count"`
	PrefillDuration    time.Duration `json:"prefill_duration_ns"`
	GenerateCount      int           `json:"generate_count"`
	GenerateDuration   time.Duration `json:"generate_duration_ns"`
	LoadDuration       time.Duration `json:"load_duration_ns"`
	TotalDuration      time.Duration `json:"total_duration_ns"`
}

type benchResult struct {
	Provider string         `json:"provider"`
	Epochs   []epochMetrics `json:"epochs"`
}

// safeTokPerSec computes tokens/sec, returning 0 if duration is too small to be meaningful.
// Some runners (ORT GenAI) report near-zero prompt eval duration because prefill
// is folded into graph execution and reflected in TTFT instead.
func safeTokPerSec(tokens int, dur time.Duration) float64 {
	if dur < time.Millisecond || tokens <= 0 {
		return 0
	}
	return float64(tokens) / dur.Seconds()
}

// Prompt word list for varied prompts per epoch.
var benchPromptWords = []string{
	"the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
	"a", "bright", "sunny", "day", "in", "the", "meadow", "where",
	"flowers", "bloom", "and", "birds", "sing", "their", "morning",
	"songs", "while", "gentle", "breeze", "carries", "sweet", "scent",
	"of", "pine", "trees", "across", "rolling", "hills", "toward",
	"distant", "mountains", "covered", "with", "fresh", "snow",
	"beneath", "clear", "blue", "sky", "children", "play", "near",
	"old", "stone", "bridge", "that", "crosses", "winding", "river",
}

func generateBenchPrompt(targetTokens int, epoch int) string {
	targetWords := int(float64(targetTokens) / 1.3)
	if targetWords < 1 {
		targetWords = 1
	}
	offset := epoch * 7
	n := len(benchPromptWords)
	words := make([]string, targetWords)
	for i := range words {
		words[i] = benchPromptWords[((i+offset)%n+n)%n]
	}
	return strings.Join(words, " ")
}

func runBenchEpochs(client *api.Client, model string, numCtx, numPredict, epochs, warmup int) ([]epochMetrics, error) {
	// Warmup
	for i := range warmup {
		fmt.Printf("  Warmup %d/%d...\n", i+1, warmup)

		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
		req := buildBenchRequest(model, numCtx, numPredict, -(i + 1))
		err := client.Generate(ctx, req, func(resp api.GenerateResponse) error {
			return nil
		})
		cancel()
		if err != nil {
			fmt.Printf("  Warmup failed: %v\n", err)
		}
	}

	// Timed epochs
	var results []epochMetrics
	for epoch := range epochs {
		fmt.Printf("  Epoch %d/%d...", epoch+1, epochs)

		var ttft time.Duration
		var ttftOnce sync.Once
		var responseMetrics *api.Metrics

		req := buildBenchRequest(model, numCtx, numPredict, epoch)
		requestStart := time.Now()

		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
		err := client.Generate(ctx, req, func(resp api.GenerateResponse) error {
			ttftOnce.Do(func() {
				if resp.Response != "" || resp.Thinking != "" {
					ttft = time.Since(requestStart)
				}
			})
			if resp.Done {
				responseMetrics = &resp.Metrics
			}
			return nil
		})
		cancel()

		if err != nil {
			fmt.Printf(" ERROR: %v\n", err)
			continue
		}
		if responseMetrics == nil {
			fmt.Println(" ERROR: no metrics")
			continue
		}

		em := epochMetrics{
			Epoch:            epoch,
			TTFT:             ttft,
			PrefillCount:     responseMetrics.PromptEvalCount,
			PrefillDuration:  responseMetrics.PromptEvalDuration,
			GenerateCount:    responseMetrics.EvalCount,
			GenerateDuration: responseMetrics.EvalDuration,
			LoadDuration:     responseMetrics.LoadDuration,
			TotalDuration:    responseMetrics.TotalDuration,
		}
		results = append(results, em)

		decodeTokS := float64(em.GenerateCount) / (float64(em.GenerateDuration.Nanoseconds())+1e-12) * 1e9
		fmt.Printf(" TTFT=%.0fms decode=%.1f tok/s\n", float64(em.TTFT.Milliseconds()), decodeTokS)
	}

	if len(results) == 0 {
		return nil, fmt.Errorf("no successful epochs")
	}

	return results, nil
}

func buildBenchRequest(model string, numCtx, numPredict, epoch int) *api.GenerateRequest {
	prompt := generateBenchPrompt(numCtx/2, epoch) // Use half context for prompt
	options := map[string]interface{}{
		"num_ctx":     numCtx,
		"num_predict": numPredict,
		"temperature": 0.0,
	}
	return &api.GenerateRequest{
		Model:   model,
		Prompt:  prompt,
		Raw:     true,
		Options: options,
	}
}

func unloadBenchModel(client *api.Client, model string) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	zero := api.Duration{Duration: 0}
	_ = client.Generate(ctx, &api.GenerateRequest{
		Model:     model,
		KeepAlive: &zero,
	}, func(resp api.GenerateResponse) error {
		return nil
	})
}

func printProfileResults(name string, metrics []epochMetrics) {
	if len(metrics) == 0 {
		return
	}

	var totalTTFT, totalPrefillDur, totalGenDur time.Duration
	var totalPrefillTok, totalGenTok int

	for _, m := range metrics {
		totalTTFT += m.TTFT
		totalPrefillDur += m.PrefillDuration
		totalGenDur += m.GenerateDuration
		totalPrefillTok += m.PrefillCount
		totalGenTok += m.GenerateCount
	}

	n := float64(len(metrics))
	avgTTFT := float64(totalTTFT.Milliseconds()) / n
	avgPrefillTokS := safeTokPerSec(totalPrefillTok, totalPrefillDur)
	avgDecodeTokS := safeTokPerSec(totalGenTok, totalGenDur)

	fmt.Printf("  Avg TTFT:       %.0f ms\n", avgTTFT)
	if avgPrefillTokS > 0 {
		fmt.Printf("  Avg prefill:    %.1f tok/s\n", avgPrefillTokS)
	} else {
		fmt.Printf("  Avg prefill:    (included in TTFT)\n")
	}
	fmt.Printf("  Avg decode:     %.1f tok/s\n", avgDecodeTokS)
	fmt.Printf("  Load duration:  %v\n", metrics[0].LoadDuration)
}

func printComparisonTable(results []benchResult) {
	fmt.Println("=== Comparison ===")
	fmt.Printf("%-25s", "Metric")
	for _, r := range results {
		fmt.Printf("%-20s", r.Provider)
	}
	fmt.Println()
	fmt.Println(strings.Repeat("-", 25+20*len(results)))

	// Compute averages per provider
	type avg struct {
		ttft       float64
		prefillTok float64
		decodeTok  float64
	}
	avgs := make([]avg, len(results))
	for i, r := range results {
		var totalTTFT time.Duration
		var totalPrefillDur, totalGenDur time.Duration
		var totalPrefillTok, totalGenTok int
		for _, m := range r.Epochs {
			totalTTFT += m.TTFT
			totalPrefillDur += m.PrefillDuration
			totalGenDur += m.GenerateDuration
			totalPrefillTok += m.PrefillCount
			totalGenTok += m.GenerateCount
		}
		n := float64(len(r.Epochs))
		avgs[i] = avg{
			ttft:       float64(totalTTFT.Milliseconds()) / n,
			prefillTok: safeTokPerSec(totalPrefillTok, totalPrefillDur),
			decodeTok:  safeTokPerSec(totalGenTok, totalGenDur),
		}
	}

	fmt.Printf("%-25s", "Avg TTFT (ms)")
	for _, a := range avgs {
		fmt.Printf("%-20.0f", a.ttft)
	}
	fmt.Println()

	fmt.Printf("%-25s", "Avg prefill (tok/s)")
	for _, a := range avgs {
		if a.prefillTok > 0 {
			fmt.Printf("%-20.1f", a.prefillTok)
		} else {
			fmt.Printf("%-20s", "(in TTFT)")
		}
	}
	fmt.Println()

	fmt.Printf("%-25s", "Avg decode (tok/s)")
	for _, a := range avgs {
		fmt.Printf("%-20.1f", a.decodeTok)
	}
	fmt.Println()
}
