//go:build windows

package cmd

import (
	"context"
	"errors"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cmd/internal/bench"
	"github.com/spf13/cobra"
)

func BenchmarkNPUHandler(cmd *cobra.Command, args []string) error {
	report := newNPUBenchmarkReport(cmd)
	numCtx, numPredict, epochs, warmup := report.Context, report.Predict, report.Epochs, report.Warmup
	jsonFile := diagnosticString(cmd, "json")
	var failure error
	if len(args) != 0 {
		failure = errors.New("benchmark npu accepts no positional arguments")
	} else {
		failure = validateBenchOptions(report.Models, numCtx, numPredict, epochs, warmup)
	}
	if failure != nil {
		report.Error = &npuDiagnosticError{"invalid_arguments", failure.Error()}
	} else {
		client, err := api.ClientFromEnvironment()
		if err != nil {
			failure = errors.New("could not create benchmark API client")
			report.Error = &npuDiagnosticError{"client_configuration_failed", failure.Error()}
		} else {
			progress := cmd.OutOrStdout()
			if jsonFile == "-" {
				progress = io.Discard
			}
			printComparisonTable(progress, report.Results)
			fmt.Fprintln(progress, "Benchmark measures model generation, not NPU execution proof. Actual provider: unknown.")
			for _, model := range report.Models {
				fmt.Fprintf(progress, "--- Model: %s; actual provider: unknown ---\n", model)
				metrics, runErr := runBenchEpochs(cmd.Context(), client, progress, model, numCtx, numPredict, epochs, warmup)
				result := benchResult{Model: model, Provider: "unknown", Status: "measured", Epochs: metrics}
				if result.Epochs == nil {
					result.Epochs = []epochMetrics{}
				}
				if runErr != nil {
					result.Status = "failed"
					result.Error = &npuDiagnosticError{"benchmark_failed", runErr.Error()}
					failure = errors.Join(failure, fmt.Errorf("model %s: %w", model, runErr))
					fmt.Fprintf(progress, "FAIL: %s\n", runErr)
				} else {
					printProfileResults(progress, metrics)
				}
				report.Results = append(report.Results, result)
				// Cleanup is best effort and never changes measurement status.
				unloadBenchModel(cmd.Context(), client, model)
			}
			if failure != nil {
				report.Error = &npuDiagnosticError{"benchmark_failed", "One or more models failed; partial measurements are not a successful benchmark"}
			}
		}
	}
	if failure != nil {
		report.Status = "failed"
	}
	if err := writeNPUBenchmarkJSON(cmd, report); err != nil {
		return fmt.Errorf("report_write_failed: %w", err)
	}
	if failure != nil {
		return fmt.Errorf("%s: %w", report.Error.Code, failure)
	}
	return nil
}

func validateBenchOptions(models []string, numCtx, numPredict, epochs, warmup int) error {
	if len(models) == 0 {
		return errors.New("--model must name at least one model")
	}
	seen := map[string]bool{}
	for _, model := range models {
		if strings.TrimSpace(model) == "" {
			return errors.New("--model must contain nonempty comma-separated model names")
		}
		if seen[model] {
			return errors.New("--model must not contain duplicate models")
		}
		seen[model] = true
	}
	if numCtx < 2 || numCtx > 1<<20 {
		return errors.New("--ctx must be between 2 and 1048576")
	}
	if numPredict < 1 || numPredict > numCtx/2 {
		return errors.New("--predict must be positive and fit the half-context generation budget")
	}
	if epochs < 1 || epochs > 10000 || warmup < 0 || warmup > 10000 {
		return errors.New("--epochs must be between 1 and 10000; --warmup must be between 0 and 10000")
	}
	return nil
}

type benchGenerator interface {
	Generate(context.Context, *api.GenerateRequest, api.GenerateResponseFunc) error
}

func runBenchEpochs(parent context.Context, client benchGenerator, progress io.Writer, model string, numCtx, numPredict, epochs, warmup int) ([]epochMetrics, error) {
	if err := validateBenchOptions([]string{model}, numCtx, numPredict, epochs, warmup); err != nil {
		return nil, err
	}
	for i := range warmup {
		fmt.Fprintf(progress, "Warmup %d/%d\n", i+1, warmup)
		if _, err := runBenchEpoch(parent, client, buildBenchRequest(model, numCtx, numPredict, -(i+1))); err != nil {
			return nil, fmt.Errorf("warmup %d: %w", i+1, err)
		}
	}
	results := make([]epochMetrics, 0, epochs)
	for epoch := range epochs {
		metrics, err := runBenchEpoch(parent, client, buildBenchRequest(model, numCtx, numPredict, epoch))
		if err != nil {
			return results, fmt.Errorf("epoch %d: %w", epoch+1, err)
		}
		metrics.Epoch = epoch + 1
		results = append(results, metrics)
		fmt.Fprintf(progress, "Epoch %d/%d: TTFT=%.3fms decode=%.1f tok/s\n",
			epoch+1, epochs, float64(metrics.TTFT)/float64(time.Millisecond),
			bench.TokensPerSecond(metrics.GenerateCount, metrics.GenerateDuration))
	}
	return results, nil
}

func runBenchEpoch(parent context.Context, client benchGenerator, request *api.GenerateRequest) (epochMetrics, error) {
	if parent == nil {
		return epochMetrics{}, errors.New("nil benchmark context")
	}
	if err := parent.Err(); err != nil {
		return epochMetrics{}, err
	}
	ctx, cancel := context.WithTimeout(parent, 5*time.Minute)
	defer cancel()
	start := time.Now()
	var ttft time.Duration
	hasToken := false
	var metrics api.Metrics
	done := false
	err := client.Generate(ctx, request, func(response api.GenerateResponse) error {
		if done {
			return errors.New("response received after the terminal generation response")
		}
		// Empty metadata chunks must not consume the first-token measurement.
		if !hasToken && (response.Response != "" || response.Thinking != "") {
			ttft = time.Since(start)
			hasToken = true
		}
		if response.Done {
			done = true
			metrics = response.Metrics
		}
		return nil
	})
	if err != nil {
		if parent.Err() != nil {
			return epochMetrics{}, parent.Err()
		}
		if ctx.Err() != nil {
			return epochMetrics{}, ctx.Err()
		}
		// Do not put server error bodies, prompts or generated text into reports.
		return epochMetrics{}, errors.New("generation request failed")
	}
	if !done {
		return epochMetrics{}, errors.New("generation stream ended without a terminal metrics response")
	}
	if !hasToken {
		return epochMetrics{}, errors.New("generation produced no visible or thinking tokens")
	}
	if metrics.EvalCount <= 0 || metrics.EvalDuration <= 0 || metrics.TotalDuration <= 0 ||
		metrics.PromptEvalCount < 0 || metrics.PromptEvalDuration < 0 || metrics.LoadDuration < 0 {
		return epochMetrics{}, errors.New("generation returned invalid token counts or durations")
	}
	if limit, ok := request.Options["num_predict"].(int); ok && metrics.EvalCount > limit {
		return epochMetrics{}, errors.New("generation token count exceeds the requested limit")
	}
	if metrics.EvalDuration > metrics.TotalDuration || metrics.PromptEvalDuration > metrics.TotalDuration || metrics.LoadDuration > metrics.TotalDuration {
		return epochMetrics{}, errors.New("generation phase duration exceeds total duration")
	}
	return epochMetrics{
		TTFT: ttft, PrefillCount: metrics.PromptEvalCount, PrefillDuration: metrics.PromptEvalDuration,
		GenerateCount: metrics.EvalCount, GenerateDuration: metrics.EvalDuration,
		LoadDuration: metrics.LoadDuration, TotalDuration: metrics.TotalDuration,
	}, nil
}

func buildBenchRequest(model string, numCtx, numPredict, epoch int) *api.GenerateRequest {
	stream := true
	return &api.GenerateRequest{
		Model: model, Prompt: bench.GeneratePromptForTokenCount(numCtx/2, epoch), Raw: true, Stream: &stream,
		Options: map[string]any{"num_ctx": numCtx, "num_predict": numPredict, "temperature": 0.0},
	}
}

func unloadBenchModel(parent context.Context, client benchGenerator, model string) {
	ctx, cancel := context.WithTimeout(parent, 30*time.Second)
	defer cancel()
	zero := api.Duration{Duration: 0}
	_ = client.Generate(ctx, &api.GenerateRequest{Model: model, KeepAlive: &zero}, func(api.GenerateResponse) error { return nil })
}

func printComparisonTable(out io.Writer, results []benchResult) {
	if len(results) < 2 {
		return
	}
	fmt.Fprintln(out, "=== Model comparison (actual providers unverified) ===")
	for _, result := range results {
		fmt.Fprintf(out, "Model: %s; provider: %s; status: %s\n", result.Model, result.Provider, result.Status)
		if result.Status == "measured" {
			printProfileResults(out, result.Epochs)
		}
	}
}

func printProfileResults(out io.Writer, metrics []epochMetrics) {
	if len(metrics) == 0 {
		return
	}
	var ttft, prefill, decode time.Duration
	var prefillTokens, decodeTokens int
	for _, metric := range metrics {
		ttft += metric.TTFT
		prefill += metric.PrefillDuration
		decode += metric.GenerateDuration
		prefillTokens += metric.PrefillCount
		decodeTokens += metric.GenerateCount
	}
	fmt.Fprintf(out, "Avg TTFT: %.3f ms\n", float64(ttft)/float64(time.Millisecond)/float64(len(metrics)))
	if prefill > 0 {
		fmt.Fprintf(out, "Prefill: %.1f tok/s\n", bench.TokensPerSecond(prefillTokens, prefill))
	} else {
		fmt.Fprintln(out, "Prefill: unavailable (no separate prefill timing reported)")
	}
	fmt.Fprintf(out, "Decode: %.1f tok/s\n", bench.TokensPerSecond(decodeTokens, decode))
}
