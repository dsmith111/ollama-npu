//go:build windows

package cmd

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/spf13/cobra"
)

type fakeBenchGenerator func(context.Context, *api.GenerateRequest, api.GenerateResponseFunc) error

func (generate fakeBenchGenerator) Generate(ctx context.Context, request *api.GenerateRequest, callback api.GenerateResponseFunc) error {
	return generate(ctx, request, callback)
}

func validBenchResponse() api.GenerateResponse {
	return api.GenerateResponse{
		Done: true, Response: "text",
		Metrics: api.Metrics{
			PromptEvalCount: 8, PromptEvalDuration: 2 * time.Millisecond,
			EvalCount: 4, EvalDuration: 5 * time.Millisecond, TotalDuration: 10 * time.Millisecond,
		},
	}
}

func TestNPUBenchmarkOptions(t *testing.T) {
	for _, options := range []struct {
		models                       []string
		ctx, predict, epochs, warmup int
	}{
		{nil, 512, 128, 1, 0}, {[]string{""}, 512, 128, 1, 0},
		{[]string{"m", ""}, 512, 128, 1, 0}, {[]string{"m", "m"}, 512, 128, 1, 0},
		{[]string{"m"}, 0, 128, 1, 0}, {[]string{"m"}, 512, 0, 1, 0},
		{[]string{"m"}, 512, 512, 1, 0}, {[]string{"m"}, 512, 128, 0, 0},
		{[]string{"m"}, 512, 128, 1, -1}, {[]string{"m"}, 1 << 30, 128, 1, 0},
	} {
		if err := validateBenchOptions(options.models, options.ctx, options.predict, options.epochs, options.warmup); err == nil {
			t.Fatalf("invalid options accepted: %+v", options)
		}
	}
	if err := validateBenchOptions([]string{"model"}, 512, 128, 1, 0); err != nil {
		t.Fatal(err)
	}
}

func TestNPUBenchmarkTTFTIgnoresMetadataChunks(t *testing.T) {
	client := fakeBenchGenerator(func(_ context.Context, _ *api.GenerateRequest, callback api.GenerateResponseFunc) error {
		if err := callback(api.GenerateResponse{}); err != nil {
			return err
		}
		time.Sleep(10 * time.Millisecond)
		if err := callback(api.GenerateResponse{Thinking: "not included in report"}); err != nil {
			return err
		}
		return callback(validBenchResponse())
	})
	metrics, err := runBenchEpoch(context.Background(), client, buildBenchRequest("model", 512, 128, 0))
	if err != nil || metrics.TTFT < 10*time.Millisecond || metrics.GenerateCount != 4 {
		t.Fatalf("metadata consumed first-token timing: %+v, %v", metrics, err)
	}
}

func TestNPUBenchmarkInvalidResponses(t *testing.T) {
	for _, mode := range []string{"request failure", "missing done", "missing token", "negative count", "excessive count", "zero eval duration", "negative prefill duration", "phase exceeds total", "extra chunk"} {
		t.Run(mode, func(t *testing.T) {
			client := fakeBenchGenerator(func(_ context.Context, _ *api.GenerateRequest, callback api.GenerateResponseFunc) error {
				response := validBenchResponse()
				switch mode {
				case "request failure":
					return errors.New("private prompt or generated text")
				case "missing done":
					response.Done = false
				case "missing token":
					response.Response = ""
				case "negative count":
					response.EvalCount = -1
				case "excessive count":
					response.EvalCount = 129
				case "zero eval duration":
					response.EvalDuration = 0
				case "negative prefill duration":
					response.PromptEvalDuration = -1
				case "phase exceeds total":
					response.EvalDuration = response.TotalDuration + 1
				}
				if err := callback(response); err != nil {
					return err
				}
				if mode == "extra chunk" {
					return callback(api.GenerateResponse{Response: "after done"})
				}
				return nil
			})
			_, err := runBenchEpoch(context.Background(), client, buildBenchRequest("m", 512, 128, 0))
			if err == nil || strings.Contains(err.Error(), "private prompt") {
				t.Fatalf("invalid response accepted or text leaked: %v", err)
			}
		})
	}
}

func TestNPUBenchmarkWarmupAndPartialFailuresFail(t *testing.T) {
	for _, warmup := range []int{0, 1} {
		calls := 0
		client := fakeBenchGenerator(func(_ context.Context, _ *api.GenerateRequest, callback api.GenerateResponseFunc) error {
			calls++
			if warmup == 1 || calls == 2 {
				return errors.New("failed")
			}
			return callback(validBenchResponse())
		})
		results, err := runBenchEpochs(context.Background(), client, io.Discard, "m", 512, 128, 3, warmup)
		if err == nil {
			t.Fatal("warmup or partial measurement failure returned success")
		}
		if warmup == 1 && (calls != 1 || len(results) != 0) || warmup == 0 && (calls != 2 || len(results) != 1) {
			t.Fatalf("continued after failure or discarded partial result: calls %d, %+v", calls, results)
		}
	}
}

func TestNPUBenchmarkContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	client := fakeBenchGenerator(func(ctx context.Context, _ *api.GenerateRequest, _ api.GenerateResponseFunc) error { return ctx.Err() })
	_, err := runBenchEpoch(ctx, client, buildBenchRequest("m", 512, 128, 0))
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation lost: %v", err)
	}
}

func TestNPUBenchmarkPartialModelFailureJSONAndFile(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var request api.GenerateRequest
		json.NewDecoder(r.Body).Decode(&request)
		if request.KeepAlive != nil {
			json.NewEncoder(w).Encode(api.GenerateResponse{Done: true})
		} else if request.Model == "bad" {
			w.WriteHeader(http.StatusInternalServerError)
			io.WriteString(w, `{"error":"failed"}`)
		} else {
			json.NewEncoder(w).Encode(validBenchResponse())
		}
	}))
	defer server.Close()
	t.Setenv("OLLAMA_HOST", server.URL)
	path := filepath.Join(t.TempDir(), "results.json")
	cmd := &cobra.Command{Use: "npu", RunE: BenchmarkNPUHandler}
	ConfigureNPUBenchmarkCommand(cmd)
	cmd.SetOut(io.Discard)
	cmd.SetErr(io.Discard)
	cmd.SetArgs([]string{"--model", "good,bad", "--epochs", "1", "--warmup", "0", "--json", path})
	if err := cmd.Execute(); err == nil {
		t.Fatal("partial model failure returned success")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var report npuBenchmarkReport
	if err := json.Unmarshal(data, &report); err != nil || report.Status != "failed" || len(report.Results) != 2 {
		t.Fatalf("partial-result JSON file: %s, %v", data, err)
	}
	if report.Results[0].Status != "measured" || report.Results[1].Status != "failed" || len(report.Results[0].Epochs) != 1 {
		t.Fatalf("successful results lost or failed model accepted: %+v", report)
	}
}

func TestNPUBenchmarkRequestAndSummary(t *testing.T) {
	request := buildBenchRequest("model", 512, 128, -1)
	if !request.Raw || request.Stream == nil || !*request.Stream || request.Options["num_predict"] != 128 || request.Prompt == "" {
		t.Fatalf("unexpected benchmark request: %+v", request)
	}
	var output bytes.Buffer
	printProfileResults(&output, []epochMetrics{
		{TTFT: 500 * time.Microsecond, GenerateCount: 1, GenerateDuration: 500 * time.Microsecond},
		{TTFT: 1500 * time.Microsecond, GenerateCount: 3, GenerateDuration: 1500 * time.Microsecond},
	})
	if !strings.Contains(output.String(), "1.000 ms") || !strings.Contains(output.String(), "2000.0 tok/s") || !strings.Contains(output.String(), "unavailable") {
		t.Fatalf("incorrect average, token rate or unmeasured prefill claim: %s", output.String())
	}
}

func TestNPUBenchmarkHandlerJSONProviderAndFailure(t *testing.T) {
	for _, failed := range []bool{false, true} {
		t.Run(map[bool]string{false: "success", true: "failure"}[failed], func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var request api.GenerateRequest
				if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
					t.Error(err)
				}
				if request.KeepAlive != nil {
					json.NewEncoder(w).Encode(api.GenerateResponse{Done: true})
					return
				}
				if failed {
					w.WriteHeader(http.StatusInternalServerError)
					io.WriteString(w, `{"error":"private prompt and output"}`)
					return
				}
				json.NewEncoder(w).Encode(api.GenerateResponse{})
				json.NewEncoder(w).Encode(validBenchResponse())
			}))
			defer server.Close()
			t.Setenv("OLLAMA_HOST", server.URL)
			cmd := &cobra.Command{Use: "npu", RunE: BenchmarkNPUHandler}
			ConfigureNPUBenchmarkCommand(cmd)
			var output bytes.Buffer
			cmd.SetOut(&output)
			cmd.SetArgs([]string{"--model", "qnn-in-model-name", "--epochs", "1", "--warmup", "0", "--json", "-"})
			err := cmd.Execute()
			if (err != nil) != failed {
				t.Fatalf("failure exit mismatch: %v", err)
			}
			var report npuBenchmarkReport
			if err := json.Unmarshal(output.Bytes(), &report); err != nil || len(report.Results) != 1 {
				t.Fatalf("JSON output: %s: %v", output.String(), err)
			}
			result := report.Results[0]
			if result.Model != "qnn-in-model-name" || result.Provider != "unknown" || result.ProviderVerified {
				t.Fatalf("model name mistaken for an actual provider: %+v", result)
			}
			if strings.Contains(output.String(), "private prompt") || strings.Contains(output.String(), "quick brown fox") {
				t.Fatal("report leaked prompts or generated text")
			}
			if failed && report.Status != "failed" || !failed && report.Status != "measured" {
				t.Fatalf("wrong aggregate status: %+v", report)
			}
		})
	}
}

func TestNPUBenchmarkHandlerInvalidArgumentsJSON(t *testing.T) {
	cmd := &cobra.Command{Use: "npu", RunE: BenchmarkNPUHandler}
	ConfigureNPUBenchmarkCommand(cmd)
	var output bytes.Buffer
	cmd.SetOut(&output)
	cmd.SetArgs([]string{"--epochs", "0", "--json", "-"})
	if err := cmd.Execute(); err == nil {
		t.Fatal("invalid benchmark returned success")
	}
	var report npuBenchmarkReport
	if err := json.Unmarshal(output.Bytes(), &report); err != nil || report.Error.Code != "invalid_arguments" {
		t.Fatalf("invalid-arguments JSON: %s, %v", output.String(), err)
	}
}
