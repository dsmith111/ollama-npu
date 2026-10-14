package bench

import (
	"bytes"
	"math"
	"strings"
	"testing"
	"time"
)

func TestTokensPerSecond(t *testing.T) {
	for _, test := range []struct {
		tokens   int
		duration time.Duration
		want     float64
	}{
		{4, 2 * time.Second, 2}, {1, 500 * time.Microsecond, 2000},
		{1, time.Nanosecond, 1e9}, {1, 0, 0}, {1, -1, 0}, {0, time.Second, 0}, {-1, time.Second, 0},
	} {
		got := TokensPerSecond(test.tokens, test.duration)
		if math.IsInf(got, 0) || math.IsNaN(got) || got != test.want {
			t.Fatalf("rate(%d,%s)=%v; want %v", test.tokens, test.duration, got, test.want)
		}
	}
}

func TestOutputMetricsMissingAndSmallTimings(t *testing.T) {
	for _, format := range []string{"csv", "benchstat"} {
		var output bytes.Buffer
		OutputMetrics(&output, format, []Metrics{
			{Model: "m", Step: "generate", Count: 1, Duration: 500 * time.Microsecond},
			{Model: "m", Step: "prefill", Count: 10, Duration: 0},
			{Model: "m", Step: "generate", Count: 1, Duration: -1},
		}, false)
		if !strings.Contains(output.String(), "2000.00") || strings.Contains(output.String(), "Inf") || strings.Contains(output.String(), "NaN") {
			t.Fatalf("invalid output math: %s", output.String())
		}
		lines := strings.Split(strings.TrimSpace(output.String()), "\n")
		if strings.Contains(lines[1], "100000000000") || strings.Contains(lines[2], "-1.00") {
			t.Fatalf("invented rates for missing or negative durations: %s", output.String())
		}
	}
}

func TestPromptGenerationNegativeEpochAndVariation(t *testing.T) {
	warmup := GeneratePromptForTokenCount(100, -1)
	timed := GeneratePromptForTokenCount(100, 0)
	if warmup == "" || timed == "" || warmup == timed || len(strings.Fields(timed)) != 76 {
		t.Fatal("warmup/timed prompt generation is empty, identical or incorrectly sized")
	}
	if GeneratePromptForTokenCount(0, math.MaxInt) == "" || GeneratePromptForTokenCount(1, math.MinInt) == "" {
		t.Fatal("extreme epoch generated no prompt")
	}
}
