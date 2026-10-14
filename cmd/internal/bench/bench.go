// Package bench provides shared benchmarking types and utilities.
package bench

import (
	"fmt"
	"io"
	"os"
	"runtime"
	"strings"
	"time"
)

// Metrics holds a single benchmark measurement.
type Metrics struct {
	Model    string
	Step     string
	Count    int
	Duration time.Duration
}

// ModelInfo holds metadata about a benchmarked model.
type ModelInfo struct {
	Name              string
	ParameterSize     string
	QuantizationLevel string
	Family            string
	SizeBytes         int64
	VRAMBytes         int64
}

// TokensPerSecond returns zero when a rate is unavailable, never infinity or a
// made-up epsilon-denominator rate for missing runner timings.
func TokensPerSecond(tokens int, duration time.Duration) float64 {
	if tokens <= 0 || duration <= 0 {
		return 0
	}
	return float64(tokens) * 1e9 / float64(duration.Nanoseconds())
}

// OutputFormatHeader writes the header for the given output format.
func OutputFormatHeader(w io.Writer, format string, verbose bool) {
	switch format {
	case "benchstat":
		if verbose {
			fmt.Fprintf(w, "goos: %s\n", runtime.GOOS)
			fmt.Fprintf(w, "goarch: %s\n", runtime.GOARCH)
		}
	case "csv":
		headings := []string{"NAME", "STEP", "COUNT", "NS_PER_COUNT", "TOKEN_PER_SEC"}
		fmt.Fprintln(w, strings.Join(headings, ","))
	}
}

// OutputModelInfo writes model metadata as a comment.
func OutputModelInfo(w io.Writer, format string, info ModelInfo) {
	params := orDefault(info.ParameterSize, "unknown")
	quant := orDefault(info.QuantizationLevel, "unknown")
	family := orDefault(info.Family, "unknown")

	memStr := ""
	if info.SizeBytes > 0 {
		memStr = fmt.Sprintf(" | Size: %d | VRAM: %d", info.SizeBytes, info.VRAMBytes)
	}
	fmt.Fprintf(w, "# Model: %s | Params: %s | Quant: %s | Family: %s%s\n",
		info.Name, params, quant, family, memStr)
}

// OutputMetrics writes benchmark metrics in the specified format.
func OutputMetrics(w io.Writer, format string, metrics []Metrics, verbose bool) {
	switch format {
	case "benchstat":
		for _, m := range metrics {
			if m.Step == "generate" || m.Step == "prefill" {
				if m.Count > 0 && m.Duration > 0 {
					nsPerToken := float64(m.Duration.Nanoseconds()) / float64(m.Count)
					tokensPerSec := TokensPerSecond(m.Count, m.Duration)
					fmt.Fprintf(w, "BenchmarkModel/name=%s/step=%s 1 %.2f ns/token %.2f token/sec\n",
						m.Model, m.Step, nsPerToken, tokensPerSec)
				} else {
					fmt.Fprintf(w, "BenchmarkModel/name=%s/step=%s 1 0 ns/token 0 token/sec\n",
						m.Model, m.Step)
				}
			} else if m.Step == "ttft" {
				fmt.Fprintf(w, "BenchmarkModel/name=%s/step=ttft 1 %d ns/op\n",
					m.Model, m.Duration.Nanoseconds())
			} else {
				fmt.Fprintf(w, "BenchmarkModel/name=%s/step=%s 1 %d ns/op\n",
					m.Model, m.Step, m.Duration.Nanoseconds())
			}
		}
	case "csv":
		for _, m := range metrics {
			if m.Step == "generate" || m.Step == "prefill" {
				var nsPerToken float64
				var tokensPerSec float64
				if m.Count > 0 && m.Duration > 0 {
					nsPerToken = float64(m.Duration.Nanoseconds()) / float64(m.Count)
					tokensPerSec = TokensPerSecond(m.Count, m.Duration)
				}
				fmt.Fprintf(w, "%s,%s,%d,%.2f,%.2f\n", m.Model, m.Step, m.Count, nsPerToken, tokensPerSec)
			} else {
				fmt.Fprintf(w, "%s,%s,1,%d,0\n", m.Model, m.Step, m.Duration.Nanoseconds())
			}
		}
	default:
		fmt.Fprintf(os.Stderr, "Unknown output format '%s'\n", format)
	}
}

// GeneratePromptForTokenCount generates a varied prompt targeting ~N tokens.
func GeneratePromptForTokenCount(targetTokens int, epoch int) string {
	targetWords := int(float64(targetTokens) / 1.3)
	if targetWords < 1 {
		targetWords = 1
	}
	n := len(promptWordList)
	offset := (epoch % n) * 7
	words := make([]string, targetWords)
	for i := range words {
		words[i] = promptWordList[((i+offset)%n+n)%n]
	}
	return strings.Join(words, " ")
}

// promptWordList provides varied words for prompt generation.
var promptWordList = []string{
	"the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
	"a", "bright", "sunny", "day", "in", "the", "meadow", "where",
	"flowers", "bloom", "and", "birds", "sing", "their", "morning",
	"songs", "while", "gentle", "breeze", "carries", "sweet", "scent",
	"of", "pine", "trees", "across", "rolling", "hills", "toward",
	"distant", "mountains", "covered", "with", "fresh", "snow",
	"beneath", "clear", "blue", "sky", "children", "play", "near",
	"old", "stone", "bridge", "that", "crosses", "winding", "river",
}

func orDefault(val, def string) string {
	if val == "" {
		return def
	}
	return val
}
