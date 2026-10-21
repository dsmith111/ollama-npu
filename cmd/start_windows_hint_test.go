package cmd

import (
	"strings"
	"testing"
)

// With no server and no app beside the executable, the portable fork says what to do
// instead of starting a stock Ollama app found elsewhere.
func TestNoServerHintNamesTheHostAndTheCommand(t *testing.T) {
	hint := noServerHint("127.0.0.1:11434")
	for _, part := range []string{"127.0.0.1:11434", "ollama.exe serve", "OLLAMA_HEXAGON=1", "OLLAMA_HOST"} {
		if !strings.Contains(hint, part) {
			t.Fatalf("hint %q lacks %q", hint, part)
		}
	}
}
