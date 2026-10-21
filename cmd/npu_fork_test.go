package cmd

import (
	"strings"
	"testing"
)

// The NPU fork's client warns when the server it reached is stock Ollama (no NPU backend).
func TestStockServerHint(t *testing.T) {
	cases := []struct {
		name, client, server string
		want                 bool
	}{
		{"fork client, stock server", "0.40.0-npu.1", "0.40.0", true},
		{"dev fork client, older stock server", "0.0.0-npu.dev.73264854ae48", "0.17.3", true},
		{"fork client, same fork server", "0.40.0-npu.1", "0.40.0-npu.1", false},
		{"fork client, another fork build", "0.40.0-npu.1", "0.0.0-npu.dev.abc", false},
		{"plain build, stock server", "0.0.0", "0.40.0", false},
		{"no server version", "0.40.0-npu.1", "", false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			hint := stockServerHint("127.0.0.1:11434", c.client, c.server)
			if (hint != "") != c.want {
				t.Fatalf("hint %q, wanted one: %v", hint, c.want)
			}
			if !c.want {
				return
			}
			for _, part := range []string{"127.0.0.1:11434", c.server, "OLLAMA_HEXAGON=1", "OLLAMA_HOST"} {
				if !strings.Contains(hint, part) {
					t.Fatalf("hint %q lacks %q", hint, part)
				}
			}
		})
	}
}
