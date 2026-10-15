//go:build windows

package oga

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestNativeQNNModelDirectoryWindowsAlias(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "model-\u00f6")
	if err := os.Mkdir(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	short, err := windowsShortPath(dir)
	if err != nil {
		t.Fatal(err)
	}
	path, err := NativeModelDirectory(dir, "qnn")
	if !asciiPath(short) {
		if path != "" || err == nil || !strings.Contains(err.Error(), "NPU_MODEL_PATH_UNSUPPORTED") {
			t.Fatalf("missing ASCII alias must fail explicitly: path=%q error=%v", path, err)
		}
		return
	}
	if err != nil || path != short {
		t.Fatalf("short path=%q, expected %q: %v", path, short, err)
	}
	original, err := os.Stat(dir)
	if err != nil {
		t.Fatal(err)
	}
	aliased, err := os.Stat(path)
	if err != nil || !os.SameFile(original, aliased) {
		t.Fatalf("native alias is not the original directory: %v", err)
	}
}
