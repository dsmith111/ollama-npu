//go:build windows

package oga

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestNativeLibraryPath(t *testing.T) {
	for _, tc := range []struct {
		name string
		path string
		want string
	}{
		{"short drive", `C:\runtime\onnxruntime-genai.dll`, `C:\runtime\onnxruntime-genai.dll`},
		{"already extended", `\\?\C:\runtime\onnxruntime-genai.dll`, `\\?\C:\runtime\onnxruntime-genai.dll`},
		{"Unicode counted as UTF16", `C:\` + strings.Repeat("\u00f6", 130) + `\onnxruntime-genai.dll`, `C:\` + strings.Repeat("\u00f6", 130) + `\onnxruntime-genai.dll`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := nativeLibraryPath(tc.path)
			if err != nil || got != tc.want {
				t.Fatalf("native path=%q, want %q: %v", got, tc.want, err)
			}

		})
	}
	path, err := nativeLibraryPath(`runtime\onnxruntime-genai.dll`)
	want, absErr := filepath.Abs(`runtime\onnxruntime-genai.dll`)
	if err != nil || absErr != nil || path != want {
		t.Fatalf("relative native path=%q, want %q: %v, %v", path, want, err, absErr)
	}
	for _, invalid := range []string{"C:\\runtime\x00\\library.dll", string([]byte{'C', ':', '\\', 0xff})} {
		if path, err := nativeLibraryPath(invalid); err == nil || path != "" {
			t.Fatalf("invalid path accepted: %q, %v", path, err)
		}
	}
}

func TestNativeLongLibraryPathUsesVerifiedAlias(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "runtime-\u00f6")
	for len(dir) < 300 {
		dir = filepath.Join(dir, strings.Repeat("component", 5))
	}
	if err := os.MkdirAll(dir, 0o700); err != nil {
		t.Fatal(err)
	}
	short, err := windowsShortPath(dir)
	if err != nil {
		t.Fatal(err)
	}
	path, err := nativeLibraryPath(filepath.Join(dir, "onnxruntime-genai.dll"))
	if !asciiPath(short) {
		if path != "" || err == nil || !strings.Contains(err.Error(), "NPU_RUNTIME_PATH_UNSUPPORTED") {
			t.Fatalf("missing alias must fail explicitly: %q, %v", path, err)
		}
		return
	}
	if err != nil || path != filepath.Join(short, "onnxruntime-genai.dll") || len(path) >= 260 {
		t.Fatalf("native library alias=%q: %v", path, err)
	}
}

func TestShortLibraryPathRejectsUnsafeAlias(t *testing.T) {
	dir := t.TempDir()
	other := t.TempDir()
	missing := errors.New("short names unavailable")
	for _, tc := range []struct {
		name  string
		alias string
		err   error
	}{
		{"missing", "", missing},
		{"relative", "runtime", nil},
		{"different directory", other, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path, err := shortLibraryPath(filepath.Join(dir, "onnxruntime-genai.dll"), func(string) (string, error) {
				return tc.alias, tc.err
			})
			if path != "" || err == nil || !strings.Contains(err.Error(), "NPU_RUNTIME_PATH_UNSUPPORTED") {
				t.Fatalf("unsafe runtime alias accepted: %q, %v", path, err)
			}
			if tc.err != nil && !errors.Is(err, tc.err) {
				t.Fatalf("lookup error lost: %v", err)
			}
		})
	}
}
