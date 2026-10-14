package ortrunner

import (
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestQNNModelDirectoryRejectsUnsafeAliases(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "model-\u00f6")
	if err := os.Mkdir(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	other := t.TempDir()
	lookupFailure := errors.New("short names unavailable")
	for _, tc := range []struct {
		name  string
		alias string
		err   error
	}{
		{"missing", "", lookupFailure},
		{"relative", "model", nil},
		{"still unicode", dir, nil},
		{"different directory", other, nil},
		{"nonexistent", filepath.Join(other, "missing"), nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			result, err := qnnModelDirectory(dir, func(path string) (string, error) {
				if path != dir {
					t.Fatalf("lookup path=%q, want %q", path, dir)
				}
				return tc.alias, tc.err
			})
			if result != "" || err == nil || !strings.Contains(err.Error(), "NPU_MODEL_PATH_UNSUPPORTED") {
				t.Fatalf("unsafe alias accepted: result=%q error=%v", result, err)
			}
			if tc.err != nil && !errors.Is(err, tc.err) {
				t.Fatalf("lookup error not preserved: %v", err)
			}
		})
	}
}

func TestNativeModelDirectoryPreservesOtherPaths(t *testing.T) {
	ascii := `C:\models\qnn`
	path, err := qnnModelDirectory(ascii, func(string) (string, error) {
		t.Fatal("ASCII path should not require short names")
		return "", nil
	})
	if err != nil || path != ascii {
		t.Fatalf("ASCII path changed: %q, %v", path, err)
	}

	dir := "model-\u00f6"
	for _, provider := range []string{"cpu", "dml"} {
		path, err := nativeModelDirectory(dir, provider)
		if err != nil || path != dir {
			t.Fatalf("%s path changed: %q, %v", provider, path, err)
		}
	}
	if runtime.GOOS != "windows" {
		path, err := nativeModelDirectory(dir, "qnn")
		if err != nil || path != dir {
			t.Fatalf("non-Windows path changed: %q, %v", path, err)
		}
	}
}

func TestQNNRelativeModelDirectoryChecksAbsolutePath(t *testing.T) {
	parent := filepath.Join(t.TempDir(), "parent-\u00f6")
	dir := filepath.Join(parent, "model")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	t.Chdir(parent)
	called := false
	_, err := qnnModelDirectory("model", func(path string) (string, error) {
		called = true
		if path != dir {
			t.Fatalf("lookup path=%q, want %q", path, dir)
		}
		return "", errors.New("short names unavailable")
	})
	if !called || err == nil {
		t.Fatalf("relative model path bypassed Unicode validation: called=%v error=%v", called, err)
	}
}
