package hexagon

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The NPU half must come from the same llama.cpp commit as ggml-hexagon; the
// hexagon_arm64 build preset checks Pinned.LlamaCppTag against
// LLAMA_CPP_VERSION at configure time, so a llama.cpp bump needs a signed set
// built from the new commit.
func TestPinnedSetShape(t *testing.T) {
	if Pinned.LlamaCppTag == "" || Pinned.URL == "" || Pinned.ArchiveSHA256 == "" {
		t.Fatalf("pinned release = %+v, want a llama.cpp tag, an archive URL and its hash", Pinned)
	}
	if len(Pinned.Files) != 6 || Pinned.Files[catalogFilename] == "" {
		t.Fatalf("pinned files = %v, want four libraries, the catalog and the .inf", Pinned.names())
	}
}

// testSet writes a set into dir and returns a release pinning it.
func testSet(t *testing.T, dir string) Release {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	rel := Release{Source: "test", LlamaCppTag: "b1", Files: map[string]string{}}
	for _, name := range []string{"libggml-htp-v73.so", "libggml-htp-v81.so", catalogFilename} {
		content := []byte("contents of " + name)
		if err := os.WriteFile(filepath.Join(dir, name), content, 0o644); err != nil {
			t.Fatal(err)
		}
		sum := sha256.Sum256(content)
		rel.Files[name] = hex.EncodeToString(sum[:])
	}
	return rel
}

func usePinned(t *testing.T, rel Release) {
	t.Helper()
	old := Pinned
	Pinned = rel
	t.Cleanup(func() { Pinned = old })
}

func TestVerifiedDirTakesTheFirstSetAndRefusesABadOne(t *testing.T) {
	empty, good, bad := t.TempDir(), filepath.Join(t.TempDir(), "good"), t.TempDir()
	usePinned(t, testSet(t, good))
	if err := os.WriteFile(filepath.Join(bad, catalogFilename), []byte("not the catalog"), 0o644); err != nil {
		t.Fatal(err)
	}

	if dir, err := VerifiedDir([]string{empty, good, bad}); err != nil || dir != good {
		t.Fatalf("VerifiedDir = %q, %v; want %q", dir, err, good)
	}
	if _, err := VerifiedDir([]string{empty, bad, good}); err == nil {
		t.Fatal("fell back past an invalid set")
	}
	if _, err := VerifiedDir([]string{empty}); !errors.Is(err, ErrNotInstalled) {
		t.Fatalf("error = %v, want ErrNotInstalled", err)
	}

	// A changed file is refused rather than loaded.
	if err := os.WriteFile(filepath.Join(good, "libggml-htp-v73.so"), []byte("other"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := VerifiedDir([]string{good}); err == nil || !strings.Contains(err.Error(), "SHA256") {
		t.Fatalf("a changed file passed: %v", err)
	}
}

func TestValidateWantsEveryPinnedFile(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "set")
	usePinned(t, testSet(t, dir))
	if err := Validate(dir); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(filepath.Join(dir, "libggml-htp-v81.so")); err != nil {
		t.Fatal(err)
	}
	if err := Validate(dir); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("a missing file passed: %v", err)
	}
}

func TestSearchDirsOrder(t *testing.T) {
	first, second, backend := t.TempDir(), t.TempDir(), t.TempDir()
	t.Setenv(PathEnv, strings.Join([]string{first, second, first, ""}, string(os.PathListSeparator)))
	want := []string{first, second, backend}
	if got := SearchDirs(backend); strings.Join(got, "|") != strings.Join(want, "|") {
		t.Fatalf("SearchDirs = %v, want %v", got, want)
	}
	t.Setenv(PathEnv, "")
	if got := SearchDirs(backend); len(got) != 1 || got[0] != backend {
		t.Fatalf("SearchDirs without %s = %v, want only the backend directory", PathEnv, got)
	}
}
