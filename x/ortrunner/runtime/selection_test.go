package runtime

import (
	"context"
	"debug/pe"
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func TestVerifiedRuntimeSelection(t *testing.T) {
	spec, cfg := offlineFixture(t)
	if _, err := install(t.Context(), cfg, []packageSpec{spec}, nil, nil); err != nil {
		t.Fatal(err)
	}
	validateFixture := func(ctx context.Context, dir string) error {
		_, err := validate(ctx, dir, []packageSpec{spec})
		return err
	}
	dir, err := verifiedDir(t.Context(), []string{t.TempDir(), cfg.Dir}, validateFixture)
	if err != nil || dir != cfg.Dir {
		t.Fatalf("selection = %q, %v", dir, err)
	}
	bad := t.TempDir()
	writeFile(t, filepath.Join(bad, "onnxruntime-genai.dll"), testPE(t, pe.IMAGE_FILE_MACHINE_ARM64))
	if _, err := verifiedDir(t.Context(), []string{bad, cfg.Dir}, validateFixture); err == nil {
		t.Fatal("invalid explicit runtime fell back to a valid later installation")
	}
	if _, err := VerifiedDir(t.Context(), []string{cfg.Dir}); err == nil {
		t.Fatal("fixture identity was accepted as an approved public runtime")
	}
	if err := os.WriteFile(filepath.Join(cfg.Dir, "onnxruntime.dll"), []byte("corrupt"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := verifiedDir(t.Context(), []string{cfg.Dir}, validateFixture); err == nil {
		t.Fatal("corrupted runtime accepted")
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := verifiedDir(ctx, []string{cfg.Dir}, validateFixture); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation = %v", err)
	}
	if _, err := VerifiedDir(t.Context(), []string{t.TempDir()}); err == nil {
		t.Fatal("missing runtime accepted")
	}
}
