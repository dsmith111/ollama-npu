package cmd

import (
	"archive/zip"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/discover/hexagon"
)

func runNPUSetup(t *testing.T, args ...string) (string, error) {
	t.Helper()
	var output bytes.Buffer
	cmd := newNPUCommand()
	cmd.SetOut(&output)
	cmd.SetErr(&output)
	cmd.SetArgs(append([]string{"setup"}, args...))
	cmd.SetContext(t.Context())
	err := cmd.Execute()
	return output.String(), err
}

// pinTestArchive writes a small archive shaped like GenieX's and pins it.
func pinTestArchive(t *testing.T) string {
	t.Helper()
	rel := hexagon.Release{Source: "test set", LlamaCppTag: "b1", ArchiveName: "set.zip", Prefix: "sdk/lib/llama_cpp/",
		Files: map[string]string{}}
	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)
	for _, name := range []string{"libggml-htp-v73.so", "libggml-htp.cat"} {
		w, err := zw.Create(rel.Prefix + name)
		if err != nil {
			t.Fatal(err)
		}
		w.Write([]byte(name))
		sum := sha256.Sum256([]byte(name))
		rel.Files[name] = hex.EncodeToString(sum[:])
	}
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
	pkgDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(pkgDir, rel.ArchiveName), buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(buf.Bytes())
	rel.ArchiveSize, rel.ArchiveSHA256 = int64(buf.Len()), hex.EncodeToString(sum[:])
	original := hexagon.Pinned
	hexagon.Pinned = rel
	t.Cleanup(func() { hexagon.Pinned = original })
	return pkgDir
}

func TestNPUSetupNeedsQualcommTerms(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "htp")
	output, err := runNPUSetup(t, "--dir", dir)
	if err == nil || !strings.Contains(err.Error(), "--accept-qualcomm-terms") {
		t.Fatalf("setup without accepting the terms: %v", err)
	}
	if !strings.Contains(output, "Responsible AI License") || !strings.Contains(output, "machine generated") {
		t.Fatalf("terms notice missing from output:\n%s", output)
	}
	if _, err := os.Stat(dir); !os.IsNotExist(err) {
		t.Fatal("setup changed files before the terms were accepted")
	}
}

func TestNPUSetupInstallsOfflineAndValidates(t *testing.T) {
	pkgDir := pinTestArchive(t)
	dir := filepath.Join(t.TempDir(), "htp")
	output, err := runNPUSetup(t, "--accept-qualcomm-terms", "--dir", dir, "--package-dir", pkgDir)
	if err != nil || !strings.Contains(output, "Pinned files verified") || !strings.Contains(output, hexagon.PathEnv+"="+dir) {
		t.Fatalf("offline install: %v\n%s", err, output)
	}
	if _, err := os.Stat(filepath.Join(dir, hexagon.ManifestFilename)); err != nil {
		t.Fatalf("manifest not installed: %v", err)
	}

	output, err = runNPUSetup(t, "--validate-only", "--dir", dir)
	if err != nil || !strings.Contains(output, dir) {
		t.Fatalf("validate-only: %v\n%s", err, output)
	}

	// Without --dir, validation looks where runners look.
	t.Setenv(hexagon.PathEnv, dir)
	if output, err := runNPUSetup(t, "--validate-only"); err != nil || !strings.Contains(output, dir) {
		t.Fatalf("validate-only search: %v\n%s", err, output)
	}

	if err := os.WriteFile(filepath.Join(dir, "libggml-htp-v73.so"), []byte("changed"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := runNPUSetup(t, "--validate-only", "--dir", dir); err == nil || !strings.Contains(err.Error(), "SHA256") {
		t.Fatalf("a changed file passed validation: %v", err)
	}
}
