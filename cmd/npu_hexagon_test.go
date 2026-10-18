//go:build windows

package cmd

import (
	"archive/zip"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ollama/ollama/x/hexagon"
	"github.com/spf13/cobra"
)

func runHexagonSetup(t *testing.T, args ...string) (npuDiagnostic, error) {
	t.Helper()
	var output bytes.Buffer
	cmd := &cobra.Command{Use: "setup", RunE: NPUSetupHandler}
	cmd.SetOut(&output)
	cmd.SetErr(&output)
	ConfigureNPUSetupCommand(cmd)
	cmd.SetArgs(append([]string{"--provider", "hexagon", "--json"}, args...))
	err := cmd.Execute()
	var report npuDiagnostic
	if jsonErr := json.Unmarshal(output.Bytes(), &report); jsonErr != nil {
		t.Fatalf("setup output is not a JSON report: %s (%v)", output.String(), jsonErr)
	}
	return report, err
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

func TestNPUSetupHexagonNeedsQualcommTerms(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "htp")
	report, err := runHexagonSetup(t, "--dir", dir)
	if err == nil || report.Error == nil || report.Error.Code != "terms_not_accepted" {
		t.Fatalf("setup without accepting the terms: %+v, %v", report.Error, err)
	}
	var notice string
	for _, check := range report.Checks {
		if check.Stage == "terms" {
			notice = check.Message
		}
	}
	if !strings.Contains(notice, "Responsible AI License") || !strings.Contains(notice, "machine generated") {
		t.Fatalf("terms notice = %q", notice)
	}
	if _, err := os.Stat(dir); !os.IsNotExist(err) {
		t.Fatal("setup changed files before the terms were accepted")
	}
}

func TestNPUSetupHexagonInstallsOfflineAndValidates(t *testing.T) {
	pkgDir := pinTestArchive(t)
	dir := filepath.Join(t.TempDir(), "htp")
	report, err := runHexagonSetup(t, "--accept-qualcomm-terms", "--dir", dir, "--package-dir", pkgDir)
	if err != nil || report.Status != "files_verified" || report.Hexagon == nil || !report.Hexagon.Installed {
		t.Fatalf("offline install: %+v, %v", report, err)
	}
	if _, err := os.Stat(filepath.Join(dir, hexagon.LicenseFilename)); err != nil {
		t.Fatalf("licence not installed: %v", err)
	}

	report, err = runHexagonSetup(t, "--validate-only", "--dir", dir)
	if err != nil || report.Status != "files_verified" || report.Dir != dir {
		t.Fatalf("validate-only: %+v, %v", report, err)
	}

	// Without --dir, validation reports the directory runners would use.
	t.Setenv(hexagon.PathEnv, dir)
	report, err = runHexagonSetup(t, "--validate-only")
	if err != nil || report.Dir != dir || len(report.SearchDirs) == 0 || report.SearchDirs[0] != dir {
		t.Fatalf("validate-only search: %+v, %v", report, err)
	}

	if err := os.WriteFile(filepath.Join(dir, "libggml-htp-v73.so"), []byte("changed"), 0o644); err != nil {
		t.Fatal(err)
	}
	if report, err := runHexagonSetup(t, "--validate-only", "--dir", dir); err == nil || report.Error.Code != "htp_validation_failed" {
		t.Fatalf("a changed file passed validation: %+v, %v", report, err)
	}
}
