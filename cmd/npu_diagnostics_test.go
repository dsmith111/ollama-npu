package cmd

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/spf13/cobra"
)

func TestNPUCommandConfiguration(t *testing.T) {
	setup := &cobra.Command{}
	setup.Flags().String("provider", "auto", "old")
	setup.Flags().Bool("force", false, "old")
	ConfigureNPUSetupCommand(setup)
	ConfigureNPUSetupCommand(setup)
	for _, name := range []string{"dir", "package-dir", "json", "provider", "force", "validate-only", "print-path"} {
		if setup.Flags().Lookup(name) == nil {
			t.Fatalf("missing setup flag %s", name)
		}
	}
	if strings.Contains(setup.Flags().Lookup("provider").Usage, "dml") || !strings.Contains(setup.Flags().Lookup("force").Usage, "never overwrite") {
		t.Fatal("setup help promises unsupported providers or replacement")
	}
	benchmark := &cobra.Command{}
	ConfigureNPUBenchmarkCommand(benchmark)
	ConfigureNPUBenchmarkCommand(benchmark)
	if !strings.Contains(benchmark.Flags().Lookup("json").Usage, "stdout") {
		t.Fatal("benchmark help omits JSON stdout")
	}
}

func TestNPUDiagnosticJSONFailureAndStages(t *testing.T) {
	var output bytes.Buffer
	cmd := &cobra.Command{}
	ConfigureNPUDoctorCommand(cmd)
	cmd.SetOut(&output)
	cmd.Flags().Set("json", "true")
	report := newNPUDiagnostic("npu doctor")
	report.check("runtime_integrity", "pass", "Files verified")
	report.untestedInference()
	err := finishNPUDiagnostic(cmd, report, "test_failed", errors.New("deliberate failure"))
	if err == nil || !cmd.SilenceErrors || !cmd.SilenceUsage {
		t.Fatal("JSON failure did not preserve nonzero exit or silence Cobra's extra output")
	}
	var actual npuDiagnostic
	if err := json.Unmarshal(output.Bytes(), &actual); err != nil {
		t.Fatalf("not a single JSON report: %s: %v", output.String(), err)
	}
	if actual.SchemaVersion != 1 || actual.Status != "checks_failed" || actual.Error.Code != "test_failed" || actual.InferenceVerified || actual.ProviderVerified {
		t.Fatalf("unsafe/inconsistent readiness report: %+v", actual)
	}
	for _, check := range actual.Checks[1:] {
		if check.Status != "not_tested" {
			t.Fatalf("untested inference stage reported as passed: %+v", check)
		}
	}
}

func TestNPUDiagnosticHumanOutputDoesNotClaimReady(t *testing.T) {
	var output bytes.Buffer
	cmd := &cobra.Command{}
	cmd.SetOut(&output)
	report := newNPUDiagnostic("npu doctor")
	report.check("hardware", "pass", "An NPU was enumerated")
	report.untestedInference()
	if err := finishNPUDiagnostic(cmd, report, "", nil); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(output.String(), "READY") || !strings.Contains(output.String(), "not_tested") {
		t.Fatalf("enumeration presented as inference proof: %s", output.String())
	}
}

func TestNPURuntimeExplicitDirectoryNoFallback(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "onnxruntime-genai.dll"), []byte("untrusted"), 0o600); err != nil {
		t.Fatal(err)
	}
	report := newNPUDiagnostic("debug-ortgenai")
	err := inspectNPURuntime(context.Background(), dir, &report)
	if err == nil || report.Dir != dir || report.Runtime == nil || report.Runtime.Valid {
		t.Fatalf("explicit invalid directory was bypassed: %+v, %v", report, err)
	}
	if len(report.SearchDirs) != 0 {
		t.Fatal("explicit directory unexpectedly searched fallback directories")
	}
}

func TestNPURuntimeSearchDoesNotCombineOrSkipInvalid(t *testing.T) {
	first, second := t.TempDir(), t.TempDir()
	if err := os.WriteFile(filepath.Join(first, "QnnHtp.dll"), []byte("incomplete"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(second, "onnxruntime-genai.dll"), []byte("other"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("OLLAMA_ORT_PATH", first+string(os.PathListSeparator)+second)
	report := newNPUDiagnostic("debug-ortgenai")
	if err := inspectNPURuntime(context.Background(), "", &report); err == nil || report.Dir != first || report.Runtime.Valid {
		t.Fatalf("partial installation skipped or combined: %+v, %v", report, err)
	}
}

func TestNPURuntimeCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	report := newNPUDiagnostic("debug-ortgenai")
	if err := inspectNPURuntime(ctx, t.TempDir(), &report); !errors.Is(err, context.Canceled) {
		t.Fatalf("validation cancellation lost: %v", err)
	}
}
