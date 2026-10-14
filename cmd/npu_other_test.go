//go:build !windows

package cmd

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/spf13/cobra"
)

func TestNPUUnsupportedPlatformFailures(t *testing.T) {
	for _, handler := range []func(*cobra.Command, []string) error{
		NPUSetupHandler, NPUDoctorHandler, DebugNPUHandler, DebugORTGenAIHandler,
	} {
		cmd := &cobra.Command{Use: "diagnostic", RunE: handler}
		ConfigureNPUSetupCommand(cmd)
		var output bytes.Buffer
		cmd.SetOut(&output)
		cmd.SetArgs([]string{"--json"})
		if err := cmd.Execute(); err == nil {
			t.Fatal("unsupported platform returned success")
		}
		var report npuDiagnostic
		if err := json.Unmarshal(output.Bytes(), &report); err != nil || report.Error.Code != "unsupported_platform" || report.InferenceVerified {
			t.Fatalf("unsupported-platform report: %s, %v", output.String(), err)
		}
	}
}

func TestNPUBenchmarkUnsupportedPlatformSchema(t *testing.T) {
	cmd := &cobra.Command{Use: "npu", RunE: BenchmarkNPUHandler}
	ConfigureNPUBenchmarkCommand(cmd)
	var output bytes.Buffer
	cmd.SetOut(&output)
	cmd.SetArgs([]string{"--model", "m", "--json", "-"})
	if err := cmd.Execute(); err == nil {
		t.Fatal("unsupported benchmark returned success")
	}
	var report npuBenchmarkReport
	if err := json.Unmarshal(output.Bytes(), &report); err != nil || report.Error.Code != "unsupported_platform" || report.Status != "failed" || len(report.Results) != 0 {
		t.Fatalf("unsupported-platform benchmark schema: %s, %v", output.String(), err)
	}
}
