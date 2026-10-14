//go:build windows

package cmd

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/spf13/cobra"
)

func TestNPUSetupPrintPathAndOfflineConfiguration(t *testing.T) {
	var output bytes.Buffer
	dir := filepath.Join(t.TempDir(), "not-created")
	cmd := &cobra.Command{Use: "setup", RunE: NPUSetupHandler}
	cmd.SetOut(&output)
	ConfigureNPUSetupCommand(cmd)
	cmd.SetArgs([]string{"--dir", dir, "--print-path", "--json"})
	if err := cmd.Execute(); err != nil {
		t.Fatal(err)
	}
	var report npuDiagnostic
	if err := json.Unmarshal(output.Bytes(), &report); err != nil || report.Dir != dir || report.Status != "path_only" {
		t.Fatalf("print path report: %s, %v", output.String(), err)
	}
	if _, err := os.Stat(dir); !os.IsNotExist(err) {
		t.Fatal("print-path changed installation files")
	}
}

func TestNPUSetupValidationOnlyFails(t *testing.T) {
	var output bytes.Buffer
	dir := t.TempDir()
	cmd := &cobra.Command{Use: "setup", RunE: NPUSetupHandler}
	cmd.SetOut(&output)
	ConfigureNPUSetupCommand(cmd)
	cmd.SetArgs([]string{"--dir", dir, "--validate-only", "--json"})
	if err := cmd.Execute(); err == nil {
		t.Fatal("missing runtime returned success")
	}
	var report npuDiagnostic
	if err := json.Unmarshal(output.Bytes(), &report); err != nil || report.Error.Code != "runtime_validation_failed" {
		t.Fatalf("missing-runtime JSON: %s, %v", output.String(), err)
	}
	entries, err := os.ReadDir(dir)
	if err != nil || len(entries) != 0 {
		t.Fatal("validate-only changed installation files")
	}
}

func TestNPUSetupOfflineMissingPackageAndCancellation(t *testing.T) {
	for _, canceled := range []bool{false, true} {
		t.Run(map[bool]string{false: "missing-package", true: "canceled"}[canceled], func(t *testing.T) {
			root := t.TempDir()
			dir := filepath.Join(root, "installed")
			cmd := &cobra.Command{Use: "setup", RunE: NPUSetupHandler}
			var output bytes.Buffer
			cmd.SetOut(&output)
			ConfigureNPUSetupCommand(cmd)
			cmd.SetArgs([]string{"--dir", dir, "--package-dir", root, "--json"})
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			if canceled {
				cancel()
			}
			cmd.SetContext(ctx)
			if err := cmd.Execute(); err == nil {
				t.Fatal("missing offline package/canceled setup returned success")
			}
			var report npuDiagnostic
			if err := json.Unmarshal(output.Bytes(), &report); err != nil || report.Error.Code != "installation_failed" {
				t.Fatalf("setup JSON: %s, %v", output.String(), err)
			}
			if _, err := os.Stat(dir); !os.IsNotExist(err) {
				t.Fatal("failed offline setup promoted an installation")
			}
		})
	}
}

func TestNPUSetupConflictingModes(t *testing.T) {
	cmd := &cobra.Command{Use: "setup", RunE: NPUSetupHandler}
	var output bytes.Buffer
	cmd.SetOut(&output)
	ConfigureNPUSetupCommand(cmd)
	cmd.SetArgs([]string{"--print-path", "--validate-only", "--json"})
	if err := cmd.Execute(); err == nil {
		t.Fatal("mutually exclusive setup modes succeeded")
	}
	var report npuDiagnostic
	if err := json.Unmarshal(output.Bytes(), &report); err != nil || report.Error.Code != "invalid_arguments" {
		t.Fatalf("conflict JSON: %s, %v", output.String(), err)
	}
}
