//go:build windows

package cmd

import (
	"errors"
	"fmt"
	"runtime"

	"github.com/ollama/ollama/discover/dxcore"
	ortruntime "github.com/ollama/ollama/x/ortrunner/runtime"
	"github.com/spf13/cobra"
)

func NPUSetupHandler(cmd *cobra.Command, args []string) error {
	report := newNPUDiagnostic("npu setup")
	dir := diagnosticString(cmd, "dir")
	if dir == "" {
		dir = ortruntime.InstallDir()
	}
	report.Dir = dir
	printPath := diagnosticBool(cmd, "print-path")
	validateOnly := diagnosticBool(cmd, "validate-only")
	if len(args) != 0 || printPath && validateOnly || diagnosticString(cmd, "package-dir") != "" && (printPath || validateOnly) {
		return finishNPUDiagnostic(cmd, report, "invalid_arguments", errors.New("setup accepts no positional arguments; --print-path, --validate-only and --package-dir cannot be combined"))
	}
	if printPath {
		if dir == "" {
			return finishNPUDiagnostic(cmd, report, "path_unavailable", errors.New("unable to determine installation directory"))
		}
		if !diagnosticBool(cmd, "json") {
			fmt.Fprintln(cmd.OutOrStdout(), dir)
			return nil
		}
		report.Status = "path_only"
		return finishNPUDiagnostic(cmd, report, "", nil)
	}
	report.untestedInference()
	if validateOnly {
		err := inspectNPURuntime(cmd.Context(), dir, &report)
		if err == nil {
			report.Status = "files_verified"
		}
		return finishNPUDiagnostic(cmd, report, "runtime_validation_failed", err)
	}
	result, err := ortruntime.SetupContext(cmd.Context(), ortruntime.Config{
		Provider: diagnosticString(cmd, "provider"), Force: diagnosticBool(cmd, "force"),
		Dir: dir, PackageDir: diagnosticString(cmd, "package-dir"),
	})
	report.Installation = result
	if result != nil {
		report.Dir = result.Dir
		report.Runtime = &result.Report
	}
	if err == nil && (result == nil || !result.Validated) {
		err = errors.New("installer did not return a verified runtime")
	}
	if err != nil {
		report.check("runtime_integrity", "fail", "Installation failed; existing runtime files were not replaced")
		return finishNPUDiagnostic(cmd, report, "installation_failed", err)
	}
	report.Status = "files_verified"
	report.check("runtime_integrity", "pass", result.Message)
	report.check("next_step", "not_tested", "Use 'ollama npu doctor'; a compatible model and verified provider execution are still required")
	return finishNPUDiagnostic(cmd, report, "", nil)
}

func NPUDoctorHandler(cmd *cobra.Command, args []string) error {
	report := newNPUDiagnostic("npu doctor")
	var failures []error
	if len(args) != 0 {
		return finishNPUDiagnostic(cmd, report, "invalid_arguments", errors.New("doctor accepts no positional arguments"))
	}
	if runtime.GOARCH != "arm64" {
		report.check("platform", "fail", "The pinned QNN bundle requires Windows ARM64")
		failures = append(failures, errors.New("pinned QNN bundle requires Windows ARM64"))
	} else {
		report.check("platform", "pass", "Windows ARM64")
	}
	if err := probeNPUHardware(&report); err != nil {
		failures = append(failures, err)
	}
	if err := inspectNPURuntime(cmd.Context(), diagnosticString(cmd, "dir"), &report); err != nil {
		failures = append(failures, err)
	}
	report.untestedInference()
	return finishNPUDiagnostic(cmd, report, "diagnostic_checks_failed", errors.Join(failures...))
}

func probeNPUHardware(report *npuDiagnostic) error {
	result := dxcore.Probe()
	hardware := &npuHardware{DXCoreAvailable: result.DXCoreAvailable, NPUCount: result.NPUCount, Adapters: []npuAdapter{}}
	for _, adapter := range result.Adapters {
		hardware.Adapters = append(hardware.Adapters, npuAdapter{
			Index: adapter.Index, Description: adapter.Description, NPU: adapter.IsNPU,
			D3D12: adapter.HasD3D12, DML: adapter.HasDML, GenericML: adapter.HasD3D12GenericML,
			DriverVersion: adapter.DriverVersion,
		})
	}
	report.Hardware = hardware
	if result.Error != "" {
		report.check("hardware", "fail", "DXCore probe failed")
		return fmt.Errorf("DXCore probe failed: %s", result.Error)
	}
	if !result.DXCoreAvailable || result.NPUCount == 0 {
		report.check("hardware", "fail", "No NPU was enumerated")
		return errors.New("no NPU adapters found")
	}
	report.check("hardware", "pass", "NPU adapter enumerated; D3D12/DirectML device creation is not QNN inference proof")
	return nil
}
