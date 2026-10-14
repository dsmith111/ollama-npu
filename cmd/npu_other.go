//go:build !windows

package cmd

import "github.com/spf13/cobra"

// NPUSetupHandler is not available on non-Windows platforms.
func NPUSetupHandler(cmd *cobra.Command, args []string) error {
	return unsupportedNPUDiagnostic(cmd, "npu setup", "NPU setup is only available on Windows ARM64")
}

// NPUDoctorHandler is not available on non-Windows platforms.
func NPUDoctorHandler(cmd *cobra.Command, args []string) error {
	return unsupportedNPUDiagnostic(cmd, "npu doctor", "NPU doctor is only available on Windows")
}
