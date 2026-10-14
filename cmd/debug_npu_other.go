//go:build !windows

package cmd

import "github.com/spf13/cobra"

// DebugNPUHandler is not available on non-Windows platforms.
func DebugNPUHandler(cmd *cobra.Command, args []string) error {
	return unsupportedNPUDiagnostic(cmd, "debug-npu", "NPU diagnostics are only available on Windows")
}
