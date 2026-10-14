//go:build !windows

package cmd

import "github.com/spf13/cobra"

// DebugORTGenAIHandler is not available on non-Windows platforms.
func DebugORTGenAIHandler(cmd *cobra.Command, args []string) error {
	return unsupportedNPUDiagnostic(cmd, "debug-ortgenai", "ORT GenAI diagnostics are only available on Windows")
}
