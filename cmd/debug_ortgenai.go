//go:build windows

package cmd

import (
	"errors"

	"github.com/spf13/cobra"
)

func DebugORTGenAIHandler(cmd *cobra.Command, args []string) error {
	report := newNPUDiagnostic("debug-ortgenai")
	if len(args) != 0 {
		return finishNPUDiagnostic(cmd, report, "invalid_arguments", errors.New("debug-ortgenai accepts no positional arguments"))
	}
	err := inspectNPURuntime(cmd.Context(), diagnosticString(cmd, "dir"), &report)
	report.untestedInference()
	return finishNPUDiagnostic(cmd, report, "runtime_validation_failed", err)
}
