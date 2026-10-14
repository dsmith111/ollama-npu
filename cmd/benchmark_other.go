//go:build !windows

package cmd

import (
	"fmt"

	"github.com/spf13/cobra"
)

// BenchmarkNPUHandler is not available on non-Windows platforms.
func BenchmarkNPUHandler(cmd *cobra.Command, args []string) error {
	message := "NPU benchmark is only available on Windows"
	report := newNPUBenchmarkReport(cmd)
	report.Status = "failed"
	report.Error = &npuDiagnosticError{"unsupported_platform", message}
	if err := writeNPUBenchmarkJSON(cmd, report); err != nil {
		return fmt.Errorf("report_write_failed: %w", err)
	}
	return fmt.Errorf("unsupported_platform: %s", message)
}
