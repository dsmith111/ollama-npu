//go:build windows

package cmd

import (
	"errors"

	"github.com/spf13/cobra"
)

func DebugNPUHandler(cmd *cobra.Command, args []string) error {
	report := newNPUDiagnostic("debug-npu")
	if len(args) != 0 {
		return finishNPUDiagnostic(cmd, report, "invalid_arguments", errors.New("debug-npu accepts no positional arguments"))
	}
	err := probeNPUHardware(&report)
	report.untestedInference()
	return finishNPUDiagnostic(cmd, report, "hardware_probe_failed", err)
}
