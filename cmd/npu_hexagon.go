//go:build windows

package cmd

import (
	"errors"
	"fmt"
	"path/filepath"
	"strings"

	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/x/hexagon"
	"github.com/spf13/cobra"
)

// hexagonSetupHandler installs or checks the signed HTP set that llama.cpp's
// Hexagon backend runs on the NPU. Qualcomm's terms attach use restrictions to
// the set, so Ollama does not ship it, and downloading it needs explicit
// acceptance.
func hexagonSetupHandler(cmd *cobra.Command, args []string) error {
	report := newNPUDiagnostic("npu setup")
	dir := diagnosticString(cmd, "dir")
	packageDir := diagnosticString(cmd, "package-dir")
	printPath := diagnosticBool(cmd, "print-path")
	validateOnly := diagnosticBool(cmd, "validate-only")
	if len(args) != 0 || printPath && validateOnly || packageDir != "" && (printPath || validateOnly) {
		return finishNPUDiagnostic(cmd, report, "invalid_arguments", errors.New("setup accepts no positional arguments; --print-path, --validate-only and --package-dir cannot be combined"))
	}
	if printPath {
		if dir == "" {
			dir = hexagon.InstallDir()
		}
		report.Dir = dir
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
		var err error
		if dir == "" {
			// The directory runners would use, in their search order.
			report.SearchDirs = hexagon.SearchDirs(filepath.Join(ml.LibOllamaPath, "hexagon"))
			dir, err = hexagon.VerifiedDir(report.SearchDirs)
		} else {
			err = hexagon.Validate(dir)
		}
		report.Dir = dir
		if err != nil {
			report.check("htp_integrity", "fail", "No verified signed Hexagon NPU library set; files were not changed")
			return finishNPUDiagnostic(cmd, report, "htp_validation_failed", err)
		}
		report.Status = "files_verified"
		report.check("htp_integrity", "pass", "Pinned "+hexagon.Pinned.Source+" files verified; loading them on the NPU is not tested")
		return finishNPUDiagnostic(cmd, report, "", nil)
	}

	if dir == "" {
		dir = hexagon.InstallDir()
	}
	report.Dir = dir
	if !diagnosticBool(cmd, "accept-qualcomm-terms") {
		report.check("terms", "fail", hexagonTermsNotice())
		return finishNPUDiagnostic(cmd, report, "terms_not_accepted", errors.New("rerun with --accept-qualcomm-terms to get Qualcomm's signed libraries"))
	}
	if packageDir == "" && !diagnosticBool(cmd, "json") {
		fmt.Fprintf(cmd.ErrOrStderr(), "Downloading %s (%d MB) from %s\n",
			hexagon.Pinned.ArchiveName, hexagon.Pinned.ArchiveSize>>20, hexagon.Pinned.URL)
	}
	result, err := hexagon.Install(cmd.Context(), dir, packageDir)
	report.Hexagon = result
	if result != nil {
		report.Dir = result.Dir
	}
	if err != nil {
		report.check("htp_integrity", "fail", "Setup failed; existing files were not replaced")
		return finishNPUDiagnostic(cmd, report, "installation_failed", err)
	}
	report.Status = "files_verified"
	report.check("htp_integrity", "pass", result.Message)
	next := "Start the server with OLLAMA_HEXAGON=1"
	if !strings.EqualFold(result.Dir, filepath.Clean(hexagon.InstallDir())) {
		next += " and " + hexagon.PathEnv + "=" + result.Dir
	}
	report.check("next_step", "not_tested", next)
	return finishNPUDiagnostic(cmd, report, "", nil)
}

func hexagonTermsNotice() string {
	return fmt.Sprintf("The signed Hexagon NPU libraries come from %s on GitHub (BSD-3-Clause). "+
		"Qualcomm also makes their use subject to its Terms of Use and Responsible AI License (%s), "+
		"which restrict how they may be used: for example, no military use, and content generated "+
		"with them must be disclosed as machine generated.",
		hexagon.Pinned.Source, strings.Join(hexagon.Terms, ", "))
}
