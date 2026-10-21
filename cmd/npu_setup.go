package cmd

import (
	"errors"
	"fmt"
	"path/filepath"
	"strings"

	"github.com/ollama/ollama/discover/hexagon"
	"github.com/ollama/ollama/ml"
	"github.com/spf13/cobra"
)

// npuSetupHandler installs or checks the signed libraries that llama.cpp's
// Hexagon backend runs on the NPU. Qualcomm's terms attach use restrictions to
// the set, so Ollama does not ship it, and downloading it needs explicit
// acceptance.
func npuSetupHandler(cmd *cobra.Command, args []string) error {
	if len(args) != 0 {
		return errors.New("setup takes no arguments")
	}
	dir, _ := cmd.Flags().GetString("dir")
	packageDir, _ := cmd.Flags().GetString("package-dir")
	validateOnly, _ := cmd.Flags().GetBool("validate-only")
	accepted, _ := cmd.Flags().GetBool("accept-qualcomm-terms")
	out := cmd.OutOrStdout()

	if validateOnly {
		var err error
		if dir == "" {
			// The directory runners would use, in their search order.
			dir, err = hexagon.VerifiedDir(hexagon.SearchDirs(filepath.Join(ml.LibOllamaPath, "hexagon")))
		} else {
			err = hexagon.Validate(dir)
		}
		if err != nil {
			return fmt.Errorf("no verified set of the signed Hexagon NPU libraries: %w", err)
		}
		fmt.Fprintf(out, "%s holds %s; loading the libraries on the NPU is not tested\n", dir, hexagon.Pinned.Source)
		return nil
	}

	if dir == "" {
		dir = hexagon.InstallDir()
	}
	if dir == "" {
		return errors.New("the signed Hexagon NPU libraries are for Windows on Snapdragon; pass --dir to install them elsewhere")
	}
	if !accepted {
		fmt.Fprintln(out, hexagonTermsNotice())
		return errors.New("rerun with --accept-qualcomm-terms to download Qualcomm's signed libraries")
	}
	if packageDir == "" {
		fmt.Fprintf(cmd.ErrOrStderr(), "Downloading %s (%d MB) from %s\n",
			hexagon.Pinned.ArchiveName, hexagon.Pinned.ArchiveSize>>20, hexagon.Pinned.URL)
	}
	result, err := hexagon.Install(cmd.Context(), dir, packageDir)
	if err != nil {
		return err
	}
	fmt.Fprintf(out, "%s: %s\n", result.Dir, result.Message)
	next := "Start the server with OLLAMA_HEXAGON=1"
	if !strings.EqualFold(result.Dir, filepath.Clean(hexagon.InstallDir())) {
		next += " and " + hexagon.PathEnv + "=" + result.Dir
	}
	fmt.Fprintln(out, next)
	return nil
}

func hexagonTermsNotice() string {
	return fmt.Sprintf("The signed Hexagon NPU libraries come from %s on GitHub (BSD-3-Clause). "+
		"Qualcomm also makes their use subject to its Terms of Use and Responsible AI License (%s), "+
		"which restrict how they may be used: for example, no military use, and content generated "+
		"with them must be disclosed as machine generated.",
		hexagon.Pinned.Source, strings.Join(hexagon.Terms, ", "))
}

func newNPUCommand() *cobra.Command {
	npuCmd := &cobra.Command{
		Use:   "npu",
		Short: "Manage the NPU backend",
	}
	setupCmd := &cobra.Command{
		Use:   "setup",
		Short: "Download the signed Hexagon NPU libraries from Qualcomm's GenieX release",
		Args:  cobra.NoArgs,
		RunE:  npuSetupHandler,
	}
	setupCmd.Flags().Bool("accept-qualcomm-terms", false, "Accept Qualcomm's terms for the signed libraries")
	setupCmd.Flags().String("dir", "", "Install into this directory instead of the per-user default")
	setupCmd.Flags().String("package-dir", "", "Use the GenieX SDK archive already downloaded into this directory")
	setupCmd.Flags().Bool("validate-only", false, "Check the installed set and change nothing")
	npuCmd.AddCommand(setupCmd)
	return npuCmd
}
