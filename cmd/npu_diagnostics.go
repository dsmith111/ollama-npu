package cmd

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/ollama/ollama/x/hexagon"
	ortruntime "github.com/ollama/ollama/x/ortrunner/runtime"
	"github.com/spf13/cobra"
)

// ConfigureNPUSetupCommand is kept separate from root command construction.
func ConfigureNPUSetupCommand(cmd *cobra.Command) {
	cmd.Short = "Acquire or verify the pinned Windows ARM64 QNN runtime or Hexagon NPU library files"
	addDiagnosticFlags(cmd)
	flags := cmd.Flags()
	if flags.Lookup("provider") == nil {
		flags.String("provider", "auto", "Pinned bundle provider: auto or qnn")
	}
	flags.Lookup("provider").Usage = "Pinned bundle provider: auto or qnn (ORT runtime), or hexagon (signed NPU libraries for GGUF models)"
	if flags.Lookup("accept-qualcomm-terms") == nil {
		flags.Bool("accept-qualcomm-terms", false, "With --provider hexagon: accept the Qualcomm terms that come with the signed libraries")
	}
	if flags.Lookup("force") == nil {
		flags.Bool("force", false, "Revalidate an immutable installation; never overwrite existing files")
	}
	flags.Lookup("force").Usage = "Revalidate an immutable installation; never overwrite existing files"
	if flags.Lookup("print-path") == nil {
		flags.Bool("print-path", false, "Print the installation directory without changing files")
	}
	if flags.Lookup("validate-only") == nil {
		flags.Bool("validate-only", false, "Verify pinned files and manifest without installing or loading libraries")
	}
	flags.Lookup("validate-only").Usage = "Verify pinned files and manifest without installing or loading libraries"
	if flags.Lookup("dir") == nil {
		flags.String("dir", "", "Explicit runtime installation or validation directory")
	}
	if flags.Lookup("package-dir") == nil {
		flags.String("package-dir", "", "Offline-only directory of approved <lowercase-package-id>.<version>.nupkg files")
	}
}

func ConfigureNPUDoctorCommand(cmd *cobra.Command) {
	cmd.Short = "Check NPU hardware and pinned runtime files; does not verify inference"
	addDiagnosticFlags(cmd)
	if cmd.Flags().Lookup("dir") == nil {
		cmd.Flags().String("dir", "", "Verify this directory instead of searching runtime paths")
	}
}

func ConfigureNPUDebugCommand(cmd *cobra.Command) { addDiagnosticFlags(cmd) }

func ConfigureORTGenAIDebugCommand(cmd *cobra.Command) {
	ConfigureNPUDoctorCommand(cmd)
	cmd.Short = "Inspect pinned runtime integrity and configuration without loading a model"
}

func ConfigureNPUBenchmarkCommand(cmd *cobra.Command) {
	cmd.Short = "Measure model generation; actual execution providers remain unverified"
	flags := cmd.Flags()
	if flags.Lookup("model") == nil {
		flags.String("model", "", "Model(s) to measure, comma-separated; names do not identify actual providers")
	}
	flags.Lookup("model").Usage = "Model(s) to measure, comma-separated; names do not identify actual providers"
	if flags.Lookup("ctx") == nil {
		flags.Int("ctx", 512, "Context length (2–1048576); half is reserved for the prompt")
	}
	flags.Lookup("ctx").Usage = "Context length (2–1048576); half is reserved for the prompt"
	if flags.Lookup("predict") == nil {
		flags.Int("predict", 128, "Positive generation token limit, at most half the context")
	}
	flags.Lookup("predict").Usage = "Positive generation token limit, at most half the context"
	if flags.Lookup("epochs") == nil {
		flags.Int("epochs", 8, "Number of timed epochs (1–10000)")
	}
	flags.Lookup("epochs").Usage = "Number of timed epochs (1–10000)"
	if flags.Lookup("warmup") == nil {
		flags.Int("warmup", 1, "Number of warmup epochs (0–10000); failures fail the benchmark")
	}
	flags.Lookup("warmup").Usage = "Number of warmup epochs (0–10000); failures fail the benchmark"
	if flags.Lookup("json") == nil {
		flags.String("json", "", "Write versioned JSON results to a file, or '-' for JSON-only stdout")
	}
	flags.Lookup("json").Usage = "Write versioned JSON results to a file, or '-' for JSON-only stdout"
}

func addDiagnosticFlags(cmd *cobra.Command) {
	if cmd.Flags().Lookup("json") == nil {
		cmd.Flags().Bool("json", false, "Write a versioned diagnostic JSON report to stdout")
	}
}

type npuCheck struct {
	Stage   string `json:"stage"`
	Status  string `json:"status"`
	Message string `json:"message"`
}

type npuDiagnosticError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

type epochMetrics struct {
	Epoch            int           `json:"epoch"`
	TTFT             time.Duration `json:"ttft_ns"`
	PrefillCount     int           `json:"prefill_count"`
	PrefillDuration  time.Duration `json:"prefill_duration_ns"`
	GenerateCount    int           `json:"generate_count"`
	GenerateDuration time.Duration `json:"generate_duration_ns"`
	LoadDuration     time.Duration `json:"load_duration_ns"`
	TotalDuration    time.Duration `json:"total_duration_ns"`
}

type benchResult struct {
	Model            string              `json:"model"`
	Provider         string              `json:"provider"`
	ProviderVerified bool                `json:"provider_execution_verified"`
	Status           string              `json:"status"`
	Epochs           []epochMetrics      `json:"epochs"`
	Error            *npuDiagnosticError `json:"error,omitempty"`
}

type npuBenchmarkReport struct {
	SchemaVersion int                 `json:"schema_version"`
	Command       string              `json:"command"`
	Status        string              `json:"status"`
	Models        []string            `json:"models"`
	Context       int                 `json:"ctx"`
	Predict       int                 `json:"predict"`
	Epochs        int                 `json:"epochs"`
	Warmup        int                 `json:"warmup"`
	Results       []benchResult       `json:"results"`
	Error         *npuDiagnosticError `json:"error,omitempty"`
}

func newNPUBenchmarkReport(cmd *cobra.Command) npuBenchmarkReport {
	numCtx, _ := cmd.Flags().GetInt("ctx")
	numPredict, _ := cmd.Flags().GetInt("predict")
	epochs, _ := cmd.Flags().GetInt("epochs")
	warmup, _ := cmd.Flags().GetInt("warmup")
	report := npuBenchmarkReport{
		SchemaVersion: 1, Command: "benchmark npu", Status: "measured",
		Models: []string{}, Context: numCtx, Predict: numPredict, Epochs: epochs, Warmup: warmup, Results: []benchResult{},
	}
	if models := diagnosticString(cmd, "model"); models != "" {
		for _, model := range strings.Split(models, ",") {
			report.Models = append(report.Models, strings.TrimSpace(model))
		}
	}
	return report
}

func writeNPUBenchmarkJSON(cmd *cobra.Command, report npuBenchmarkReport) error {
	path := diagnosticString(cmd, "json")
	if path == "" {
		return nil
	}
	encoder := json.NewEncoder(cmd.OutOrStdout())
	if path == "-" {
		if report.Error != nil {
			cmd.SilenceErrors, cmd.SilenceUsage = true, true
		}
		encoder.SetIndent("", "  ")
		return encoder.Encode(report)
	}
	file, err := os.OpenFile(path, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o600)
	if err != nil {
		return err
	}
	encoder = json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	err = encoder.Encode(report)
	if err == nil {
		err = file.Sync()
	}
	return errors.Join(err, file.Close())
}

type npuAdapter struct {
	Index         int    `json:"index"`
	Description   string `json:"description"`
	NPU           bool   `json:"npu"`
	D3D12         bool   `json:"d3d12_device"`
	DML           bool   `json:"directml_device"`
	GenericML     bool   `json:"generic_ml"`
	DriverVersion uint64 `json:"driver_version"`
}

type npuHardware struct {
	DXCoreAvailable bool         `json:"dxcore_available"`
	NPUCount        int          `json:"npu_count"`
	Adapters        []npuAdapter `json:"adapters"`
}

type npuDiagnostic struct {
	SchemaVersion     int                          `json:"schema_version"`
	Command           string                       `json:"command"`
	Status            string                       `json:"status"`
	InferenceVerified bool                         `json:"inference_verified"`
	ProviderVerified  bool                         `json:"provider_execution_verified"`
	Checks            []npuCheck                   `json:"checks"`
	SearchDirs        []string                     `json:"search_dirs,omitempty"`
	Dir               string                       `json:"dir,omitempty"`
	Hardware          *npuHardware                 `json:"hardware,omitempty"`
	Runtime           *ortruntime.ValidationReport `json:"runtime,omitempty"`
	Installation      *ortruntime.InstallResult    `json:"installation,omitempty"`
	Hexagon           *hexagon.InstallResult       `json:"hexagon,omitempty"`
	Configuration     map[string]string            `json:"configuration,omitempty"`
	Error             *npuDiagnosticError          `json:"error,omitempty"`
}

func newNPUDiagnostic(command string) npuDiagnostic {
	report := npuDiagnostic{SchemaVersion: 1, Command: command, Status: "checks_passed", Checks: []npuCheck{}}
	// Deliberately allowlist runtime configuration, never arbitrary environment,
	// prompts, generated text, authorization headers or model request bodies.
	report.Configuration = map[string]string{}
	for _, name := range []string{
		"OLLAMA_ONNX_PROVIDER", "OLLAMA_ORT_QNN_BACKEND_TYPE", "OLLAMA_ORT_QNN_BACKEND_PATH",
		"OLLAMA_ORT_QNN_HTP_ARCH", "OLLAMA_ORT_QNN_SOC_MODEL",
	} {
		if value := os.Getenv(name); value != "" {
			report.Configuration[name] = value
		}
	}
	return report
}

func (report *npuDiagnostic) check(stage, status, message string) {
	report.Checks = append(report.Checks, npuCheck{stage, status, message})
}

func (report *npuDiagnostic) untestedInference() {
	report.check("library_load", "not_tested", "Native libraries were not loaded by this file/hardware diagnostic")
	report.check("model_load", "not_tested", "No model was loaded")
	report.check("generation", "not_tested", "No generation request was made")
	report.check("provider_execution", "not_tested", "Actual execution provider is unverified; file or device checks are not inference proof")
}

func diagnosticBool(cmd *cobra.Command, name string) bool {
	value, _ := cmd.Flags().GetBool(name)
	return value
}

func diagnosticString(cmd *cobra.Command, name string) string {
	value, _ := cmd.Flags().GetString(name)
	return value
}

func finishNPUDiagnostic(cmd *cobra.Command, report npuDiagnostic, code string, cause error) error {
	if cause != nil {
		report.Status = "checks_failed"
		report.Error = &npuDiagnosticError{code, cause.Error()}
		cmd.SilenceUsage = true
	}
	if diagnosticBool(cmd, "json") {
		encoder := json.NewEncoder(cmd.OutOrStdout())
		encoder.SetIndent("", "  ")
		if err := encoder.Encode(report); err != nil {
			return fmt.Errorf("write diagnostic JSON: %w", err)
		}
		// Cobra must still return a failing exit status without adding usage or
		// a second unstructured copy of the report to stdout/stderr.
		if cause != nil {
			cmd.SilenceErrors, cmd.SilenceUsage = true, true
		}
	} else {
		out := cmd.OutOrStdout()
		fmt.Fprintf(out, "=== %s ===\nStatus: %s\n", report.Command, report.Status)
		if report.Dir != "" {
			fmt.Fprintf(out, "Directory: %s\n", report.Dir)
		}
		for _, name := range []string{
			"OLLAMA_ONNX_PROVIDER", "OLLAMA_ORT_QNN_BACKEND_TYPE", "OLLAMA_ORT_QNN_BACKEND_PATH",
			"OLLAMA_ORT_QNN_HTP_ARCH", "OLLAMA_ORT_QNN_SOC_MODEL",
		} {
			if value := report.Configuration[name]; value != "" {
				fmt.Fprintf(out, "Configured %s: %s (not actual execution proof)\n", name, value)
			}
		}
		for _, dir := range report.SearchDirs {
			fmt.Fprintf(out, "Search: %s\n", dir)
		}
		if report.Hardware != nil {
			for _, adapter := range report.Hardware.Adapters {
				fmt.Fprintf(out, "Adapter #%d: %s (NPU=%t, D3D12 device=%t, DirectML device=%t)\n",
					adapter.Index, adapter.Description, adapter.NPU, adapter.D3D12, adapter.DML)
			}
		}
		for _, check := range report.Checks {
			fmt.Fprintf(out, "%s: %s — %s\n", check.Stage, check.Status, check.Message)
		}
		if cause != nil {
			fmt.Fprintf(out, "Error [%s]: %s\n", code, cause)
		}
	}
	if cause != nil {
		return fmt.Errorf("%s: %w", code, cause)
	}
	return nil
}

func unsupportedNPUDiagnostic(cmd *cobra.Command, command, message string) error {
	report := newNPUDiagnostic(command)
	report.check("platform", "fail", message)
	report.untestedInference()
	return finishNPUDiagnostic(cmd, report, "unsupported_platform", errors.New(message))
}

// Choose the first runtime-bearing directory, never combine DLLs across paths or
// skip an invalid/partially installed candidate in favor of a later valid one.
func inspectNPURuntime(ctx context.Context, dir string, report *npuDiagnostic) error {
	if ctx == nil {
		return errors.New("nil diagnostic context")
	}
	if err := ctx.Err(); err != nil {
		report.check("runtime_integrity", "fail", "Runtime validation was canceled")
		return err
	}
	if dir == "" {
		report.SearchDirs = ortruntime.SearchDirs()
		markers := append([]string{ortruntime.ManifestFilename}, ortruntime.RequiredNativeFiles()...)
		for _, candidate := range report.SearchDirs {
			for _, marker := range markers {
				_, err := os.Lstat(filepath.Join(candidate, marker))
				if err == nil || !errors.Is(err, os.ErrNotExist) {
					dir = candidate
					break
				}
			}
			if dir != "" {
				break
			}
		}
	}
	if dir == "" {
		report.check("runtime_integrity", "fail", "No pinned runtime found; use 'ollama npu setup' or an explicit --dir")
		return errors.New("no runtime found in the configured search directories")
	}
	report.Dir = dir
	validation, err := ortruntime.ValidateInstallDirContext(ctx, dir)
	report.Runtime = &validation
	if err != nil {
		report.check("runtime_integrity", "fail", "Runtime failed pinned manifest, hash or architecture validation; files were not changed")
		return err
	}
	report.check("runtime_integrity", "pass", "Approved package provenance, raw file hashes, native architecture and notices verified")
	report.check("external_dependencies", "not_tested", "OEM DSP RPC transport and ARM64 VC++ dependencies are driver/system-owned; install validation does not test them")
	return nil
}
