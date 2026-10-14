package runtime

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

const (
	ProviderAuto = "auto"
	ProviderQNN  = "qnn"
	ProviderDML  = "dml"
	ProviderCPU  = "cpu"
)

// InstallConfig controls acquisition. PackageDir selects offline-only acquisition
// from <lowercase-package-id>.<version>.nupkg files. Dir overrides InstallDir.
// Force revalidates but never deletes or overwrites an existing installation.
type InstallConfig struct {
	Provider   string
	Force      bool
	Dir        string
	PackageDir string
	Context    context.Context
}

// Config and Setup retain compatibility with callers using the setup API.
type Config = InstallConfig

type InstallResult struct {
	Dir       string           `json:"dir"`
	Provider  string           `json:"provider"`
	Installed bool             `json:"installed"`
	Validated bool             `json:"validated"` // file integrity only, not device/driver readiness
	Message   string           `json:"message"`
	Report    ValidationReport `json:"report"`
}

var ErrUnsafeReplacement = errors.New("refusing to replace an existing runtime installation")

// InstallDir returns the immutable, versioned per-user QNN installation path.
func InstallDir() string {
	if runtime.GOOS != "windows" {
		return ""
	}
	base := os.Getenv("LOCALAPPDATA")
	if base == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return ""
		}
		base = filepath.Join(home, "AppData", "Local")
	}
	return filepath.Join(base, "Ollama", "runtimes", "ortgenai", "win-arm64-qnn",
		PinnedGenAIVersion+"_"+PinnedORTVersion)
}

func DefaultInstallDir() string { return InstallDir() }

// SearchDirs shares the loader/diagnostic search order without changing PATH.
// Callers must validate a directory before loading native code from it.
func SearchDirs() []string {
	dirs := filepath.SplitList(os.Getenv("OLLAMA_ORT_PATH"))
	dirs = append(dirs, InstallDir())
	if exe, err := os.Executable(); err == nil {
		if resolved, err := filepath.EvalSymlinks(exe); err == nil {
			exe = resolved
		}
		dirs = append(dirs, filepath.Dir(exe))
	}
	var out []string
	seen := make(map[string]bool)
	for _, dir := range dirs {
		if dir == "" {
			continue
		}
		dir = filepath.Clean(dir)
		key := dir
		if runtime.GOOS == "windows" {
			key = strings.ToLower(key)
		}
		if !seen[key] {
			out = append(out, dir)
			seen[key] = true
		}
	}
	return out
}

// VerifiedDir selects the first declared runtime containing GenAI, refusing an
// invalid candidate instead of silently falling back to another installation.
func VerifiedDir(ctx context.Context, dirs []string) (string, error) {
	return verifiedDir(ctx, dirs, func(ctx context.Context, dir string) error {
		_, err := ValidateInstallDirContext(ctx, dir)
		return err
	})
}

func verifiedDir(ctx context.Context, dirs []string, validate func(context.Context, string) error) (string, error) {
	for _, dir := range dirs {
		if err := ctx.Err(); err != nil {
			return "", err
		}
		_, err := os.Lstat(filepath.Join(dir, "onnxruntime-genai.dll"))
		if errors.Is(err, os.ErrNotExist) {
			continue
		}
		if err != nil {
			return "", fmt.Errorf("inspect selected QNN runtime: %w", err)
		}
		if err := validate(ctx, dir); err != nil {
			return "", fmt.Errorf("QNN runtime %s is not the approved intact installation; run ollama npu setup in a fresh directory and select it with OLLAMA_ORT_PATH: %w", dir, err)
		}
		return dir, nil
	}
	return "", fmt.Errorf("no verified QNN runtime found; run ollama npu setup --provider qnn and select it with OLLAMA_ORT_PATH")
}

func IsInstalled() bool {
	dir := InstallDir()
	if dir == "" {
		return false
	}
	report, err := ValidateInstallDir(dir)
	return err == nil && report.Valid
}

func Setup(cfg Config) (*InstallResult, error) { return Install(cfg) }

func Install(cfg InstallConfig) (*InstallResult, error) {
	ctx := cfg.Context
	if ctx == nil {
		ctx = context.Background()
	}
	return InstallContext(ctx, cfg)
}

func SetupContext(ctx context.Context, cfg Config) (*InstallResult, error) {
	return InstallContext(ctx, cfg)
}

func InstallContext(ctx context.Context, cfg InstallConfig) (*InstallResult, error) {
	if runtime.GOOS != "windows" || runtime.GOARCH != "arm64" {
		return nil, fmt.Errorf("ORT GenAI QNN acquisition requires Windows ARM64")
	}
	client := downloadClient()
	defer client.CloseIdleConnections()
	return install(ctx, cfg, approvedPackages(), client, nugetDownloadURL)
}

func install(ctx context.Context, cfg InstallConfig, specs []packageSpec, client *http.Client, urlFor func(NuGetPackage) string) (*InstallResult, error) {
	if ctx == nil {
		return nil, errors.New("nil installation context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if cfg.Provider != "" && cfg.Provider != ProviderAuto && cfg.Provider != ProviderQNN {
		return nil, fmt.Errorf("provider %q is not supported by the pinned QNN bundle", cfg.Provider)
	}
	dir := cfg.Dir
	if dir == "" {
		dir = InstallDir()
	}
	if dir == "" {
		return nil, errors.New("unable to determine runtime installation directory")
	}
	dir, err := filepath.Abs(dir)
	if err != nil {
		return nil, err
	}
	parent := filepath.Dir(dir)
	if dir == parent {
		return nil, errors.New("runtime installation cannot be a filesystem root")
	}
	if err := makeSafeDir(parent); err != nil {
		return nil, fmt.Errorf("installation parent: %w", err)
	}
	unlock, err := acquireInstallLock(ctx, dir+".lock")
	if err != nil {
		return nil, fmt.Errorf("installation lock: %w", err)
	}
	defer unlock()

	if _, err := os.Lstat(dir); err == nil {
		report, validationErr := validate(ctx, dir, specs)
		result := &InstallResult{Dir: dir, Provider: ProviderQNN, Validated: report.Valid, Report: report}
		if validationErr != nil {
			result.Message = "Existing runtime is invalid; no files were changed"
			return result, fmt.Errorf("%w: %s; use a new empty directory or explicitly quarantine it after stopping all users: %w", ErrUnsafeReplacement, dir, validationErr)
		}
		result.Message = "Runtime already verified; immutable installation retained"
		return result, nil
	} else if !errors.Is(err, os.ErrNotExist) {
		return nil, fmt.Errorf("inspect installation: %w", err)
	}

	stage, err := os.MkdirTemp(parent, ".ortgenai-install-")
	if err != nil {
		return nil, fmt.Errorf("create staging directory: %w", err)
	}
	defer os.RemoveAll(stage)
	manifest := newManifest(specs)
	for _, spec := range specs {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if cfg.PackageDir != "" {
			path := filepath.Join(cfg.PackageDir, packageFilename(spec.Package))
			if err := verifyPackage(ctx, path, spec.Package); err != nil {
				return nil, fmt.Errorf("offline package %s: %w", spec.Package.ID, err)
			}
			if err := extractPackage(ctx, path, stage, spec); err != nil {
				return nil, fmt.Errorf("extract %s: %w", spec.Package.ID, err)
			}
		} else {
			if err := acquirePackage(ctx, client, urlFor(spec.Package), spec, stage); err != nil {
				return nil, fmt.Errorf("acquire %s: %w", spec.Package.ID, err)
			}
		}
	}
	if err := writeManifest(stage, manifest); err != nil {
		return nil, err
	}
	report, err := validate(ctx, stage, specs)
	if err != nil {
		return nil, fmt.Errorf("staging validation: %w", err)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	// Never remove a destination: an installer not respecting our lock may have
	// created it, and Windows DLLs may already be loaded from that directory.
	if _, err := os.Lstat(dir); err == nil {
		return nil, fmt.Errorf("%w: destination appeared during acquisition", ErrUnsafeReplacement)
	} else if !errors.Is(err, os.ErrNotExist) {
		return nil, err
	}
	if err := os.Rename(stage, dir); err != nil {
		return nil, fmt.Errorf("atomic promotion (destination was not removed): %w", err)
	}
	report.Dir = dir
	return &InstallResult{
		Dir: dir, Provider: ProviderQNN, Installed: true, Validated: true, Report: report,
		Message: "Pinned runtime files verified; QNN device and driver readiness not tested",
	}, nil
}
