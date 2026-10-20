package ortrunner

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"unicode/utf8"

	"github.com/ollama/ollama/x/ortrunner/oga"
	ortruntime "github.com/ollama/ollama/x/ortrunner/runtime"
)

// Runner manages ORT GenAI model loading and inference.
type Runner struct {
	nativeMu              sync.Mutex
	stateMu               sync.RWMutex
	model                 *oga.Model
	tokenizer             *oga.Tokenizer
	config                *oga.Config
	modelDir              string
	contextLength         int
	eosTokens             []int32
	modelType             string
	vocabSize             int
	promptPaddingMultiple int
	chatTemplate          string // chat_template.jinja; empty uses the tokenizer config's
	status                StatusResponse
}

// modelMemory sizes a model for status reporting: its ONNX graphs, external data
// and compiled context binaries. With the QNN provider, everything but the
// CPU stages' graphs and data is the NPU's share: the compiled contexts and the
// stages that wrap them.
func modelMemory(dir string, meta modelMetadata, provider string) (total, npu uint64) {
	size := func(name string) uint64 {
		info, err := os.Lstat(filepath.Join(dir, filepath.FromSlash(name)))
		if err != nil || !info.Mode().IsRegular() {
			return 0
		}
		return uint64(info.Size())
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return 0, 0
	}
	for _, entry := range entries {
		name := strings.ToLower(entry.Name())
		if strings.HasSuffix(name, ".onnx") || strings.HasSuffix(name, ".bin") || strings.Contains(name, ".onnx.data") {
			total += size(entry.Name())
		}
	}
	if provider != "qnn" {
		return total, 0
	}
	var cpu uint64
	for _, stage := range meta.stages {
		if stage.filename != "" && stage.providers["qnn"] == nil {
			cpu += size(stage.filename) + size(stage.filename+".data")
		}
	}
	if cpu > total {
		return total, 0
	}
	return total, total - cpu
}

// maxChatTemplateBytes bounds chat_template.jinja (the largest known templates
// are tens of kilobytes).
const maxChatTemplateBytes = 1 << 20

// readChatTemplate returns the model's chat_template.jinja, or "" without one.
func readChatTemplate(modelDir string) (string, error) {
	path := filepath.Join(modelDir, "chat_template.jinja")
	info, err := os.Lstat(path)
	if os.IsNotExist(err) {
		return "", nil
	}
	if err != nil {
		return "", err
	}
	if !info.Mode().IsRegular() || info.Size() > maxChatTemplateBytes {
		return "", fmt.Errorf("chat_template.jinja must be a regular file of at most %d bytes", maxChatTemplateBytes)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	if !utf8.Valid(data) || strings.ContainsRune(string(data), 0) {
		return "", fmt.Errorf("chat_template.jinja must be UTF-8 without NUL")
	}
	return string(data), nil
}

// Load loads the ORT GenAI model from a directory.
func (r *Runner) Load(modelDir string) error {
	r.nativeMu.Lock()
	defer r.nativeMu.Unlock()
	if r.model != nil {
		return fmt.Errorf("a model is already loaded")
	}
	meta, err := readModelMetadata(modelDir)
	if err != nil {
		return err
	}
	chatTemplate, err := readChatTemplate(modelDir)
	if err != nil {
		return err
	}
	profile, err := ReadCompatibilityProfile(modelDir)
	if err != nil {
		return err
	}
	if profile != nil {
		if err := profile.validateMetadata(meta); err != nil {
			return err
		}
		if err := checkProfileEnvironment(os.Getenv); err != nil {
			return err
		}
	}
	plan, err := planProvider(meta, modelDir, os.Getenv)
	if err != nil {
		return err
	}
	if profile != nil {
		if err := profile.CheckHost(plan.name); err != nil {
			return err
		}
	}
	nativeDir, err := oga.NativeModelDirectory(modelDir, plan.name)
	if err != nil {
		return err
	}
	if profile != nil {
		if err := oga.CheckInitVerified(); err != nil {
			return err
		}
	} else {
		slog.Warn("ORT model has no compatibility profile; target/runtime compatibility is unverified", "profile", CompatibilityFilename)
		if err := oga.CheckInit(); err != nil {
			return err
		}
	}
	cfg, model, tok, err := loadConfiguredModel(
		func() (*oga.Config, error) { return oga.NewConfig(nativeDir) },
		func(cfg *oga.Config) error {
			if !plan.preserve {
				if err := cfg.ClearProviders(); err != nil {
					return fmt.Errorf("clear execution providers: %w", err)
				}
			}
			if plan.appendName == "" {
				return nil
			}
			if err := cfg.AppendProvider(plan.appendName); err != nil {
				return fmt.Errorf("append %s provider: %w", plan.name, err)
			}
			keys := make([]string, 0, len(plan.options))
			for key := range plan.options {
				keys = append(keys, key)
			}
			sort.Strings(keys)
			for _, key := range keys {
				if err := cfg.SetProviderOption(plan.appendName, key, plan.options[key]); err != nil {
					return fmt.Errorf("set %s option %s: %w", plan.name, key, err)
				}
			}
			return nil
		},
		oga.NewModel,
		oga.NewTokenizer,
	)
	if err != nil {
		if profile != nil {
			return fmt.Errorf("NPU_PROFILE_LOAD_FAILED: declared compatibility is not execution proof; check the publisher's compiled target and OEM driver, preserve the intact model config and verified runtime: %w", err)
		}
		return classifyLoadError(err, plan.name)
	}
	r.model, r.tokenizer, r.config = model, tok, cfg
	r.chatTemplate = chatTemplate
	r.modelDir, r.contextLength, r.eosTokens = modelDir, meta.contextLength, meta.eosTokens
	r.modelType = meta.modelType
	r.vocabSize = meta.vocabSize
	r.promptPaddingMultiple = meta.promptPaddingMultiple
	r.stateMu.Lock()
	memory, npuMemory := modelMemory(modelDir, meta, plan.name)
	r.status = StatusResponse{
		ProtocolVersion: ProtocolVersion,
		Status:          0, Progress: 100, ContextLength: meta.contextLength,
		Memory: memory, NPUMemory: npuMemory,
		PromptPaddingMultiple: meta.promptPaddingMultiple,
		Provider:              plan.name, ProviderState: plan.state,
		NativeCancelAvailable: oga.NativeCancellationAvailable(),
		CancellationMode:      "between-native-calls", CancellationGraceMS: int(cancellationGrace.Milliseconds()),
		WatchdogRequired:   true,
		ProviderConfigured: true, ProviderLoaded: true,
		Hybrid: plan.hybrid, StrictQNN: plan.strictQNN,
	}
	r.stateMu.Unlock()
	slog.Info("ORT GenAI model loaded", "provider", plan.name, "state", plan.state, "context_length", meta.contextLength)
	return nil
}

// classifyLoadError examines an ORT GenAI model load error and returns a more
// actionable error message keyed to known failure signatures.
func classifyLoadError(err error, provider string) error {
	msg := err.Error()

	switch {
	case strings.Contains(msg, "DML provider requested") && strings.Contains(msg, "not been built with DML support"):
		return fmt.Errorf("NPU_PROVIDER_MISMATCH: provider=%s but your onnxruntime-genai.dll was not built with DML support. "+
			"Either install a DML-enabled GenAI build, or set OLLAMA_ONNX_PROVIDER=qnn for Snapdragon NPU: %w", provider, err)

	case strings.Contains(msg, "openSession") || strings.Contains(msg, "0x80000406") || strings.Contains(msg, "DspTransport"):
		missing := checkMissingQNNDLLs()
		hint := "QNN HTP transport failed to open a DSP session. "
		if missing != "" {
			hint += "Missing DLLs in OLLAMA_ORT_PATH: " + missing + ". "
		}
		hint += "Check: (1) all QNN runtime DLLs are present, (2) DSP RPC transport DLLs are available, " +
			"(3) try setting OLLAMA_ORT_QNN_HTP_ARCH and OLLAMA_ORT_QNN_SOC_MODEL explicitly"
		return fmt.Errorf("NPU_QNN_TRANSPORT_OPENSESSION_FAILED: %s: %w", hint, err)

	case strings.Contains(msg, "Unable to load lib"):
		missing := checkMissingQNNDLLs()
		hint := "QNN backend library load failed. "
		if missing != "" {
			hint += "Missing DLLs: " + missing + ". "
		}
		hint += "Ensure all QNN/HTP DLLs are in OLLAMA_ORT_PATH and that directory is first in PATH"
		return fmt.Errorf("NPU_QNN_LIBRARY_LOAD_FAILED: %s: %w", hint, err)

	case strings.Contains(msg, "requested API version") && strings.Contains(msg, "not available"):
		return fmt.Errorf("ORT_API_VERSION_MISMATCH: onnxruntime-genai.dll and onnxruntime.dll versions are incompatible. "+
			"Ensure both DLLs come from the same NuGet package and OLLAMA_ORT_PATH is first in PATH: %w", err)
	}

	return err
}

// checkMissingQNNDLLs scans the ORT directory for expected QNN runtime DLLs
// and returns a comma-separated list of missing ones.
func checkMissingQNNDLLs() string {
	searchDirs := ortSearchDirs()

	// Expected DLLs for a complete QNN HTP runtime
	expected := []string{
		"onnxruntime.dll",
		"onnxruntime-genai.dll",
		"onnxruntime_providers_shared.dll",
		"onnxruntime_providers_qnn.dll",
		"QnnHtp.dll",
		"QnnSystem.dll",
		"QnnHtpPrepare.dll",
	}

	var missing []string
	for _, dll := range expected {
		found := false
		for _, dir := range searchDirs {
			if _, err := os.Stat(filepath.Join(dir, dll)); err == nil {
				found = true
				break
			}
		}
		if !found {
			missing = append(missing, dll)
		}
	}

	return strings.Join(missing, ", ")
}

// ortSearchDirs returns the list of directories to scan for ORT/QNN DLLs.
func ortSearchDirs() []string {
	return ortruntime.SearchDirs()
}

// DefaultRuntimeInstallDir returns the standard per-user install directory
// for the ORT GenAI runtime bundle on Windows.
func DefaultRuntimeInstallDir() string {
	return ortruntime.DefaultInstallDir()
}

// ORTDirReport describes the contents and health of an ORT runtime directory.
type ORTDirReport struct {
	SearchDirs   []string
	FoundDLLs    []string
	MissingDLLs  []string
	HasQNN       bool
	HasDML       bool
	HasTransport bool // cdsprpc / DSP transport DLL found
	Provider     string
	BackendType  string
	BackendPath  string
	HTPArch      string
	SocModel     string
}

// ValidateORTDir inspects the ORT runtime directory and returns a diagnostic report.
func ValidateORTDir() ORTDirReport {
	dirs := ortSearchDirs()

	report := ORTDirReport{
		SearchDirs:  dirs,
		Provider:    os.Getenv("OLLAMA_ONNX_PROVIDER"),
		BackendType: os.Getenv("OLLAMA_ORT_QNN_BACKEND_TYPE"),
		BackendPath: os.Getenv("OLLAMA_ORT_QNN_BACKEND_PATH"),
		HTPArch:     os.Getenv("OLLAMA_ORT_QNN_HTP_ARCH"),
		SocModel:    os.Getenv("OLLAMA_ORT_QNN_SOC_MODEL"),
	}

	// All DLLs we check for
	allDLLs := []string{
		"onnxruntime.dll",
		"onnxruntime-genai.dll",
		"onnxruntime_providers_shared.dll",
		"onnxruntime_providers_qnn.dll",
		"QnnHtp.dll",
		"QnnSystem.dll",
		"QnnHtpPrepare.dll",
		"DirectML.dll",
	}

	for _, dll := range allDLLs {
		found := false
		for _, dir := range dirs {
			if _, err := os.Stat(filepath.Join(dir, dll)); err == nil {
				found = true
				report.FoundDLLs = append(report.FoundDLLs, dll)
				break
			}
		}
		if !found {
			report.MissingDLLs = append(report.MissingDLLs, dll)
		}
	}

	// Check for transport/RPC DLLs (glob for cdsprpc patterns)
	for _, dir := range dirs {
		matches, _ := filepath.Glob(filepath.Join(dir, "*cdsprpc*"))
		if len(matches) > 0 {
			report.HasTransport = true
			for _, m := range matches {
				report.FoundDLLs = append(report.FoundDLLs, filepath.Base(m))
			}
		}
		// Also check for QnnHtpV*Stub.dll
		stubs, _ := filepath.Glob(filepath.Join(dir, "QnnHtpV*Stub.dll"))
		for _, s := range stubs {
			report.FoundDLLs = append(report.FoundDLLs, filepath.Base(s))
		}
	}

	for _, dll := range report.FoundDLLs {
		switch {
		case dll == "onnxruntime_providers_qnn.dll":
			report.HasQNN = true
		case dll == "DirectML.dll":
			report.HasDML = true
		}
	}

	return report
}

// Close frees all ORT GenAI resources.
func (r *Runner) Close() {
	r.nativeMu.Lock()
	defer r.nativeMu.Unlock()
	if r.tokenizer != nil {
		r.tokenizer.Close()
		r.tokenizer = nil
	}
	if r.model != nil {
		r.model.Close()
		r.model = nil
	}
	if r.config != nil {
		r.config.Close()
		r.config = nil
	}
	r.stateMu.Lock()
	r.status = StatusResponse{ProtocolVersion: ProtocolVersion, Status: 1, ProviderState: "unloaded"}
	r.stateMu.Unlock()
}

func (r *Runner) Status() StatusResponse {
	r.stateMu.RLock()
	defer r.stateMu.RUnlock()
	return r.status
}

// IsStrictQNN describes configured CPU fallback policy, not hardware verification.
func (r *Runner) IsStrictQNN() bool { return r.Status().StrictQNN }

func (r *Runner) markOperationVerified() {
	r.stateMu.Lock()
	r.status.OperationVerified = true
	r.stateMu.Unlock()
}

func (r *Runner) Tokenize(ctx context.Context, text string) ([]int32, error) {
	r.nativeMu.Lock()
	defer r.nativeMu.Unlock()
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if r.tokenizer == nil {
		return nil, fmt.Errorf("model is not loaded")
	}
	tokens, err := r.tokenizer.Encode(text)
	if err == nil {
		for _, token := range tokens {
			if token < 0 {
				return nil, fmt.Errorf("tokenizer returned a negative token ID")
			}
		}
	}
	return tokens, err
}

// ApplyChatTemplate renders a JSON message list with the model's own chat
// template, ending with the assistant's generation prompt.
func (r *Runner) ApplyChatTemplate(ctx context.Context, messages string) (string, error) {
	r.nativeMu.Lock()
	defer r.nativeMu.Unlock()
	if err := ctx.Err(); err != nil {
		return "", err
	}
	if r.tokenizer == nil {
		return "", fmt.Errorf("model is not loaded")
	}
	return r.tokenizer.ApplyChatTemplate(r.chatTemplate, messages)
}

func (r *Runner) Detokenize(ctx context.Context, tokens []int32) (string, error) {
	r.nativeMu.Lock()
	defer r.nativeMu.Unlock()
	if err := ctx.Err(); err != nil {
		return "", err
	}
	if r.tokenizer == nil {
		return "", fmt.Errorf("model is not loaded")
	}
	for _, token := range tokens {
		if token < 0 || int(token) >= r.vocabSize {
			return "", ResponseError{StatusCode: 400, ErrorMessage: fmt.Sprintf("token ID %d is outside model vocabulary", token)}
		}
	}
	return r.tokenizer.Decode(tokens)
}
