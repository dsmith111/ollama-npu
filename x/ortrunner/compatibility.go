package ortrunner

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"runtime"
	"slices"
	"strings"
	"unicode"
	"unicode/utf8"

	ortruntime "github.com/ollama/ollama/x/ortrunner/runtime"
)

const CompatibilityFilename = "ollama_npu.json"
const maxCompatibilityBytes = 64 << 10

// CompatibilityProfile is a publisher declaration, not execution or licensing proof.
// Artifact identity belongs to the enclosing content-addressed model manifest.
type CompatibilityProfile struct {
	SchemaVersion  int                  `json:"schema_version"`
	Platform       string               `json:"platform"`
	Provider       string               `json:"provider"`
	GenAIVersion   string               `json:"genai_version"`
	ORTVersion     string               `json:"ort_version"`
	ProcessorNames []string             `json:"processor_names"`
	ContextLength  int                  `json:"context_length"`
	Precision      string               `json:"precision"`
	Capabilities   []string             `json:"capabilities"`
	Stages         []CompatibilityStage `json:"stages"`
	Source         *CompatibilitySource `json:"source,omitempty"`
}

type CompatibilityStage struct {
	Filename string `json:"filename"`
	Provider string `json:"provider"`
}

type CompatibilitySource struct {
	Repository string `json:"repository"`
	Revision   string `json:"revision"`
}

var compatibilityVersion = regexp.MustCompile(`^[0-9]+\.[0-9]+\.[0-9]+$`)
var compatibilityRevision = regexp.MustCompile(`^[a-f0-9]{40}([a-f0-9]{24})?$`)

func profileText(s string, limit int) bool {
	return s != "" && len(s) <= limit && strings.TrimSpace(s) == s &&
		utf8.ValidString(s) && !strings.ContainsFunc(s, unicode.IsControl)
}

func ParseCompatibilityProfile(data []byte) (*CompatibilityProfile, error) {
	if len(data) > maxCompatibilityBytes {
		return nil, fmt.Errorf("%s exceeds %d bytes", CompatibilityFilename, maxCompatibilityBytes)
	}
	if !utf8.Valid(data) || len(bytes.TrimSpace(data)) == 0 || bytes.TrimSpace(data)[0] != '{' {
		return nil, fmt.Errorf("%s must be a UTF-8 JSON object", CompatibilityFilename)
	}
	if err := validateJSON(json.NewDecoder(bytes.NewReader(data)), 0); err != nil {
		return nil, fmt.Errorf("invalid %s: %w", CompatibilityFilename, err)
	}
	var p CompatibilityProfile
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&p); err != nil {
		return nil, fmt.Errorf("invalid %s: %w", CompatibilityFilename, err)
	}
	if err := decoder.Decode(new(any)); err != io.EOF {
		return nil, fmt.Errorf("%s contains trailing data", CompatibilityFilename)
	}
	if p.SchemaVersion != 1 || p.Platform != "windows/arm64" || p.Provider != "qnn" {
		return nil, fmt.Errorf("%s requires schema_version=1, platform=windows/arm64 and provider=qnn; other profiles are not supported", CompatibilityFilename)
	}
	if !compatibilityVersion.MatchString(p.GenAIVersion) || !compatibilityVersion.MatchString(p.ORTVersion) {
		return nil, fmt.Errorf("%s requires exact genai_version and ort_version (major.minor.patch)", CompatibilityFilename)
	}
	if p.ContextLength <= 0 || p.ContextLength > math.MaxInt32 || !profileText(p.Precision, 64) ||
		!slices.Equal(p.Capabilities, []string{"completion"}) {
		return nil, fmt.Errorf("%s requires positive int32 context_length, declared precision and text-only capabilities=[completion]", CompatibilityFilename)
	}
	if len(p.ProcessorNames) == 0 || len(p.ProcessorNames) > 32 {
		return nil, fmt.Errorf("%s requires 1..32 exact Windows processor_names", CompatibilityFilename)
	}
	seen := make(map[string]bool)
	for _, name := range p.ProcessorNames {
		key := strings.ToLower(name)
		if !profileText(name, 256) || seen[key] {
			return nil, fmt.Errorf("%s has an invalid or duplicate processor name", CompatibilityFilename)
		}
		seen[key] = true
	}
	if len(p.Stages) == 0 || len(p.Stages) > 64 {
		return nil, fmt.Errorf("%s requires 1..64 graph stages", CompatibilityFilename)
	}
	seen = make(map[string]bool)
	hasQNN := false
	for _, stage := range p.Stages {
		key := strings.ToLower(stage.Filename)
		if !profileText(stage.Filename, 512) || strings.ContainsAny(stage.Filename, `\:`) ||
			path.IsAbs(stage.Filename) || path.Clean(stage.Filename) != stage.Filename ||
			stage.Filename == ".." || strings.HasPrefix(stage.Filename, "../") ||
			!strings.HasSuffix(stage.Filename, ".onnx") || seen[key] ||
			(stage.Provider != "cpu" && stage.Provider != "qnn") {
			return nil, fmt.Errorf("%s has an invalid or duplicate graph stage %q", CompatibilityFilename, stage.Filename)
		}
		seen[key] = true
		hasQNN = hasQNN || stage.Provider == "qnn"
	}
	if !hasQNN {
		return nil, fmt.Errorf("%s requires a QNN graph stage", CompatibilityFilename)
	}
	if p.Source != nil {
		u, err := url.Parse(p.Source.Repository)
		if err != nil || !profileText(p.Source.Repository, 1024) || u.Scheme != "https" ||
			u.Host == "" || u.User != nil || u.Path == "" || u.Path == "/" ||
			u.RawQuery != "" || u.ForceQuery || u.Fragment != "" ||
			!compatibilityRevision.MatchString(p.Source.Revision) {
			return nil, fmt.Errorf("%s source requires a credential-free HTTPS repository and a full lowercase Git revision", CompatibilityFilename)
		}
	}
	return &p, nil
}

func ReadCompatibilityProfileFile(filename string) (*CompatibilityProfile, error) {
	info, err := os.Lstat(filename)
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() || info.Size() > maxCompatibilityBytes {
		return nil, fmt.Errorf("%s must be a regular file of at most %d bytes", CompatibilityFilename, maxCompatibilityBytes)
	}
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	data, err := io.ReadAll(io.LimitReader(f, maxCompatibilityBytes+1))
	if err != nil {
		return nil, err
	}
	return ParseCompatibilityProfile(data)
}

// Absence preserves legacy bundles; malformed declarations are never ignored.
func ReadCompatibilityProfile(dir string) (*CompatibilityProfile, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	for _, entry := range entries {
		if strings.EqualFold(entry.Name(), CompatibilityFilename) && entry.Name() != CompatibilityFilename {
			return nil, fmt.Errorf("compatibility profile filename must be exactly %s, not %q", CompatibilityFilename, entry.Name())
		}
	}
	p, err := ReadCompatibilityProfileFile(filepath.Join(dir, CompatibilityFilename))
	if os.IsNotExist(err) {
		return nil, nil
	}
	return p, err
}

func (p *CompatibilityProfile) ValidateModelConfig(filename string) error {
	meta, err := readModelMetadataFile(filename)
	if err != nil {
		return err
	}
	return p.validateMetadata(meta)
}

func (p *CompatibilityProfile) validateMetadata(meta modelMetadata) error {
	pipeline := len(meta.stages) > 1
	if pipeline != (meta.modelType == "decoder-pipeline") || pipeline && meta.stages[0].filename != "" {
		return fmt.Errorf("%s requires model.type=decoder-pipeline with pipeline-only graphs, or a decoder-only model type with just decoder.filename", CompatibilityFilename)
	}
	if p.ContextLength != meta.contextLength {
		return fmt.Errorf("%s context_length=%d does not match genai_config.json=%d", CompatibilityFilename, p.ContextLength, meta.contextLength)
	}
	actual := make(map[string]string)
	for _, stage := range meta.stages {
		if stage.filename == "" {
			continue
		}
		providers := stage.providers
		if !stage.primary && !stage.sessionDeclared {
			providers = meta.stages[0].providers
		}
		provider := "cpu"
		for name, options := range providers {
			provider = name
			if name != "qnn" {
				continue
			}
			for key, allowed := range map[string]string{"backend_type": "htp", "backend_path": "QnnHtp.dll"} {
				if raw, ok := options[key]; ok {
					var value string
					if err := json.Unmarshal(raw, &value); err != nil || value != allowed {
						return fmt.Errorf("%s stage %q requires the verified HTP backend, not %s=%q", CompatibilityFilename, stage.filename, key, value)
					}
				}
			}
		}
		if _, ok := actual[stage.filename]; ok {
			return fmt.Errorf("%s cannot describe duplicate graph filename %q", CompatibilityFilename, stage.filename)
		}
		actual[stage.filename] = provider
	}
	if len(actual) != len(p.Stages) || len(meta.filenames) != len(actual) {
		return fmt.Errorf("%s stages must describe every genai_config.json graph exactly once", CompatibilityFilename)
	}
	for _, name := range meta.filenames {
		if _, ok := actual[name]; !ok {
			return fmt.Errorf("%s schema 1 only supports decoder graphs; unaccounted graph %q", CompatibilityFilename, name)
		}
	}
	for _, stage := range p.Stages {
		if actual[stage.Filename] != stage.Provider {
			return fmt.Errorf("%s stage %q provider=%q does not match genai_config.json", CompatibilityFilename, stage.Filename, stage.Provider)
		}
	}
	return nil
}

func (p *CompatibilityProfile) ValidateHost(goos, arch, provider, processor string) error {
	if p.Platform != goos+"/"+arch || p.Provider != provider {
		return fmt.Errorf("NPU_MODEL_INCOMPATIBLE: model requires %s/%s, selected %s/%s; acquire a compatible model", p.Platform, p.Provider, goos+"/"+arch, provider)
	}
	if p.GenAIVersion != ortruntime.PinnedGenAIVersion || p.ORTVersion != ortruntime.PinnedORTVersion {
		return fmt.Errorf("NPU_RUNTIME_INCOMPATIBLE: model requires GenAI QNN %s / ORT QNN %s, this fork supports %s / %s; use a compatible model/fork, do not mix DLLs",
			p.GenAIVersion, p.ORTVersion, ortruntime.PinnedGenAIVersion, ortruntime.PinnedORTVersion)
	}
	if !slices.ContainsFunc(p.ProcessorNames, func(name string) bool { return strings.EqualFold(name, strings.TrimSpace(processor)) }) {
		return fmt.Errorf("NPU_TARGET_INCOMPATIBLE: processor %q is not in the model's declared targets; acquire a model for this device", processor)
	}
	return nil
}

func (p *CompatibilityProfile) CheckHost(provider string) error {
	if runtime.GOOS != "windows" || runtime.GOARCH != "arm64" {
		return p.ValidateHost(runtime.GOOS, runtime.GOARCH, provider, "")
	}
	processor, err := compatibilityProcessorName()
	if err != nil {
		return fmt.Errorf("NPU_TARGET_UNKNOWN: cannot identify the Windows processor: %w", err)
	}
	return p.ValidateHost(runtime.GOOS, runtime.GOARCH, provider, processor)
}

func checkProfileEnvironment(env func(string) string) error {
	for _, key := range []string{"OLLAMA_ORT_QNN_BACKEND_TYPE", "OLLAMA_ORT_QNN_BACKEND_PATH",
		"OLLAMA_ORT_QNN_HTP_ARCH", "OLLAMA_ORT_QNN_SOC_MODEL"} {
		if env(key) != "" {
			return fmt.Errorf("NPU_MODEL_INCOMPATIBLE: %s cannot override a declared compatibility profile; use the intact model config", key)
		}
	}
	return nil
}

// CheckCompatibility does not load native code or assert accelerator execution.
func CheckCompatibility(modelDir, provider string) (*CompatibilityProfile, error) {
	p, err := ReadCompatibilityProfile(modelDir)
	if err != nil || p == nil {
		return p, err
	}
	if err := p.ValidateModelConfig(filepath.Join(modelDir, "genai_config.json")); err != nil {
		return nil, err
	}
	if err := p.CheckHost(provider); err != nil {
		return nil, err
	}
	if err := checkProfileEnvironment(os.Getenv); err != nil {
		return nil, err
	}
	return p, nil
}
