package ortrunner

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"
)

const maxConfigBytes = 4 << 20

type modelMetadata struct {
	contextLength         int
	eosTokens             []int32
	stages                []modelStage
	filenames             []string
	modelType             string
	vocabSize             int
	promptPaddingMultiple int
}

type modelStage struct {
	name                string
	filename            string
	primary             bool
	providers           map[string]map[string]json.RawMessage
	sessionDeclared     bool
	cpuFallbackDisabled bool
}

type providerPlan struct {
	name       string
	state      string
	appendName string
	options    map[string]string
	preserve   bool
	hybrid     bool
	strictQNN  bool
}

func readModelMetadata(dir string) (modelMetadata, error) {
	return readModelMetadataFile(filepath.Join(dir, "genai_config.json"))
}

func readModelMetadataFile(filename string) (modelMetadata, error) {
	f, err := os.Open(filename)
	if err != nil {
		return modelMetadata{}, err
	}
	defer f.Close()
	data, err := io.ReadAll(io.LimitReader(f, maxConfigBytes+1))
	if err != nil {
		return modelMetadata{}, err
	}
	if len(data) > maxConfigBytes {
		return modelMetadata{}, fmt.Errorf("genai_config.json exceeds %d bytes", maxConfigBytes)
	}
	return parseModelMetadata(data)
}

func parseModelMetadata(data []byte) (modelMetadata, error) {
	if !utf8.Valid(data) || len(bytes.TrimSpace(data)) == 0 || bytes.TrimSpace(data)[0] != '{' {
		return modelMetadata{}, fmt.Errorf("genai_config.json must be a UTF-8 JSON object")
	}
	if err := validateJSON(json.NewDecoder(bytes.NewReader(data)), 0); err != nil {
		return modelMetadata{}, fmt.Errorf("invalid genai_config.json: %w", err)
	}
	var config struct {
		Model struct {
			ContextLength int             `json:"context_length"`
			VocabSize     int             `json:"vocab_size"`
			Type          string          `json:"type"`
			EOS           json.RawMessage `json:"eos_token_id"`
			Pad           int32           `json:"pad_token_id"`
			Decoder       json.RawMessage `json:"decoder"`
		} `json:"model"`
	}
	var result modelMetadata
	if err := json.Unmarshal(data, &config); err != nil {
		return result, fmt.Errorf("invalid genai_config.json: %w", err)
	}
	if config.Model.ContextLength <= 0 || config.Model.ContextLength > math.MaxInt32 {
		return result, fmt.Errorf("model.context_length must be a positive int32")
	}
	if config.Model.VocabSize <= 0 || config.Model.VocabSize > math.MaxInt32 || config.Model.Type == "" {
		return result, fmt.Errorf("model.type and positive int32 model.vocab_size are required")
	}
	result.contextLength = config.Model.ContextLength
	result.modelType = config.Model.Type
	result.vocabSize = config.Model.VocabSize
	if len(config.Model.EOS) == 0 {
		result.eosTokens = []int32{config.Model.Pad}
	} else if err := json.Unmarshal(config.Model.EOS, &result.eosTokens); err != nil {
		var eos int32
		if err := json.Unmarshal(config.Model.EOS, &eos); err != nil {
			return result, fmt.Errorf("invalid model.eos_token_id: %w", err)
		}
		result.eosTokens = []int32{eos}
	}
	if len(result.eosTokens) == 0 {
		return result, fmt.Errorf("model.eos_token_id must not be empty")
	}
	for _, token := range append(append([]int32(nil), result.eosTokens...), config.Model.Pad) {
		if token < 0 || int(token) >= config.Model.VocabSize {
			return result, fmt.Errorf("special token ID %d is outside model vocabulary", token)
		}
	}
	var decoder struct {
		Filename      string                       `json:"filename"`
		Session       json.RawMessage              `json:"session_options"`
		Pipeline      []map[string]json.RawMessage `json:"pipeline"`
		SlidingWindow *struct {
			WindowSize  int    `json:"window_size"`
			SlideInputs *bool  `json:"slide_inputs"`
			Alignment   string `json:"alignment"`
		} `json:"sliding_window"`
	}
	if err := json.Unmarshal(config.Model.Decoder, &decoder); err != nil {
		return result, fmt.Errorf("invalid model.decoder: %w", err)
	}
	if window := decoder.SlidingWindow; window != nil {
		if window.WindowSize <= 0 || window.WindowSize > result.contextLength {
			return result, fmt.Errorf("sliding_window.window_size must be between 1 and model context length")
		}
		if window.Alignment != "" && window.Alignment != "left" && window.Alignment != "right" {
			return result, fmt.Errorf("sliding_window.alignment must be left or right")
		}
		if optionValue(window.SlideInputs, true) {
			result.promptPaddingMultiple = window.WindowSize
		}
	}
	top, err := parseModelStage("decoder", decoder.Filename, decoder.Session, true)
	if err != nil {
		return result, err
	}
	result.stages = append(result.stages, top)
	if len(decoder.Pipeline) == 0 {
		if decoder.Filename == "" {
			return result, fmt.Errorf("model.decoder.filename is required for a non-pipeline model")
		}
	} else {
		seen := make(map[string]bool)
		for _, group := range decoder.Pipeline {
			if len(group) == 0 {
				return result, fmt.Errorf("pipeline entries must not be empty")
			}
			// Deterministic validation; native OGA retains the config's original stage order.
			names := make([]string, 0, len(group))
			for name := range group {
				names = append(names, name)
			}
			sort.Strings(names)
			for _, name := range names {
				if seen[name] {
					return result, fmt.Errorf("duplicate pipeline stage %q", name)
				}
				seen[name] = true
				var stage struct {
					Filename string          `json:"filename"`
					Session  json.RawMessage `json:"session_options"`
				}
				if err := json.Unmarshal(group[name], &stage); err != nil {
					return result, fmt.Errorf("invalid pipeline stage %q: %w", name, err)
				}
				if stage.Filename == "" {
					return result, fmt.Errorf("pipeline stage %q requires a filename", name)
				}
				s, err := parseModelStage(name, stage.Filename, stage.Session, false)
				if err != nil {
					return result, err
				}
				result.stages = append(result.stages, s)
			}
		}
	}
	result.filenames, err = ModelFileReferences(data)
	if err != nil {
		return result, err
	}
	return result, nil
}

// ModelFileReferences returns every nonempty filename under the model config.
// Callers apply their own artifact path and graph/provider validation.
func ModelFileReferences(data []byte) ([]string, error) {
	var document struct {
		Model map[string]any `json:"model"`
	}
	if err := json.Unmarshal(data, &document); err != nil {
		return nil, err
	}
	var filenames []string
	var visit func(any) error
	visit = func(value any) error {
		switch value := value.(type) {
		case map[string]any:
			for key, child := range value {
				if key == "filename" {
					name, ok := child.(string)
					if !ok {
						return fmt.Errorf("model filename references must be strings")
					}
					if name != "" {
						filenames = append(filenames, name)
					}
				} else if err := visit(child); err != nil {
					return err
				}
			}
		case []any:
			for _, child := range value {
				if err := visit(child); err != nil {
					return err
				}
			}
		}
		return nil
	}
	if err := visit(document.Model); err != nil {
		return nil, err
	}
	sort.Strings(filenames)
	return filenames, nil
}

func parseModelStage(name, filename string, session json.RawMessage, primary bool) (modelStage, error) {
	s := modelStage{name: name, filename: filename, primary: primary, providers: make(map[string]map[string]json.RawMessage)}
	if len(session) == 0 {
		return s, nil
	}
	s.sessionDeclared = true
	var options struct {
		Providers  []map[string]map[string]json.RawMessage `json:"provider_options"`
		DisableCPU string                                  `json:"session.disable_cpu_ep_fallback"`
	}
	if err := json.Unmarshal(session, &options); err != nil {
		return s, fmt.Errorf("invalid %s session options: %w", name, err)
	}
	if options.DisableCPU != "" && options.DisableCPU != "0" && options.DisableCPU != "1" {
		return s, fmt.Errorf("invalid session.disable_cpu_ep_fallback in stage %s", name)
	}
	s.cpuFallbackDisabled = options.DisableCPU == "1"
	for _, group := range options.Providers {
		for provider, values := range group {
			provider = strings.ToLower(provider)
			switch provider {
			case "qnn", "dml", "cpu":
			default:
				return s, fmt.Errorf("unsupported provider %q in stage %q", provider, name)
			}
			if values == nil || s.providers[provider] != nil {
				return s, fmt.Errorf("invalid or duplicate provider %q in stage %q", provider, name)
			}
			for key, raw := range values {
				if key == "device_filtering_options" {
					var device struct {
						Type   string  `json:"hardware_device_type"`
						ID     *uint32 `json:"hardware_device_id"`
						Vendor *uint32 `json:"hardware_vendor_id"`
					}
					decoder := json.NewDecoder(bytes.NewReader(raw))
					decoder.DisallowUnknownFields()
					if err := decoder.Decode(&device); err != nil {
						return s, fmt.Errorf("invalid %s device_filtering_options: %w", provider, err)
					}
					if device.Type != "" && !strings.EqualFold(device.Type, "cpu") && !strings.EqualFold(device.Type, "gpu") && !strings.EqualFold(device.Type, "npu") {
						return s, fmt.Errorf("unsupported hardware_device_type %q", device.Type)
					}
					continue
				}
				var value string
				if err := json.Unmarshal(raw, &value); err != nil {
					return s, fmt.Errorf("provider option %s.%s must be a string", provider, key)
				}
			}
			s.providers[provider] = values
		}
	}
	if len(s.providers) > 1 {
		return s, fmt.Errorf("multiple providers in stage %q are not supported", name)
	}
	return s, nil
}

func modelFile(dir, filename string) (string, error) {
	if filename == "" || strings.ContainsRune(filename, 0) || filepath.IsAbs(filename) || filepath.VolumeName(filename) != "" {
		return "", fmt.Errorf("model filename must be relative to its model directory: %q", filename)
	}
	clean := filepath.Clean(filename)
	if clean == "." || clean == ".." || strings.HasPrefix(clean, ".."+string(filepath.Separator)) || strings.Contains(clean, ":") {
		return "", fmt.Errorf("model filename escapes its directory: %q", filename)
	}
	path := filepath.Join(dir, clean)
	root, err := filepath.EvalSymlinks(dir)
	if err != nil {
		return "", err
	}
	resolved, err := filepath.EvalSymlinks(path)
	if err != nil {
		return "", err
	}
	rel, err := filepath.Rel(root, resolved)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return "", fmt.Errorf("model artifact resolves outside its directory: %s", path)
	}
	info, err := os.Stat(path)
	if err != nil {
		return "", err
	}
	if !info.Mode().IsRegular() || info.Size() == 0 {
		return "", fmt.Errorf("model artifact is not a nonempty regular file: %s", path)
	}
	return path, nil
}

func planProvider(meta modelMetadata, dir string, env func(string) string) (providerPlan, error) {
	p := providerPlan{options: make(map[string]string)}
	if len(meta.stages) == 0 {
		return p, fmt.Errorf("model decoder stages are required")
	}
	hasQNN := false
	hasDML := false
	hasCompiledQNN := false
	stages := append([]modelStage(nil), meta.stages...)
	for i := range stages {
		s := &stages[i]
		if !s.primary && !s.sessionDeclared {
			s.providers = stages[0].providers
			s.cpuFallbackDisabled = stages[0].cpuFallbackDisabled
		}
		hasQNN = hasQNN || s.providers["qnn"] != nil
		hasDML = hasDML || s.providers["dml"] != nil
		if s.filename != "" {
			path, err := modelFile(dir, s.filename)
			if err != nil {
				return p, fmt.Errorf("stage %s: %w", s.name, err)
			}
			compiled, err := inspectQNNContext(path, filepath.Dir(path))
			if err != nil {
				return p, fmt.Errorf("stage %s: %w", s.name, err)
			}
			if compiled && s.providers["qnn"] == nil {
				return p, fmt.Errorf("stage %s contains QNN EPContext nodes but does not declare the QNN provider; CPU/DML fallback is unsupported", s.name)
			}
			if s.providers["qnn"] != nil && !compiled {
				return p, fmt.Errorf("stage %s requires a compiled QNN EPContext graph, not a portable ONNX graph", s.name)
			}
			hasCompiledQNN = hasCompiledQNN || compiled
		}
	}
	p.name = strings.ToLower(env("OLLAMA_ONNX_PROVIDER"))
	if p.name == "" {
		if hasQNN || env("OLLAMA_ONNX_NPU") == "1" {
			p.name = "qnn"
		} else {
			// A portable graph remains portable even when QNN DLLs happen to be installed.
			p.name = "cpu"
		}
	} else if env("OLLAMA_ONNX_NPU") == "1" && p.name != "qnn" {
		return p, fmt.Errorf("OLLAMA_ONNX_NPU conflicts with OLLAMA_ONNX_PROVIDER")
	}
	if p.name != "cpu" && p.name != "dml" && p.name != "qnn" {
		return p, fmt.Errorf("unsupported execution provider %q", p.name)
	}
	if hasQNN && p.name != "qnn" {
		return p, fmt.Errorf("QNN-compiled models require provider=qnn; CPU/DML fallback is not supported")
	}
	if p.name == "qnn" && !hasCompiledQNN {
		return p, fmt.Errorf("provider=qnn requires a compiled QNN model with QNN provider options and EPContext graphs; portable ONNX graphs are not supported")
	}
	deviceType := strings.ToLower(env("OLLAMA_ORT_DEVICE_TYPE"))
	deviceID := env("OLLAMA_ORT_DEVICE_ID")
	if deviceID != "" {
		id, err := strconv.ParseUint(deviceID, 10, 31)
		if err != nil {
			return p, fmt.Errorf("invalid OLLAMA_ORT_DEVICE_ID: %w", err)
		}
		if p.name != "dml" {
			return p, fmt.Errorf("explicit device IDs are supported only for DML")
		}
		p.options["device_index"] = strconv.FormatUint(id, 10)
	}
	if deviceType != "" && !(p.name == "qnn" && deviceType == "npu") {
		return p, fmt.Errorf("device_type=%q is not supported by the pinned %s provider API; for DML use device_id", deviceType, p.name)
	}
	qnnEnv := map[string]string{
		"backend_type": env("OLLAMA_ORT_QNN_BACKEND_TYPE"),
		"backend_path": env("OLLAMA_ORT_QNN_BACKEND_PATH"),
		"htp_arch":     env("OLLAMA_ORT_QNN_HTP_ARCH"),
		"soc_model":    env("OLLAMA_ORT_QNN_SOC_MODEL"),
	}
	if qnnEnv["backend_type"] != "" && qnnEnv["backend_path"] != "" {
		return p, fmt.Errorf("QNN backend_type and backend_path are mutually exclusive")
	}
	for key, value := range qnnEnv {
		if value == "" {
			continue
		}
		if p.name != "qnn" {
			return p, fmt.Errorf("QNN option %s cannot be applied to provider=%s", key, p.name)
		}
		if strings.ContainsAny(value, "\"\x00\r\n") {
			return p, fmt.Errorf("invalid QNN option %s", key)
		}
		if key == "backend_type" && strings.ToLower(value) != "htp" {
			return p, fmt.Errorf("compiled QNN models require the HTP backend")
		}
		if key == "htp_arch" || key == "soc_model" {
			if _, err := strconv.ParseUint(value, 10, 31); err != nil {
				return p, fmt.Errorf("QNN %s must be a nonnegative integer", key)
			}
		}
		p.options[key] = value
	}
	switch p.name {
	case "cpu":
		p.state = "portable-cpu"
		// Immutable non-primary providers cannot be removed with ClearProviders.
		if hasDML {
			for _, s := range meta.stages {
				if !s.primary && s.providers["dml"] != nil {
					return p, fmt.Errorf("CPU selection cannot override the immutable DML provider in pipeline stage %s", s.name)
				}
			}
		}
	case "dml":
		p.appendName = "DML"
		p.state = "portable-dml"
		for _, s := range meta.stages {
			if s.primary || !s.sessionDeclared {
				continue
			}
			if s.providers["dml"] == nil {
				p.hybrid = true
				continue
			}
			if requested := p.options["device_index"]; requested != "" {
				var configured string
				_ = json.Unmarshal(s.providers["dml"]["device_index"], &configured)
				if configured != requested {
					return p, fmt.Errorf("cannot override immutable DML device_index in pipeline stage %s; update genai_config.json", s.name)
				}
			}
		}
		if p.hybrid {
			p.state = "portable-dml-with-cpu-stages"
		}
	case "qnn":
		p.state = "qnn-compiled"
		p.preserve = true
		p.strictQNN = true
		for _, s := range stages {
			if s.providers["dml"] != nil {
				return p, fmt.Errorf("mixed QNN/DML pipeline stage %s is not supported", s.name)
			}
			options := s.providers["qnn"]
			if options == nil {
				if s.filename != "" {
					p.hybrid = true
					p.strictQNN = false
				}
				continue
			}
			if s.filename != "" && !s.cpuFallbackDisabled {
				p.strictQNN = false
			}
			read := func(key string) string {
				var value string
				_ = json.Unmarshal(options[key], &value)
				return value
			}
			if read("backend_type") != "" && read("backend_path") != "" {
				return p, fmt.Errorf("stage %s specifies both QNN backend_type and backend_path", s.name)
			}
			if value := read("backend_type"); value != "" && !strings.EqualFold(value, "htp") {
				return p, fmt.Errorf("stage %s does not select the QNN HTP backend", s.name)
			}
			if value := read("backend_path"); strings.Contains(strings.ToLower(filepath.Base(value)), "qnncpu") || strings.Contains(strings.ToLower(filepath.Base(value)), "qnngpu") {
				return p, fmt.Errorf("stage %s does not select the QNN HTP backend", s.name)
			}
			for key, value := range p.options {
				if !s.primary && read(key) != value {
					return p, fmt.Errorf("cannot override immutable QNN option %s in pipeline stage %s; update genai_config.json", key, s.name)
				}
				if key == "backend_type" && read("backend_path") != "" || key == "backend_path" && read("backend_type") != "" {
					return p, fmt.Errorf("QNN backend override conflicts with stage %s config", s.name)
				}
			}
		}
		if p.hybrid {
			p.state = "qnn-compiled-with-cpu-stages"
		}
		if meta.stages[0].providers["qnn"] != nil {
			p.appendName = "QNN"
			top := meta.stages[0].providers["qnn"]
			if top["backend_type"] == nil && top["backend_path"] == nil && p.options["backend_type"] == "" && p.options["backend_path"] == "" {
				p.options["backend_type"] = "htp"
			}
		} else {
			// The compiled pipeline's baked providers are intentionally preserved.
			p.options = nil
		}
	}
	return p, nil
}
