package server

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"slices"
	"strings"

	"github.com/ollama/ollama/x/ortrunner"
)

type ortGenAIConfig struct {
	contextLength int
	filenames     []string
	providers     []string
	immutableDML  bool
}

func readORTGenAIConfig(filename string) (ortGenAIConfig, error) {
	var result ortGenAIConfig
	f, err := os.Open(filename)
	if err != nil {
		return result, err
	}
	defer f.Close()
	const maxConfigBytes = 4 << 20
	data, err := io.ReadAll(io.LimitReader(f, maxConfigBytes+1))
	if err != nil {
		return result, err
	}
	if len(data) > maxConfigBytes {
		return result, fmt.Errorf("genai_config.json exceeds %d bytes", maxConfigBytes)
	}
	var config struct {
		Model struct {
			Type          string          `json:"type"`
			ContextLength int64           `json:"context_length"`
			VocabSize     int64           `json:"vocab_size"`
			EOS           json.RawMessage `json:"eos_token_id"`
			Pad           int32           `json:"pad_token_id"`
			Decoder       struct {
				Filename string                       `json:"filename"`
				Session  json.RawMessage              `json:"session_options"`
				Pipeline []map[string]json.RawMessage `json:"pipeline"`
			} `json:"decoder"`
		} `json:"model"`
	}
	if err := json.Unmarshal(data, &config); err != nil {
		return result, fmt.Errorf("invalid genai_config.json: %w", err)
	}
	m := config.Model
	if m.Type == "" || m.ContextLength <= 0 || m.ContextLength > math.MaxInt32 || m.VocabSize <= 0 || m.VocabSize > math.MaxInt32 {
		return result, fmt.Errorf("model.type, positive int32 context_length and vocab_size are required")
	}
	result.contextLength = int(m.ContextLength)
	tokens := []int32{m.Pad}
	if len(m.EOS) > 0 {
		var eos []int32
		if err := json.Unmarshal(m.EOS, &eos); err != nil {
			var token int32
			if err := json.Unmarshal(m.EOS, &token); err != nil {
				return result, fmt.Errorf("invalid model.eos_token_id: %w", err)
			}
			eos = []int32{token}
		}
		if len(eos) == 0 {
			return result, fmt.Errorf("model.eos_token_id must not be empty")
		}
		tokens = append(tokens, eos...)
	}
	for _, token := range tokens {
		if token < 0 || int64(token) >= m.VocabSize {
			return result, fmt.Errorf("special token ID %d is outside model vocabulary", token)
		}
	}
	session := func(name string, raw json.RawMessage, primary bool) error {
		var options struct {
			Providers []map[string]map[string]json.RawMessage `json:"provider_options"`
		}
		if len(raw) == 0 {
			return nil
		}
		if err := json.Unmarshal(raw, &options); err != nil {
			return fmt.Errorf("invalid session options for %s: %w", name, err)
		}
		seen := make(map[string]bool)
		for _, group := range options.Providers {
			for provider, values := range group {
				provider = strings.ToLower(provider)
				if !slices.Contains([]string{"cpu", "dml", "qnn"}, provider) || values == nil || seen[provider] {
					return fmt.Errorf("invalid or duplicate provider %q in stage %q", provider, name)
				}
				for key, value := range values {
					if key == "device_filtering_options" {
						continue
					}
					if err := json.Unmarshal(value, new(string)); err != nil {
						return fmt.Errorf("provider option %s.%s must be a string", provider, key)
					}
				}
				seen[provider] = true
				result.immutableDML = result.immutableDML || !primary && provider == "dml"
				if !slices.Contains(result.providers, provider) {
					result.providers = append(result.providers, provider)
				}
			}
		}
		if len(seen) > 1 {
			return fmt.Errorf("multiple providers in stage %q are unsupported", name)
		}
		return nil
	}
	if err := session("decoder", m.Decoder.Session, true); err != nil {
		return result, err
	}
	if len(m.Decoder.Pipeline) == 0 && m.Decoder.Filename == "" {
		return result, fmt.Errorf("model.decoder.filename is required for a non-pipeline model")
	}
	stages := make(map[string]bool)
	for _, group := range m.Decoder.Pipeline {
		for name, raw := range group {
			if stages[name] {
				return result, fmt.Errorf("duplicate pipeline stage %q", name)
			}
			stages[name] = true
			var stage struct {
				Filename string          `json:"filename"`
				Session  json.RawMessage `json:"session_options"`
			}
			if err := json.Unmarshal(raw, &stage); err != nil {
				return result, err
			}
			if stage.Filename == "" {
				return result, fmt.Errorf("pipeline stage %q requires a filename", name)
			}
			if err := session(name, stage.Session, false); err != nil {
				return result, err
			}
		}
	}
	if len(m.Decoder.Pipeline) > 0 && len(stages) == 0 {
		return result, fmt.Errorf("decoder pipeline requires at least one stage")
	}
	result.filenames, err = ortrunner.ModelFileReferences(data)
	if err != nil {
		return result, err
	}
	for _, name := range result.filenames {
		if err := validateORTGenAIPath(name); err != nil {
			return result, err
		}
	}
	slices.Sort(result.providers)
	return result, nil
}
