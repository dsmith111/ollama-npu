package server

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/types/model"
)

func TestORTProviderPolicy(t *testing.T) {
	for _, test := range []struct {
		name, provider, configured, mode, format, device, id, legacyNPU, npuID string
		numGPU                                                                 int
		wantProvider, wantDevice, wantID                                       string
		wantError                                                              bool
	}{
		{name: "portable defaults CPU", numGPU: -1, wantProvider: "cpu"},
		{name: "compiled QNN policy", configured: "QNN", numGPU: -1, wantProvider: "qnn", wantDevice: "npu"},
		{name: "primary DML is replaceable", configured: "dml", numGPU: -1, wantProvider: "cpu"},
		{name: "explicit DML", provider: "dml", numGPU: -1, wantProvider: "dml"},
		{name: "explicit CPU", configured: "dml", provider: "cpu", numGPU: -1, wantProvider: "cpu"},
		{name: "CPU request", configured: "dml", numGPU: 0, wantProvider: "cpu"},
		{name: "off portable", mode: "off", numGPU: -1, wantProvider: "cpu"},
		{name: "off compiled", mode: "off", configured: "qnn", numGPU: -1, wantError: true},
		{name: "NPU requires compiled provider", mode: "npu", numGPU: -1, wantError: true},
		{name: "explicit QNN portable rejected", provider: "qnn", numGPU: -1, wantError: true},
		{name: "explicit NPU QNN", configured: "QNN", provider: "qnn", mode: "npu", numGPU: -1, wantProvider: "qnn", wantDevice: "npu"},
		{name: "CPU QNN contradiction", configured: "qnn", provider: "cpu", numGPU: -1, wantError: true},
		{name: "CPU NPU contradiction", provider: "cpu", mode: "npu", numGPU: -1, wantError: true},
		{name: "CPU GPU contradiction", provider: "dml", numGPU: 0, wantError: true},
		{name: "GPU DML", provider: "dml", mode: "gpu", numGPU: -1, wantProvider: "dml"},
		{name: "GPU defaults DML", mode: "gpu", numGPU: -1, wantProvider: "dml"},
		{name: "DML NPU cannot be guaranteed", provider: "dml", mode: "npu", numGPU: -1, wantError: true},
		{name: "QNN bundle cannot CPU", format: "genai-qnn", provider: "cpu", numGPU: -1, wantError: true},
		{name: "invalid provider", provider: "bogus", numGPU: -1, wantError: true},
		{name: "DML device index", provider: "dml", id: "002", numGPU: -1, wantProvider: "dml", wantID: "2"},
		{name: "DML invalid device index", provider: "dml", id: "-1", numGPU: -1, wantError: true},
		{name: "DML device type unsupported", provider: "dml", device: "gpu", numGPU: -1, wantError: true},
		{name: "QNN device index unsupported", configured: "qnn", id: "0", numGPU: -1, wantError: true},
		{name: "DXCore index not QNN ID", configured: "qnn", npuID: "0", numGPU: -1, wantError: true},
		{name: "legacy NPU flag", configured: "qnn", legacyNPU: "1", numGPU: -1, wantProvider: "qnn", wantDevice: "npu"},
		{name: "legacy NPU flag conflicts", configured: "qnn", legacyNPU: "1", provider: "cpu", numGPU: -1, wantError: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Setenv("OLLAMA_ONNX_MODEL", "")
			t.Setenv("OLLAMA_ONNX_NPU", test.legacyNPU)
			t.Setenv("OLLAMA_ORT_DEVICE_TYPE", test.device)
			t.Setenv("OLLAMA_ORT_DEVICE_ID", test.id)
			t.Setenv("OLLAMA_NPU_DEVICE_ID", test.npuID)
			t.Setenv("OLLAMA_ACCEL_MODE", test.mode)
			t.Setenv("OLLAMA_ONNX_PROVIDER", test.provider)
			dir := t.TempDir()
			session := ""
			if test.configured != "" {
				session = `,"session_options":{"provider_options":[{"` + test.configured + `":{}}]}`
			}
			config := `{"model":{"type":"llama","context_length":128,"vocab_size":8,"decoder":{"filename":"model.onnx"` + session + `}}}`
			if err := os.WriteFile(filepath.Join(dir, "genai_config.json"), []byte(config), 0o644); err != nil {
				t.Fatal(err)
			}
			format := test.format
			if format == "" {
				format = "ortgenai"
			}
			opts, err := resolveORTOptions(&Model{ModelPath: dir, Config: model.ConfigV2{ModelFormat: format}}, api.Options{Runner: api.Runner{NumGPU: test.numGPU}})
			if test.wantError {
				if err == nil {
					t.Fatalf("expected error, got %+v", opts)
				}
			} else if err != nil || opts == nil || opts.Provider != test.wantProvider || opts.DeviceType != test.wantDevice || opts.DeviceID != test.wantID {
				t.Fatalf("policy = %+v, %v", opts, err)
			}
		})
	}
}

func TestORTPolicyLeavesOrdinaryGGUFNative(t *testing.T) {
	t.Setenv("OLLAMA_ONNX_MODEL", "")
	t.Setenv("OLLAMA_ONNX_NPU", "")
	t.Setenv("OLLAMA_ONNX_PROVIDER", "qnn")
	for _, mode := range []string{"", "auto", "off", "cpu", "gpu", "invalid"} {
		t.Setenv("OLLAMA_ACCEL_MODE", mode)
		opts, err := resolveORTOptions(&Model{ModelPath: "model.gguf"}, api.DefaultOptions())
		if err != nil || opts != nil {
			t.Fatalf("GGUF mode %q: %+v, %v", mode, opts, err)
		}
	}
	t.Setenv("OLLAMA_ACCEL_MODE", "npu")
	if _, err := resolveORTOptions(&Model{ModelPath: "model.gguf"}, api.DefaultOptions()); err == nil {
		t.Fatal("native GGUF must not masquerade as NPU inference")
	}
}

func TestORTGenAIConfigPipelineAndStrictValidation(t *testing.T) {
	dir := t.TempDir()
	filename := filepath.Join(dir, "genai_config.json")
	pipeline := `{"model":{"type":"llama","context_length":128,"vocab_size":8,"decoder":{"filename":"","pipeline":[{"cpu":{"filename":"cpu.onnx"}},{"compiled":{"filename":"nested/qnn.onnx","session_options":{"provider_options":[{"QNN":{"backend_type":"htp"}}]}}}]}}}`
	if err := os.WriteFile(filename, []byte(pipeline), 0o644); err != nil {
		t.Fatal(err)
	}
	config, err := readORTGenAIConfig(filename)
	if err != nil || !slices.Equal(config.providers, []string{"qnn"}) || !slices.Equal(config.filenames, []string{"cpu.onnx", "nested/qnn.onnx"}) {
		t.Fatalf("pipeline metadata = %+v, %v", config, err)
	}
	for _, invalid := range []string{
		pipeline + `{}`,
		strings.Replace(pipeline, `"context_length":128`, `"context_length":0`, 1),
		strings.Replace(pipeline, `"vocab_size":8`, `"vocab_size":2147483648`, 1),
		strings.Replace(pipeline, `"type":"llama"`, `"type":"llama","eos_token_id":[]`, 1),
		strings.Replace(pipeline, `"type":"llama"`, `"type":"llama","eos_token_id":8`, 1),
		strings.Replace(pipeline, `"backend_type":"htp"`, `"backend_type":123`, 1),
		strings.Replace(pipeline, `"QNN":`, `"QNNExecutionProvider":`, 1),
		strings.Replace(pipeline, `"backend_type":"htp"`, `"backend_type":"htp"}},{"qnn":{`, 1),
		strings.Repeat(" ", 4<<20) + pipeline,
	} {
		if err := os.WriteFile(filename, []byte(invalid), 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := readORTGenAIConfig(filename); err == nil {
			t.Fatal("accepted malformed/unsupported model configuration")
		}
	}
	dmlPipeline := strings.Replace(pipeline, `"QNN":{"backend_type":"htp"}`, `"DML":{}`, 1)
	if err := os.WriteFile(filename, []byte(dmlPipeline), 0o644); err != nil {
		t.Fatal(err)
	}
	for _, key := range []string{"OLLAMA_ACCEL_MODE", "OLLAMA_ONNX_PROVIDER", "OLLAMA_ONNX_NPU", "OLLAMA_ONNX_MODEL", "OLLAMA_ORT_DEVICE_TYPE", "OLLAMA_ORT_DEVICE_ID", "OLLAMA_NPU_DEVICE_ID"} {
		t.Setenv(key, "")
	}
	m := &Model{ModelPath: dir, Config: model.ConfigV2{ModelFormat: "ortgenai"}}
	if _, err := resolveORTOptions(m, api.DefaultOptions()); err == nil {
		t.Fatal("portable CPU default cannot remove an immutable pipeline DML provider")
	}
	t.Setenv("OLLAMA_ONNX_PROVIDER", "dml")
	if opts, err := resolveORTOptions(m, api.DefaultOptions()); err != nil || opts.Provider != "dml" {
		t.Fatalf("explicit pipeline DML = %+v, %v", opts, err)
	}
}
