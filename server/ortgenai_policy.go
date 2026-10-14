package server

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/x/ortrunner"
)

// Select one provider before scheduling. A child load failure is terminal,
// never a reason to silently route an NPU request to native GGUF or CPU.
func resolveORTOptions(m *Model, opts api.Options) (*ortrunner.ClientOptions, error) {
	mode := strings.ToLower(strings.TrimSpace(envconfig.AccelMode()))
	npu := os.Getenv("OLLAMA_ONNX_NPU") == "1"
	modelDir := os.Getenv("OLLAMA_ONNX_MODEL")
	if modelDir == "" && m.IsORTGenAI() {
		modelDir = m.ModelPath
	}
	if modelDir == "" {
		if m.IsORTGenAI() {
			return nil, fmt.Errorf("ORT GenAI model has no materialized model directory")
		}
		if mode == "npu" || npu {
			return nil, fmt.Errorf("NPU acceleration requires an ORT GenAI model; GGUF models use the native CPU/GPU runner")
		}
		return nil, nil
	}
	switch mode {
	case "", "auto", "off", "cpu", "gpu", "npu":
	default:
		return nil, fmt.Errorf("unsupported OLLAMA_ACCEL_MODE %q", mode)
	}
	config, err := readORTGenAIConfig(filepath.Join(modelDir, "genai_config.json"))
	if err != nil {
		return nil, fmt.Errorf("read ORT GenAI model config: %w", err)
	}
	requested := strings.ToLower(strings.TrimSpace(os.Getenv("OLLAMA_ONNX_PROVIDER")))
	if requested == "auto" {
		requested = ""
	}
	if requested != "" && !slices.Contains([]string{"cpu", "dml", "qnn"}, requested) {
		return nil, fmt.Errorf("unsupported OLLAMA_ONNX_PROVIDER %q", requested)
	}
	if npu {
		if mode == "off" || mode == "cpu" || mode == "gpu" || requested != "" && requested != "qnn" {
			return nil, fmt.Errorf("OLLAMA_ONNX_NPU conflicts with provider/acceleration selection")
		}
		mode = "npu"
		requested = "qnn"
	}
	deviceType := strings.ToLower(strings.TrimSpace(os.Getenv("OLLAMA_ORT_DEVICE_TYPE")))
	if deviceType != "" && deviceType != "npu" {
		return nil, fmt.Errorf("device_type=%q is unsupported by the pinned provider API; for DML use OLLAMA_ORT_DEVICE_ID", deviceType)
	}
	if opts.NumGPU == 0 || mode == "cpu" || mode == "off" {
		if deviceType != "" || mode == "npu" || mode == "gpu" || requested != "" && requested != "cpu" {
			return nil, fmt.Errorf("CPU-only request conflicts with ORT provider/device selection")
		}
		requested = "cpu"
	}
	hasQNN := slices.Contains(config.providers, "qnn")
	provider := requested
	if provider == "" {
		if hasQNN {
			provider = "qnn"
		} else if mode == "gpu" {
			provider = "dml"
		} else {
			// Match the child's portable-CPU default, independent of installed DLLs
			// or a replaceable primary DML provider in the model config.
			provider = "cpu"
		}
	}
	if hasQNN && provider != "qnn" {
		return nil, fmt.Errorf("QNN-compiled models require provider=qnn; CPU/DML fallback is not supported")
	}
	if provider == "qnn" && !hasQNN {
		return nil, fmt.Errorf("provider=qnn requires a compiled model with QNN provider options and EPContext graphs")
	}
	if m.Config.ModelFormat == "genai-qnn" && provider != "qnn" {
		return nil, fmt.Errorf("genai-qnn model requires the qnn provider")
	}
	if provider == "cpu" && config.immutableDML {
		return nil, fmt.Errorf("CPU selection cannot override an immutable DML pipeline stage")
	}
	if (mode == "npu" || deviceType == "npu") && provider != "qnn" {
		return nil, fmt.Errorf("NPU selection requires compiled QNN; the pinned DML API cannot guarantee NPU device selection")
	}
	if mode == "gpu" && provider != "dml" {
		return nil, fmt.Errorf("GPU selection requires provider=dml")
	}
	if provider == "qnn" {
		deviceType = "npu"
	}
	deviceID := strings.TrimSpace(os.Getenv("OLLAMA_ORT_DEVICE_ID"))
	if deviceID != "" {
		id, err := strconv.ParseUint(deviceID, 10, 31)
		if err != nil || provider != "dml" {
			return nil, fmt.Errorf("explicit nonnegative ORT device IDs are supported only for DML")
		}
		deviceID = strconv.FormatUint(id, 10)
	}
	if envconfig.NPUDeviceID() != "" && provider == "qnn" {
		return nil, fmt.Errorf("OLLAMA_NPU_DEVICE_ID is a DXCore adapter index and cannot select a QNN device")
	}
	return &ortrunner.ClientOptions{
		ModelDir: modelDir, Provider: provider, DeviceType: deviceType, DeviceID: deviceID,
	}, nil
}
