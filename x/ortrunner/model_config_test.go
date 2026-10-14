package ortrunner

import (
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func testModelDir(t *testing.T) string {
	t.Helper()
	path := filepath.Join(".", fmt.Sprintf(".ort-test-%d", time.Now().UnixNano()))
	if err := os.Mkdir(path, 0700); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := os.RemoveAll(path); err != nil {
			t.Error(err)
		}
	})
	return path
}

func writeModelArtifact(t *testing.T, dir, name string, data []byte) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, name), data, 0600); err != nil {
		t.Fatal(err)
	}
}

func testModelJSON(decoder, context string) []byte {
	return []byte(`{"model":{"type":"phi3","vocab_size":100,"context_length":` + context +
		`,"pad_token_id":0,"eos_token_id":[99],"decoder":` + decoder + `}}`)
}

func TestModelMetadataValidation(t *testing.T) {
	valid := testModelJSON(`{"filename":"model.onnx","session_options":{"provider_options":[{"dml":{}}]}}`, "8192")
	meta, err := parseModelMetadata(valid)
	if err != nil || meta.contextLength != 8192 || len(meta.eosTokens) != 1 || meta.eosTokens[0] != 99 {
		t.Fatalf("metadata=%+v error=%v", meta, err)
	}
	for _, config := range []string{
		string(testModelJSON(`{"filename":"model.onnx"}`, "0")),
		string(testModelJSON(`{"filename":"model.onnx"}`, "-1")),
		string(testModelJSON(`{"filename":"model.onnx"}`, "1.5")),
		string(testModelJSON(`{"filename":"model.onnx"}`, `"4096"`)),
		string(testModelJSON(`{"filename":"model.onnx"}`, "2147483648")),
		string(testModelJSON(`{}`, "4096")),
		string(testModelJSON(`{"filename":"m","session_options":{"provider_options":[{"cuda":{}}]}}`, "4096")),
		string(testModelJSON(`{"filename":"m","session_options":{"provider_options":[{"qnn":{"soc_model":60}}]}}`, "4096")),
		string(testModelJSON(`{"filename":"m","session_options":{"provider_options":[{"qnn":{}},{"QNN":{}}]}}`, "4096")),
		string(testModelJSON(`{"filename":"m","session_options":{"provider_options":[{"qnn":{}},{"dml":{}}]}}`, "4096")),
		`{"model":{"context_length":4096}}`,
		strings.Replace(string(valid), `"eos_token_id":[99]`, `"eos_token_id":[]`, 1),
		strings.Replace(string(valid), `"eos_token_id":[99]`, `"eos_token_id":[100]`, 1),
		"not-json",
		strings.Replace(string(valid), `"context_length":8192`, `"context_length":1,"context_length":8192`, 1),
		strings.Replace(string(valid), `"decoder":`, `"decoder":null,"decoder":`, 1),
		string(testModelJSON(`{"filename":"m","session_options":null}`, "4096")),
		string(testModelJSON(`{"filename":"m","session_options":{"provider_options":[{"qnn":{"device_filtering_options":{"hardware_device_id":-1}}}]}}`, "4096")),
		string(testModelJSON(`{"filename":"m","session_options":{"provider_options":[{"qnn":{"device_filtering_options":{"hardware_device_type":"auto"}}}]}}`, "4096")),
		string(testModelJSON(`{"pipeline":[{}]}`, "4096")),
	} {
		if _, err := parseModelMetadata([]byte(config)); err == nil {
			t.Errorf("accepted invalid config %s", config)
		}
	}
}

func envMap(values map[string]string) func(string) string {
	return func(key string) string { return values[key] }
}

func TestPortableProviderCompatibility(t *testing.T) {
	dir := testModelDir(t)
	writeModelArtifact(t, dir, "model.onnx", portableGraph())
	meta, err := parseModelMetadata(testModelJSON(`{"filename":"model.onnx","session_options":{"provider_options":[{"dml":{}}]}}`, "4096"))
	if err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		env  map[string]string
		name string
		bad  bool
	}{
		{nil, "cpu", false},
		{map[string]string{"OLLAMA_ONNX_PROVIDER": "cpu"}, "cpu", false},
		{map[string]string{"OLLAMA_ONNX_PROVIDER": "DML", "OLLAMA_ORT_DEVICE_ID": "2"}, "dml", false},
		{map[string]string{"OLLAMA_ONNX_PROVIDER": "qnn"}, "", true},
		{map[string]string{"OLLAMA_ONNX_NPU": "1"}, "", true},
		{map[string]string{"OLLAMA_ONNX_PROVIDER": "cuda"}, "", true},
		{map[string]string{"OLLAMA_ONNX_PROVIDER": "dml", "OLLAMA_ORT_DEVICE_ID": "-1"}, "", true},
		{map[string]string{"OLLAMA_ONNX_PROVIDER": "dml", "OLLAMA_ORT_DEVICE_TYPE": "npu"}, "", true},
		{map[string]string{"OLLAMA_ONNX_PROVIDER": "dml", "OLLAMA_ORT_DEVICE_TYPE": "gpu"}, "", true},
		{map[string]string{"OLLAMA_ONNX_PROVIDER": "cpu", "OLLAMA_ORT_DEVICE_ID": "0"}, "", true},
		{map[string]string{"OLLAMA_ONNX_PROVIDER": "cpu", "OLLAMA_ORT_QNN_SOC_MODEL": "60"}, "", true},
		{map[string]string{"OLLAMA_ONNX_PROVIDER": "cpu", "OLLAMA_ONNX_NPU": "1"}, "", true},
	} {
		p, err := planProvider(meta, dir, envMap(test.env))
		if (err != nil) != test.bad {
			t.Errorf("env=%v error=%v expected invalid=%v", test.env, err, test.bad)
		}
		if err == nil && (p.name != test.name || test.name == "dml" && p.options["device_index"] != "2") {
			t.Errorf("plan=%+v", p)
		}
	}
}

func protoBytes(field int, data []byte) []byte {
	result := binary.AppendUvarint(nil, uint64(field<<3|2))
	result = binary.AppendUvarint(result, uint64(len(data)))
	return append(result, data...)
}

func protoNumber(field int, value uint64) []byte {
	return binary.AppendUvarint(binary.AppendUvarint(nil, uint64(field<<3)), value)
}

func portableGraph() []byte {
	// The pinned GenAI allocator's valid tiny Constant ONNX model.
	return []byte{
		0x08, 0x0a, 0x12, 0x01, 0x61, 0x3a, 0x53, 0x0a, 0x38, 0x12, 0x06, 0x76, 0x61, 0x6c, 0x75, 0x65,
		0x73, 0x22, 0x08, 0x43, 0x6f, 0x6e, 0x73, 0x74, 0x61, 0x6e, 0x74, 0x2a, 0x24, 0x0a, 0x05, 0x76,
		0x61, 0x6c, 0x75, 0x65, 0x2a, 0x18, 0x08, 0x01, 0x10, 0x01, 0x42, 0x0c, 0x63, 0x6f, 0x6e, 0x73,
		0x74, 0x5f, 0x74, 0x65, 0x6e, 0x73, 0x6f, 0x72, 0x4a, 0x04, 0x00, 0x00, 0x00, 0x00, 0xa0, 0x01,
		0x04, 0x12, 0x01, 0x62, 0x62, 0x14, 0x0a, 0x06, 0x76, 0x61, 0x6c, 0x75, 0x65, 0x73, 0x12, 0x0a,
		0x0a, 0x08, 0x08, 0x01, 0x12, 0x04, 0x0a, 0x02, 0x08, 0x01, 0x42, 0x04, 0x0a, 0x00, 0x10, 0x15,
	}
}

func contextGraph(external string) []byte {
	embed := uint64(1)
	cache := []byte("embedded compiled QNN context")
	if external != "" {
		embed, cache = 0, []byte(external)
	}
	attrs := protoBytes(1, []byte("embed_mode"))
	attrs = append(attrs, protoNumber(3, embed)...)
	cacheAttrs := protoBytes(1, []byte("ep_cache_context"))
	cacheAttrs = append(cacheAttrs, protoBytes(4, cache)...)
	node := protoBytes(4, []byte("EPContext"))
	node = append(node, protoBytes(7, []byte("com.microsoft"))...)
	node = append(node, protoBytes(5, attrs)...)
	node = append(node, protoBytes(5, cacheAttrs)...)
	source := append(protoBytes(1, []byte("source")), protoBytes(4, []byte("Qnn"))...)
	node = append(node, protoBytes(5, source)...)
	main := append(protoBytes(1, []byte("main_context")), protoNumber(3, 1)...)
	node = append(node, protoBytes(5, main)...)
	return protoBytes(7, protoBytes(1, node))
}

func TestQNNRequiresCompiledContextWithoutFallback(t *testing.T) {
	dir := testModelDir(t)
	writeModelArtifact(t, dir, "model.onnx", contextGraph(""))
	meta, err := parseModelMetadata(testModelJSON(`{"filename":"model.onnx","session_options":{"provider_options":[{"qnn":{"soc_model":"60"}}]}}`, "4096"))
	if err != nil {
		t.Fatal(err)
	}
	p, err := planProvider(meta, dir, envMap(nil))
	if err != nil || p.name != "qnn" || p.appendName != "QNN" || p.options["backend_type"] != "htp" {
		t.Fatalf("plan=%+v err=%v", p, err)
	}
	for _, env := range []map[string]string{
		{"OLLAMA_ONNX_PROVIDER": "cpu"},
		{"OLLAMA_ONNX_PROVIDER": "dml"},
		{"OLLAMA_ONNX_PROVIDER": "qnn", "OLLAMA_ORT_QNN_BACKEND_TYPE": "cpu"},
		{"OLLAMA_ONNX_PROVIDER": "qnn", "OLLAMA_ORT_QNN_BACKEND_TYPE": "htp", "OLLAMA_ORT_QNN_BACKEND_PATH": "QnnHtp.dll"},
		{"OLLAMA_ONNX_PROVIDER": "qnn", "OLLAMA_ORT_QNN_SOC_MODEL": "not-an-integer"},
	} {
		if _, err := planProvider(meta, dir, envMap(env)); err == nil {
			t.Errorf("accepted incompatible QNN configuration %v", env)
		}
	}
	writeModelArtifact(t, dir, "model.onnx", protoBytes(7, protoBytes(1, protoBytes(4, []byte("MatMul")))))
	if _, err := planProvider(meta, dir, envMap(nil)); err == nil {
		t.Fatal("accepted portable graph as compiled QNN model")
	}
}

func TestQNNPipelinePreservesCompiledProviders(t *testing.T) {
	dir := testModelDir(t)
	writeModelArtifact(t, dir, "embeddings.onnx", portableGraph())
	writeModelArtifact(t, dir, "context_ctx.onnx", contextGraph(""))
	meta, err := parseModelMetadata(testModelJSON(`{"session_options":{"provider_options":[]},"pipeline":[{"embeddings":{"filename":"embeddings.onnx"},"context_ctx":{"filename":"context_ctx.onnx","session_options":{"provider_options":[{"qnn":{"soc_model":"60"}}]}}}]}`, "4096"))
	if err != nil {
		t.Fatal(err)
	}
	p, err := planProvider(meta, dir, envMap(nil))
	if err != nil || p.appendName != "" || p.state != "qnn-compiled-with-cpu-stages" {
		t.Fatalf("pipeline plan=%+v error=%v", p, err)
	}
	if _, err := planProvider(meta, dir, envMap(map[string]string{"OLLAMA_ORT_QNN_SOC_MODEL": "73"})); err == nil {
		t.Fatal("silently accepted immutable pipeline option override")
	}
	if _, err := planProvider(meta, dir, envMap(map[string]string{"OLLAMA_ORT_QNN_SOC_MODEL": "60"})); err != nil {
		t.Fatal(err)
	}
}

func TestQNNExternalContextArtifacts(t *testing.T) {
	dir := testModelDir(t)
	path := filepath.Join(dir, "context.onnx")
	writeModelArtifact(t, dir, "context.onnx", contextGraph("compiled.bin"))
	if _, err := inspectQNNContext(path, dir); err == nil {
		t.Fatal("accepted missing compiled context")
	}
	writeModelArtifact(t, dir, "compiled.bin", []byte("compiled context"))
	if compiled, err := inspectQNNContext(path, dir); err != nil || !compiled {
		t.Fatalf("compiled=%v error=%v", compiled, err)
	}
	writeModelArtifact(t, dir, "context.onnx", contextGraph("..\\compiled.bin"))
	if _, err := inspectQNNContext(path, dir); err == nil {
		t.Fatal("accepted escaping context artifact")
	}
	writeModelArtifact(t, dir, "context.onnx", []byte{0x3a, 0x10, 0x0a})
	if _, err := inspectQNNContext(path, dir); err == nil {
		t.Fatal("accepted truncated protobuf")
	}
}

func TestModelConfigReadLimitsAndArtifacts(t *testing.T) {
	dir := testModelDir(t)
	if _, err := readModelMetadata(dir); err == nil {
		t.Fatal("accepted missing config")
	}
	writeModelArtifact(t, dir, "genai_config.json", []byte(strings.Repeat(" ", maxConfigBytes+1)))
	if _, err := readModelMetadata(dir); err == nil {
		t.Fatal("accepted oversized config")
	}
	for _, name := range []string{"", "..\\escape.onnx", "C:\\model.onnx", "missing.onnx"} {
		if _, err := modelFile(dir, name); err == nil {
			t.Errorf("accepted invalid model artifact %q", name)
		}
	}
}
