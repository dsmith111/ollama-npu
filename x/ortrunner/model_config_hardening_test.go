package ortrunner

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

func TestActualEPContextCannotBeReclassifiedAsPortable(t *testing.T) {
	dir := testModelDir(t)
	writeModelArtifact(t, dir, "model.onnx", contextGraph(""))
	meta, err := parseModelMetadata(testModelJSON(`{"filename":"model.onnx"}`, "4096"))
	if err != nil {
		t.Fatal(err)
	}
	for _, provider := range []string{"cpu", "dml", "qnn", ""} {
		if _, err := planProvider(meta, dir, envMap(map[string]string{"OLLAMA_ONNX_PROVIDER": provider})); err == nil {
			t.Fatalf("accepted undeclared compiled context with provider=%q", provider)
		}
	}
}

func TestQNNPrimaryPipelineProviderInheritanceAndCPUHead(t *testing.T) {
	dir := testModelDir(t)
	writeModelArtifact(t, dir, "head.onnx", portableGraph())
	writeModelArtifact(t, dir, "context.onnx", contextGraph(""))
	meta, err := parseModelMetadata(testModelJSON(`{"session_options":{"provider_options":[{"qnn":{"backend_type":"htp"}}]},"pipeline":[{"context":{"filename":"context.onnx"}},{"head":{"filename":"head.onnx","session_options":{"provider_options":[]}}}]}`, "4096"))
	if err != nil {
		t.Fatal(err)
	}
	p, err := planProvider(meta, dir, envMap(nil))
	if err != nil || !p.preserve || !p.hybrid || p.strictQNN || p.appendName != "QNN" {
		t.Fatalf("pipeline=%+v error=%v", p, err)
	}
}

func TestQNNStrictPolicyIsNotInferredFromDLLsOrProviderName(t *testing.T) {
	dir := testModelDir(t)
	writeModelArtifact(t, dir, "model.onnx", contextGraph(""))
	for _, disable := range []string{"0", "1", "garbage"} {
		meta, err := parseModelMetadata(testModelJSON(`{"filename":"model.onnx","session_options":{"session.disable_cpu_ep_fallback":"`+disable+`","provider_options":[{"qnn":{"backend_type":"htp"}}]}}`, "4096"))
		if disable == "garbage" {
			if err == nil {
				t.Fatal("accepted malformed CPU fallback policy")
			}
			continue
		}
		if err != nil {
			t.Fatal(err)
		}
		p, err := planProvider(meta, dir, envMap(nil))
		if err != nil || p.strictQNN != (disable == "1") || p.hybrid {
			t.Fatalf("plan=%+v err=%v", p, err)
		}
	}
}

func TestONNXInspectionRejectsMalformedAndWrongSource(t *testing.T) {
	dir := testModelDir(t)
	path := filepath.Join(dir, "model.onnx")
	cases := [][]byte{
		{},
		[]byte("not protobuf"),
		protoBytes(7, nil),
		append(protoBytes(7, nil), protoBytes(7, nil)...),
		{0x3a, 0x80},
		bytes.Replace(contextGraph(""), []byte("Qnn"), []byte("Dml"), 1),
		bytes.Replace(contextGraph(""), []byte("EPContext"), []byte("NotAContext"), 1),
	}
	for _, data := range cases {
		if err := os.WriteFile(path, data, 0600); err != nil {
			t.Fatal(err)
		}
		if _, err := inspectQNNContext(path, dir); err == nil {
			t.Fatalf("accepted malformed ONNX graph %x", data)
		}
	}
	writeModelArtifact(t, dir, "model.onnx", portableGraph())
	if compiled, err := inspectQNNContext(path, dir); err != nil || compiled {
		t.Fatalf("valid tiny portable graph: compiled=%v err=%v", compiled, err)
	}
}
