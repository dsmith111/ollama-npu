package ortrunner

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	ortruntime "github.com/ollama/ollama/x/ortrunner/runtime"
)

func compatibilityFixture() CompatibilityProfile {
	return CompatibilityProfile{
		SchemaVersion: 1, Platform: "windows/arm64", Provider: "qnn",
		GenAIVersion: ortruntime.PinnedGenAIVersion, ORTVersion: ortruntime.PinnedORTVersion,
		ProcessorNames: []string{"Fixture Snapdragon"}, ContextLength: 4096,
		Precision: "int4 decoder; fp32 CPU stages", Capabilities: []string{"completion"},
		Stages: []CompatibilityStage{{"model.onnx", "qnn"}},
	}
}

func profileJSON(t *testing.T, p CompatibilityProfile) []byte {
	t.Helper()
	data, err := json.Marshal(p)
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func TestCompatibilityProfileValidation(t *testing.T) {
	base := string(profileJSON(t, compatibilityFixture()))
	for name, data := range map[string]string{
		"unsupported schema": strings.Replace(base, `"schema_version":1`, `"schema_version":2`, 1),
		"unknown field":      strings.Replace(base, `"schema_version":1`, `"schema_version":1,"qualified":true`, 1),
		"duplicate":          strings.Replace(base, `"schema_version":1`, `"schema_version":1,"SCHEMA_VERSION":1`, 1),
		"trailing":           base + "{}",
		"null":               "null",
		"null source":        strings.Replace(base, `"schema_version":1`, `"schema_version":1,"source":null`, 1),
		"oversize":           strings.Repeat(" ", maxCompatibilityBytes) + base,
		"UTF8":               base + "\xff",
		"other backend":      strings.Replace(base, `"qnn"`, `"hexagon"`, 1),
		"version range":      strings.Replace(base, `"genai_version":"0.12.2"`, `"genai_version":">=0.12.2"`, 1),
		"unknown target":     strings.Replace(base, `["Fixture Snapdragon"]`, `[]`, 1),
		"duplicate target":   strings.Replace(base, `["Fixture Snapdragon"]`, `["Fixture Snapdragon","fixture snapdragon"]`, 1),
		"empty precision":    strings.Replace(base, `"int4 decoder; fp32 CPU stages"`, `""`, 1),
		"false capability":   strings.Replace(base, `["completion"]`, `["completion","tools"]`, 1),
		"escaping graph":     strings.Replace(base, `"model.onnx"`, `"../model.onnx"`, 1),
		"backslash graph":    strings.Replace(base, `"model.onnx"`, `"dir\\model.onnx"`, 1),
		"cpu only":           strings.Replace(base, `"provider":"qnn"}`, `"provider":"cpu"}`, 1),
	} {
		t.Run(name, func(t *testing.T) {
			if p, err := ParseCompatibilityProfile([]byte(data)); err == nil {
				t.Fatalf("accepted invalid profile: %+v", p)
			}
		})
	}
	p := compatibilityFixture()
	p.Source = &CompatibilitySource{"https://huggingface.co/microsoft/Phi-3-mini-4k-instruct", strings.Repeat("a", 40)}
	if _, err := ParseCompatibilityProfile(profileJSON(t, p)); err != nil {
		t.Fatal(err)
	}
	for _, source := range []CompatibilitySource{
		{"https://user:secret@example.com/repo", strings.Repeat("a", 40)},
		{"http://example.com/repo", strings.Repeat("a", 40)},
		{"https://example.com/repo?token=secret", strings.Repeat("a", 40)},
		{"https://example.com/repo", "main"},
	} {
		p.Source = &source
		if _, err := ParseCompatibilityProfile(profileJSON(t, p)); err == nil {
			t.Fatal("accepted mutable/credentialed source identity")
		}
	}
}

func TestCompatibilityHostAndOverrides(t *testing.T) {
	p := compatibilityFixture()
	if err := p.ValidateHost("windows", "arm64", "qnn", " fixture snapdragon "); err != nil {
		t.Fatal(err)
	}
	for _, host := range [][4]string{
		{"linux", "arm64", "qnn", "Fixture Snapdragon"},
		{"windows", "amd64", "qnn", "Fixture Snapdragon"},
		{"windows", "arm64", "cpu", "Fixture Snapdragon"},
		{"windows", "arm64", "qnn", "Different Snapdragon"},
		{"windows", "arm64", "qnn", ""},
	} {
		if err := p.ValidateHost(host[0], host[1], host[2], host[3]); err == nil {
			t.Fatalf("accepted incompatible host %v", host)
		}
	}
	p.GenAIVersion = "0.12.3"
	if err := p.ValidateHost("windows", "arm64", "qnn", "Fixture Snapdragon"); err == nil {
		t.Fatal("accepted newer unqualified runtime")
	}
	p = compatibilityFixture()
	p.ORTVersion = "1.24.2"
	if err := p.ValidateHost("windows", "arm64", "qnn", "Fixture Snapdragon"); err == nil {
		t.Fatal("accepted mismatched ORT tuple")
	}
	for _, key := range []string{"OLLAMA_ORT_QNN_BACKEND_TYPE", "OLLAMA_ORT_QNN_BACKEND_PATH", "OLLAMA_ORT_QNN_HTP_ARCH", "OLLAMA_ORT_QNN_SOC_MODEL"} {
		if err := checkProfileEnvironment(envMap(map[string]string{key: "override"})); err == nil {
			t.Fatalf("accepted %s override", key)
		}
	}
}

func TestCompatibilityConfigAndLegacy(t *testing.T) {
	dir := t.TempDir()
	p := compatibilityFixture()
	if profile, err := ReadCompatibilityProfile(dir); profile != nil || err != nil {
		t.Fatalf("legacy absence = %+v, %v", profile, err)
	}
	config := testModelJSON(`{"filename":"model.onnx","session_options":{"provider_options":[{"QNN":{"backend_type":"htp"}}]}}`, "4096")
	writeModelArtifact(t, dir, "genai_config.json", config)
	writeModelArtifact(t, dir, CompatibilityFilename, profileJSON(t, p))
	if err := p.ValidateModelConfig(filepath.Join(dir, "genai_config.json")); err != nil {
		t.Fatal(err)
	}
	for _, bad := range [][]byte{
		testModelJSON(`{"filename":"model.onnx"}`, "4096"),
		testModelJSON(`{"filename":"other.onnx","session_options":{"provider_options":[{"qnn":{}}]}}`, "4096"),
		testModelJSON(`{"filename":"model.onnx","session_options":{"provider_options":[{"qnn":{}}]}}`, "512"),
		testModelJSON(`{"filename":"model.onnx","session_options":{"provider_options":[{"qnn":{"backend_path":"C:/private/QnnHtp.dll"}}]}}`, "4096"),
		testModelJSON(`{"filename":"model.onnx","session_options":{"provider_options":[{"qnn":{"backend_type":"cpu"}}]}}`, "4096"),
		[]byte(strings.Replace(string(config), `"decoder":`, `"encoder":{"filename":"encoder.onnx"},"decoder":`, 1)),
	} {
		writeModelArtifact(t, dir, "genai_config.json", bad)
		if err := p.ValidateModelConfig(filepath.Join(dir, "genai_config.json")); err == nil {
			t.Fatal("accepted contradictory graph/config profile")
		}
	}
	p.Stages = append(p.Stages, CompatibilityStage{"cpu.onnx", "cpu"})
	decoder := `{"session_options":{"provider_options":[{"qnn":{}}]},"pipeline":[{"compiled":{"filename":"model.onnx"}},{"embedding":{"filename":"cpu.onnx","session_options":{"provider_options":[]}}}]}`
	pipelineConfig := strings.Replace(string(testModelJSON(decoder, "4096")), `"type":"phi3"`, `"type":"decoder-pipeline"`, 1)
	writeModelArtifact(t, dir, "genai_config.json", []byte(pipelineConfig))
	if err := p.ValidateModelConfig(filepath.Join(dir, "genai_config.json")); err != nil {
		t.Fatalf("effective inherited/explicit CPU stage providers: %v", err)
	}
	writeModelArtifact(t, dir, CompatibilityFilename, []byte(`{}`))
	if _, err := ReadCompatibilityProfile(dir); err == nil {
		t.Fatal("malformed optional profile was silently ignored")
	}
	if err := os.Remove(filepath.Join(dir, CompatibilityFilename)); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(filepath.Join(dir, CompatibilityFilename), 0o755); err != nil {
		t.Fatal(err)
	}
	if _, err := ReadCompatibilityProfile(dir); err == nil {
		t.Fatal("profile directory accepted as absent")
	}
}

func TestCompatibilityRejectsBeforeNativeInitialization(t *testing.T) {
	for _, key := range []string{"OLLAMA_ONNX_PROVIDER", "OLLAMA_ONNX_NPU", "OLLAMA_ORT_DEVICE_TYPE", "OLLAMA_ORT_DEVICE_ID",
		"OLLAMA_ORT_QNN_BACKEND_TYPE", "OLLAMA_ORT_QNN_BACKEND_PATH", "OLLAMA_ORT_QNN_HTP_ARCH", "OLLAMA_ORT_QNN_SOC_MODEL"} {
		t.Setenv(key, "")
	}

	dir := t.TempDir()
	p := compatibilityFixture()
	p.GenAIVersion = "99.0.0"
	writeModelArtifact(t, dir, CompatibilityFilename, profileJSON(t, p))
	writeModelArtifact(t, dir, "genai_config.json", testModelJSON(`{"filename":"model.onnx","session_options":{"provider_options":[{"qnn":{}}]}}`, "4096"))
	writeModelArtifact(t, dir, "model.onnx", contextGraph(""))
	if err := new(Runner).Load(dir); err == nil || (!strings.Contains(err.Error(), "NPU_RUNTIME_INCOMPATIBLE") && !strings.Contains(err.Error(), "NPU_MODEL_INCOMPATIBLE")) {
		t.Fatalf("load must reject compatibility before consulting/loading native runtime: %v", err)
	}
}

func TestCompatibilityProfileCaseAliases(t *testing.T) {
	dir := t.TempDir()
	writeModelArtifact(t, dir, strings.ToUpper(CompatibilityFilename), profileJSON(t, compatibilityFixture()))
	if _, err := ReadCompatibilityProfile(dir); err == nil {
		t.Fatal("case-aliased declaration bypassed canonical-profile enforcement")
	}
}

func TestCompatibilityActualGraphLayout(t *testing.T) {
	p := compatibilityFixture()
	p.Stages = append(p.Stages, CompatibilityStage{"cpu.onnx", "cpu"})
	// GenAI's phi3 constructor ignores pipeline and executes only decoder.filename.
	decoder := `{"filename":"cpu.onnx","pipeline":[{"inactive":{"filename":"model.onnx","session_options":{"provider_options":[{"qnn":{}}]}}}]}`
	base := string(testModelJSON(decoder, "4096"))
	for _, config := range []string{base, strings.Replace(base, `"type":"phi3"`, `"type":"decoder-pipeline"`, 1)} {
		meta, err := parseModelMetadata([]byte(config))
		if err != nil {
			t.Fatal(err)
		}
		if err := p.validateMetadata(meta); err == nil || !strings.Contains(err.Error(), "model.type") {
			t.Fatalf("declared stages must match actual native model-type graph selection: %v", err)
		}
	}
}
