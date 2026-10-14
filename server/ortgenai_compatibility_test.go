package server

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
	"github.com/ollama/ollama/x/ortrunner"
	ortruntime "github.com/ollama/ollama/x/ortrunner/runtime"
)

func ortProfileFiles(t *testing.T) (map[string]string, *ortrunner.CompatibilityProfile) {
	t.Helper()
	files := ortTestFiles(t)
	delete(files, "optional-empty.txt")
	p := &ortrunner.CompatibilityProfile{
		SchemaVersion: 1, Platform: "windows/arm64", Provider: "qnn",
		GenAIVersion: ortruntime.PinnedGenAIVersion, ORTVersion: ortruntime.PinnedORTVersion,
		ProcessorNames: []string{"Synthetic fixture target"}, ContextLength: 128,
		Precision: "synthetic; not inference-qualified", Capabilities: []string{"completion"},
		Stages: []ortrunner.CompatibilityStage{{Filename: "model.onnx", Provider: "qnn"}},
	}
	data, err := json.Marshal(p)
	if err != nil {
		t.Fatal(err)
	}
	for name, content := range map[string]string{
		ortrunner.CompatibilityFilename: string(data),
		"genai_config.json": `{"model":{"type":"phi3","context_length":128,"vocab_size":8,"eos_token_id":2,
			"decoder":{"filename":"model.onnx","session_options":{"provider_options":[{"QNN":{"backend_type":"htp"}}]}}}}`,
	} {
		layer, err := manifest.NewLayer(strings.NewReader(content), MediaTypeORTGenAI)
		if err != nil {
			t.Fatal(err)
		}
		files[name] = layer.Digest
	}
	return files, p
}

func TestORTCompatibilityCreateCopyAndShow(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	files, p := ortProfileFiles(t)
	progress := func(api.ProgressResponse) {}
	layers, err := convertFromORTGenAI(t.Context(), files, progress)
	if err != nil {
		t.Fatal(err)
	}
	name := model.ParseName("profiled:qnn")
	if err := createModel(t.Context(), api.CreateRequest{Template: "{{ .Prompt }}", Parameters: map[string]any{"num_ctx": 128}}, name, layers, new(model.ConfigV2), progress); err != nil {
		t.Fatal(err)
	}
	derivedLayers, config, err := parseFromModel(t.Context(), name, progress)
	if err != nil {
		t.Fatal(err)
	}
	derived := model.ParseName("profile-derived:qnn")
	if err := createModel(t.Context(), api.CreateRequest{System: "derived"}, derived, derivedLayers, &config, progress); err != nil {
		t.Fatal(err)
	}
	copied := model.ParseName("profile-copy:qnn")
	if _, err := CopyModel(derived, copied); err != nil {
		t.Fatal(err)
	}
	for _, tag := range []model.Name{name, derived, copied} {
		m, err := GetModel(tag.String())
		if err != nil || !reflect.DeepEqual(m.ORTCompatibility, p) {
			t.Fatalf("lost profile for %s: model=%+v error=%v", tag.String(), m, err)
		}
		info, err := GetModelInfo(api.ShowRequest{Model: tag.String()})
		if err != nil || info.ModelInfo["npu.compatibility_status"] != "declared" ||
			!reflect.DeepEqual(info.ModelInfo["npu.compatibility"], p) {
			t.Fatalf("Show declared profile: %+v, %v", info, err)
		}
		data, err := os.ReadFile(filepath.Join(m.ModelPath, ortrunner.CompatibilityFilename))
		if err != nil || fmt.Sprintf("sha256:%x", sha256.Sum256(data)) != files[ortrunner.CompatibilityFilename] {
			t.Fatalf("profile bytes/digest changed: %v", err)
		}
	}
	legacyLayers, err := convertFromORTGenAI(t.Context(), ortTestFiles(t), progress)
	if err != nil {
		t.Fatal(err)
	}
	legacy := model.ParseName("legacy:qnn")
	if err := createModel(t.Context(), api.CreateRequest{}, legacy, legacyLayers, new(model.ConfigV2), progress); err != nil {
		t.Fatal(err)
	}
	info, err := GetModelInfo(api.ShowRequest{Model: legacy.String()})
	if err != nil || info.ModelInfo["npu.compatibility_status"] != "legacy-unverified" {
		t.Fatalf("legacy Show = %+v, %v", info, err)
	}
}

func TestORTCompatibilityRejectsInvalidImportAndOverride(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	for _, content := range []string{`{}`, `{"schema_version":99}`, strings.Repeat(" ", 64<<10)} {
		files, _ := ortProfileFiles(t)
		layer, err := manifest.NewLayer(strings.NewReader(content), MediaTypeORTGenAI)
		if err != nil {
			t.Fatal(err)
		}
		files[ortrunner.CompatibilityFilename] = layer.Digest
		if _, err := convertFromORTGenAI(t.Context(), files, func(api.ProgressResponse) {}); !errors.Is(err, errInvalidORTGenAI) {
			t.Fatalf("malformed profile import error = %v", err)
		}
	}
	files, p := ortProfileFiles(t)
	files["OLLAMA_NPU.JSON"] = files[ortrunner.CompatibilityFilename]
	delete(files, ortrunner.CompatibilityFilename)
	if _, err := convertFromORTGenAI(t.Context(), files, func(api.ProgressResponse) {}); err == nil {
		t.Fatal("case-aliased profile bypassed validation")
	}
	t.Setenv("OLLAMA_ONNX_MODEL", "different-model")
	if _, err := resolveORTOptions(&Model{ModelPath: "profiled-model", ORTCompatibility: p}, api.DefaultOptions()); err == nil {
		t.Fatal("environment model override bypassed persisted profile")
	}
}

// This fixture is deliberately loopback-only. It proves client transport, not
// hosted-registry acceptance, redistribution rights or QNN execution.
func TestORTCompatibilityRegistryRoundTrip(t *testing.T) {
	producer := t.TempDir()
	t.Setenv("OLLAMA_MODELS", producer)
	files, profile := ortProfileFiles(t)
	layers, err := convertFromORTGenAI(t.Context(), files, func(api.ProgressResponse) {})
	if err != nil {
		t.Fatal(err)
	}
	var mu sync.Mutex
	blobs := make(map[string][]byte)
	uploads := make(map[string][]byte)
	var manifestData []byte
	var count int
	registry := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		switch {
		case r.URL.Path == "/v2/library/profile/blobs/uploads/" && r.Method == http.MethodPost:
			count++
			id := strconv.Itoa(count)
			uploads[id] = nil
			w.Header().Set("Location", "http://"+r.Host+"/upload/"+id)
			w.WriteHeader(http.StatusAccepted)
		case strings.HasPrefix(r.URL.Path, "/upload/"):
			id := strings.TrimPrefix(r.URL.Path, "/upload/")
			if _, ok := uploads[id]; !ok {
				http.NotFound(w, r)
				return
			}
			switch r.Method {
			case http.MethodPatch:
				data, err := io.ReadAll(io.LimitReader(r.Body, 8<<20))
				if err != nil {
					t.Error(err)
					http.Error(w, "read upload", http.StatusBadRequest)
					return
				}
				uploads[id] = append(uploads[id], data...)
				w.Header().Set("Location", "http://"+r.Host+r.URL.Path)
				w.WriteHeader(http.StatusAccepted)
			case http.MethodPut:
				data := uploads[id]
				digest := fmt.Sprintf("sha256:%x", sha256.Sum256(data))
				if r.URL.Query().Get("digest") != digest {
					t.Error("upload digest mismatch")
					http.Error(w, "digest mismatch", http.StatusBadRequest)
					return
				}
				blobs[digest] = bytes.Clone(data)
				delete(uploads, id)
				w.WriteHeader(http.StatusCreated)
			default:
				http.NotFound(w, r)
			}
		case strings.HasPrefix(r.URL.Path, "/v2/library/profile/blobs/"):
			data, ok := blobs[strings.TrimPrefix(r.URL.Path, "/v2/library/profile/blobs/")]
			if !ok {
				http.NotFound(w, r)
				return
			}
			w.Header().Set("Location", "http://"+r.Host+r.URL.Path)
			http.ServeContent(w, r, "blob", time.Time{}, bytes.NewReader(data))
		case r.URL.Path == "/v2/library/profile/manifests/qnn" && r.Method == http.MethodPut:
			data, err := io.ReadAll(io.LimitReader(r.Body, 1<<20))
			if err != nil {
				t.Error(err)
				http.Error(w, "read manifest", http.StatusBadRequest)
				return
			}
			manifestData = data
			w.WriteHeader(http.StatusCreated)
		case r.URL.Path == "/v2/library/profile/manifests/qnn" && r.Method == http.MethodGet && manifestData != nil:
			w.Header().Set("Content-Type", manifest.MediaTypeManifest)
			w.Header().Set("Content-Length", strconv.Itoa(len(manifestData)))
			_, _ = w.Write(manifestData)
		default:
			http.NotFound(w, r)
		}
	}))
	defer registry.Close()
	name := model.ParseName(strings.TrimPrefix(registry.URL, "http://") + "/library/profile:qnn")
	request := api.CreateRequest{
		Template: "{{ .Prompt }}", System: "fixture system", License: []string{"synthetic fixture notice"},
		Parameters: map[string]any{"num_ctx": 128, "temperature": 0},
	}
	if err := createModel(t.Context(), request, name, layers, new(model.ConfigV2), func(api.ProgressResponse) {}); err != nil {
		t.Fatal(err)
	}
	original, err := manifest.ParseNamedManifest(name)
	if err != nil {
		t.Fatal(err)
	}
	if err := PushModel(t.Context(), name.String(), &registryOptions{Insecure: true}, func(api.ProgressResponse) {}); err != nil {
		t.Fatal(err)
	}
	consumer := t.TempDir()
	t.Setenv("OLLAMA_MODELS", consumer)
	if err := PullModel(t.Context(), name.String(), "", &registryOptions{Insecure: true}, func(api.ProgressResponse) {}); err != nil {
		t.Fatal(err)
	}
	pulled, err := manifest.ParseNamedManifest(name)
	if err != nil || !reflect.DeepEqual(original.Layers, pulled.Layers) || original.Config.Digest != pulled.Config.Digest {
		t.Fatalf("transport changed descriptors: %v", err)
	}
	m, err := GetModel(name.String())
	if err != nil || !reflect.DeepEqual(m.ORTCompatibility, profile) || m.Template.String() != request.Template ||
		m.System != request.System || !reflect.DeepEqual(m.License, request.License) || m.Options["num_ctx"] != float64(128) {
		t.Fatalf("empty-store model/profile/template/parameters/license mismatch: %+v, %v", m, err)
	}
	for _, key := range []string{"OLLAMA_ONNX_MODEL", "OLLAMA_ONNX_NPU", "OLLAMA_ONNX_PROVIDER", "OLLAMA_ACCEL_MODE",
		"OLLAMA_NPU_DEVICE_ID", "OLLAMA_ORT_DEVICE_TYPE", "OLLAMA_ORT_DEVICE_ID"} {
		t.Setenv(key, "")
	}
	if _, err := resolveORTOptions(m, api.DefaultOptions()); err == nil || !strings.Contains(err.Error(), "INCOMPATIBLE") {
		t.Fatalf("synthetic wrong-target model must be rejected before native load: %v", err)
	}
	// Corrupt bytes must not publish a tag, even when the descriptor was valid.
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	mu.Lock()
	profileLayer := files[ortrunner.CompatibilityFilename]
	profile.GenAIVersion = "99.0.0"
	bad, err := json.Marshal(profile)
	if err != nil {
		mu.Unlock()
		t.Fatal(err)
	}
	// Keep the digest descriptor intact: corruption must fail rather than look valid.
	blobs[profileLayer] = bad
	mu.Unlock()
	if err := PullModel(t.Context(), name.String(), "", &registryOptions{Insecure: true}, func(api.ProgressResponse) {}); err == nil {
		t.Fatal("corrupt downloaded profile accepted")
	}
	if _, err := manifest.ParseNamedManifest(name); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("failed pull published a model tag: %v", err)
	}
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	profile.GenAIVersion = ortruntime.PinnedGenAIVersion
	profile.ContextLength++
	bad, err = json.Marshal(profile)
	if err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	var mismatched manifest.Manifest
	if err := json.Unmarshal(manifestData, &mismatched); err != nil {
		mu.Unlock()
		t.Fatal(err)
	}
	badDigest := fmt.Sprintf("sha256:%x", sha256.Sum256(bad))
	blobs[badDigest] = bad
	for i := range mismatched.Layers {
		if mismatched.Layers[i].Name == ortrunner.CompatibilityFilename {
			mismatched.Layers[i].Digest = badDigest
			mismatched.Layers[i].Size = int64(len(bad))
		}
	}
	manifestData, err = json.Marshal(mismatched)
	mu.Unlock()
	if err != nil {
		t.Fatal(err)
	}
	if err := PullModel(t.Context(), name.String(), "", &registryOptions{Insecure: true}, func(api.ProgressResponse) {}); err == nil || !strings.Contains(err.Error(), "context_length") {
		t.Fatalf("well-hashed contradictory profile must fail static validation: %v", err)
	}
	if _, err := manifest.ParseNamedManifest(name); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("contradictory declaration published a tag: %v", err)
	}
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	mismatched.Layers = append(mismatched.Layers, manifest.Layer{
		MediaType: manifest.MediaTypeImageTensor, Digest: fmt.Sprintf("sha256:%x", sha256.Sum256([]byte("native tensor fixture"))), Size: 1,
	})
	if err := validateDeclaredORTProfile(mismatched.Layers); err == nil || !strings.Contains(err.Error(), "native tensor layers") {
		t.Fatalf("mixed-format push must reject declared ORT before reading artifacts: %v", err)
	}
	mu.Lock()
	manifestData, err = json.Marshal(mismatched)
	mu.Unlock()
	if err != nil {
		t.Fatal(err)
	}
	if err := PullModel(t.Context(), name.String(), "", &registryOptions{Insecure: true}, func(api.ProgressResponse) {}); err == nil || !strings.Contains(err.Error(), "native tensor layers") {
		t.Fatalf("native fast-transfer must not bypass ORT declaration validation: %v", err)
	}
	if _, err := manifest.ParseNamedManifest(name); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("mixed-format declaration published a tag: %v", err)
	}
}

func TestORTCompatibilityManifestListPreflight(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	files, _ := ortProfileFiles(t)
	layers, err := convertFromORTGenAI(t.Context(), files, func(api.ProgressResponse) {})
	if err != nil {
		t.Fatal(err)
	}
	childName := model.ParseName("profile-list-child:qnn")
	if err := createModel(t.Context(), api.CreateRequest{}, childName, layers, new(model.ConfigV2), func(api.ProgressResponse) {}); err != nil {
		t.Fatal(err)
	}
	child, err := manifest.ParseNamedManifest(childName)
	if err != nil {
		t.Fatal(err)
	}
	validRef, err := manifest.NewManifestReference(child.BlobDigest(), child.Runner, child.Format)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := pushLayersForManifestList(manifest.Manifest{Manifests: []manifest.Manifest{validRef}}); err != nil {
		t.Fatalf("valid profiled child preflight: %v", err)
	}
	bad, err := manifest.NewLayer(strings.NewReader(`{}`), MediaTypeORTGenAI)
	if err != nil {
		t.Fatal(err)
	}
	bad.Name = ortrunner.CompatibilityFilename
	badLayers := append([]manifest.Layer(nil), child.Layers...)
	for i := range badLayers {
		if badLayers[i].Name == ortrunner.CompatibilityFilename {
			badLayers[i] = bad
		}
	}
	if err := manifest.WriteManifestWithMetadata(childName, child.Config, badLayers, child.Runner, child.Format); err != nil {
		t.Fatal(err)
	}
	child, err = manifest.ParseNamedManifest(childName)
	if err != nil {
		t.Fatal(err)
	}
	badRef, err := manifest.NewManifestReference(child.BlobDigest(), child.Runner, child.Format)
	if err != nil {
		t.Fatal(err)
	}
	var requests atomic.Int32
	registry := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests.Add(1)
		http.Error(w, "preflight must finish before registry access", http.StatusBadRequest)
	}))
	defer registry.Close()
	parentName := model.ParseName(strings.TrimPrefix(registry.URL, "http://") + "/library/profile-list:qnn")
	if _, err := manifest.WriteManifestList(parentName, []manifest.Manifest{validRef, badRef}); err != nil {
		t.Fatal(err)
	}
	if err := PushModel(t.Context(), parentName.String(), &registryOptions{Insecure: true}, func(api.ProgressResponse) {}); err == nil || !strings.Contains(err.Error(), "declared ORT profile") {
		t.Fatalf("invalid list child must fail declaration preflight: %v", err)
	}
	if requests.Load() != 0 {
		t.Fatalf("invalid manifest list performed %d registry requests", requests.Load())
	}
}
