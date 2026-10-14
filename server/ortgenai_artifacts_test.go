package server

import (
	"context"
	"errors"
	"maps"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/types/model"
)

func ortTestManifest(t *testing.T) *manifest.Manifest {
	t.Helper()
	layers, err := convertFromORTGenAI(t.Context(), ortTestFiles(t), func(api.ProgressResponse) {})
	if err != nil {
		t.Fatal(err)
	}
	mf := new(manifest.Manifest)
	for _, layer := range layers {
		mf.Layers = append(mf.Layers, layer.Layer)
	}
	return mf
}

func TestORTGenAIArtifactReferenceValidation(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	base := ortTestFiles(t)
	for _, test := range []struct {
		name, filename, content, remove string
	}{
		{name: "missing nested graph", remove: "nested/model.onnx"},
		{name: "malformed config", filename: "genai_config.json", content: "{invalid"},
		{name: "missing model metadata", filename: "genai_config.json", content: "{}"},
		{name: "trailing config", filename: "genai_config.json", content: `{"model":{}} {}`},
		{name: "missing decoder ref", filename: "genai_config.json", content: `{"model":{"type":"llama","context_length":128,"vocab_size":8,"decoder":{"filename":"missing.onnx"}}}`},
		{name: "escaping decoder ref", filename: "genai_config.json", content: `{"model":{"type":"llama","context_length":128,"vocab_size":8,"decoder":{"filename":"../model.onnx"}}}`},
		{name: "missing pipeline ref", filename: "genai_config.json", content: `{"model":{"type":"llama","context_length":128,"vocab_size":8,"decoder":{"pipeline":[{"first":{"filename":"model.onnx"}},{"second":{"filename":"missing.onnx"}}]}}}`},
		{name: "null tokenizer", filename: "tokenizer.json", content: `null`},
		{name: "invalid tokenizer", filename: "tokenizer.json", content: `{invalid`},
		{name: "empty graph", filename: "model.onnx", content: ""},
	} {
		t.Run(test.name, func(t *testing.T) {
			files := maps.Clone(base)
			delete(files, test.remove)
			if test.filename != "" {
				layer, err := manifest.NewLayer(strings.NewReader(test.content), MediaTypeORTGenAI)
				if err != nil {
					t.Fatal(err)
				}
				files[test.filename] = layer.Digest
			}
			if _, err := convertFromORTGenAI(t.Context(), files, func(api.ProgressResponse) {}); !errors.Is(err, errInvalidORTGenAI) {
				t.Fatalf("validation error = %v", err)
			}
		})
	}
}

func TestORTGenAIRejectsNonportableAndCollidingNames(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	base := ortTestManifest(t)
	for _, name := range []string{"CON.onnx", "com1.bin", "LPT².data", "nested/NUL", "model?.onnx", "trailing./file", "trailing /file", "bad\x00.onnx", "MODEL.onnx", "nested"} {
		t.Run(name, func(t *testing.T) {
			layers := append([]manifest.Layer(nil), base.Layers...)
			extra := layers[0]
			extra.Name = name
			layers = append(layers, extra)
			if _, err := materializeORTGenAIDir(&manifest.Manifest{Layers: layers}); err == nil {
				t.Fatalf("accepted unsafe/colliding name %q", name)
			}
		})
	}
	if isORTGenAIDirComplete(t.TempDir(), new(manifest.Manifest)) {
		t.Fatal("an empty manifest is not a complete model")
	}
}

func TestORTGenAIMaterializationRepairsSameSizeCorruption(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	mf := ortTestManifest(t)
	dir, err := materializeORTGenAIDir(mf)
	if err != nil {
		t.Fatal(err)
	}
	cached := filepath.Join(dir, "model.onnx")
	original, err := os.ReadFile(cached)
	if err != nil {
		t.Fatal(err)
	}
	corrupt := append([]byte(nil), original...)
	corrupt[len(corrupt)-1] ^= 1
	if err := os.WriteFile(cached, corrupt, 0o644); err != nil {
		t.Fatal(err)
	}
	if isORTGenAIDirComplete(dir, mf) {
		t.Fatal("same-size corruption was accepted")
	}
	if again, err := materializeORTGenAIDir(mf); err != nil || again != dir {
		t.Fatalf("repair = %q, %v", again, err)
	}
	if !isORTGenAIDirComplete(dir, mf) {
		t.Fatal("repaired cache is incomplete")
	}
	quarantined, err := filepath.Glob(dir + ".invalid-*")
	if err != nil || len(quarantined) != 1 {
		t.Fatalf("corrupt cache must be retained for diagnosis: %v, %v", quarantined, err)
	}
	for _, layer := range mf.Layers {
		if layer.Name != "model.onnx" {
			continue
		}
		blob, err := manifest.BlobsPath(layer.Digest)
		if err != nil {
			t.Fatal(err)
		}
		if err := verifyORTGenAIFile(blob, layer, nil); err != nil {
			t.Fatalf("cache writes must not mutate content-addressed blobs: %v", err)
		}
	}
}

func TestORTGenAIMaterializationRequiresCompleteNestedTree(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	mf := ortTestManifest(t)
	dir, err := materializeORTGenAIDir(mf)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(filepath.Join(dir, "nested", "model.onnx")); err != nil {
		t.Fatal(err)
	}
	if isORTGenAIDirComplete(dir, mf) {
		t.Fatal("missing nested graph was accepted")
	}
	if _, err := materializeORTGenAIDir(mf); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "unknown.bin"), []byte("stale"), 0o644); err != nil {
		t.Fatal(err)
	}
	if isORTGenAIDirComplete(dir, mf) {
		t.Fatal("foreign cached artifact was accepted")
	}
}

func TestORTGenAIMaterializationRejectsCorruptBlobAtomically(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	mf := ortTestManifest(t)
	layer := mf.Layers[0]
	blob, err := manifest.BlobsPath(layer.Digest)
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(blob)
	if err != nil {
		t.Fatal(err)
	}
	data[0] ^= 1
	if err := os.WriteFile(blob, data, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := materializeORTGenAIDir(mf); !errors.Is(err, errInvalidORTGenAI) {
		t.Fatalf("corrupt blob error = %v", err)
	}
	entries, err := os.ReadDir(filepath.Join(os.Getenv("OLLAMA_MODELS"), "ortgenai"))
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range entries {
		if entry.IsDir() {
			t.Fatalf("published partial model directory: %s", entry.Name())
		}
	}
}

func TestORTGenAIMaterializationConcurrent(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	mf := ortTestManifest(t)
	var wg sync.WaitGroup
	results := make(chan string, 8)
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			dir, err := materializeORTGenAIDir(mf)
			if err != nil {
				t.Error(err)
				return
			}
			results <- dir
		}()
	}
	wg.Wait()
	close(results)
	var expected string
	for dir := range results {
		if expected != "" && dir != expected || !isORTGenAIDirComplete(dir, mf) {
			t.Fatalf("inconsistent concurrent materialization: %q", dir)
		}
		expected = dir
	}
}

func TestORTGenAILockProcess(t *testing.T) {
	if path := os.Getenv("OLLAMA_TEST_ORT_LOCK"); path != "" {
		ctx, cancel := context.WithTimeout(t.Context(), 200*time.Millisecond)
		defer cancel()
		unlock, err := acquireORTGenAILock(ctx, path)
		if os.Getenv("OLLAMA_TEST_ORT_LOCK_BLOCKED") == "1" {
			if !errors.Is(err, context.DeadlineExceeded) {
				if unlock != nil {
					unlock()
				}
				t.Fatalf("another process bypassed the lock: %v", err)
			}
		} else {
			if err != nil {
				t.Fatal(err)
			}
			unlock()
		}
		return
	}
	path := filepath.Join(t.TempDir(), "model.lock")
	unlock, err := acquireORTGenAILock(t.Context(), path)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if unlock != nil {
			unlock()
		}
	}()
	for _, blocked := range []string{"1", "0"} {
		if blocked == "0" {
			unlock()
			unlock = nil
		}
		cmd := exec.CommandContext(t.Context(), os.Args[0], "-test.run=^TestORTGenAILockProcess$")
		cmd.Env = append(os.Environ(), "OLLAMA_TEST_ORT_LOCK="+path, "OLLAMA_TEST_ORT_LOCK_BLOCKED="+blocked)
		if output, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("cross-process lock test: %v\n%s", err, output)
		}
	}
}

func TestORTGenAIFromRejectsNativeLayers(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	mf := ortTestManifest(t)
	var layers []*modelLayer
	for _, layer := range mf.Layers {
		layers = append(layers, &modelLayer{Layer: layer})
	}

	native := mf.Layers[0]
	native.MediaType = manifest.MediaTypeImageModel
	layers = append(layers, &modelLayer{Layer: native})
	if err := createModel(t.Context(), api.CreateRequest{}, model.ParseName("invalid-mixed"), layers, new(model.ConfigV2), func(api.ProgressResponse) {}); !errors.Is(err, errInvalidORTGenAI) {
		t.Fatalf("mixed FROM error = %v", err)
	}
}

func TestORTGenAIExternalDataIsNotDetectedAsAnotherModel(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())
	files := ortTestFiles(t)
	layer, err := manifest.NewLayer(strings.NewReader("GGUFexternal tensor bytes"), MediaTypeORTGenAI)
	if err != nil {
		t.Fatal(err)
	}
	files["nested/weights.bin"] = layer.Digest
	if kind, err := detectModelTypeFromFiles(files); err != nil || kind != "ortgenai" {
		t.Fatalf("external data misclassified: %q, %v", kind, err)
	}
}
