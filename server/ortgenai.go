package server

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/manifest"
)

const MediaTypeORTGenAI = "application/vnd.ollama.image.ortgenai"

var verifiedORTFiles = struct {
	sync.Mutex
	files map[string]verifiedORTFile
}{files: make(map[string]verifiedORTFile)}

type verifiedORTFile struct {
	info   os.FileInfo
	digest string
}

func sameORTFile(a, b os.FileInfo) bool {
	return os.SameFile(a, b) && a.Mode() == b.Mode() && a.Size() == b.Size() && a.ModTime().Equal(b.ModTime())
}

func convertFromORTGenAI(ctx context.Context, files map[string]string, fn func(api.ProgressResponse)) ([]*modelLayer, error) {
	if err := validateCreateFiles(files); err != nil {
		return nil, err
	}
	names := make([]string, 0, len(files))
	for name := range files {
		names = append(names, name)
	}
	slices.Sort(names)
	layers := make([]*modelLayer, 0, len(files))
	var artifacts []manifest.Layer
	for _, name := range names {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		layer, err := manifest.NewLayerFromLayer(files[name], MediaTypeORTGenAI, "")
		if err != nil {
			return nil, fmt.Errorf("package %s: %w", name, err)
		}
		layer.Name = name
		artifacts = append(artifacts, layer)
		layers = append(layers, &modelLayer{Layer: layer})
	}
	if err := validateORTGenAIArtifacts(artifacts); err != nil {
		return nil, err
	}
	fn(api.ProgressResponse{Status: "packaging ORT GenAI model"})
	return layers, nil
}

func validateORTGenAIPath(name string) error {
	if err := validateCreateFilePath(name); err != nil {
		return err
	}
	for _, part := range strings.Split(name, "/") {
		if strings.ContainsAny(part, `<>:"|?*`) || strings.TrimRight(part, ". ") != part {
			return fmt.Errorf("%w: nonportable artifact name %q", errInvalidORTGenAI, name)
		}
		for _, ch := range part {
			if ch < 32 || ch == 127 {
				return fmt.Errorf("%w: invalid artifact name %q", errInvalidORTGenAI, name)
			}
		}
		base, _, _ := strings.Cut(strings.ToUpper(part), ".")
		if slices.Contains([]string{"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}, base) ||
			(strings.HasPrefix(base, "COM") || strings.HasPrefix(base, "LPT")) && slices.Contains([]string{"1", "2", "3", "4", "5", "6", "7", "8", "9", "¹", "²", "³"}, base[3:]) {
			return fmt.Errorf("%w: reserved artifact name %q", errInvalidORTGenAI, name)
		}
	}
	return nil
}

func ortGenAIArtifacts(layers []manifest.Layer) (map[string]manifest.Layer, error) {
	files := make(map[string]manifest.Layer)
	folded := make(map[string]bool)
	hasONNX := false
	for _, layer := range layers {
		if layer.MediaType != MediaTypeORTGenAI {
			continue
		}
		if err := validateORTGenAIPath(layer.Name); err != nil {
			return nil, err
		}
		key := strings.ToLower(layer.Name)
		if folded[key] {
			return nil, fmt.Errorf("%w: duplicate artifact name %q", errInvalidORTGenAI, layer.Name)
		}
		required := strings.HasSuffix(layer.Name, ".onnx") || slices.Contains([]string{"genai_config.json", "tokenizer.json", "tokenizer.model"}, layer.Name)
		if layer.Size < 0 || required && layer.Size == 0 || len(layer.Digest) != len("sha256:")+sha256.Size*2 {
			return nil, fmt.Errorf("%w: invalid size/digest for %q", errInvalidORTGenAI, layer.Name)
		}
		if _, err := manifest.BlobsPath(layer.Digest); err != nil {
			return nil, err
		}
		files[layer.Name] = layer
		folded[key] = true
		hasONNX = hasONNX || strings.HasSuffix(layer.Name, ".onnx")
	}
	for name := range folded {
		for parent := filepath.ToSlash(filepath.Dir(filepath.FromSlash(name))); parent != "."; parent = filepath.ToSlash(filepath.Dir(filepath.FromSlash(parent))) {
			if folded[parent] {
				return nil, fmt.Errorf("%w: artifact %q is also a directory", errInvalidORTGenAI, parent)
			}
		}
	}
	if len(files) > maxCreateFiles {
		return nil, fmt.Errorf("%w: too many artifacts", errInvalidORTGenAI)
	}
	if !hasONNX || files["genai_config.json"].Name == "" || files["tokenizer.json"].Name == "" && files["tokenizer.model"].Name == "" {
		return nil, fmt.Errorf("%w: requires ONNX graphs, genai_config.json, and tokenizer.json or tokenizer.model", errInvalidORTGenAI)
	}
	return files, nil
}

func verifyORTGenAIFile(filename string, layer manifest.Layer, dst io.Writer) error {
	info, err := os.Lstat(filename)
	if err != nil {
		return err
	}
	if !info.Mode().IsRegular() || info.Size() != layer.Size {
		return fmt.Errorf("%w: artifact %q is not a regular file of size %d", errInvalidORTGenAI, layer.Name, layer.Size)
	}
	// Blobs and derived assets are immutable within Ollama. Rehash on any file
	// identity/metadata change, but do not reread gigabytes on every API request.
	if dst == nil && layer.Size >= 1<<20 {
		verifiedORTFiles.Lock()
		verified, ok := verifiedORTFiles.files[filename]
		verifiedORTFiles.Unlock()
		if ok && verified.digest == layer.Digest && sameORTFile(info, verified.info) {
			return nil
		}
	}
	f, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer f.Close()
	opened, err := f.Stat()
	if err != nil {
		return err
	}
	if !os.SameFile(info, opened) {
		return fmt.Errorf("%w: artifact %q changed during validation", errInvalidORTGenAI, layer.Name)
	}
	hash := sha256.New()
	writer := io.Writer(hash)
	if dst != nil {
		writer = io.MultiWriter(dst, hash)
	}
	n, err := io.Copy(writer, f)
	if err != nil {
		return err
	}
	if n != layer.Size || fmt.Sprintf("sha256:%x", hash.Sum(nil)) != layer.Digest {
		return fmt.Errorf("%w: SHA256 mismatch for %q", errInvalidORTGenAI, layer.Name)
	}
	after, err := f.Stat()
	if err != nil {
		return err
	}
	if !sameORTFile(opened, after) {
		return fmt.Errorf("%w: artifact %q changed during hashing", errInvalidORTGenAI, layer.Name)
	}
	if dst == nil && layer.Size >= 1<<20 {
		verifiedORTFiles.Lock()
		if len(verifiedORTFiles.files) >= 1024 {
			clear(verifiedORTFiles.files)
		}
		verifiedORTFiles.files[filename] = verifiedORTFile{info: after, digest: layer.Digest}
		verifiedORTFiles.Unlock()
	}
	return nil
}

func validateORTGenAIArtifacts(layers []manifest.Layer) error {
	files, err := ortGenAIArtifacts(layers)
	if err != nil {
		return err
	}
	for _, layer := range files {
		path, err := manifest.BlobsPath(layer.Digest)
		if err != nil {
			return err
		}
		if err := verifyORTGenAIFile(path, layer, nil); err != nil {
			return fmt.Errorf("validate %s: %w", layer.Name, err)
		}
	}
	configPath, err := manifest.BlobsPath(files["genai_config.json"].Digest)
	if err != nil {
		return err
	}
	config, err := readORTGenAIConfig(configPath)
	if err != nil {
		return fmt.Errorf("%w: %w", errInvalidORTGenAI, err)
	}
	for _, name := range config.filenames {
		if err := validateORTGenAIPath(name); err != nil {
			return err
		}
		if files[name].Name == "" {
			return fmt.Errorf("%w: genai_config.json references missing artifact %q", errInvalidORTGenAI, name)
		}
		if files[name].Size == 0 {
			return fmt.Errorf("%w: genai_config.json references empty artifact %q", errInvalidORTGenAI, name)
		}
	}
	if tokenizer := files["tokenizer.json"]; tokenizer.Name != "" {
		if tokenizer.Size > 64<<20 {
			return fmt.Errorf("%w: tokenizer.json exceeds 64 MiB", errInvalidORTGenAI)
		}
		path, err := manifest.BlobsPath(tokenizer.Digest)
		if err != nil {
			return err
		}
		f, err := os.Open(path)
		if err != nil {
			return err
		}
		defer f.Close()
		decoder := json.NewDecoder(io.LimitReader(f, 64<<20))
		var value map[string]json.RawMessage
		if err := decoder.Decode(&value); err != nil || value == nil {
			return fmt.Errorf("%w: tokenizer.json must contain a JSON object", errInvalidORTGenAI)
		}
		if err := decoder.Decode(new(any)); err != io.EOF {
			return fmt.Errorf("%w: invalid trailing tokenizer.json data", errInvalidORTGenAI)
		}
	}
	// ONNX graph semantics and external-data/EPContext references are checked by
	// the child loader. Packaging validates config references, not inference.
	return nil
}

func materializeORTGenAIDir(mf *manifest.Manifest) (string, error) {
	files, err := ortGenAIArtifacts(mf.Layers)
	if err != nil {
		return "", err
	}
	names := make([]string, 0, len(files))
	for name := range files {
		names = append(names, name)
	}
	slices.Sort(names)
	hash := sha256.New()
	for _, name := range names {
		fmt.Fprintf(hash, "%q %s %d\n", name, files[name].Digest, files[name].Size)
	}
	root := filepath.Join(envconfig.Models(), "ortgenai")
	if err := os.MkdirAll(root, 0o755); err != nil {
		return "", err
	}
	info, err := os.Lstat(root)
	if err != nil {
		return "", err
	}
	if !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return "", fmt.Errorf("%w: unsafe ORT GenAI cache root", errInvalidORTGenAI)
	}
	cacheDir := filepath.Join(root, fmt.Sprintf("%x", hash.Sum(nil)))
	unlock, err := acquireORTGenAILock(context.Background(), cacheDir+".lock")
	if err != nil {
		return "", fmt.Errorf("lock ORT GenAI cache: %w", err)
	}
	defer unlock()
	if err := validateORTGenAIArtifacts(mf.Layers); err != nil {
		return "", err
	}
	if isORTGenAIDirComplete(cacheDir, mf) {
		return cacheDir, nil
	}
	staging, err := os.MkdirTemp(root, filepath.Base(cacheDir)+"-building-")
	if err != nil {
		return "", err
	}
	defer os.RemoveAll(staging)
	for _, name := range names {
		layer := files[name]
		blob, err := manifest.BlobsPath(layer.Digest)
		if err != nil {
			return "", err
		}
		dst := filepath.Join(staging, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
			return "", err
		}
		f, err := os.OpenFile(dst, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
		if err != nil {
			return "", err
		}
		copyErr := verifyORTGenAIFile(blob, layer, f)
		syncErr := f.Sync()
		closeErr := f.Close()
		for _, err := range []error{copyErr, syncErr, closeErr} {
			if err != nil {
				return "", fmt.Errorf("materialize %s: %w", name, err)
			}
		}
	}
	if !isORTGenAIDirComplete(staging, mf) {
		return "", fmt.Errorf("%w: incomplete materialization", errInvalidORTGenAI)
	}
	// Retain an invalid old cache for diagnosis; never modify it in place or
	// delete user/model artifacts. Only a fully verified tree is published.
	if _, err := os.Lstat(cacheDir); err == nil {
		if err := os.Rename(cacheDir, cacheDir+".invalid-"+filepath.Base(staging)); err != nil {
			return "", fmt.Errorf("quarantine invalid ORT GenAI cache: %w", err)
		}
	} else if !os.IsNotExist(err) {
		return "", err
	}
	if err := os.Rename(staging, cacheDir); err != nil {
		return "", fmt.Errorf("publish ORT GenAI cache: %w", err)
	}
	return cacheDir, nil
}

func isORTGenAIDirComplete(dir string, mf *manifest.Manifest) bool {
	files, err := ortGenAIArtifacts(mf.Layers)
	if err != nil {
		return false
	}
	info, err := os.Lstat(dir)
	if err != nil || !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return false
	}
	count := 0
	err = filepath.WalkDir(dir, func(filename string, entry os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if entry.IsDir() {
			return nil
		}
		rel, err := filepath.Rel(dir, filename)
		if err != nil {
			return err
		}
		layer, ok := files[filepath.ToSlash(rel)]
		if !ok {
			return fmt.Errorf("unexpected cached artifact %s", rel)
		}
		if err := verifyORTGenAIFile(filename, layer, nil); err != nil {
			return err
		}
		count++
		return nil
	})
	return err == nil && count == len(files)
}
