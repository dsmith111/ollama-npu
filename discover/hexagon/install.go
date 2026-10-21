package hexagon

import (
	"archive/zip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

// maxFileSize bounds each extracted library; the pinned ones are under 1 MiB.
const maxFileSize = 64 << 20

// InstallResult reports what setup did.
type InstallResult struct {
	Dir       string `json:"dir"`
	Source    string `json:"source"`
	Installed bool   `json:"installed"`
	Validated bool   `json:"validated"` // file integrity only, not NPU readiness
	Message   string `json:"message"`
}

// Install puts the pinned set in dir, from the release archive in packageDir
// when given or else downloaded from Qualcomm's release. It never replaces an
// existing directory; one that already holds the set is reported as verified.
func Install(ctx context.Context, dir, packageDir string) (*InstallResult, error) {
	client := downloadClient()
	defer client.CloseIdleConnections()
	return install(ctx, Pinned, dir, packageDir, client)
}

func install(ctx context.Context, rel Release, dir, packageDir string, client *http.Client) (*InstallResult, error) {
	if dir == "" {
		return nil, errors.New("unable to determine the installation directory")
	}
	dir, err := filepath.Abs(dir)
	if err != nil {
		return nil, err
	}
	parent := filepath.Dir(dir)
	if parent == dir {
		return nil, errors.New("the installation directory cannot be a filesystem root")
	}
	result := &InstallResult{Dir: dir, Source: rel.Source}
	if _, err := os.Lstat(dir); err == nil {
		if err := validate(rel, dir); err != nil {
			result.Message = "The existing directory does not hold the pinned set; no files were changed"
			return result, fmt.Errorf("refusing to replace %s; use a new directory: %w", dir, err)
		}
		result.Validated = true
		result.Message = "Pinned files already verified"
		return result, nil
	} else if !errors.Is(err, os.ErrNotExist) {
		return nil, err
	}

	if err := os.MkdirAll(parent, 0o755); err != nil {
		return nil, fmt.Errorf("installation parent: %w", err)
	}
	stage, err := os.MkdirTemp(parent, ".hexagon-htp-install-")
	if err != nil {
		return nil, fmt.Errorf("create staging directory: %w", err)
	}
	defer os.RemoveAll(stage)

	archive := filepath.Join(packageDir, rel.ArchiveName)
	if packageDir == "" {
		archive = filepath.Join(stage, rel.ArchiveName)
		if err := download(ctx, client, rel, archive); err != nil {
			return nil, err
		}
	}
	if err := checkArchive(rel, archive); err != nil {
		return nil, err
	}
	if err := extract(rel, archive, stage); err != nil {
		return nil, err
	}
	if packageDir == "" {
		if err := os.Remove(archive); err != nil {
			return nil, err
		}
	}
	if err := writeManifest(rel, stage); err != nil {
		return nil, err
	}
	if err := validate(rel, stage); err != nil {
		return nil, fmt.Errorf("staging validation: %w", err)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	// Never remove a destination: Windows may already have libraries loaded from it.
	if _, err := os.Lstat(dir); err == nil {
		return nil, fmt.Errorf("refusing to replace %s, which appeared during setup", dir)
	} else if !errors.Is(err, os.ErrNotExist) {
		return nil, err
	}
	if err := os.Rename(stage, dir); err != nil {
		return nil, fmt.Errorf("promote staging directory: %w", err)
	}
	result.Installed, result.Validated = true, true
	result.Message = "Pinned files verified; loading them on the NPU is not tested"
	return result, nil
}

func downloadClient() *http.Client {
	return &http.Client{
		Timeout: 30 * time.Minute,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			if len(via) >= 5 {
				return errors.New("too many redirects")
			}
			if req.URL.Scheme != "https" || !releaseHost(req.URL.Hostname()) {
				return fmt.Errorf("unexpected redirect to %s", req.URL.Redacted())
			}
			return nil
		},
	}
}

// releaseHost reports whether GitHub serves release assets from host.
func releaseHost(host string) bool {
	switch host {
	case "github.com", "objects.githubusercontent.com", "release-assets.githubusercontent.com":
		return true
	}
	return false
}

func download(ctx context.Context, client *http.Client, rel Release, path string) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, rel.URL, nil)
	if err != nil {
		return err
	}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("download %s: %w", rel.URL, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download %s: %s", rel.URL, resp.Status)
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
	if err != nil {
		return err
	}
	n, err := io.Copy(f, io.LimitReader(resp.Body, rel.ArchiveSize+1))
	if closeErr := f.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return fmt.Errorf("download %s: %w", rel.URL, err)
	}
	if n != rel.ArchiveSize {
		return fmt.Errorf("download %s: got %d bytes, want %d", rel.URL, n, rel.ArchiveSize)
	}
	return nil
}

func checkArchive(rel Release, path string) error {
	info, err := os.Stat(path)
	if err != nil {
		return err
	}
	if info.Size() != rel.ArchiveSize {
		return fmt.Errorf("%s is %d bytes, want %d", path, info.Size(), rel.ArchiveSize)
	}
	got, err := fileSHA256(path)
	if err != nil {
		return err
	}
	if got != rel.ArchiveSHA256 {
		return fmt.Errorf("%s has SHA256 %s, want %s", path, got, rel.ArchiveSHA256)
	}
	return nil
}

// extract copies only the pinned files out of the archive, checking each hash.
func extract(rel Release, archive, dir string) error {
	r, err := zip.OpenReader(archive)
	if err != nil {
		return err
	}
	defer r.Close()
	entries := make(map[string]*zip.File, len(r.File))
	for _, f := range r.File {
		entries[f.Name] = f
	}
	for _, name := range rel.names() {
		f := entries[rel.Prefix+name]
		if f == nil {
			return fmt.Errorf("%s lacks %s", rel.ArchiveName, rel.Prefix+name)
		}
		if err := extractFile(f, filepath.Join(dir, name), rel.Files[name]); err != nil {
			return err
		}
	}
	return nil
}

func extractFile(f *zip.File, path, want string) error {
	if f.UncompressedSize64 > maxFileSize {
		return fmt.Errorf("%s is too large (%d bytes)", f.Name, f.UncompressedSize64)
	}
	rc, err := f.Open()
	if err != nil {
		return err
	}
	defer rc.Close()
	out, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	h := sha256.New()
	_, err = io.Copy(io.MultiWriter(out, h), io.LimitReader(rc, maxFileSize))
	if closeErr := out.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return fmt.Errorf("extract %s: %w", f.Name, err)
	}
	if got := hex.EncodeToString(h.Sum(nil)); got != want {
		return fmt.Errorf("%s has SHA256 %s, want %s", f.Name, got, want)
	}
	return nil
}

// writeManifest records the set's origin, licence and terms next to it.
func writeManifest(rel Release, dir string) error {
	manifest, err := json.MarshalIndent(map[string]any{
		"source":         rel.Source,
		"url":            rel.URL,
		"archive_sha256": rel.ArchiveSHA256,
		"llama_cpp":      rel.LlamaCppTag,
		"files":          rel.Files,
		"license":        rel.License,
		"terms":          Terms,
	}, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, ManifestFilename), append(manifest, '\n'), 0o644)
}
