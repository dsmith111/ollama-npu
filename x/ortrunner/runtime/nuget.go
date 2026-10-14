package runtime

import (
	"archive/zip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"
	"unicode/utf8"
)

type NuGetPackage struct {
	ID      string
	Version string
	SHA256  string // mandatory SHA-256 of the exact .nupkg
	Size    int64  // exact size when known
}

const (
	PinnedGenAIVersion = "0.12.2"
	PinnedORTVersion   = "1.24.1"
	maxPackageSize     = 128 << 20
	maxFileSize        = 256 << 20
	maxExpandedSize    = 1 << 30
	maxZIPEntries      = 4096
	downloadTimeout    = 5 * time.Minute
	nativePrefix       = "runtimes/win-arm64/native/"
)

// PinnedPackages is informational. Acquisition uses a private fresh copy, so
// mutating this compatibility variable cannot weaken the trust anchors.
var PinnedPackages = pinnedPackages()

// Hashes reproduced from separate official NuGet v2 and globalcdn downloads,
// 2026-10-06. These are hashes of the entire signed .nupkg, not unpacked ZIPs.
func pinnedPackages() []NuGetPackage {
	return []NuGetPackage{
		{"Microsoft.ML.OnnxRuntimeGenAI.QNN", PinnedGenAIVersion, "a77f31d2458a5d4d57fb4920ae8f65ab53258899c4fe1f6fe897a078650eec19", 2309592},
		{"Microsoft.ML.OnnxRuntime.QNN", PinnedORTVersion, "27e0a4a1ea9f9f8348ab183ca4886195b655bf3543cf1c9938733fc6d6b42796", 96523158},
	}
}

func packageFilename(pkg NuGetPackage) string {
	return strings.ToLower(pkg.ID) + "." + pkg.Version + ".nupkg"
}

func nugetDownloadURL(pkg NuGetPackage) string {
	return "https://www.nuget.org/api/v2/package/" + url.PathEscape(pkg.ID) + "/" + url.PathEscape(pkg.Version)
}

func downloadClient() *http.Client {
	return &http.Client{
		Timeout: downloadTimeout,
		Transport: &http.Transport{
			Proxy:               http.ProxyFromEnvironment,
			DialContext:         (&net.Dialer{Timeout: 15 * time.Second, KeepAlive: 30 * time.Second}).DialContext,
			TLSHandshakeTimeout: 15 * time.Second, ResponseHeaderTimeout: 30 * time.Second,
		},
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			host := strings.ToLower(req.URL.Hostname())
			if len(via) >= 5 || req.URL.Scheme != "https" || req.URL.User != nil ||
				(req.URL.Port() != "" && req.URL.Port() != "443") ||
				(host != "www.nuget.org" && host != "api.nuget.org" && host != "globalcdn.nuget.org") {
				return errors.New("unsafe NuGet redirect")
			}
			return nil
		},
	}
}

// DownloadAndExtractNuGet writes one approved package into a caller-owned staging
// directory, without replacing any file. Use Install for atomic bundle promotion.
func DownloadAndExtractNuGet(pkg NuGetPackage, destDir string) error {
	return DownloadAndExtractNuGetContext(context.Background(), pkg, destDir)
}

func DownloadAndExtractNuGetContext(ctx context.Context, pkg NuGetPackage, destDir string) error {
	if ctx == nil {
		return errors.New("nil acquisition context")
	}
	for _, spec := range approvedPackages() {
		if pkg == spec.Package {
			if err := makeSafeDir(destDir); err != nil {
				return err
			}
			client := downloadClient()
			defer client.CloseIdleConnections()
			return acquirePackage(ctx, client, nugetDownloadURL(pkg), spec, destDir)
		}
	}
	return errors.New("package is not an exact approved pin (including SHA256 and size)")
}

func acquirePackage(ctx context.Context, client *http.Client, url string, spec packageSpec, destDir string) error {
	ctx, cancel := context.WithTimeout(ctx, downloadTimeout)
	defer cancel()
	tmp, err := os.CreateTemp(destDir, ".nupkg-")
	if err != nil {
		return err
	}
	path := tmp.Name()
	defer os.Remove(path)
	err = downloadPackage(ctx, client, url, spec.Package, tmp)
	closeErr := tmp.Close()
	if err != nil {
		return err
	}
	if closeErr != nil {
		return closeErr
	}
	return extractPackage(ctx, path, destDir, spec)
}

func downloadPackage(ctx context.Context, client *http.Client, url string, pkg NuGetPackage, out *os.File) error {
	if err := checkPackagePin(pkg); err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("download: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download: HTTP %d", resp.StatusCode)
	}
	if resp.ContentLength > maxPackageSize || (pkg.Size > 0 && resp.ContentLength >= 0 && resp.ContentLength != pkg.Size) {
		return fmt.Errorf("unexpected package Content-Length %d", resp.ContentLength)
	}
	h := sha256.New()
	n, err := io.Copy(io.MultiWriter(out, h), io.LimitReader(contextReader{ctx, resp.Body}, packageSizeLimit(pkg)+1))
	if err != nil {
		return fmt.Errorf("download body: %w", err)
	}
	if err := checkDigest(pkg, n, hex.EncodeToString(h.Sum(nil))); err != nil {
		return err
	}
	return out.Sync()
}

func checkPackagePin(pkg NuGetPackage) error {
	hash, err := hex.DecodeString(pkg.SHA256)
	if err != nil || len(hash) != sha256.Size {
		return errors.New("a valid expected package SHA256 is required")
	}
	if pkg.Size < 0 || pkg.Size > maxPackageSize {
		return errors.New("invalid expected package size")
	}
	return nil
}

func packageSizeLimit(pkg NuGetPackage) int64 {
	if pkg.Size > 0 {
		return pkg.Size
	}
	return maxPackageSize
}

func checkDigest(pkg NuGetPackage, size int64, actual string) error {
	if size > maxPackageSize || (pkg.Size > 0 && size != pkg.Size) {
		return fmt.Errorf("package size mismatch: got %d, expected %d (maximum %d)", size, pkg.Size, maxPackageSize)
	}
	if !strings.EqualFold(actual, pkg.SHA256) {
		return fmt.Errorf("package SHA256 mismatch: expected %s, got %s", pkg.SHA256, actual)
	}
	return nil
}

func verifyPackage(ctx context.Context, path string, pkg NuGetPackage) error {
	if err := checkPackagePin(pkg); err != nil {
		return err
	}
	f, err := openRegular(path)
	if err != nil {
		return err
	}
	defer f.Close()
	h := sha256.New()
	n, err := io.Copy(h, io.LimitReader(contextReader{ctx, f}, packageSizeLimit(pkg)+1))
	if err != nil {
		return err
	}
	return checkDigest(pkg, n, hex.EncodeToString(h.Sum(nil)))
}

type contextReader struct {
	ctx context.Context
	r   io.Reader
}

func (r contextReader) Read(p []byte) (int, error) {
	if err := r.ctx.Err(); err != nil {
		return 0, err
	}
	return r.r.Read(p)
}

func extractPackage(ctx context.Context, zipPath, destDir string, spec packageSpec) error {
	r, err := zip.OpenReader(zipPath)
	if err != nil {
		return fmt.Errorf("open nupkg: %w", err)
	}
	defer r.Close()
	if err := preflightZIP(r.File); err != nil {
		return err
	}
	expected := make(map[string]approvedFile)
	for _, file := range spec.Files {
		expected[file.Source] = file
	}
	selected := make(map[string]*zip.File)
	for _, f := range r.File {
		if file, ok := expected[f.Name]; ok {
			if f.FileInfo().IsDir() || f.UncompressedSize64 != uint64(file.Size) {
				return fmt.Errorf("invalid approved entry size/type: %s", f.Name)
			}
			selected[f.Name] = f
		} else if strings.HasPrefix(strings.ToLower(f.Name), nativePrefix) && !f.FileInfo().IsDir() {
			ext := strings.ToLower(filepath.Ext(f.Name))
			if ext != ".lib" && ext != ".pdb" {
				return fmt.Errorf("unapproved native entry: %s", f.Name)
			}
		}
	}
	for _, file := range spec.Files {
		if selected[file.Source] == nil {
			return fmt.Errorf("required package entry missing: %s", file.Source)
		}
	}
	if err := makeSafeDir(destDir); err != nil {
		return err
	}
	for _, file := range spec.Files {
		destPath := filepath.Join(destDir, filepath.FromSlash(file.Path))
		if err := makeSafeDir(filepath.Dir(destPath)); err != nil {
			return err
		}
		if err := extractZIPFile(ctx, selected[file.Source], destPath, file); err != nil {
			return fmt.Errorf("extract %s: %w", file.Source, err)
		}
		if err := validateNative(destPath); err != nil {
			return err
		}
	}
	return nil
}

func preflightZIP(files []*zip.File) error {
	if len(files) == 0 || len(files) > maxZIPEntries {
		return errors.New("ZIP entry count outside permitted bounds")
	}
	type entry struct {
		name      string
		directory bool
	}
	seen := make(map[string]entry)
	var expanded uint64
	for _, f := range files {
		if err := safeArchiveName(f.Name); err != nil {
			return err
		}
		mode := f.Mode()
		if !mode.IsRegular() && !mode.IsDir() {
			return fmt.Errorf("non-regular ZIP entry: %s", f.Name)
		}
		key := strings.ToLower(strings.TrimSuffix(f.Name, "/"))
		if _, exists := seen[key]; exists {
			return fmt.Errorf("duplicate or case-aliased ZIP entry: %s", f.Name)
		}
		seen[key] = entry{strings.TrimSuffix(f.Name, "/"), mode.IsDir()}
		if f.UncompressedSize64 > maxFileSize || f.CompressedSize64 > maxPackageSize {
			return fmt.Errorf("oversize ZIP entry: %s", f.Name)
		}
		if f.UncompressedSize64 > 1<<20 && (f.CompressedSize64 == 0 || f.UncompressedSize64/f.CompressedSize64 > 250) {
			return fmt.Errorf("ZIP compression ratio exceeds limit: %s", f.Name)
		}
		expanded += f.UncompressedSize64
		if expanded > maxExpandedSize {
			return errors.New("ZIP expanded size exceeds limit")
		}
	}
	for _, e := range seen {
		parts := strings.Split(e.name, "/")
		for i := 1; i < len(parts); i++ {
			name := strings.Join(parts[:i], "/")
			if ancestor, ok := seen[strings.ToLower(name)]; ok && (!ancestor.directory || ancestor.name != name) {
				return fmt.Errorf("ZIP file/directory or case conflict: %s", e.name)
			}
		}
	}
	// Implicit directories can also alias (e.g. A/a.dll and a/b.dll).
	dirs := make(map[string]string)
	for _, e := range seen {
		parts := strings.Split(e.name, "/")
		for i := 1; i < len(parts); i++ {
			name := strings.Join(parts[:i], "/")
			key := strings.ToLower(name)
			if old, ok := dirs[key]; ok && old != name {
				return fmt.Errorf("case-aliased ZIP directory: %s", name)
			}
			dirs[key] = name
		}
	}
	return nil
}

func safeArchiveName(name string) error {
	if name == "" || !utf8.ValidString(name) || strings.ContainsAny(name, "\\:") || strings.HasPrefix(name, "/") || len(name) > 512 {
		return fmt.Errorf("unsafe ZIP path: %q", name)
	}
	for _, part := range strings.Split(strings.TrimSuffix(name, "/"), "/") {
		if part == "" || part == "." || part == ".." || strings.TrimRight(part, ". ") != part {
			return fmt.Errorf("unsafe ZIP path component: %q", name)
		}
		for _, c := range part {
			if c < 32 || c > 126 || strings.ContainsRune("<>\"|?*", c) {
				return fmt.Errorf("unsafe Windows ZIP path: %q", name)
			}
		}
		base := strings.ToUpper(strings.TrimRight(strings.SplitN(part, ".", 2)[0], " "))
		if base == "CON" || base == "PRN" || base == "AUX" || base == "NUL" ||
			(len(base) == 4 && (strings.HasPrefix(base, "COM") || strings.HasPrefix(base, "LPT")) && base[3] >= '0' && base[3] <= '9') {
			return fmt.Errorf("reserved Windows ZIP path: %q", name)
		}
	}
	return nil
}

func extractZIPFile(ctx context.Context, z *zip.File, path string, expected approvedFile) error {
	in, err := z.Open()
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	h := sha256.New()
	n, copyErr := io.Copy(io.MultiWriter(out, h), io.LimitReader(contextReader{ctx, in}, expected.Size+1))
	if copyErr == nil && (n != expected.Size || hex.EncodeToString(h.Sum(nil)) != expected.SHA256) {
		copyErr = errors.New("installed entry size or SHA256 mismatch")
	}
	if copyErr == nil {
		copyErr = out.Sync()
	}
	closeErr := out.Close()
	if copyErr != nil {
		return copyErr
	}
	return closeErr
}

func makeSafeDir(path string) error {
	path, err := filepath.Abs(path)
	if err != nil {
		return err
	}
	parent := filepath.Dir(path)
	if parent != path {
		if err := makeSafeDir(parent); err != nil {
			return err
		}
	}
	info, err := os.Lstat(path)
	if errors.Is(err, os.ErrNotExist) {
		if err := os.Mkdir(path, 0o755); err != nil && !errors.Is(err, os.ErrExist) {
			return err
		}
		info, err = os.Lstat(path)
	}
	if err != nil {
		return err
	}
	if !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return fmt.Errorf("not a real directory (links are not allowed): %s", path)
	}
	return nil
}

func openRegular(path string) (*os.File, error) {
	info, err := os.Lstat(path)
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() {
		return nil, fmt.Errorf("not a regular file: %s", path)
	}
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	actual, err := f.Stat()
	if err != nil || !os.SameFile(info, actual) {
		f.Close()
		return nil, fmt.Errorf("file changed while opening: %s", path)
	}
	return f, nil
}
