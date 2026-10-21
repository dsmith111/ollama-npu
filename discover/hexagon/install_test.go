package hexagon

import (
	"archive/zip"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// testRelease writes an archive like GenieX's, with the set under a prefix and
// an unrelated file beside it, and returns a release pinning it.
func testRelease(t *testing.T) (Release, string) {
	t.Helper()
	rel := Release{Source: "test", LlamaCppTag: "b1", License: "test licence", ArchiveName: "set.zip",
		Prefix: "sdk/lib/llama_cpp/", Files: map[string]string{}}
	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)
	for _, name := range []string{"libggml-htp-v73.so", "libggml-htp-v81.so", catalogFilename, "other.dll"} {
		content := []byte("contents of " + name)
		w, err := zw.Create(rel.Prefix + name)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := w.Write(content); err != nil {
			t.Fatal(err)
		}
		if name != "other.dll" {
			sum := sha256.Sum256(content)
			rel.Files[name] = hex.EncodeToString(sum[:])
		}
	}
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
	pkgDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(pkgDir, rel.ArchiveName), buf.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(buf.Bytes())
	rel.ArchiveSize, rel.ArchiveSHA256 = int64(buf.Len()), hex.EncodeToString(sum[:])
	return rel, pkgDir
}

func TestInstallFromAnArchiveAndKeepTheResult(t *testing.T) {
	rel, pkgDir := testRelease(t)
	usePinned(t, rel)
	dir := filepath.Join(t.TempDir(), "runtimes", "hexagon-htp")
	result, err := install(t.Context(), rel, dir, pkgDir, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !result.Installed || !result.Validated {
		t.Fatalf("result = %+v, want installed and validated", result)
	}
	if err := Validate(dir); err != nil {
		t.Fatal(err)
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, e := range entries {
		names = append(names, e.Name())
	}
	want := []string{"libggml-htp-v73.so", "libggml-htp-v81.so", catalogFilename, ManifestFilename}
	if strings.Join(names, ",") != strings.Join(want, ",") {
		t.Fatalf("installed %v, want %v (only the pinned files and the manifest)", names, want)
	}
	var manifest map[string]any
	data, _ := os.ReadFile(filepath.Join(dir, ManifestFilename))
	if err := json.Unmarshal(data, &manifest); err != nil || manifest["archive_sha256"] != rel.ArchiveSHA256 ||
		manifest["license"] != rel.License {
		t.Fatalf("manifest = %s, %v", data, err)
	}
	staging, _ := filepath.Glob(filepath.Join(filepath.Dir(dir), ".hexagon-htp-install-*"))
	if len(staging) != 0 {
		t.Fatalf("staging left behind: %v", staging)
	}

	again, err := install(t.Context(), rel, dir, pkgDir, nil)
	if err != nil || again.Installed || !again.Validated {
		t.Fatalf("second install = %+v, %v; want the existing set kept and verified", again, err)
	}
}

func TestInstallDownloadsTheArchive(t *testing.T) {
	rel, pkgDir := testRelease(t)
	archive, err := os.ReadFile(filepath.Join(pkgDir, rel.ArchiveName))
	if err != nil {
		t.Fatal(err)
	}
	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/"+rel.ArchiveName {
			http.NotFound(w, r)
			return
		}
		w.Write(archive)
	}))
	defer server.Close()
	rel.URL = server.URL + "/" + rel.ArchiveName
	dir := filepath.Join(t.TempDir(), "htp")
	if _, err := install(t.Context(), rel, dir, "", server.Client()); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(dir, rel.ArchiveName)); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("the downloaded archive was kept in the installation: %v", err)
	}

	rel.URL = server.URL + "/missing.zip"
	if _, err := install(t.Context(), rel, filepath.Join(t.TempDir(), "htp"), "", server.Client()); err == nil ||
		!strings.Contains(err.Error(), "404") {
		t.Fatalf("missing archive error = %v", err)
	}
}

func TestInstallRefusesBadArchivesAndForeignDirectories(t *testing.T) {
	rel, pkgDir := testRelease(t)
	parent := t.TempDir()

	tampered := rel
	tampered.ArchiveSHA256 = strings.Repeat("0", 64)
	if _, err := install(t.Context(), tampered, filepath.Join(parent, "a"), pkgDir, nil); err == nil {
		t.Fatal("installed from an archive with the wrong hash")
	}
	wrongFile := rel
	wrongFile.Files = map[string]string{catalogFilename: strings.Repeat("0", 64)}
	if _, err := install(t.Context(), wrongFile, filepath.Join(parent, "b"), pkgDir, nil); err == nil {
		t.Fatal("installed a file with the wrong hash")
	}
	missing := rel
	missing.Files = map[string]string{"libggml-htp-v99.so": rel.Files[catalogFilename]}
	if _, err := install(t.Context(), missing, filepath.Join(parent, "c"), pkgDir, nil); err == nil {
		t.Fatal("installed although the archive lacks a pinned file")
	}
	for _, name := range []string{"a", "b", "c"} {
		if _, err := os.Stat(filepath.Join(parent, name)); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("a failed install created %s", name)
		}
	}

	foreign := filepath.Join(parent, "foreign")
	if err := os.MkdirAll(foreign, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(foreign, "keep.txt"), []byte("mine"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := install(t.Context(), rel, foreign, pkgDir, nil); err == nil {
		t.Fatal("replaced a directory that does not hold the pinned set")
	}
	if data, err := os.ReadFile(filepath.Join(foreign, "keep.txt")); err != nil || string(data) != "mine" {
		t.Fatalf("existing directory changed: %q, %v", data, err)
	}
}

func TestDownloadsFollowOnlyGitHubReleaseRedirects(t *testing.T) {
	client := downloadClient()
	for target, ok := range map[string]bool{
		"https://release-assets.githubusercontent.com/github-production-release-asset/1": true,
		"https://objects.githubusercontent.com/github-production-release-asset-2e65be/1": true,
		"http://objects.githubusercontent.com/insecure":                                  false,
		"https://example.com/set.zip":                                                    false,
	} {
		u, _ := url.Parse(target)
		err := client.CheckRedirect(&http.Request{URL: u}, []*http.Request{{}})
		if (err == nil) != ok {
			t.Errorf("redirect to %s: err = %v, want allowed=%t", target, err, ok)
		}
	}
}
