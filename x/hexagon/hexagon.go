// Package hexagon finds and installs the signed Hexagon NPU libraries that
// llama.cpp's Hexagon backend runs on the NPU.
//
// Windows loads the HTP ops libraries (libggml-htp-v*.so) only when a trusted
// catalog signs them, and they must come from the same llama.cpp commit as the
// host backend (ggml-hexagon). Qualcomm publishes a Microsoft-signed set in its
// GenieX releases, but its terms attach use restrictions to it, so Ollama does
// not ship the set: `ollama npu setup --provider hexagon` downloads it from
// Qualcomm's release into a per-user directory, and runners find it there.
package hexagon

import (
	"crypto/sha256"
	_ "embed"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
)

// PathEnv lists directories holding the signed set, searched before the
// backend directory and the setup directory.
const PathEnv = "OLLAMA_HEXAGON_PATH"

// LicenseFilename is GenieX's licence, written next to the installed files
// because the release archives omit it.
const LicenseFilename = "GenieX-LICENSE.txt"

// ManifestFilename records where an installed set came from.
const ManifestFilename = "ollama_hexagon.json"

// catalogFilename marks a directory that holds, or claims to hold, a set.
const catalogFilename = "libggml-htp.cat"

//go:embed GenieX-LICENSE.txt
var license []byte

// Terms are the conditions Qualcomm attaches to the set beyond its BSD-3
// licence: its site terms bring in its Responsible AI License, which restricts
// how the libraries may be used.
var Terms = []string{
	"https://www.qualcomm.com/site/terms-of-use",
	"https://www.qualcomm.com/site/responsible-ai-license",
}

// Release pins a signed set and the archive it comes from.
type Release struct {
	Source        string // who publishes the set
	LlamaCppTag   string // llama.cpp tag the set was built from; must match LLAMA_CPP_VERSION
	URL           string
	ArchiveName   string
	ArchiveSize   int64
	ArchiveSHA256 string
	Prefix        string            // folder inside the archive that holds the files
	Files         map[string]string // file name -> SHA256
}

// Pinned is the set this build loads: Qualcomm GenieX v0.9.1, built from
// llama.cpp e467e5b (b11644). Tests may replace it.
var Pinned = Release{
	Source:        "Qualcomm GenieX v0.9.1",
	LlamaCppTag:   "b11644",
	URL:           "https://github.com/qualcomm/GenieX/releases/download/v0.9.1/geniex-sdk-windows-arm64-v0.9.1.zip",
	ArchiveName:   "geniex-sdk-windows-arm64-v0.9.1.zip",
	ArchiveSize:   91542153,
	ArchiveSHA256: "2ed2bf63f042086c525b442081ed8d7ad4ff76bcce8bad6eeabe1cf38cc0a1b5",
	Prefix:        "sdk-windows-arm64/lib/llama_cpp/",
	Files: map[string]string{
		"libggml-htp-v73.so": "e93616d06df4ec812a66ca2b05e3bd1706ef588926383b94091bebf007ed913e",
		"libggml-htp-v75.so": "44c064535682df7b71a9f21193aa83c55bc846f3db74f35eb1ac0ab1814d15ed",
		"libggml-htp-v79.so": "279187f7a4b0df53deedc18d265268f52ef0f20aa63140f8e150841861442620",
		"libggml-htp-v81.so": "3207bd9ce22844c1d9fb4f3ab1c3acd34819de9811f9b4337a7ad1a14965a71e",
		"libggml-htp.cat":    "bdd10a100e426eb48398a8bc5751c125579f6ec52c15e25106ba57e669444b41",
		"libggml-htp.inf":    "e19530fffe17a934f0bd5f35a9ef78935305c2349229b9b3552cddf68e42230d",
	},
}

func (r Release) names() []string {
	names := make([]string, 0, len(r.Files))
	for name := range r.Files {
		names = append(names, name)
	}
	slices.Sort(names)
	return names
}

// InstallDir is where `ollama npu setup --provider hexagon` puts the pinned
// set: a versioned per-user directory beside the QNN runtime's. It is empty
// where setup is unsupported.
func InstallDir() string {
	if runtime.GOOS != "windows" {
		return ""
	}
	base := os.Getenv("LOCALAPPDATA")
	if base == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return ""
		}
		base = filepath.Join(home, "AppData", "Local")
	}
	return filepath.Join(base, "Ollama", "runtimes", "hexagon-htp", "win-arm64", Pinned.LlamaCppTag)
}

// SearchDirs lists where runners look for the set: OLLAMA_HEXAGON_PATH, then
// the backend's own directory (local builds may bundle the set), then
// InstallDir.
func SearchDirs(backendDir string) []string {
	dirs := filepath.SplitList(os.Getenv(PathEnv))
	dirs = append(dirs, backendDir, InstallDir())
	var out []string
	seen := make(map[string]bool)
	for _, dir := range dirs {
		if dir == "" {
			continue
		}
		dir = filepath.Clean(dir)
		key := dir
		if runtime.GOOS == "windows" {
			key = strings.ToLower(key)
		}
		if !seen[key] {
			seen[key] = true
			out = append(out, dir)
		}
	}
	return out
}

// ErrNotInstalled reports that no search directory holds a set.
var ErrNotInstalled = errors.New("the signed Hexagon NPU libraries are not installed; run 'ollama npu setup --provider hexagon'")

// VerifiedDir returns the first directory holding a set, checked against
// Pinned. It refuses an invalid candidate instead of falling back to another.
func VerifiedDir(dirs []string) (string, error) {
	for _, dir := range dirs {
		if _, err := os.Lstat(filepath.Join(dir, catalogFilename)); errors.Is(err, os.ErrNotExist) {
			continue
		} else if err != nil {
			return "", err
		}
		if err := Validate(dir); err != nil {
			return "", fmt.Errorf("%s does not hold the pinned Hexagon NPU libraries (%s): %w", dir, Pinned.Source, err)
		}
		return dir, nil
	}
	return "", ErrNotInstalled
}

// Validate checks that dir holds every pinned file, as a regular file with the
// pinned hash.
func Validate(dir string) error {
	return validate(Pinned, dir)
}

func validate(rel Release, dir string) error {
	for _, name := range rel.names() {
		if err := checkFile(filepath.Join(dir, name), rel.Files[name]); err != nil {
			return err
		}
	}
	return nil
}

func checkFile(path, want string) error {
	info, err := os.Lstat(path)
	if err != nil {
		return err
	}
	if !info.Mode().IsRegular() {
		return fmt.Errorf("%s is not a regular file", path)
	}
	got, err := fileSHA256(path)
	if err != nil {
		return err
	}
	if got != want {
		return fmt.Errorf("%s has SHA256 %s, want %s", path, got, want)
	}
	return nil
}

func fileSHA256(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
