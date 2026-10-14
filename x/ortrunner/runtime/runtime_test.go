package runtime

import (
	"archive/zip"
	"bytes"
	"context"
	"crypto/sha256"
	"debug/pe"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

type zipEntry struct {
	name string
	data []byte
	mode os.FileMode
}

func testZIP(t *testing.T, entries []zipEntry) []byte {
	t.Helper()
	var b bytes.Buffer
	w := zip.NewWriter(&b)
	for _, entry := range entries {
		header := &zip.FileHeader{Name: entry.name, Method: zip.Deflate}
		if entry.mode != 0 {
			header.SetMode(entry.mode)
		}
		f, err := w.CreateHeader(header)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := f.Write(entry.data); err != nil {
			t.Fatal(err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	return b.Bytes()
}

func digest(b []byte) string {
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}

func testPE(t *testing.T, machine uint16) []byte {
	t.Helper()
	b := make([]byte, 1024)
	copy(b, "MZ")
	binary.LittleEndian.PutUint32(b[60:], 128)
	copy(b[128:], "PE\x00\x00")
	header := pe.FileHeader{
		Machine: machine, NumberOfSections: 1, SizeOfOptionalHeader: 240,
		Characteristics: pe.IMAGE_FILE_DLL | pe.IMAGE_FILE_EXECUTABLE_IMAGE | pe.IMAGE_FILE_LARGE_ADDRESS_AWARE,
	}
	optional := pe.OptionalHeader64{
		Magic: 0x20b, ImageBase: 0x180000000, SectionAlignment: 4096, FileAlignment: 512,
		SizeOfImage: 8192, SizeOfHeaders: 512, Subsystem: 3, NumberOfRvaAndSizes: 16,
	}
	var headers bytes.Buffer
	if err := binary.Write(&headers, binary.LittleEndian, header); err != nil {
		t.Fatal(err)
	}
	if err := binary.Write(&headers, binary.LittleEndian, optional); err != nil {
		t.Fatal(err)
	}
	copy(b[132:], headers.Bytes())
	section := b[392:432]
	copy(section, ".text")
	binary.LittleEndian.PutUint32(section[8:], 512)
	binary.LittleEndian.PutUint32(section[12:], 4096)
	binary.LittleEndian.PutUint32(section[16:], 512)
	binary.LittleEndian.PutUint32(section[20:], 512)
	return b
}

func fixture(t *testing.T, machine uint16) (packageSpec, []byte) {
	t.Helper()
	spec := packageSpec{Package: NuGetPackage{ID: "Fixture.QNN", Version: "1.0"}}
	var entries []zipEntry
	for _, name := range []string{"onnxruntime.dll", "onnxruntime-genai.dll", "onnxruntime_providers_qnn.dll"} {
		data := testPE(t, machine)
		source := nativePrefix + name
		entries = append(entries, zipEntry{source, data, 0})
		spec.Files = append(spec.Files, approvedFile{source, name, int64(len(data)), digest(data)})
	}
	for _, name := range []string{"LICENSE", "ThirdPartyNotices.txt"} {
		data := []byte("Fixture notice, preserved byte for byte: " + name)
		entries = append(entries, zipEntry{name, data, 0})
		spec.Files = append(spec.Files, approvedFile{name, "licenses/fixture.qnn/" + name, int64(len(data)), digest(data)})
	}
	data := testZIP(t, entries)
	spec.Package.SHA256, spec.Package.Size = digest(data), int64(len(data))
	return spec, data
}

func writeFile(t *testing.T, path string, data []byte) {
	t.Helper()
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
}

func offlineFixture(t *testing.T) (packageSpec, InstallConfig) {
	t.Helper()
	root := t.TempDir()
	packages := filepath.Join(root, "packages")
	if err := os.Mkdir(packages, 0o755); err != nil {
		t.Fatal(err)
	}
	spec, data := fixture(t, pe.IMAGE_FILE_MACHINE_ARM64)
	writeFile(t, filepath.Join(packages, packageFilename(spec.Package)), data)
	return spec, InstallConfig{Dir: filepath.Join(root, "installed"), PackageDir: packages}
}

func TestDownloadVerification(t *testing.T) {
	spec, data := fixture(t, pe.IMAGE_FILE_MACHINE_ARM64)
	for _, test := range []struct {
		name string
		pin  NuGetPackage
		body []byte
		code int
		want string
	}{
		{"valid", spec.Package, data, 200, ""},
		{"hash", NuGetPackage{SHA256: strings.Repeat("0", 64), Size: int64(len(data))}, data, 200, "SHA256 mismatch"},
		{"missing hash", NuGetPackage{}, data, 200, "SHA256 is required"},
		{"truncated", spec.Package, data[:len(data)-1], 200, "Content-Length"},
		{"HTTP error", spec.Package, data, 503, "HTTP 503"},
	} {
		t.Run(test.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(test.code)
				w.Write(test.body)
			}))
			defer server.Close()
			f, err := os.Create(filepath.Join(t.TempDir(), "download"))
			if err != nil {
				t.Fatal(err)
			}
			defer f.Close()
			err = downloadPackage(context.Background(), server.Client(), server.URL, test.pin, f)
			if test.want == "" && err != nil || test.want != "" && (err == nil || !strings.Contains(err.Error(), test.want)) {
				t.Fatalf("download error = %v; want %q", err, test.want)
			}
		})
	}
}

func TestDownloadBoundsAndCancellation(t *testing.T) {
	for _, mode := range []string{"oversize", "chunked oversize", "interrupt", "cancel", "deadline"} {
		t.Run(mode, func(t *testing.T) {
			spec, data := fixture(t, pe.IMAGE_FILE_MACHINE_ARM64)
			started := make(chan struct{})
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch mode {
				case "oversize":
					w.Header().Set("Content-Length", fmt.Sprint(maxPackageSize+1))
				case "chunked oversize":
					w.(http.Flusher).Flush()
					w.Write(append(data, 0))
				case "interrupt":
					w.Header().Set("Content-Length", fmt.Sprint(len(data)))
					w.Write(data[:10])
				case "cancel", "deadline":
					w.(http.Flusher).Flush()
					close(started)
					<-r.Context().Done()
				}
			}))
			defer server.Close()
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			if mode == "cancel" {
				go func() { <-started; cancel() }()
			} else if mode == "deadline" {
				var cancelDeadline context.CancelFunc
				ctx, cancelDeadline = context.WithTimeout(ctx, 40*time.Millisecond)
				defer cancelDeadline()
			}
			dir := t.TempDir()
			err := acquirePackage(ctx, server.Client(), server.URL, spec, dir)
			if err == nil {
				t.Fatal("unsafe/interrupted download succeeded")
			}
			if mode == "cancel" && !errors.Is(err, context.Canceled) {
				t.Fatalf("cancellation not preserved: %v", err)
			}
			if mode == "deadline" && !errors.Is(err, context.DeadlineExceeded) {
				t.Fatalf("deadline not preserved: %v", err)
			}
			entries, err := os.ReadDir(dir)
			if err != nil || len(entries) != 0 {
				t.Fatalf("download artifacts remain: %v, %v", entries, err)
			}
		})
	}
}

func TestPEAndDSPArchitecture(t *testing.T) {
	dir := t.TempDir()
	for _, test := range []struct {
		name string
		data []byte
		want string
	}{
		{"ARM64.dll", testPE(t, pe.IMAGE_FILE_MACHINE_ARM64), ""},
		{"AMD64.dll", testPE(t, pe.IMAGE_FILE_MACHINE_AMD64), "not a native PE ARM64"},
		{"notPE.dll", []byte("not a PE file"), "not a valid PE"},
		{"notELF.so", []byte("not an ELF"), "not a valid DSP ELF"},
	} {
		path := filepath.Join(dir, test.name)
		writeFile(t, path, test.data)
		err := validateNative(path)
		if test.want == "" && err != nil || test.want != "" && (err == nil || !strings.Contains(err.Error(), test.want)) {
			t.Fatalf("%s: %v; want %s", test.name, err, test.want)
		}
	}
	// Minimal ELF32 header, with no sections, is enough to identify the machine.
	for _, machine := range []uint16{164, 183} {
		b := make([]byte, 52)
		copy(b, "\x7fELF\x01\x01\x01")
		binary.LittleEndian.PutUint16(b[16:], 3)
		binary.LittleEndian.PutUint16(b[18:], machine)
		binary.LittleEndian.PutUint32(b[20:], 1)
		binary.LittleEndian.PutUint16(b[40:], 52)
		path := filepath.Join(dir, fmt.Sprintf("machine%d.so", machine))
		writeFile(t, path, b)
		err := validateNative(path)
		if machine == 164 && err != nil || machine != 164 && err == nil {
			t.Fatalf("ELF machine %d: %v", machine, err)
		}
	}
}

func TestExtractionDoesNotOverwrite(t *testing.T) {
	spec, data := fixture(t, pe.IMAGE_FILE_MACHINE_ARM64)
	root := t.TempDir()
	path := filepath.Join(root, "package.zip")
	writeFile(t, path, data)
	dest := filepath.Join(root, "dest")
	if err := os.Mkdir(dest, 0o755); err != nil {
		t.Fatal(err)
	}
	native := filepath.Join(dest, "onnxruntime.dll")
	writeFile(t, native, []byte("must not overwrite"))
	if err := extractPackage(context.Background(), path, dest, spec); !errors.Is(err, os.ErrExist) {
		t.Fatalf("existing native file not rejected: %v", err)
	}
	actual, err := os.ReadFile(native)
	if err != nil || string(actual) != "must not overwrite" {
		t.Fatal("existing file was modified")
	}
}

func TestExtractionHashMismatch(t *testing.T) {
	spec, data := fixture(t, pe.IMAGE_FILE_MACHINE_ARM64)
	spec.Files[0].SHA256 = strings.Repeat("0", 64)
	root := t.TempDir()
	writeFile(t, filepath.Join(root, "package.zip"), data)
	err := extractPackage(context.Background(), filepath.Join(root, "package.zip"), filepath.Join(root, "dest"), spec)
	if err == nil || !strings.Contains(err.Error(), "SHA256 mismatch") {
		t.Fatalf("extraction hash mismatch accepted: %v", err)
	}
}

func TestDownloadRedirectPolicy(t *testing.T) {
	client := downloadClient()
	defer client.CloseIdleConnections()
	for _, target := range []string{
		"https://globalcdn.nuget.org/packages/package.nupkg",
		"http://globalcdn.nuget.org/packages/package.nupkg",
		"https://example.invalid/package.nupkg",
		"https://globalcdn.nuget.org:8080/package.nupkg",
		"https://user@globalcdn.nuget.org/package.nupkg",
	} {
		req, err := http.NewRequest(http.MethodGet, target, nil)
		if err != nil {
			t.Fatal(err)
		}
		err = client.CheckRedirect(req, nil)
		if strings.HasPrefix(target, "https://globalcdn.nuget.org/") {
			if err != nil {
				t.Fatal(err)
			}
		} else if err == nil {
			t.Fatalf("unsafe redirect accepted: %s", target)
		}
	}
}

func TestUnsafeZIP(t *testing.T) {
	cases := map[string][]zipEntry{
		"traversal":                {{nativePrefix + "../escape.dll", []byte("x"), 0}},
		"absolute":                 {{"/escape.dll", nil, 0}},
		"drive":                    {{"C:/escape.dll", nil, 0}},
		"backslash":                {{`runtimes\win-arm64\native\bad.dll`, nil, 0}},
		"ADS":                      {{nativePrefix + "a.dll:stream", nil, 0}},
		"reserved":                 {{nativePrefix + "NUL.dll", nil, 0}},
		"trailing dot":             {{nativePrefix + "a.dll.", nil, 0}},
		"trailing space":           {{nativePrefix + "a.dll ", nil, 0}},
		"case alias":               {{nativePrefix + "A.dll", nil, 0}, {nativePrefix + "a.dll", nil, 0}},
		"duplicate":                {{nativePrefix + "a.dll", nil, 0}, {nativePrefix + "a.dll", nil, 0}},
		"file directory conflict":  {{"a", nil, 0}, {"a/b", nil, 0}},
		"implicit directory alias": {{"a/b", nil, 0}, {"A/c", nil, 0}},
		"symlink":                  {{nativePrefix + "a.dll", []byte("target"), os.ModeSymlink | 0o777}},
	}
	for name, entries := range cases {
		t.Run(name, func(t *testing.T) {
			root := t.TempDir()
			path := filepath.Join(root, "package.zip")
			writeFile(t, path, testZIP(t, entries))
			spec, _ := fixture(t, pe.IMAGE_FILE_MACHINE_ARM64)
			if err := extractPackage(context.Background(), path, filepath.Join(root, "dest"), spec); err == nil {
				t.Fatal("unsafe ZIP accepted")
			}
			if _, err := os.Stat(filepath.Join(root, "dest")); !errors.Is(err, os.ErrNotExist) {
				t.Fatalf("unsafe ZIP wrote destination: %v", err)
			}
		})
	}
}

func TestZIPBombBounds(t *testing.T) {
	for _, files := range [][]*zip.File{
		{{FileHeader: zip.FileHeader{Name: "big", UncompressedSize64: maxFileSize + 1}}},
		{{FileHeader: zip.FileHeader{Name: "ratio", UncompressedSize64: 2 << 20, CompressedSize64: 10}}},
		make([]*zip.File, maxZIPEntries+1),
	} {
		if err := preflightZIP(files); err == nil {
			t.Fatal("ZIP bomb accepted")
		}
	}
	var files []*zip.File
	for i := 0; i < 5; i++ {
		files = append(files, &zip.File{FileHeader: zip.FileHeader{Name: fmt.Sprint(i), UncompressedSize64: maxFileSize, CompressedSize64: maxPackageSize}})
	}
	if err := preflightZIP(files); err == nil {
		t.Fatal("aggregate ZIP bomb accepted")
	}
}

func TestMissingAndUnapprovedNative(t *testing.T) {
	spec, data := fixture(t, pe.IMAGE_FILE_MACHINE_ARM64)
	r, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		t.Fatal(err)
	}
	var entries []zipEntry
	for _, f := range r.File {
		if !strings.HasSuffix(f.Name, "onnxruntime_providers_qnn.dll") {
			if strings.HasPrefix(f.Name, nativePrefix) {
				entries = append(entries, zipEntry{f.Name, testPE(t, pe.IMAGE_FILE_MACHINE_ARM64), 0})
			} else {
				entries = append(entries, zipEntry{f.Name, []byte("Fixture notice, preserved byte for byte: " + f.Name), 0})
			}
		}
	}
	for _, unapproved := range []bool{false, true} {
		t.Run(fmt.Sprint(unapproved), func(t *testing.T) {
			current := append([]zipEntry(nil), entries...)
			want := "required package entry missing"
			if unapproved {
				current = append(current, zipEntry{nativePrefix + "unexpected.dll", nil, 0})
				want = "unapproved native entry"
			}
			path := filepath.Join(t.TempDir(), "package.zip")
			writeFile(t, path, testZIP(t, current))
			err := extractPackage(context.Background(), path, filepath.Join(t.TempDir(), "dest"), spec)
			if err == nil || !strings.Contains(err.Error(), want) {
				t.Fatalf("error = %v; want %s", err, want)
			}
		})
	}
}

func TestInstallOfflineImmutableAndCorruption(t *testing.T) {
	spec, cfg := offlineFixture(t)
	specs := []packageSpec{spec}
	noNetwork := func(NuGetPackage) string { t.Fatal("offline acquisition tried network"); return "" }
	result, err := install(context.Background(), cfg, specs, nil, noNetwork)
	if err != nil || !result.Installed || !result.Validated {
		t.Fatalf("install: %v, %v", result, err)
	}
	report, err := validate(context.Background(), cfg.Dir, specs)
	if err != nil || !report.Valid || len(report.VerifiedFiles) != len(spec.Files) {
		t.Fatalf("validation: %+v, %v", report, err)
	}
	for _, file := range spec.Files {
		if strings.HasPrefix(file.Path, "licenses/") {
			data, err := os.ReadFile(filepath.Join(cfg.Dir, filepath.FromSlash(file.Path)))
			if err != nil || digest(data) != file.SHA256 {
				t.Fatalf("notice not preserved: %s, %v", file.Path, err)
			}
		}
	}
	path := filepath.Join(cfg.Dir, "onnxruntime.dll")
	before, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	cfg.Force = true
	result, err = install(context.Background(), cfg, specs, nil, noNetwork)
	after, statErr := os.Stat(path)
	if err != nil || statErr != nil || result.Installed || !os.SameFile(before, after) {
		t.Fatalf("force replaced valid installation: %v, %v", result, err)
	}
	bad := testPE(t, pe.IMAGE_FILE_MACHINE_ARM64)
	bad[len(bad)-1] = 1
	writeFile(t, path, bad)
	// Rewriting the manifest to match a corrupt native file cannot bless it.
	manifest := newManifest(specs)
	manifest.Files[0].SHA256 = digest(bad)
	b, err := json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	writeFile(t, filepath.Join(cfg.Dir, ManifestFilename), b)
	report, err = validate(context.Background(), cfg.Dir, specs)
	if err == nil || report.Valid || report.ManifestValid || len(report.InvalidFiles) < 2 {
		t.Fatalf("corrupt installation accepted: %+v, %v", report, err)
	}
	result, err = install(context.Background(), cfg, specs, nil, noNetwork)
	if !errors.Is(err, ErrUnsafeReplacement) || result.Validated {
		t.Fatalf("unsafe force replacement not rejected: %v, %v", result, err)
	}
	stillBad, err := os.ReadFile(path)
	if err != nil || !bytes.Equal(stillBad, bad) {
		t.Fatal("invalid existing installation was modified")
	}
}

func TestArchitectureAndInterruptedInstall(t *testing.T) {
	for _, test := range []string{"wrong architecture", "canceled", "missing offline", "bad offline hash", "missing installed provider", "extra installed DLL"} {
		t.Run(test, func(t *testing.T) {
			spec, cfg := offlineFixture(t)
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			switch test {
			case "wrong architecture":
				var data []byte
				spec, data = fixture(t, pe.IMAGE_FILE_MACHINE_AMD64)
				writeFile(t, filepath.Join(cfg.PackageDir, packageFilename(spec.Package)), data)
			case "canceled":
				cancel()
			case "missing offline":
				if err := os.Remove(filepath.Join(cfg.PackageDir, packageFilename(spec.Package))); err != nil {
					t.Fatal(err)
				}
			case "bad offline hash":
				writeFile(t, filepath.Join(cfg.PackageDir, packageFilename(spec.Package)), []byte("corrupt"))
			}
			_, err := install(ctx, cfg, []packageSpec{spec}, nil, nil)
			if test == "missing installed provider" || test == "extra installed DLL" {
				if err != nil {
					t.Fatal(err)
				}
				if test == "missing installed provider" {
					if err := os.Remove(filepath.Join(cfg.Dir, "onnxruntime_providers_qnn.dll")); err != nil {
						t.Fatal(err)
					}
				} else {
					writeFile(t, filepath.Join(cfg.Dir, "unapproved.dll"), testPE(t, pe.IMAGE_FILE_MACHINE_ARM64))
				}
				report, err := validate(context.Background(), cfg.Dir, []packageSpec{spec})
				if err == nil || report.Valid {
					t.Fatal("incomplete/augmented installation accepted")
				}
				return
			}
			if err == nil {
				t.Fatal("invalid/interrupted installation succeeded")
			}
			if _, err := os.Stat(cfg.Dir); !errors.Is(err, os.ErrNotExist) {
				t.Fatal("failed install was promoted")
			}
			stages, err := filepath.Glob(filepath.Join(filepath.Dir(cfg.Dir), ".ortgenai-install-*"))
			if err != nil || len(stages) != 0 {
				t.Fatalf("staging artifacts remain: %v, %v", stages, err)
			}
		})
	}
}

func TestConcurrentInstall(t *testing.T) {
	spec, data := fixture(t, pe.IMAGE_FILE_MACHINE_ARM64)
	var downloads atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		downloads.Add(1)
		time.Sleep(40 * time.Millisecond)
		w.Write(data)
	}))
	defer server.Close()
	cfg := InstallConfig{Dir: filepath.Join(t.TempDir(), "installed"), Force: true}
	var group sync.WaitGroup
	var installed atomic.Int32
	errors := make(chan error, 8)
	for i := 0; i < 8; i++ {
		group.Go(func() {
			result, err := install(context.Background(), cfg, []packageSpec{spec}, server.Client(), func(NuGetPackage) string { return server.URL })
			if err != nil {
				errors <- err
			} else if result.Installed {
				installed.Add(1)
			}
		})
	}
	group.Wait()
	close(errors)
	for err := range errors {
		t.Error(err)
	}
	if installed.Load() != 1 || downloads.Load() != 1 {
		t.Fatalf("installed %d times, downloaded %d times", installed.Load(), downloads.Load())
	}
}

func TestCrossProcessLock(t *testing.T) {
	if path := os.Getenv("OLLAMA_RUNTIME_TEST_LOCK"); path != "" {
		unlock, err := acquireInstallLock(context.Background(), path)
		if err != nil {
			t.Fatal(err)
		}
		defer unlock()
		fmt.Println("LOCK-ACQUIRED")
		return
	}
	path := filepath.Join(t.TempDir(), "runtime.lock")
	unlock, err := acquireInstallLock(context.Background(), path)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 40*time.Millisecond)
	defer cancel()
	if release, err := acquireInstallLock(ctx, path); !errors.Is(err, context.DeadlineExceeded) {
		if release != nil {
			release()
		}
		t.Fatalf("held lock did not honor context: %v", err)
	}
	childCtx, childCancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer childCancel()
	cmd := exec.CommandContext(childCtx, os.Args[0], "-test.run=^TestCrossProcessLock$", "-test.v")
	cmd.Env = append(os.Environ(), "OLLAMA_RUNTIME_TEST_LOCK="+path)
	var output bytes.Buffer
	cmd.Stdout, cmd.Stderr = &output, &output
	if err := cmd.Start(); err != nil {
		unlock()
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	select {
	case err := <-done:
		unlock()
		t.Fatalf("child acquired held lock: %v, %s", err, output.String())
	case <-time.After(100 * time.Millisecond):
	}
	unlock()
	if err := <-done; err != nil || !strings.Contains(output.String(), "LOCK-ACQUIRED") {
		t.Fatalf("child failed after release: %v, %s", err, output.String())
	}
}

func TestCrossProcessSetup(t *testing.T) {
	if root := os.Getenv("OLLAMA_RUNTIME_TEST_SETUP"); root != "" {
		spec, _ := fixture(t, pe.IMAGE_FILE_MACHINE_ARM64)
		cfg := InstallConfig{Dir: filepath.Join(root, "installed"), PackageDir: filepath.Join(root, "packages"), Force: true}
		result, err := install(context.Background(), cfg, []packageSpec{spec}, nil, nil)
		if err != nil || !result.Validated {
			t.Fatalf("child setup: %v, %v", result, err)
		}
		if result.Installed {
			fmt.Println("NEW-INSTALL")
		}
		return
	}
	spec, cfg := offlineFixture(t)
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	var commands []*exec.Cmd
	var outputs []*bytes.Buffer
	for i := 0; i < 3; i++ {
		cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestCrossProcessSetup$", "-test.v")
		cmd.Env = append(os.Environ(), "OLLAMA_RUNTIME_TEST_SETUP="+filepath.Dir(cfg.Dir))
		output := new(bytes.Buffer)
		cmd.Stdout, cmd.Stderr = output, output
		if err := cmd.Start(); err != nil {
			t.Fatal(err)
		}
		commands, outputs = append(commands, cmd), append(outputs, output)
	}
	newInstalls := 0
	for i, cmd := range commands {
		if err := cmd.Wait(); err != nil {
			t.Fatalf("child setup: %v, %s", err, outputs[i].String())
		}
		newInstalls += strings.Count(outputs[i].String(), "NEW-INSTALL")
	}
	if report, err := validate(context.Background(), cfg.Dir, []packageSpec{spec}); err != nil || !report.Valid || newInstalls != 1 {
		t.Fatalf("concurrent cross-process setup: %+v, %v; new installs %d", report, err, newInstalls)
	}
}

func TestRejectUnsupportedProvider(t *testing.T) {
	for _, provider := range []string{ProviderDML, ProviderCPU, "unknown"} {
		_, err := install(context.Background(), InstallConfig{Provider: provider, Dir: filepath.Join(t.TempDir(), "installed")}, nil, nil, nil)
		if err == nil {
			t.Fatalf("provider %q accepted", provider)
		}
	}
}

func TestPublicPinsCannotWeakenAcquisition(t *testing.T) {
	original := append([]NuGetPackage(nil), PinnedPackages...)
	t.Cleanup(func() { PinnedPackages = original })
	PinnedPackages[0].SHA256 = strings.Repeat("0", 64)
	dir := filepath.Join(t.TempDir(), "not-created")
	err := DownloadAndExtractNuGetContext(context.Background(), PinnedPackages[0], dir)
	if err == nil || !strings.Contains(err.Error(), "not an exact approved pin") {
		t.Fatalf("mutable public package pin accepted: %v", err)
	}
	if _, err := os.Stat(dir); !errors.Is(err, os.ErrNotExist) {
		t.Fatal("unapproved package changed the destination")
	}
	if approvedPackages()[0].Package.SHA256 != original[0].SHA256 {
		t.Fatal("mutable public pins changed private trust anchors")
	}
}

func TestExplicitInstallationRetainsPermissionsAndIdentity(t *testing.T) {
	spec, cfg := offlineFixture(t)
	if _, err := install(context.Background(), cfg, []packageSpec{spec}, nil, nil); err != nil {
		t.Fatal(err)
	}
	dirBefore, err := os.Stat(cfg.Dir)
	if err != nil {
		t.Fatal(err)
	}
	filePath := filepath.Join(cfg.Dir, "onnxruntime.dll")
	fileBefore, err := os.Stat(filePath)
	if err != nil {
		t.Fatal(err)
	}
	cfg.Force = true
	result, err := install(context.Background(), cfg, []packageSpec{spec}, nil, nil)
	dirAfter, dirErr := os.Stat(cfg.Dir)
	fileAfter, fileErr := os.Stat(filePath)
	if err != nil || dirErr != nil || fileErr != nil || result.Installed || !result.Validated {
		t.Fatalf("explicit force revalidation: %+v, %v, %v, %v", result, err, dirErr, fileErr)
	}
	if !os.SameFile(dirBefore, dirAfter) || !os.SameFile(fileBefore, fileAfter) ||
		dirBefore.Mode() != dirAfter.Mode() || fileBefore.Mode() != fileAfter.Mode() {
		t.Fatal("explicit installation revalidation replaced files or changed permissions")
	}
}

func TestSearchDirsExplicitOrderAndDeduplication(t *testing.T) {
	first, second := t.TempDir(), t.TempDir()
	alias := first
	if runtime.GOOS == "windows" {
		alias = strings.ToUpper(first)
	}
	t.Setenv("OLLAMA_ORT_PATH", strings.Join([]string{first, second, alias, ""}, string(os.PathListSeparator)))
	dirs := SearchDirs()
	if len(dirs) < 2 || dirs[0] != first || dirs[1] != second {
		t.Fatalf("explicit paths lost priority: %v", dirs)
	}
	seen := map[string]bool{}
	for _, dir := range dirs {
		key := dir
		if runtime.GOOS == "windows" {
			key = strings.ToLower(key)
		}
		if dir == "" || seen[key] {
			t.Fatalf("empty or duplicate search directory: %v", dirs)
		}
		seen[key] = true
	}
	if runtime.GOOS == "windows" && dirs[2] != InstallDir() {
		t.Fatalf("versioned installation lost priority over executable directory: %v", dirs)
	}
}

func TestValidationCanceledBeforeIO(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	report, err := ValidateInstallDirContext(ctx, filepath.Join(t.TempDir(), "missing"))
	if !errors.Is(err, context.Canceled) || report.Valid {
		t.Fatalf("validation ignored cancellation: %+v, %v", report, err)
	}
}

func TestOfficialPackageInventory(t *testing.T) {
	dir := os.Getenv("OLLAMA_RUNTIME_TEST_PACKAGES")
	if dir == "" {
		t.Skip("optional offline verification of independently downloaded official packages")
	}
	// Uses a private test destination, never the user's runtime directory.
	stage := t.TempDir()
	for i, spec := range approvedPackages() {
		filename := []string{"genai.nupkg", "ort.nupkg"}[i]
		path := filepath.Join(dir, filename)
		if err := verifyPackage(context.Background(), path, spec.Package); err != nil {
			t.Fatal(err)
		}
		if err := extractPackage(context.Background(), path, stage, spec); err != nil {
			t.Fatal(err)
		}
	}
	if err := writeManifest(stage, newManifest(approvedPackages())); err != nil {
		t.Fatal(err)
	}
	if report, err := ValidateInstallDir(stage); err != nil || !report.Valid {
		t.Fatalf("official pinned bundle failed validation: %+v, %v", report, err)
	}
	if len(RequiredNativeFiles()) != 15 || !reflect.DeepEqual(PinnedPackages, pinnedPackages()) {
		t.Fatal("pinned inventory changed")
	}
	if runtime.GOOS == "windows" && runtime.GOARCH == "arm64" {
		root := t.TempDir()
		packages := filepath.Join(root, "packages")
		if err := os.Mkdir(packages, 0o755); err != nil {
			t.Fatal(err)
		}
		for i, spec := range approvedPackages() {
			data, err := os.ReadFile(filepath.Join(dir, []string{"genai.nupkg", "ort.nupkg"}[i]))
			if err != nil {
				t.Fatal(err)
			}
			writeFile(t, filepath.Join(packages, packageFilename(spec.Package)), data)
		}
		cfg := Config{Dir: filepath.Join(root, "installed"), PackageDir: packages}
		result, err := Setup(cfg)
		if err != nil || !result.Installed || !result.Validated || !result.Report.Valid {
			t.Fatalf("public offline Setup: %+v, %v", result, err)
		}
		cfg.Force = true
		result, err = SetupContext(context.Background(), cfg)
		if err != nil || result.Installed || !result.Validated {
			t.Fatalf("public immutable force Setup: %+v, %v", result, err)
		}
	}
}
