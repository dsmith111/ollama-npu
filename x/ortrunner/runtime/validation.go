package runtime

import (
	"context"
	"crypto/sha256"
	"debug/elf"
	"debug/pe"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"reflect"
	"strings"
)

const ManifestFilename = "runtime-manifest.json"

type ManifestFile struct {
	Path      string
	PackageID string
	Source    string
	Size      int64
	SHA256    string
}

// Manifest records package provenance and hashes of every installed native file
// and notice. Validation compares it with compiled trust anchors, not just itself.
type Manifest struct {
	Schema   int
	OS       string
	Arch     string
	Provider string
	Packages []NuGetPackage
	Files    []ManifestFile
}

type ValidationReport struct {
	Dir                  string   `json:"dir"`
	Provider             string   `json:"provider"`
	Valid                bool     `json:"valid"`
	ManifestValid        bool     `json:"manifest_valid"`
	VerifiedFiles        []string `json:"verified_files"`
	MissingFiles         []string `json:"missing_files"`
	InvalidFiles         []string `json:"invalid_files"`
	Problems             []string `json:"problems"`
	ExternalDependencies []string `json:"external_dependencies"` // inventory only, not a device readiness test
}

func ValidateInstallDir(dir string) (ValidationReport, error) {
	return ValidateInstallDirContext(context.Background(), dir)
}

func ValidateInstallDirContext(ctx context.Context, dir string) (ValidationReport, error) {
	return validate(ctx, dir, approvedPackages())
}

func newManifest(specs []packageSpec) Manifest {
	m := Manifest{Schema: 1, OS: "windows", Arch: "arm64", Provider: ProviderQNN}
	for _, spec := range specs {
		m.Packages = append(m.Packages, spec.Package)
		for _, file := range spec.Files {
			m.Files = append(m.Files, ManifestFile{file.Path, spec.Package.ID, file.Source, file.Size, file.SHA256})
		}
	}
	return m
}

func writeManifest(dir string, manifest Manifest) error {
	f, err := os.OpenFile(filepath.Join(dir, ManifestFilename), os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
	if err != nil {
		return fmt.Errorf("create manifest: %w", err)
	}
	encoder := json.NewEncoder(f)
	encoder.SetIndent("", "  ")
	err = encoder.Encode(manifest)
	if err == nil {
		err = f.Sync()
	}
	closeErr := f.Close()
	if err != nil {
		return err
	}
	return closeErr
}

func readManifest(dir string) (Manifest, error) {
	var manifest Manifest
	f, err := openRegular(filepath.Join(dir, ManifestFilename))
	if err != nil {
		return manifest, err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return manifest, err
	}
	if info.Size() > 512<<10 {
		return manifest, errors.New("oversize runtime manifest")
	}
	decoder := json.NewDecoder(io.LimitReader(f, (512<<10)+1))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&manifest); err != nil {
		return manifest, err
	}
	var extra any
	if err := decoder.Decode(&extra); !errors.Is(err, io.EOF) {
		return manifest, errors.New("trailing data in runtime manifest")
	}
	return manifest, nil
}

func validate(ctx context.Context, dir string, specs []packageSpec) (ValidationReport, error) {
	report := ValidationReport{
		Dir: dir, Provider: ProviderQNN, ExternalDependencies: ExternalDependencies(),
		VerifiedFiles: []string{}, MissingFiles: []string{}, InvalidFiles: []string{}, Problems: []string{},
	}
	if ctx == nil {
		return report, errors.New("nil validation context")
	}
	if err := ctx.Err(); err != nil {
		report.Problems = append(report.Problems, err.Error())
		return report, err
	}
	fail := func(path string, err error) {
		if errors.Is(err, os.ErrNotExist) {
			report.MissingFiles = append(report.MissingFiles, path)
		} else {
			report.InvalidFiles = append(report.InvalidFiles, path)
		}
		report.Problems = append(report.Problems, path+": "+err.Error())
	}
	if err := inspectRealDir(dir); err != nil {
		fail(dir, err)
		return report, errors.New(strings.Join(report.Problems, "; "))
	}
	expected := newManifest(specs)
	actual, err := readManifest(dir)
	if err != nil {
		fail(ManifestFilename, err)
	} else if !reflect.DeepEqual(actual, expected) {
		fail(ManifestFilename, errors.New("manifest does not match the approved package identity and complete file inventory"))
	} else {
		report.ManifestValid = true
	}
	allowed := map[string]bool{ManifestFilename: true}
	for _, file := range expected.Files {
		if err := ctx.Err(); err != nil {
			fail(file.Path, err)
			return report, err
		}
		allowed[file.Path] = true
		path := filepath.Join(dir, filepath.FromSlash(file.Path))
		if err := inspectRealDir(filepath.Dir(path)); err != nil {
			fail(file.Path, err)
			continue
		}
		if err := validateFile(ctx, path, file); err != nil {
			fail(file.Path, err)
		} else {
			report.VerifiedFiles = append(report.VerifiedFiles, file.Path)
		}
	}
	err = filepath.WalkDir(dir, func(path string, entry fs.DirEntry, walkErr error) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		if walkErr != nil {
			return walkErr
		}
		relative, err := filepath.Rel(dir, path)
		if err != nil {
			return err
		}
		if entry.Type()&os.ModeSymlink != 0 {
			fail(relative, errors.New("links are not permitted in an installation"))
			return nil
		}
		if !entry.IsDir() && !allowed[filepath.ToSlash(relative)] {
			fail(relative, errors.New("unexpected or case-aliased installed file"))
		}
		return nil
	})
	if err != nil {
		fail(dir, err)
	}
	report.Valid = len(report.Problems) == 0
	if !report.Valid {
		return report, errors.New(strings.Join(report.Problems, "; "))
	}
	return report, nil
}

func validateFile(ctx context.Context, path string, expected ManifestFile) error {
	f, err := openRegular(path)
	if err != nil {
		return err
	}
	defer f.Close()
	info, err := f.Stat()
	if err != nil {
		return err
	}
	if info.Size() != expected.Size {
		return fmt.Errorf("size mismatch: expected %d, got %d", expected.Size, info.Size())
	}
	h := sha256.New()
	n, err := io.Copy(h, io.LimitReader(contextReader{ctx, f}, expected.Size+1))
	if err != nil {
		return err
	}
	if n != expected.Size || hex.EncodeToString(h.Sum(nil)) != expected.SHA256 {
		return errors.New("installed SHA256 mismatch")
	}
	return validateNative(path)
}

func validateNative(path string) error {
	switch strings.ToLower(filepath.Ext(path)) {
	case ".dll":
		f, err := pe.Open(path)
		if err != nil {
			return fmt.Errorf("%s is not a valid PE DLL: %w", filepath.Base(path), err)
		}
		defer f.Close()
		header, ok := f.OptionalHeader.(*pe.OptionalHeader64)
		if f.Machine != pe.IMAGE_FILE_MACHINE_ARM64 || !ok || header.Magic != 0x20b ||
			f.Characteristics&pe.IMAGE_FILE_DLL == 0 || len(f.Sections) == 0 {
			return fmt.Errorf("%s is not a native PE ARM64 DLL", filepath.Base(path))
		}
	case ".so":
		f, err := elf.Open(path)
		if err != nil {
			return fmt.Errorf("%s is not a valid DSP ELF: %w", filepath.Base(path), err)
		}
		defer f.Close()
		// The standard library does not name Qualcomm's ELF e_machine value.
		if f.Machine != elf.Machine(164) || f.Class != elf.ELFCLASS32 || f.Data != elf.ELFDATA2LSB {
			return fmt.Errorf("%s is not a Hexagon DSP skeleton", filepath.Base(path))
		}
	}
	return nil
}

func inspectRealDir(path string) error {
	path, err := filepath.Abs(path)
	if err != nil {
		return err
	}
	parent := filepath.Dir(path)
	if parent != path {
		if err := inspectRealDir(parent); err != nil {
			return err
		}
	}
	info, err := os.Lstat(path)
	if err != nil {
		return err
	}
	if !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return fmt.Errorf("not a real directory (links are not permitted): %s", path)
	}
	return nil
}
