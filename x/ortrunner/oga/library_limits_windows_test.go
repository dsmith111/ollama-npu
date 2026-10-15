//go:build windows

package oga

import (
	"path/filepath"
	"strings"
	"testing"
)

func TestLibraryPathsFitIncludesLongestProviderFilename(t *testing.T) {
	dir := `C:\` + strings.Repeat("d", 230)
	path := filepath.Join(dir, "onnxruntime-genai.dll")
	if len(path)+1 > 260 {
		t.Fatal("fixture must fit the main DLL while exceeding the provider path limit")
	}
	fits, err := libraryPathsFit(path)
	if err != nil || fits {
		t.Fatalf("longer provider dependency was not included: fits=%v error=%v", fits, err)
	}
}
