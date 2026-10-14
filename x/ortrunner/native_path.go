package ortrunner

import (
	"fmt"
	"os"
	"path/filepath"
	"unicode/utf8"
)

func asciiPath(path string) bool {
	for _, r := range path {
		if r >= utf8.RuneSelf {
			return false
		}
	}
	return true
}

func qnnModelDirectory(dir string, shortPath func(string) (string, error)) (string, error) {
	pathError := func(cause error) error {
		return fmt.Errorf("NPU_MODEL_PATH_UNSUPPORTED: the pinned Windows QNN file mapper requires an ASCII model path; Windows must supply an alias for the same directory, otherwise select an ASCII OLLAMA_MODELS directory or a separately qualified runtime with Unicode support: %w", cause)
	}
	absolute, err := filepath.Abs(dir)
	if err != nil {
		return "", pathError(err)
	}
	if asciiPath(absolute) {
		return dir, nil
	}
	alias, err := shortPath(absolute)
	if err != nil {
		return "", pathError(err)
	}
	if !filepath.IsAbs(alias) || !asciiPath(alias) {
		return "", pathError(fmt.Errorf("Windows did not supply an absolute ASCII alias"))
	}
	original, err := os.Stat(absolute)
	if err != nil {
		return "", pathError(err)
	}
	aliased, err := os.Stat(alias)
	if err != nil {
		return "", pathError(err)
	}
	if !original.IsDir() || !aliased.IsDir() || !os.SameFile(original, aliased) {
		return "", pathError(fmt.Errorf("alias does not identify the original model directory"))
	}
	return alias, nil
}
