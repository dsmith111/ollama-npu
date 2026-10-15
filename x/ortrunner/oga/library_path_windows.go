//go:build windows

package oga

import (
	"errors"
	"fmt"
	"log/slog"
	"path/filepath"
	"unicode/utf8"

	"golang.org/x/sys/windows"
)

func nativeLibraryPath(path string) (string, error) {
	if !utf8.ValidString(path) {
		return "", errors.New("native library path is not valid UTF-8")
	}
	absolute, err := filepath.Abs(path)
	if err != nil {
		return "", err
	}
	wide, err := windows.UTF16FromString(absolute)
	if err != nil {
		return "", err
	}
	if len(wide) <= 260 {
		return absolute, nil
	}
	// The pinned ORT provider discovery truncates module paths at MAX_PATH.
	return shortLibraryPath(absolute, windowsShortPath)
}

func shortLibraryPath(absolute string, lookup func(string) (string, error)) (string, error) {
	pathError := func(cause error) error {
		return fmt.Errorf("NPU_RUNTIME_PATH_UNSUPPORTED: the pinned Windows runtime requires a module path within MAX_PATH; Windows must supply an ASCII alias for the same directory, otherwise use the default setup location or a separately qualified runtime: %w", cause)
	}
	dir, err := directoryAlias(filepath.Dir(absolute), lookup)
	if err != nil {
		return "", pathError(err)
	}
	path := filepath.Join(dir, filepath.Base(absolute))
	wide, err := windows.UTF16FromString(path)
	if err != nil {
		return "", pathError(err)
	}
	if len(wide) > 260 {
		return "", pathError(errors.New("verified directory alias is still too long"))
	}
	slog.Info("using verified Windows runtime-directory alias", "library", absolute, "native_library", path)
	return path, nil
}
