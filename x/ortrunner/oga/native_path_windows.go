//go:build windows

package oga

import (
	"fmt"
	"log/slog"

	"golang.org/x/sys/windows"
)

func NativeModelDirectory(dir, provider string) (string, error) {
	if provider != "qnn" {
		return dir, nil
	}
	// ORT 1.24.1's QNN mapper widens narrow path bytes instead of decoding them.
	path, err := qnnModelDirectory(dir, windowsShortPath)
	if err == nil && path != dir {
		slog.Info("using verified Windows QNN model-directory alias", "model_dir", dir, "native_dir", path)
	}
	return path, err
}

func windowsShortPath(path string) (string, error) {
	wide, err := windows.UTF16PtrFromString(path)
	if err != nil {
		return "", err
	}
	size, err := windows.GetShortPathName(wide, nil, 0)
	if err != nil {
		return "", err
	}
	if size == 0 || size > 32768 {
		return "", fmt.Errorf("invalid Windows short-path size %d", size)
	}
	buffer := make([]uint16, size)
	length, err := windows.GetShortPathName(wide, &buffer[0], uint32(len(buffer)))
	if err != nil {
		return "", err
	}
	if length == 0 || length >= uint32(len(buffer)) {
		return "", fmt.Errorf("Windows short path changed during lookup")
	}
	return windows.UTF16ToString(buffer[:length]), nil
}
