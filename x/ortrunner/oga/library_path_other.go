//go:build !windows

package oga

func nativeLibraryPath(path string) (string, error) {
	return path, nil
}
