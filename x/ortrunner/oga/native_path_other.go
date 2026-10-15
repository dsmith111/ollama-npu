//go:build !windows

package oga

func NativeModelDirectory(dir, provider string) (string, error) {
	return dir, nil
}
