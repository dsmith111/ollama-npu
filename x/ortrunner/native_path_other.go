//go:build !windows

package ortrunner

func nativeModelDirectory(dir, provider string) (string, error) {
	return dir, nil
}
