//go:build !windows

package ortrunner

import "fmt"

func compatibilityProcessorName() (string, error) {
	return "", fmt.Errorf("processor profile detection requires Windows ARM64")
}
