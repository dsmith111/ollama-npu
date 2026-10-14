//go:build windows

package ortrunner

import "golang.org/x/sys/windows/registry"

func compatibilityProcessorName() (string, error) {
	key, err := registry.OpenKey(registry.LOCAL_MACHINE, `HARDWARE\DESCRIPTION\System\CentralProcessor\0`, registry.QUERY_VALUE)
	if err != nil {
		return "", err
	}
	defer key.Close()
	name, _, err := key.GetStringValue("ProcessorNameString")
	return name, err
}
