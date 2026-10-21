package cmd

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"strings"
	"syscall"
	"unsafe"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"golang.org/x/sys/windows"
)

const (
	Installer = "OllamaSetup.exe"
)

// startApp is upstream's fallback when no server answers: start the desktop app, which runs
// a server. The portable NPU fork ships no app, and upstream's search would find and start a
// stock Ollama app installed elsewhere on the machine, handing the user a server without the
// NPU backend. The fork starts only an app beside this executable and otherwise says what to do.
func startApp(ctx context.Context, client *api.Client) error {
	if len(isProcRunning(Installer)) > 0 {
		return fmt.Errorf("upgrade in progress...")
	}
	exe, err := os.Executable()
	if err != nil {
		return err
	}
	appExe := filepath.Join(filepath.Dir(exe), "ollama app.exe")
	if _, err := os.Stat(appExe); err != nil {
		if !errors.Is(err, os.ErrNotExist) {
			return err
		}
		return errors.New(noServerHint(envconfig.Host().Host))
	}

	cmd_path := "c:\\Windows\\system32\\cmd.exe"
	cmd := exec.Command(cmd_path, "/c", appExe, "--hide", "--fast-startup")
	cmd.SysProcAttr = &syscall.SysProcAttr{CreationFlags: 0x08000000, HideWindow: true}

	cmd.Stdin = strings.NewReader("")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("unable to start ollama app %w", err)
	}

	if cmd.Process != nil {
		defer cmd.Process.Release() //nolint:errcheck
	}
	return waitForServer(ctx, client)
}

// noServerHint tells a user of the portable package what to do when nothing answers at host.
func noServerHint(host string) string {
	return fmt.Sprintf("no Ollama server is running at %s. This portable NPU fork has no desktop app to start one: "+
		"run `.\\ollama.exe serve` from the fork's folder in another window first (with OLLAMA_HEXAGON=1 to use "+
		"the NPU), or point OLLAMA_HOST at a running server", host)
}

func isProcRunning(procName string) []uint32 {
	pids := make([]uint32, 2048)
	var ret uint32
	if err := windows.EnumProcesses(pids, &ret); err != nil || ret == 0 {
		slog.Debug("failed to check for running installers", "error", err)
		return nil
	}
	if ret > uint32(len(pids)) {
		pids = make([]uint32, ret+10)
		if err := windows.EnumProcesses(pids, &ret); err != nil || ret == 0 {
			slog.Debug("failed to check for running installers", "error", err)
			return nil
		}
	}
	if ret < uint32(len(pids)) {
		pids = pids[:ret]
	}
	var matches []uint32
	for _, pid := range pids {
		if pid == 0 {
			continue
		}
		hProcess, err := windows.OpenProcess(windows.PROCESS_QUERY_INFORMATION|windows.PROCESS_VM_READ, false, pid)
		if err != nil {
			continue
		}
		defer windows.CloseHandle(hProcess)
		var module windows.Handle
		var cbNeeded uint32
		cb := uint32(unsafe.Sizeof(module))
		if err := windows.EnumProcessModules(hProcess, &module, cb, &cbNeeded); err != nil {
			continue
		}
		var sz uint32 = 1024 * 8
		moduleName := make([]uint16, sz)
		cb = uint32(len(moduleName)) * uint32(unsafe.Sizeof(uint16(0)))
		if err := windows.GetModuleBaseName(hProcess, module, &moduleName[0], cb); err != nil && err != syscall.ERROR_INSUFFICIENT_BUFFER {
			continue
		}
		exeFile := path.Base(strings.ToLower(syscall.UTF16ToString(moduleName)))
		if strings.EqualFold(exeFile, procName) {
			matches = append(matches, pid)
		}
	}
	return matches
}
