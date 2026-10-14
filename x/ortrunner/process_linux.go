//go:build linux

package ortrunner

import (
	"os"
	"os/exec"
	"syscall"
)

type processGuard struct{}

func newProcessGuard(cmd *exec.Cmd) (*processGuard, error) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Pdeathsig: syscall.SIGKILL}
	return &processGuard{}, nil
}

func (*processGuard) Attach(*os.Process) error { return nil }
func (*processGuard) Close() error             { return nil }
