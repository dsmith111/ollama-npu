//go:build !windows && !linux

package ortrunner

import (
	"os"
	"os/exec"
)

type processGuard struct{}

func newProcessGuard(*exec.Cmd) (*processGuard, error) { return &processGuard{}, nil }
func (*processGuard) Attach(*os.Process) error         { return nil }
func (*processGuard) Close() error                     { return nil }
