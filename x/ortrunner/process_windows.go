//go:build windows

package ortrunner

import (
	"fmt"
	"os"
	"os/exec"
	"unsafe"

	"golang.org/x/sys/windows"
)

type processGuard struct {
	job windows.Handle
}

func newProcessGuard(_ *exec.Cmd) (*processGuard, error) {
	job, err := windows.CreateJobObject(nil, nil)
	if err != nil {
		return nil, fmt.Errorf("create runner job: %w", err)
	}
	info := windows.JOBOBJECT_EXTENDED_LIMIT_INFORMATION{}
	info.BasicLimitInformation.LimitFlags = windows.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
	if _, err := windows.SetInformationJobObject(job, windows.JobObjectExtendedLimitInformation,
		uintptr(unsafe.Pointer(&info)), uint32(unsafe.Sizeof(info))); err != nil {
		windows.CloseHandle(job)
		return nil, fmt.Errorf("configure runner job: %w", err)
	}
	return &processGuard{job: job}, nil
}

func (g *processGuard) Attach(process *os.Process) error {
	handle, err := windows.OpenProcess(windows.PROCESS_SET_QUOTA|windows.PROCESS_TERMINATE, false, uint32(process.Pid))
	if err != nil {
		return err
	}
	defer windows.CloseHandle(handle)
	return windows.AssignProcessToJobObject(g.job, handle)
}

func (g *processGuard) Close() error {
	if g.job == 0 {
		return nil
	}
	err := windows.CloseHandle(g.job)
	g.job = 0
	return err
}
