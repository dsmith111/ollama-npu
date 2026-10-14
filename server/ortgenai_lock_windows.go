//go:build windows

package server

import (
	"context"
	"errors"
	"fmt"
	"time"

	"golang.org/x/sys/windows"
)

func acquireORTGenAILock(ctx context.Context, path string) (func(), error) {
	name, err := windows.UTF16PtrFromString(path)
	if err != nil {
		return nil, err
	}
	handle, err := windows.CreateFile(name, windows.GENERIC_READ|windows.GENERIC_WRITE,
		windows.FILE_SHARE_READ|windows.FILE_SHARE_WRITE, nil, windows.OPEN_ALWAYS,
		windows.FILE_ATTRIBUTE_NORMAL|windows.FILE_FLAG_OPEN_REPARSE_POINT, 0)
	if err != nil {
		return nil, err
	}
	var info windows.ByHandleFileInformation
	if err := windows.GetFileInformationByHandle(handle, &info); err != nil {
		windows.CloseHandle(handle)
		return nil, err
	}
	if info.FileAttributes&(windows.FILE_ATTRIBUTE_REPARSE_POINT|windows.FILE_ATTRIBUTE_DIRECTORY) != 0 {
		windows.CloseHandle(handle)
		return nil, fmt.Errorf("unsafe ORT GenAI lock file: %s", path)
	}
	var overlapped windows.Overlapped
	for {
		if err := ctx.Err(); err != nil {
			windows.CloseHandle(handle)
			return nil, err
		}
		err := windows.LockFileEx(handle, windows.LOCKFILE_EXCLUSIVE_LOCK|windows.LOCKFILE_FAIL_IMMEDIATELY, 0, 1, 0, &overlapped)
		if err == nil {
			// Do not unlink the persistent lock: waiters must lock the same inode.
			return func() {
				windows.UnlockFileEx(handle, 0, 1, 0, &overlapped)
				windows.CloseHandle(handle)
			}, nil
		}
		if !errors.Is(err, windows.ERROR_LOCK_VIOLATION) {
			windows.CloseHandle(handle)
			return nil, err
		}
		select {
		case <-ctx.Done():
		case <-time.After(25 * time.Millisecond):
		}
	}
}
