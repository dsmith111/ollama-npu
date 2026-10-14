//go:build !windows

package runtime

import (
	"context"
	"errors"
	"fmt"
	"time"

	"golang.org/x/sys/unix"
)

func acquireInstallLock(ctx context.Context, path string) (func(), error) {
	fd, err := unix.Open(path, unix.O_CREAT|unix.O_RDWR|unix.O_NOFOLLOW|unix.O_CLOEXEC, 0o600)
	if err != nil {
		return nil, err
	}
	var info unix.Stat_t
	if err := unix.Fstat(fd, &info); err != nil {
		unix.Close(fd)
		return nil, err
	}
	if info.Mode&unix.S_IFMT != unix.S_IFREG {
		unix.Close(fd)
		return nil, fmt.Errorf("unsafe lock file: %s", path)
	}
	for {
		if err := ctx.Err(); err != nil {
			unix.Close(fd)
			return nil, err
		}
		err := unix.Flock(fd, unix.LOCK_EX|unix.LOCK_NB)
		if err == nil {
			return func() {
				unix.Flock(fd, unix.LOCK_UN)
				unix.Close(fd)
			}, nil
		}
		if !errors.Is(err, unix.EWOULDBLOCK) && !errors.Is(err, unix.EAGAIN) {
			unix.Close(fd)
			return nil, err
		}
		select {
		case <-ctx.Done():
			unix.Close(fd)
			return nil, ctx.Err()
		case <-time.After(25 * time.Millisecond):
		}
	}
}
