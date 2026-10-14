package oga

import (
	"os"
	"os/exec"
	"testing"
)

func TestShutdownDoesNotInitializeRuntime(t *testing.T) {
	if os.Getenv("OLLAMA_TEST_OGA_SHUTDOWN") == "1" {
		Shutdown()
		uninitialized := false
		initOnce.Do(func() { uninitialized = true })
		if !uninitialized {
			t.Fatal("shutdown triggered native runtime initialization")
		}
		return
	}
	cmd := exec.Command(os.Args[0], "-test.run=^TestShutdownDoesNotInitializeRuntime$")
	cmd.Env = append(os.Environ(), "OLLAMA_TEST_OGA_SHUTDOWN=1")
	if output, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("uninitialized shutdown subprocess: %v\n%s", err, output)
	}
}
