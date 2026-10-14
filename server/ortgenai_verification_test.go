package server

import (
	"crypto/sha256"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/manifest"
)

func TestORTFileVerificationMemoDoesNotBypassIntegrity(t *testing.T) {
	for _, test := range []string{"changed timestamp", "different digest", "copy"} {
		t.Run(test, func(t *testing.T) {
			filename := filepath.Join(t.TempDir(), "weights.bin")
			data := []byte(strings.Repeat("a", 2<<20))
			layer := manifest.Layer{Name: "weights.bin", Size: int64(len(data)), Digest: fmt.Sprintf("sha256:%x", sha256.Sum256(data))}
			if err := os.WriteFile(filename, data, 0o644); err != nil {
				t.Fatal(err)
			}
			if err := verifyORTGenAIFile(filename, layer, nil); err != nil {
				t.Fatal(err)
			}
			if test == "different digest" {
				layer.Digest = "sha256:" + strings.Repeat("0", 64)
			} else {
				original, err := os.Stat(filename)
				if err != nil {
					t.Fatal(err)
				}
				data[0] ^= 1
				if err := os.WriteFile(filename, data, 0o644); err != nil {
					t.Fatal(err)
				}
				modified := original.ModTime()
				if test == "changed timestamp" {
					modified = modified.Add(time.Hour)
				}
				if err := os.Chtimes(filename, modified, modified); err != nil {
					t.Fatal(err)
				}
			}
			var dst io.Writer
			if test == "copy" {
				dst = io.Discard
			}
			if err := verifyORTGenAIFile(filename, layer, dst); err == nil {
				t.Fatal("cached verification bypassed SHA256 mismatch")
			}
		})
	}
}

func TestORTFileVerificationInvalidatesChangedLargeFile(t *testing.T) {
	filename := filepath.Join(t.TempDir(), "weights.bin")
	data := []byte(strings.Repeat("a", 2<<20))
	layer := manifest.Layer{Name: "weights.bin", Size: int64(len(data)), Digest: fmt.Sprintf("sha256:%x", sha256.Sum256(data))}
	if err := os.WriteFile(filename, data, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := verifyORTGenAIFile(filename, layer, nil); err != nil {
		t.Fatal(err)
	}
	if err := verifyORTGenAIFile(filename, layer, nil); err != nil {
		t.Fatal(err)
	}
	original, err := os.Stat(filename)
	if err != nil {
		t.Fatal(err)
	}
	replacement := filename + ".new"
	data[0] ^= 1
	if err := os.WriteFile(replacement, data, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(replacement, original.ModTime(), original.ModTime()); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(filename); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(replacement, filename); err != nil {
		t.Fatal(err)
	}
	if err := verifyORTGenAIFile(filename, layer, nil); err == nil {
		t.Fatal("replacement with matching size/time reused old SHA256 verification")
	}
}

func BenchmarkORTCachedFileVerification(b *testing.B) {
	filename := filepath.Join(b.TempDir(), "weights.bin")
	data := []byte(strings.Repeat("a", 2<<20))
	layer := manifest.Layer{Name: "weights.bin", Size: int64(len(data)), Digest: fmt.Sprintf("sha256:%x", sha256.Sum256(data))}
	if err := os.WriteFile(filename, data, 0o644); err != nil {
		b.Fatal(err)
	}
	if err := verifyORTGenAIFile(filename, layer, nil); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for b.Loop() {
		if err := verifyORTGenAIFile(filename, layer, nil); err != nil {
			b.Fatal(err)
		}
	}
}
