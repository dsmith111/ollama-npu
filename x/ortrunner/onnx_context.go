package ortrunner

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"strings"
)

// Inspect only protobuf structure and EPContext attributes, without loading weights.
func inspectQNNContext(filename, modelDir string) (bool, error) {
	f, err := os.Open(filename)
	if err != nil {
		return false, err
	}
	defer f.Close()
	found := false
	graphs, nodes, caches := 0, 0, 0
	err = walkProto(bufio.NewReader(f), func(field int, wire int, value uint64, data *io.LimitedReader) error {
		if field != 7 || wire != 2 {
			return nil
		}
		graphs++
		return walkProto(data, func(field int, wire int, value uint64, data *io.LimitedReader) error {
			if field != 1 || wire != 2 {
				return nil
			}
			nodes++
			var op, domain string
			embed := uint64(1)
			var contextPath string
			contextPresent := false
			var source string
			main := uint64(0)
			attributes := make(map[string]bool)
			err := walkProto(data, func(field int, wire int, value uint64, data *io.LimitedReader) error {
				switch {
				case wire == 2 && (field == 4 || field == 7):
					text, err := protoText(data, 4096)
					if field == 4 {
						op = text
					} else {
						domain = text
					}
					return err
				case wire == 2 && field == 5:
					var name, text string
					var number uint64
					present := false
					err := walkProto(data, func(field int, wire int, value uint64, data *io.LimitedReader) error {
						if field == 1 && wire == 2 {
							var err error
							name, err = protoText(data, 4096)
							return err
						}
						if field == 3 && wire == 0 {
							number = value
						}
						if field == 4 && wire == 2 {
							present = data.N > 0
							if data.N <= 4096 {
								var err error
								text, err = protoText(data, 4096)
								return err
							}
						}
						return nil
					})
					if name == "embed_mode" {
						embed = number
					}
					if attributes[name] {
						return fmt.Errorf("duplicate ONNX attribute %q", name)
					}
					attributes[name] = true
					if name == "source" {
						source = text
					}
					if name == "main_context" {
						main = number
					}
					if name == "ep_cache_context" {
						contextPath, contextPresent = text, present
					}
					return err
				}
				return nil
			})
			if err != nil {
				return err
			}
			if op != "EPContext" || domain != "com.microsoft" {
				return nil
			}
			if !strings.EqualFold(source, "QNN") && !strings.EqualFold(source, "QNNExecutionProvider") {
				return fmt.Errorf("EPContext source %q is not supported by the QNN provider", source)
			}
			found = true
			if embed > 1 || main > 1 {
				return fmt.Errorf("invalid EPContext embed_mode or main_context")
			}
			// Secondary EPContext nodes may share the main node's cache.
			if !contextPresent {
				if main == 1 {
					return fmt.Errorf("main QNN EPContext has no ep_cache_context")
				}
				return nil
			}
			caches++
			if embed == 0 {
				if _, err := modelFile(modelDir, contextPath); err != nil {
					return fmt.Errorf("missing external QNN context: %w", err)
				}
			}
			return nil
		})
	})
	if err != nil {
		return false, fmt.Errorf("invalid ONNX EPContext graph: %w", err)
	}
	if graphs != 1 || nodes == 0 {
		return false, fmt.Errorf("ONNX model must contain one nonempty graph")
	}
	if found && caches == 0 {
		return false, fmt.Errorf("QNN EPContext graph has no embedded or external context")
	}
	return found, nil
}

type byteReader struct{ io.Reader }

func (r byteReader) ReadByte() (byte, error) {
	var b [1]byte
	_, err := io.ReadFull(r.Reader, b[:])
	return b[0], err
}

func walkProto(r io.Reader, visit func(int, int, uint64, *io.LimitedReader) error) error {
	for {
		tag, err := binary.ReadUvarint(byteReader{r})
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
		field, wire := int(tag>>3), int(tag&7)
		if field == 0 || tag>>3 > (1<<29)-1 {
			return fmt.Errorf("protobuf field zero")
		}
		var value uint64
		var data *io.LimitedReader
		switch wire {
		case 0:
			value, err = binary.ReadUvarint(byteReader{r})
		case 1:
			_, err = io.CopyN(io.Discard, r, 8)
		case 2:
			value, err = binary.ReadUvarint(byteReader{r})
			if err == nil {
				if value > uint64(^uint64(0)>>1) {
					return fmt.Errorf("protobuf length overflow")
				}
				data = &io.LimitedReader{R: r, N: int64(value)}
			}
		case 5:
			_, err = io.CopyN(io.Discard, r, 4)
		default:
			return fmt.Errorf("unsupported protobuf wire type %d", wire)
		}
		if err != nil {
			return err
		}
		if err := visit(field, wire, value, data); err != nil {
			return err
		}
		if data != nil {
			if _, err := io.CopyN(io.Discard, data, data.N); err != nil {
				return err
			}
		}
	}
}

func protoText(r *io.LimitedReader, limit int64) (string, error) {
	if r.N > limit {
		return "", fmt.Errorf("protobuf string exceeds %d bytes", limit)
	}
	data, err := io.ReadAll(r)
	return string(data), err
}
