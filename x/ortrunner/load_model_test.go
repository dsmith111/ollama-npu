package ortrunner

import (
	"errors"
	"reflect"
	"testing"
)

type fakeModelResource struct {
	name   string
	closed *[]string
}

func (r *fakeModelResource) Close() { *r.closed = append(*r.closed, r.name) }

func TestFailedModelLoadResourceCleanup(t *testing.T) {
	for _, stage := range []string{"config", "clear-providers", "append-provider", "provider-option", "model", "tokenizer", "success"} {
		t.Run(stage, func(t *testing.T) {
			var closed []string
			fail := errors.New("load failed")
			cfg, model, tok, err := loadConfiguredModel(
				func() (*fakeModelResource, error) {
					if stage == "config" {
						return nil, fail
					}
					return &fakeModelResource{"config", &closed}, nil
				},
				func(*fakeModelResource) error {
					if stage == "clear-providers" || stage == "append-provider" || stage == "provider-option" {
						return fail
					}
					return nil
				},
				func(*fakeModelResource) (*fakeModelResource, error) {
					if stage == "model" {
						return nil, fail
					}
					return &fakeModelResource{"model", &closed}, nil
				},
				func(*fakeModelResource) (*fakeModelResource, error) {
					if stage == "tokenizer" {
						return nil, fail
					}
					return &fakeModelResource{"tokenizer", &closed}, nil
				},
			)
			var want []string
			switch stage {
			case "config", "success":
			case "tokenizer":
				want = []string{"model", "config"}
			default:
				want = []string{"config"}
			}
			if !reflect.DeepEqual(closed, want) || (err == nil) != (stage == "success") {
				t.Fatalf("closed=%v want=%v error=%v", closed, want, err)
			}
			if stage == "success" {
				tok.Close()
				model.Close()
				cfg.Close()
				if !reflect.DeepEqual(closed, []string{"tokenizer", "model", "config"}) {
					t.Fatal(closed)
				}
			}
		})
	}
}
