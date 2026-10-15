//go:build integration

package integration

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
)

func TestNPUQueuedRecovery(t *testing.T) {
	c, enabled, err := npuConfig(os.LookupEnv)
	if !enabled {
		t.Skip("requires the same explicit local model/endpoint opt-in as TestNPUQualification")
	}
	if err != nil {
		t.Fatal(err)
	}
	npuLogSource(t)
	h := npuNewHTTP(c.endpoint)
	defer h.client.CloseIdleConnections()
	ctx, cancel := context.WithTimeout(t.Context(), 20*time.Minute)
	defer cancel()
	var inventory api.ListResponse
	if err := h.json(ctx, "/api/tags", nil, &inventory); err != nil {
		t.Fatal(err)
	}
	digest, err := npuDigest(inventory, c.npuModel)
	if err != nil {
		t.Fatal(err)
	}
	var show api.ShowResponse
	if err := h.json(ctx, "/api/show", api.ShowRequest{Model: c.npuModel}, &show); err != nil {
		t.Fatal(err)
	}
	if show.RemoteHost != "" || show.RemoteModel != "" || show.Details.Format != "ortgenai" {
		t.Fatal("queued recovery requires an existing local ORT GenAI model")
	}
	if err := npuValidateTemplate(show); err != nil {
		t.Fatal(err)
	}
	t.Logf("endpoint=%s model=%s digest=%s; no prompt replay or NPU placement certification", c.endpoint, c.npuModel, digest)
	t.Cleanup(func() {
		if err := h.unload(t, c.npuModel); err != nil {
			t.Error(err)
		}
		if err := h.waitUnloaded(t, c.npuModel); err != nil {
			t.Error(err)
		}
	})
	for round := range 10 {
		t.Run(fmt.Sprintf("disconnect_%02d", round+1), func(t *testing.T) {
			streamCtx, stop := context.WithTimeout(ctx, npuRequestTimeout)
			defer stop()
			request := npuGenerateRequest(c.npuModel, "Write all the numbers from 1 to 200 separated by spaces.")
			request.Stream = new(true)
			request.Options["num_predict"] = 256
			response, err := h.open(streamCtx, "/api/generate", request)
			if err != nil {
				t.Fatal(err)
			}
			defer response.Body.Close()
			first, _, err := npuReadStream(response.Body, true)
			if err != nil || first.Done || first.Response == "" {
				t.Fatalf("active nonterminal stream prerequisite: %+v, %v", first, err)
			}
			type result struct {
				response api.GenerateResponse
				want     string
				err      error
			}
			results := make(chan result, len(npuPrompts))
			for _, prompt := range npuPrompts {
				go func() {
					requestCtx, finish := context.WithTimeout(ctx, npuRequestTimeout)
					defer finish()
					req := npuGenerateRequest(c.npuModel, prompt.text)
					// Exercise the terminal Tokenize callback while requests queue.
					req.Raw, req.Prompt = false, prompt.text
					var r api.GenerateResponse
					err := h.json(requestCtx, "/api/generate", req, &r)
					results <- result{r, prompt.want, err}
				}()
			}
			deadlineCtx, finish := context.WithTimeout(ctx, 100*time.Millisecond)
			var expired api.GenerateResponse
			err = h.json(deadlineCtx, "/api/generate", npuGenerateRequest(c.npuModel, npuPrompts[0].text), &expired)
			finish()
			if !errors.Is(err, context.DeadlineExceeded) {
				t.Fatalf("queued deadline request did not expire: %v", err)
			}
			select {
			case r := <-results:
				t.Fatalf("queued request returned while stream remained active: %+v", r)
			default:
			}
			stop()
			response.Body.Close()
			for range npuPrompts {
				select {
				case r := <-results:
					if r.err != nil {
						t.Fatal(r.err)
					}
					if err := npuValidateGeneration(r.response, c.npuModel, r.want, "stop", 32); err != nil {
						t.Fatal(err)
					}
					if len(r.response.Context) == 0 {
						t.Fatal("queued normal Generate lost its conversation tokens")
					}
					t.Logf("queued_response=%q context_tokens=%d", strings.TrimSpace(r.response.Response), len(r.response.Context))
				case <-time.After(npuRequestTimeout):
					t.Fatal("queued recovery exceeded its bound")
				}
			}
		})
	}
}
