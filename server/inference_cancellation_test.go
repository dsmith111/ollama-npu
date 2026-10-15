package server

import (
	"context"
	"testing"
	"time"
)

func TestInferenceResponseStopsWhenReceiverDisconnects(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	ch := make(chan any)
	finished := make(chan struct{})
	go func() {
		sendInferenceResponse(ctx, ch, "first")
		sendInferenceResponse(ctx, ch, "second")
		close(finished)
	}()
	select {
	case response := <-ch:
		if response != "first" {
			t.Fatalf("unexpected response %v", response)
		}
	case <-time.After(time.Second):
		t.Fatal("live response was not delivered")
	}
	cancel()
	select {
	case <-finished:
	case <-time.After(time.Second):
		t.Fatal("disconnected receiver blocked the completion callback")
	}
}
