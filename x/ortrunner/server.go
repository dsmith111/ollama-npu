package ortrunner

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"

	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/x/ortrunner/oga"
)

// Execute permits unauthenticated loopback access only when neither --auth-token
// nor OLLAMA_ORT_RUNNER_TOKEN is set, for legacy direct runner invocations.
func Execute(args []string) error {
	slog.SetDefault(logutil.NewLogger(os.Stderr, envconfig.LogLevel()))
	var modelDir, token string
	var port int
	flags := flag.NewFlagSet("ortrunner", flag.ContinueOnError)
	_ = flags.Bool("ortgenai-engine", false, "Select the internal ORT GenAI runner")
	flags.StringVar(&modelDir, "model", "", "Path to ONNX model directory")
	flags.IntVar(&port, "port", 0, "Loopback port; zero chooses an available port")
	flags.StringVar(&token, "auth-token", os.Getenv("OLLAMA_ORT_RUNNER_TOKEN"), "Per-child bearer token; omitted means legacy unauthenticated loopback access")
	_ = flags.Bool("verbose", false, "Enable debug logging")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if modelDir == "" || port < 0 || port > 65535 || flags.NArg() != 0 {
		return fmt.Errorf("a model directory and a port between 0 and 65535 are required; positional arguments are unsupported")
	}
	if token != "" && (strings.TrimSpace(token) != token || strings.ContainsAny(token, "\x00\r\n \t")) {
		return fmt.Errorf("invalid runner bearer token")
	}
	if token == "" {
		slog.Warn("legacy direct ORT runner allows unauthenticated loopback requests; set OLLAMA_ORT_RUNNER_TOKEN for child isolation")
	}
	if err := oga.CheckInit(); err != nil {
		return fmt.Errorf("ORT GenAI not available: %w", err)
	}
	sigCtx, sigCancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer sigCancel()
	var runner Runner
	if err := runner.Load(modelDir); err != nil {
		runner.Close()
		oga.Shutdown()
		return fmt.Errorf("failed to load model: %w", err)
	}
	service := newRunnerService(sigCtx, &runner, requestQueueCapacity)
	cleanupSafe := true
	defer func() {
		service.stop()
		if cleanupSafe {
			runner.Close()
			oga.Shutdown()
		}
	}()
	listener, err := net.Listen("tcp", net.JoinHostPort("127.0.0.1", strconv.Itoa(port)))
	if err != nil {
		service.stop()
		<-service.done
		return err
	}
	defer listener.Close()
	server := newHTTPServer(service.ctx, service.handler(token))
	server.Addr = listener.Addr().String()
	ready := ReadyResponse{Event: "ready", Port: listener.Addr().(*net.TCPAddr).Port, StatusResponse: service.status()}
	if err := json.NewEncoder(os.Stdout).Encode(ready); err != nil {
		service.stop()
		<-service.done
		return fmt.Errorf("announce runner readiness: %w", err)
	}
	serveErr := make(chan error, 1)
	go func() {
		serveErr <- server.Serve(listener)
	}()
	select {
	case err = <-serveErr:
		service.stop()
	case <-service.ctx.Done():
	}
	service.stop()
	shutdownCtx, cancel := context.WithTimeout(context.Background(), cancellationGrace)
	defer cancel()
	shutdownErr := server.Shutdown(shutdownCtx)
	if shutdownErr != nil {
		shutdownErr = errors.Join(shutdownErr, server.Close())
	}
	select {
	case <-service.done:
	case <-shutdownCtx.Done():
		// Never destroy a model while a native call may still be using it.
		cleanupSafe = false
		return errors.Join(err, shutdownErr, fmt.Errorf("native ORT work exceeded cancellation grace; parent must terminate the child process"))
	}
	if errors.Is(err, http.ErrServerClosed) {
		err = nil
	}
	return errors.Join(err, shutdownErr)
}
