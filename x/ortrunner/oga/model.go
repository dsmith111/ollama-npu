package oga

// #include "oga.h"
// #include <stdlib.h>
import "C"

import (
	"encoding/json"
	"fmt"
	"strings"
	"unsafe"
)

// ogaError converts an OgaResult to a Go error and frees the result.
func ogaError(result *C.OgaResult) error {
	if result == nil {
		return nil
	}
	msg := C.GoString(C.OgaResultGetError(result))
	C.OgaDestroyResult(result)
	return fmt.Errorf("ort genai: %s", msg)
}

// Config wraps an OgaConfig handle.
type Config struct {
	ctx *C.OgaConfig
}

// NewConfig creates a new ORT GenAI config from a model directory path.
func NewConfig(modelDir string) (*Config, error) {
	if strings.ContainsRune(modelDir, 0) {
		return nil, fmt.Errorf("ort genai: model directory contains NUL")
	}
	if err := CheckInit(); err != nil {
		return nil, err
	}
	cPath := C.CString(modelDir)
	defer C.free(unsafe.Pointer(cPath))

	var cfg *C.OgaConfig
	if err := ogaError(C.OgaCreateConfig(cPath, &cfg)); err != nil {
		return nil, err
	}
	return &Config{ctx: cfg}, nil
}

// ClearProviders removes the primary decoder's provider list only, not pipeline stages or options.
func (c *Config) ClearProviders() error {
	if c.ctx == nil {
		return fmt.Errorf("ort genai: config is closed")
	}
	return ogaError(C.OgaConfigClearProviders(c.ctx))
}

// AppendProvider adds an execution provider to the config.
func (c *Config) AppendProvider(name string) error {
	if c.ctx == nil || strings.ContainsAny(name, "\"\\\x00\r\n") {
		return fmt.Errorf("ort genai: config is closed or provider name is invalid")
	}
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return ogaError(C.OgaConfigAppendProvider(c.ctx, cName))
}

// SetProviderOption sets a key/value option on an execution provider.
func (c *Config) SetProviderOption(provider, key, value string) error {
	if c.ctx == nil || strings.ContainsAny(provider+key, "\"\\\x00\r\n") {
		return fmt.Errorf("ort genai: config is closed or provider/key is invalid")
	}
	cProvider := C.CString(provider)
	defer C.free(unsafe.Pointer(cProvider))
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))
	// v0.12.2 interpolates values into JSON without escaping them.
	encoded, err := json.Marshal(value)
	if err != nil {
		return err
	}
	cValue := C.CString(string(encoded[1 : len(encoded)-1]))
	defer C.free(unsafe.Pointer(cValue))
	return ogaError(C.OgaConfigSetProviderOption(c.ctx, cProvider, cKey, cValue))
}

// Close frees the config.
func (c *Config) Close() {
	if c.ctx != nil {
		C.OgaDestroyConfig(c.ctx)
		c.ctx = nil
	}
}

// Model wraps an OgaModel handle.
type Model struct {
	ctx *C.OgaModel
}

// NewModel creates a model from a config.
func NewModel(config *Config) (*Model, error) {
	if err := CheckInit(); err != nil {
		return nil, err
	}
	if config == nil || config.ctx == nil {
		return nil, fmt.Errorf("ort genai: config is closed")
	}
	var m *C.OgaModel
	if err := ogaError(C.OgaCreateModelFromConfig(config.ctx, &m)); err != nil {
		return nil, err
	}
	return &Model{ctx: m}, nil
}

// Close frees the model.
func (m *Model) Close() {
	if m.ctx != nil {
		C.OgaDestroyModel(m.ctx)
		m.ctx = nil
	}
}
