package oga

// #include "oga.h"
// #include <stdlib.h>
import "C"

import (
	"fmt"
	"strings"
	"unicode/utf8"
	"unsafe"
)

// Tokenizer wraps an OgaTokenizer handle.
type Tokenizer struct {
	ctx *C.OgaTokenizer
}

// NewTokenizer creates a tokenizer from a model.
func NewTokenizer(model *Model) (*Tokenizer, error) {
	if err := CheckInit(); err != nil {
		return nil, err
	}
	if model == nil || model.ctx == nil {
		return nil, fmt.Errorf("ort genai: model is closed")
	}
	var tok *C.OgaTokenizer
	if err := ogaError(C.OgaCreateTokenizer(model.ctx, &tok)); err != nil {
		return nil, err
	}
	return &Tokenizer{ctx: tok}, nil
}

// Encode tokenizes a string and returns the token IDs.
func (t *Tokenizer) Encode(text string) ([]int32, error) {
	if t == nil || t.ctx == nil || strings.ContainsRune(text, 0) || !utf8.ValidString(text) {
		return nil, fmt.Errorf("ort genai: tokenizer is closed or text is not UTF-8 without NUL")
	}
	var seqs *C.OgaSequences
	if err := ogaError(C.OgaCreateSequences(&seqs)); err != nil {
		return nil, err
	}
	defer C.OgaDestroySequences(seqs)

	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	if err := ogaError(C.OgaTokenizerEncode(t.ctx, cText, seqs)); err != nil {
		return nil, err
	}
	if C.OgaSequencesCount(seqs) != 1 {
		return nil, fmt.Errorf("ort genai: expected exactly one encoded sequence")
	}

	count := C.OgaSequencesGetSequenceCount(seqs, 0)
	data := C.OgaSequencesGetSequenceData(seqs, 0)
	if count == 0 {
		return nil, nil
	}
	if data == nil {
		return nil, fmt.Errorf("ort genai: encoded sequence data is nil")
	}

	tokens := make([]int32, count)
	copy(tokens, unsafe.Slice((*int32)(unsafe.Pointer(data)), count))
	return tokens, nil
}

// Decode converts token IDs back to a string.
func (t *Tokenizer) Decode(tokens []int32) (string, error) {
	if t == nil || t.ctx == nil {
		return "", fmt.Errorf("ort genai: tokenizer is closed")
	}
	for _, token := range tokens {
		if token < 0 {
			return "", fmt.Errorf("ort genai: token IDs must be nonnegative")
		}
	}
	if len(tokens) == 0 {
		return "", nil
	}
	var out *C.char
	if err := ogaError(C.OgaTokenizerDecode(t.ctx, (*C.int32_t)(&tokens[0]), C.size_t(len(tokens)), &out)); err != nil {
		return "", err
	}
	result := C.GoString(out)
	C.OgaDestroyString(out)
	return result, nil
}

// ApplyChatTemplate renders messages, a JSON array of {"role", "content"}
// objects, with a Jinja chat template, or with the tokenizer config's own when
// template is empty, ending with the assistant's generation prompt.
func (t *Tokenizer) ApplyChatTemplate(template, messages string) (string, error) {
	if t == nil || t.ctx == nil {
		return "", fmt.Errorf("ort genai: tokenizer is closed")
	}
	if C.OgaHasChatTemplates() == 0 {
		return "", fmt.Errorf("ort genai: this runtime cannot render chat templates")
	}
	if strings.ContainsRune(template, 0) || strings.ContainsRune(messages, 0) {
		return "", fmt.Errorf("ort genai: chat template input contains NUL")
	}
	var cTemplate *C.char
	if template != "" {
		cTemplate = C.CString(template)
		defer C.free(unsafe.Pointer(cTemplate))
	}
	cMessages := C.CString(messages)
	defer C.free(unsafe.Pointer(cMessages))
	var out *C.char
	if err := ogaError(C.OgaTokenizerApplyChatTemplate(t.ctx, cTemplate, cMessages, nil, C.bool(true), &out)); err != nil {
		return "", err
	}
	result := C.GoString(out)
	C.OgaDestroyString(out)
	if !utf8.ValidString(result) {
		return "", fmt.Errorf("ort genai: chat template rendered invalid UTF-8")
	}
	return result, nil
}

// Close frees the tokenizer.
func (t *Tokenizer) Close() {
	if t.ctx != nil {
		C.OgaDestroyTokenizer(t.ctx)
		t.ctx = nil
	}
}

// TokenStream wraps an OgaTokenizerStream for incremental decoding.
type TokenStream struct {
	ctx *C.OgaTokenizerStream
}

// NewTokenStream creates a streaming token decoder from a tokenizer.
func NewTokenStream(tok *Tokenizer) (*TokenStream, error) {
	if err := CheckInit(); err != nil {
		return nil, err
	}
	if tok == nil || tok.ctx == nil {
		return nil, fmt.Errorf("ort genai: tokenizer is closed")
	}
	var stream *C.OgaTokenizerStream
	if err := ogaError(C.OgaCreateTokenizerStream(tok.ctx, &stream)); err != nil {
		return nil, err
	}
	return &TokenStream{ctx: stream}, nil
}

// Decode decodes a single token into text. Returns partial text as tokens
// accumulate to form complete word pieces.
func (s *TokenStream) Decode(token int32) (string, error) {
	if s.ctx == nil || token < 0 {
		return "", fmt.Errorf("ort genai: stream is closed or token is negative")
	}
	var out *C.char
	if err := ogaError(C.OgaTokenizerStreamDecode(s.ctx, C.int32_t(token), &out)); err != nil {
		return "", err
	}
	// The returned pointer is owned by the stream, don't free it
	return C.GoString(out), nil
}

// Close frees the token stream.
func (s *TokenStream) Close() {
	if s.ctx != nil {
		C.OgaDestroyTokenizerStream(s.ctx)
		s.ctx = nil
	}
}
