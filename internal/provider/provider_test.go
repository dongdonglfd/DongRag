package provider

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestHTTPErrorRetryable(t *testing.T) {
	for status, want := range map[int]bool{400: false, 401: false, 408: true, 429: true, 500: true, 503: true} {
		err := &HTTPError{Service: "test", StatusCode: status}
		if err.Retryable() != want {
			t.Fatalf("status %d retryable = %t, want %t", status, err.Retryable(), want)
		}
		var marked interface{ Retryable() bool }
		if !errors.As(err, &marked) {
			t.Fatalf("status %d does not expose retryability", status)
		}
	}
}

func TestOllamaEmbedder(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/embed" || r.Method != http.MethodPost {
			t.Fatalf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"embeddings":[[0.1,0.2],[0.3,0.4]]}`))
	}))
	defer server.Close()

	values, err := (OllamaEmbedder{BaseURL: server.URL, Model: "test"}).Embed(context.Background(), []string{"a", "b"})
	if err != nil || len(values) != 2 || len(values[0]) != 2 {
		t.Fatalf("unexpected embeddings: %#v, %v", values, err)
	}
}

func TestBailianChat(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" || r.Header.Get("Authorization") != "Bearer key" {
			t.Fatalf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"answer"}}]}`))
	}))
	defer server.Close()

	answer, err := (BailianChat{BaseURL: server.URL, APIKey: "key", Model: "test"}).Complete(context.Background(), "system", "user")
	if err != nil || answer != "answer" {
		t.Fatalf("unexpected answer: %q, %v", answer, err)
	}
}

func TestBailianChatStream(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" || r.Header.Get("Authorization") != "Bearer key" {
			t.Fatalf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		var request struct {
			Stream bool `json:"stream"`
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil || !request.Stream {
			t.Fatalf("stream flag missing: err=%v stream=%t", err, request.Stream)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		flusher := w.(http.Flusher)
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"hello \"}}]}\n\n"))
		flusher.Flush()
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"world\"}}]}\n\n"))
		flusher.Flush()
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
		flusher.Flush()
	}))
	defer server.Close()

	var chunks []string
	err := (BailianChat{BaseURL: server.URL, APIKey: "key", Model: "test"}).Stream(context.Background(), "system", "user", func(chunk string) error {
		chunks = append(chunks, chunk)
		return nil
	})
	if err != nil || len(chunks) != 2 || chunks[0]+chunks[1] != "hello world" {
		t.Fatalf("stream chunks=%q err=%v", chunks, err)
	}
}

func TestBailianChatStreamCancellation(t *testing.T) {
	started := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"first\"}}]}\n\n"))
		w.(http.Flusher).Flush()
		close(started)
		<-r.Context().Done()
	}))
	defer server.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	err := (BailianChat{BaseURL: server.URL, APIKey: "key", Model: "test"}).Stream(ctx, "system", "user", func(string) error {
		cancel()
		return context.Canceled
	})
	<-started
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("stream cancellation error=%v, want context.Canceled", err)
	}
}

func TestBailianChatStreamIgnoresNonDataSSELines(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		writer := bufio.NewWriter(w)
		_, _ = writer.WriteString(": keep-alive\n\nevent: message\ndata: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n")
		_ = writer.Flush()
	}))
	defer server.Close()
	var got string
	err := (BailianChat{BaseURL: server.URL, APIKey: "key", Model: "test"}).Stream(context.Background(), "system", "user", func(chunk string) error { got += chunk; return nil })
	if err != nil || got != "ok" {
		t.Fatalf("stream got=%q err=%v", got, err)
	}
}

func TestOllamaHealthChecksConfiguredModel(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/tags" || r.Method != http.MethodGet {
			t.Fatalf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"models":[{"name":"bge-m3:latest"}]}`))
	}))
	defer server.Close()

	if err := (OllamaEmbedder{BaseURL: server.URL, Model: "bge-m3"}).Health(context.Background()); err != nil {
		t.Fatalf("configured model reported unhealthy: %v", err)
	}
	if err := (OllamaEmbedder{BaseURL: server.URL, Model: "missing"}).Health(context.Background()); err == nil {
		t.Fatal("missing model reported healthy")
	}
}

func TestBailianHealthUsesModelsEndpoint(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/models" || r.Header.Get("Authorization") != "Bearer key" {
			t.Fatalf("unexpected request: %s %s auth=%q", r.Method, r.URL.Path, r.Header.Get("Authorization"))
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	client := BailianChat{BaseURL: server.URL, APIKey: "key", Model: "test"}
	if err := client.Health(context.Background()); err != nil {
		t.Fatalf("healthy models endpoint returned error: %v", err)
	}
	server.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
	})
	if err := client.Health(context.Background()); err == nil {
		t.Fatal("unauthorized models endpoint reported healthy")
	}
}

func TestRerankerHealth(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/health" || r.Method != http.MethodGet {
			t.Fatalf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()
	client := LlamaCppReranker{BaseURL: server.URL}
	if err := client.Health(context.Background()); err != nil {
		t.Fatalf("healthy reranker returned error: %v", err)
	}
	server.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	})
	if err := client.Health(context.Background()); err == nil {
		t.Fatal("unavailable reranker reported healthy")
	}
}
