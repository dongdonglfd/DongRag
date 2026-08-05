package web

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestReadyChecksDependencies(t *testing.T) {
	server := &Server{
		PostgresHealth:  func(context.Context) error { return nil },
		EmbeddingHealth: func(context.Context) error { return nil },
		LLMHealth:       func(context.Context) error { return nil },
		WorkerReady:     func() bool { return true },
	}
	recorder := httptest.NewRecorder()
	server.ready(recorder, httptest.NewRequest(http.MethodGet, "/readyz", nil))
	if recorder.Code != http.StatusOK {
		t.Fatalf("ready returned %d: %s", recorder.Code, recorder.Body.String())
	}
	server.LLMHealth = func(context.Context) error { return errors.New("offline") }
	recorder = httptest.NewRecorder()
	server.ready(recorder, httptest.NewRequest(http.MethodGet, "/readyz", nil))
	if recorder.Code != http.StatusServiceUnavailable {
		t.Fatalf("unready returned %d: %s", recorder.Code, recorder.Body.String())
	}
}

func TestReadyReportsEachUnavailableDependency(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*Server)
		want   string
	}{
		{name: "postgres", want: "postgres", mutate: func(s *Server) { s.PostgresHealth = failingHealth }},
		{name: "embedding", want: "embedding", mutate: func(s *Server) { s.EmbeddingHealth = failingHealth }},
		{name: "llm", want: "llm", mutate: func(s *Server) { s.LLMHealth = failingHealth }},
		{name: "reranker", want: "reranker", mutate: func(s *Server) { s.RerankerHealth = failingHealth }},
		{name: "worker", want: "worker", mutate: func(s *Server) { s.WorkerReady = func() bool { return false } }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			server := readyTestServer()
			test.mutate(server)
			recorder := httptest.NewRecorder()
			server.ready(recorder, httptest.NewRequest(http.MethodGet, "/readyz", nil))
			if recorder.Code != http.StatusServiceUnavailable {
				t.Fatalf("status = %d, want %d: %s", recorder.Code, http.StatusServiceUnavailable, recorder.Body.String())
			}
			var response struct {
				Status       string            `json:"status"`
				Dependencies map[string]string `json:"dependencies"`
			}
			if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
				t.Fatal(err)
			}
			if response.Status != "not_ready" || response.Dependencies[test.want] != "not_ready" {
				t.Fatalf("response = %#v", response)
			}
		})
	}
}

func TestReadyTreatsDisabledRerankerAsReady(t *testing.T) {
	server := readyTestServer()
	server.RerankerHealth = nil
	recorder := httptest.NewRecorder()
	server.ready(recorder, httptest.NewRequest(http.MethodGet, "/readyz", nil))
	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d: %s", recorder.Code, recorder.Body.String())
	}
	var response struct {
		Dependencies map[string]string `json:"dependencies"`
	}
	if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	if response.Dependencies["reranker"] != "disabled" {
		t.Fatalf("reranker = %q, want disabled", response.Dependencies["reranker"])
	}
}

func readyTestServer() *Server {
	return &Server{
		PostgresHealth:  func(context.Context) error { return nil },
		EmbeddingHealth: func(context.Context) error { return nil },
		LLMHealth:       func(context.Context) error { return nil },
		RerankerHealth:  func(context.Context) error { return nil },
		WorkerReady:     func() bool { return true },
	}
}

func failingHealth(context.Context) error { return errors.New("offline") }

func TestDocumentReindexID(t *testing.T) {
	tests := []struct {
		path string
		id   string
		ok   bool
	}{
		{path: "/v1/documents/doc-123/reindex", id: "doc-123", ok: true},
		{path: "/v1/documents/doc-123/reindex/", id: "doc-123", ok: true},
		{path: "/v1/documents/doc-123", ok: false},
		{path: "/v1/documents/reindex", ok: false},
	}
	for _, test := range tests {
		id, ok := documentReindexID(test.path)
		if id != test.id || ok != test.ok {
			t.Errorf("documentReindexID(%q) = (%q, %v), want (%q, %v)", test.path, id, ok, test.id, test.ok)
		}
	}
}

func TestHandlerServesEmbeddedFrontend(t *testing.T) {
	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodGet, "/", nil)
	(&Server{}).Handler().ServeHTTP(recorder, request)
	if recorder.Code != http.StatusOK || recorder.Header().Get("Content-Type") != "text/html; charset=utf-8" {
		t.Fatalf("unexpected response: %d %s", recorder.Code, recorder.Header().Get("Content-Type"))
	}

	recorder = httptest.NewRecorder()
	request = httptest.NewRequest(http.MethodGet, "/static/app.js", nil)
	(&Server{}).Handler().ServeHTTP(recorder, request)
	if recorder.Code != http.StatusOK {
		t.Fatalf("static asset returned %d", recorder.Code)
	}
}

func TestWriteSSE(t *testing.T) {
	recorder := httptest.NewRecorder()
	if err := writeSSE(recorder, recorder, "token", map[string]string{"content": "hello\nworld"}); err != nil {
		t.Fatal(err)
	}
	want := "event: token\ndata: {\"content\":\"hello\\nworld\"}\n\n"
	if recorder.Body.String() != want || !recorder.Flushed {
		t.Fatalf("SSE response = %q flushed=%t", recorder.Body.String(), recorder.Flushed)
	}
}
