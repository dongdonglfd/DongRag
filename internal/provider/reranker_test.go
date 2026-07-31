package provider

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestLlamaCppReranker(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/rerank" || r.Method != http.MethodPost {
			t.Fatalf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"results":[{"index":0,"relevance_score":0.2},{"index":1,"relevance_score":0.9}]}`))
	}))
	defer server.Close()
	got, err := (LlamaCppReranker{BaseURL: server.URL}).Rerank(context.Background(), "q", []string{"a", "b"})
	if err != nil || len(got) != 2 || got[0].Index != 1 || got[0].Score != 0.9 {
		t.Fatalf("unexpected result: %#v, %v", got, err)
	}
}
