package observability

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestMetricsMiddlewareAndHandler(t *testing.T) {
	metrics := NewMetrics()
	handler := metrics.Middleware(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusCreated) }), nil)
	handler.ServeHTTP(httptest.NewRecorder(), httptest.NewRequest(http.MethodPost, "/v1/query", nil))
	recorder := httptest.NewRecorder()
	metrics.Handler().ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, "/metrics", nil))
	text := recorder.Body.String()
	if !strings.Contains(text, `minirag_http_requests_total{method="POST",route="/v1/query",status="201"} 1`) {
		t.Fatalf("HTTP counter missing:\n%s", text)
	}
	if !strings.Contains(text, "minirag_http_request_duration_seconds_count") {
		t.Fatalf("HTTP duration missing:\n%s", text)
	}
}

func TestTraceCarrierRoundTrip(t *testing.T) {
	telemetry, err := New(context.Background(), Config{ServiceName: "test"})
	if err != nil {
		t.Fatal(err)
	}
	ctx, span := Start(context.Background(), "parent")
	extracted := Extract(context.Background(), Inject(ctx))
	_, child := Start(extracted, "child")
	if span.SpanContext().TraceID() != child.SpanContext().TraceID() {
		t.Fatal("trace carrier did not preserve trace ID")
	}
	child.End()
	span.End()
	if err := telemetry.Shutdown(context.Background()); err != nil {
		t.Fatal(err)
	}
}

func TestRouteNormalizesIdentifiers(t *testing.T) {
	for path, want := range map[string]string{"/v1/jobs/job-123": "/v1/jobs/:id", "/v1/documents/doc-123/reindex": "/v1/documents/:id", "/unknown": "/other"} {
		if got := Route(path); got != want {
			t.Fatalf("Route(%q) = %q, want %q", path, got, want)
		}
	}
}
