package observability

import (
	"context"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/trace"
)

const tracerName = "github.com/lfd/minirag"

type Config struct {
	ServiceName string
	Endpoint    string
	Insecure    bool
}

type Telemetry struct {
	Metrics  *Metrics
	Tracer   trace.Tracer
	provider *sdktrace.TracerProvider
}

func New(ctx context.Context, cfg Config) (*Telemetry, error) {
	name := cfg.ServiceName
	if strings.TrimSpace(name) == "" {
		name = "minirag"
	}
	options := []sdktrace.TracerProviderOption{
		sdktrace.WithResource(resource.NewSchemaless(attribute.String("service.name", name))),
		sdktrace.WithSampler(sdktrace.AlwaysSample()),
	}
	if strings.TrimSpace(cfg.Endpoint) != "" {
		endpoint := cfg.Endpoint
		var exporterOptions []otlptracehttp.Option
		if strings.Contains(endpoint, "://") {
			exporterOptions = append(exporterOptions, otlptracehttp.WithEndpointURL(endpoint))
		} else {
			exporterOptions = append(exporterOptions, otlptracehttp.WithEndpoint(endpoint))
		}
		if cfg.Insecure || strings.HasPrefix(endpoint, "http://") || !strings.Contains(endpoint, "://") {
			exporterOptions = append(exporterOptions, otlptracehttp.WithInsecure())
		}
		exporter, err := otlptracehttp.New(ctx, exporterOptions...)
		if err != nil {
			return nil, fmt.Errorf("create OTLP trace exporter: %w", err)
		}
		options = append(options, sdktrace.WithBatcher(exporter))
	}
	provider := sdktrace.NewTracerProvider(options...)
	otel.SetTracerProvider(provider)
	otel.SetTextMapPropagator(propagation.TraceContext{})
	return &Telemetry{Metrics: NewMetrics(), Tracer: provider.Tracer(tracerName), provider: provider}, nil
}

func (t *Telemetry) Shutdown(ctx context.Context) error {
	if t == nil || t.provider == nil {
		return nil
	}
	return t.provider.Shutdown(ctx)
}

func Start(ctx context.Context, name string, options ...trace.SpanStartOption) (context.Context, trace.Span) {
	return otel.Tracer(tracerName).Start(ctx, name, options...)
}

func End(span trace.Span, err error) {
	if err != nil {
		span.RecordError(err)
		span.SetStatus(codes.Error, err.Error())
	}
	span.End()
}

func Inject(ctx context.Context) map[string]string {
	carrier := propagation.MapCarrier{}
	otel.GetTextMapPropagator().Inject(ctx, carrier)
	result := make(map[string]string, len(carrier))
	for key, value := range carrier {
		result[key] = value
	}
	return result
}

func Extract(ctx context.Context, carrier map[string]string) context.Context {
	if len(carrier) == 0 {
		return ctx
	}
	return otel.GetTextMapPropagator().Extract(ctx, propagation.MapCarrier(carrier))
}

type Metrics struct {
	Registry      *prometheus.Registry
	HTTPRequests  *prometheus.CounterVec
	HTTPDuration  *prometheus.HistogramVec
	QueryDuration prometheus.Histogram
	StageDuration *prometheus.HistogramVec
	QueueJobs     *prometheus.CounterVec
	QueueDepth    prometheus.Gauge
	QueueRunning  prometheus.Gauge
}

func NewMetrics() *Metrics {
	registry := prometheus.NewRegistry()
	m := &Metrics{
		Registry:      registry,
		HTTPRequests:  prometheus.NewCounterVec(prometheus.CounterOpts{Name: "minirag_http_requests_total", Help: "Total HTTP requests handled by MiniRAG."}, []string{"method", "route", "status"}),
		HTTPDuration:  prometheus.NewHistogramVec(prometheus.HistogramOpts{Name: "minirag_http_request_duration_seconds", Help: "HTTP request duration in seconds.", Buckets: prometheus.DefBuckets}, []string{"method", "route"}),
		QueryDuration: prometheus.NewHistogram(prometheus.HistogramOpts{Name: "minirag_query_duration_seconds", Help: "End-to-end query pipeline duration in seconds.", Buckets: prometheus.DefBuckets}),
		StageDuration: prometheus.NewHistogramVec(prometheus.HistogramOpts{Name: "minirag_stage_duration_seconds", Help: "MiniRAG pipeline stage duration in seconds.", Buckets: prometheus.DefBuckets}, []string{"pipeline", "stage"}),
		QueueJobs:     prometheus.NewCounterVec(prometheus.CounterOpts{Name: "minirag_queue_jobs_total", Help: "Durable Queue job outcomes."}, []string{"outcome"}),
		QueueDepth:    prometheus.NewGauge(prometheus.GaugeOpts{Name: "minirag_queue_depth", Help: "Number of queued Durable Queue jobs."}),
		QueueRunning:  prometheus.NewGauge(prometheus.GaugeOpts{Name: "minirag_queue_processing", Help: "Number of processing Durable Queue jobs."}),
	}
	registry.MustRegister(m.HTTPRequests, m.HTTPDuration, m.QueryDuration, m.StageDuration, m.QueueJobs, m.QueueDepth, m.QueueRunning, prometheus.NewGoCollector())
	return m
}

func (m *Metrics) Handler() http.Handler {
	if m == nil || m.Registry == nil {
		return http.NotFoundHandler()
	}
	return promhttp.HandlerFor(m.Registry, promhttp.HandlerOpts{})
}

func (m *Metrics) ObserveStage(pipeline, stage string, started time.Time) {
	if m == nil || m.StageDuration == nil {
		return
	}
	m.StageDuration.WithLabelValues(pipeline, stage).Observe(time.Since(started).Seconds())
}

func (m *Metrics) ObserveQueue(outcome string) {
	if m != nil && m.QueueJobs != nil {
		m.QueueJobs.WithLabelValues(outcome).Inc()
	}
}

func (m *Metrics) SetQueueState(queued, processing int) {
	if m == nil {
		return
	}
	m.QueueDepth.Set(float64(queued))
	m.QueueRunning.Set(float64(processing))
}

func (m *Metrics) Middleware(next http.Handler, route func(*http.Request) string) http.Handler {
	if route == nil {
		route = func(req *http.Request) string { return Route(req.URL.Path) }
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		started := time.Now()
		name := route(r)
		parent := otel.GetTextMapPropagator().Extract(r.Context(), propagation.HeaderCarrier(r.Header))
		ctx, span := Start(parent, "HTTP "+r.Method+" "+name, trace.WithSpanKind(trace.SpanKindServer))
		defer span.End()
		r = r.WithContext(ctx)
		writer := &responseWriter{ResponseWriter: w}
		next.ServeHTTP(writer, r)
		if writer.status == 0 {
			writer.status = http.StatusOK
		}
		status := strconv.Itoa(writer.status)
		m.HTTPRequests.WithLabelValues(r.Method, name, status).Inc()
		m.HTTPDuration.WithLabelValues(r.Method, name).Observe(time.Since(started).Seconds())
		span.SetAttributes(attribute.String("http.method", r.Method), attribute.String("http.route", name), attribute.Int("http.status_code", writer.status))
		if writer.status >= 500 {
			span.SetStatus(codes.Error, http.StatusText(writer.status))
		}
	})
}

func Route(path string) string {
	switch {
	case path == "/healthz":
		return "/healthz"
	case path == "/readyz":
		return "/readyz"
	case path == "/metrics":
		return "/metrics"
	case path == "/v1/documents":
		return "/v1/documents"
	case strings.HasPrefix(path, "/v1/documents/"):
		return "/v1/documents/:id"
	case strings.HasPrefix(path, "/v1/jobs/"):
		return "/v1/jobs/:id"
	case path == "/v1/query":
		return "/v1/query"
	case path == "/v1/chat/stream":
		return "/v1/chat/stream"
	default:
		return "/other"
	}
}

type responseWriter struct {
	http.ResponseWriter
	status int
}

func (w *responseWriter) WriteHeader(status int) {
	if w.status != 0 {
		return
	}
	w.status = status
	w.ResponseWriter.WriteHeader(status)
}
func (w *responseWriter) Write(p []byte) (int, error) {
	if w.status == 0 {
		w.WriteHeader(http.StatusOK)
	}
	return w.ResponseWriter.Write(p)
}
func (w *responseWriter) Flush() {
	if w.status == 0 {
		w.WriteHeader(http.StatusOK)
	}
	if f, ok := w.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}
