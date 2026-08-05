package rag

import (
	"context"
	"errors"
	"log/slog"
	"sync"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgconn"
	"github.com/lfd/minirag/internal/observability"
	"github.com/lfd/minirag/internal/store"
	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
)

type markedRetryableError bool

func (e markedRetryableError) Error() string   { return "marked error" }
func (e markedRetryableError) Retryable() bool { return bool(e) }

func TestDefaultRetryDelays(t *testing.T) {
	want := []time.Duration{time.Second, 5 * time.Second, 20 * time.Second, time.Minute}
	if len(defaultRetryDelays) != len(want) {
		t.Fatalf("got %d retry delays, want %d", len(defaultRetryDelays), len(want))
	}
	for i := range want {
		if defaultRetryDelays[i] != want[i] {
			t.Fatalf("retry delay %d = %s, want %s", i, defaultRetryDelays[i], want[i])
		}
	}
}

func TestIsRetryableIngestionError(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want bool
	}{
		{name: "deadline", err: context.DeadlineExceeded, want: true},
		{name: "cancel", err: context.Canceled, want: false},
		{name: "marked transient", err: markedRetryableError(true), want: true},
		{name: "marked permanent", err: markedRetryableError(false), want: false},
		{name: "serialization", err: &pgconn.PgError{Code: "40001"}, want: true},
		{name: "invalid data", err: &pgconn.PgError{Code: "22000"}, want: false},
		{name: "plain error", err: errors.New("invalid document"), want: false},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := isRetryableIngestionError(test.err); got != test.want {
				t.Fatalf("isRetryableIngestionError() = %t, want %t", got, test.want)
			}
		})
	}
}

func TestTruncateError(t *testing.T) {
	if got := truncateError("123456", 4); got != "1234" {
		t.Fatalf("truncateError() = %q", got)
	}
}

func TestWorkerStartRecoversExpiredJobs(t *testing.T) {
	queue := &fakeIngestionJobStore{recovered: 2}
	worker := newTestWorker(queue, fakeIngestionService{})
	ctx, cancel := context.WithCancel(context.Background())
	if err := worker.Start(ctx); err != nil {
		t.Fatal(err)
	}
	cancel()
	worker.Wait()
	if queue.recoverCalls != 1 {
		t.Fatalf("RecoverExpiredIngestionJobs called %d times, want 1", queue.recoverCalls)
	}
}

func TestWorkerSchedulesRetryableFailure(t *testing.T) {
	queue := &fakeIngestionJobStore{}
	service := fakeIngestionService{indexErr: markedRetryableError(true)}
	worker := newTestWorker(queue, service)
	job := &store.IngestionJob{ID: "job-retry", Name: "retry.txt", RetryCount: 1}

	worker.process(context.Background(), job)
	if queue.failedID != job.ID || !queue.failedRetry || queue.failedDelay != 5*time.Second {
		t.Fatalf("retry update = id %q retry %t delay %s", queue.failedID, queue.failedRetry, queue.failedDelay)
	}
}

func TestWorkerMarksPermanentFailure(t *testing.T) {
	queue := &fakeIngestionJobStore{}
	service := fakeIngestionService{indexErr: errors.New("invalid document")}
	worker := newTestWorker(queue, service)
	job := &store.IngestionJob{ID: "job-failed", Name: "failed.txt"}

	worker.process(context.Background(), job)
	if queue.failedID != job.ID || queue.failedRetry || queue.failedMessage != "invalid document" {
		t.Fatalf("failure update = id %q retry %t message %q", queue.failedID, queue.failedRetry, queue.failedMessage)
	}
}

func TestWorkerCancellationReleasesLease(t *testing.T) {
	queue := &fakeIngestionJobStore{}
	service := fakeIngestionService{waitForCancel: true}
	worker := newTestWorker(queue, service)
	job := &store.IngestionJob{ID: "job-shutdown", Name: "shutdown.txt"}
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		worker.process(ctx, job)
		close(done)
	}()
	cancel()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("worker did not stop after cancellation")
	}
	if queue.releasedID != job.ID {
		t.Fatalf("released job %q, want %q", queue.releasedID, job.ID)
	}
}

func TestWorkerJobSpanInheritsAcquireSpan(t *testing.T) {
	exporter := tracetest.NewInMemoryExporter()
	provider := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exporter))
	previous := otel.GetTracerProvider()
	otel.SetTracerProvider(provider)
	t.Cleanup(func() {
		_ = provider.Shutdown(context.Background())
		otel.SetTracerProvider(previous)
	})

	ctx, acquireSpan := observability.Start(context.Background(), "Acquire Job")
	acquireSpan.End()
	worker := newTestWorker(&fakeIngestionJobStore{}, fakeIngestionService{})
	worker.process(ctx, &store.IngestionJob{ID: "job-trace", Name: "trace.txt"})

	spans := exporter.GetSpans()
	var acquire, job tracetest.SpanStub
	for _, span := range spans {
		switch span.Name {
		case "Acquire Job":
			acquire = span
		case "Durable Queue Job":
			job = span
		}
	}
	if !acquire.SpanContext.IsValid() || !job.SpanContext.IsValid() {
		t.Fatalf("missing worker spans: %#v", spans)
	}
	if job.Parent.SpanID() != acquire.SpanContext.SpanID() {
		t.Fatalf("job parent = %s, want acquire span %s", job.Parent.SpanID(), acquire.SpanContext.SpanID())
	}
}

type fakeIngestionJobStore struct {
	mu            sync.Mutex
	recovered     int64
	recoverCalls  int
	failedID      string
	failedMessage string
	failedRetry   bool
	failedDelay   time.Duration
	releasedID    string
}

func (s *fakeIngestionJobStore) RecoverExpiredIngestionJobs(context.Context) (int64, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.recoverCalls++
	return s.recovered, nil
}

func (*fakeIngestionJobStore) ClaimNextIngestionJob(context.Context, string, time.Duration) (*store.IngestionJob, error) {
	return nil, nil
}

func (*fakeIngestionJobStore) QueueState(context.Context) (int, int, error) { return 0, 0, nil }

func (*fakeIngestionJobStore) RenewIngestionJobLease(context.Context, string, string, time.Duration) (bool, error) {
	return true, nil
}

func (*fakeIngestionJobStore) CompleteIngestionJob(context.Context, string, string, string) (bool, error) {
	return true, nil
}

func (s *fakeIngestionJobStore) FailIngestionJob(_ context.Context, id, _ string, message string, retry bool, delay time.Duration) (bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.failedID = id
	s.failedMessage = message
	s.failedRetry = retry
	s.failedDelay = delay
	return true, nil
}

func (s *fakeIngestionJobStore) ReleaseIngestionJob(_ context.Context, id, _ string) (bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.releasedID = id
	return true, nil
}

type fakeIngestionService struct {
	indexErr      error
	waitForCancel bool
}

func (s fakeIngestionService) IndexDocument(ctx context.Context, _, _ string, _ []byte) (store.Document, error) {
	if s.waitForCancel {
		<-ctx.Done()
		return store.Document{}, ctx.Err()
	}
	return store.Document{ID: "doc-test"}, s.indexErr
}

func (fakeIngestionService) ReindexDocument(context.Context, string, string, []byte) error {
	return nil
}

func (fakeIngestionService) ReindexStoredChunks(context.Context, string) error {
	return nil
}

func newTestWorker(queue ingestionJobStore, service ingestionService) *IngestionWorker {
	return &IngestionWorker{
		Store:            queue,
		Service:          service,
		Logger:           slog.New(slog.NewTextHandler(testDiscardWriter{}, nil)),
		WorkerID:         "worker-test",
		LeaseDuration:    time.Hour,
		PollInterval:     time.Millisecond,
		RecoveryInterval: time.Hour,
		JobTimeout:       time.Minute,
		RetryDelays:      append([]time.Duration(nil), defaultRetryDelays...),
		done:             make(chan struct{}),
	}
}

type testDiscardWriter struct{}

func (testDiscardWriter) Write(p []byte) (int, error) { return len(p), nil }
