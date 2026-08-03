package rag

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/jackc/pgx/v5/pgconn"
	"github.com/lfd/minirag/internal/observability"
	"github.com/lfd/minirag/internal/store"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

var defaultRetryDelays = []time.Duration{time.Second, 5 * time.Second, 20 * time.Second, time.Minute}

type ingestionJobStore interface {
	RecoverExpiredIngestionJobs(context.Context) (int64, error)
	ClaimNextIngestionJob(context.Context, string, time.Duration) (*store.IngestionJob, error)
	RenewIngestionJobLease(context.Context, string, string, time.Duration) (bool, error)
	CompleteIngestionJob(context.Context, string, string, string) (bool, error)
	FailIngestionJob(context.Context, string, string, string, bool, time.Duration) (bool, error)
	ReleaseIngestionJob(context.Context, string, string) (bool, error)
	QueueState(context.Context) (int, int, error)
}

type ingestionService interface {
	IndexDocument(context.Context, string, string, []byte) (store.Document, error)
	ReindexDocument(context.Context, string, string, []byte) error
	ReindexStoredChunks(context.Context, string) error
}

type IngestionWorker struct {
	Store            ingestionJobStore
	Service          ingestionService
	Logger           *slog.Logger
	WorkerID         string
	LeaseDuration    time.Duration
	PollInterval     time.Duration
	RecoveryInterval time.Duration
	JobTimeout       time.Duration
	RetryDelays      []time.Duration
	Metrics          *observability.Metrics

	done      chan struct{}
	startOnce sync.Once
	running   atomic.Bool
}

func NewIngestionWorker(database *store.Store, service *Service, logger *slog.Logger) *IngestionWorker {
	return &IngestionWorker{
		Store:            database,
		Service:          service,
		Logger:           logger,
		WorkerID:         newID("worker"),
		LeaseDuration:    2 * time.Minute,
		PollInterval:     500 * time.Millisecond,
		RecoveryInterval: 15 * time.Second,
		JobTimeout:       10 * time.Minute,
		RetryDelays:      append([]time.Duration(nil), defaultRetryDelays...),
		done:             make(chan struct{}),
	}
}

func (w *IngestionWorker) Start(ctx context.Context) error {
	if w.Store == nil || w.Service == nil {
		return fmt.Errorf("ingestion worker requires store and service")
	}
	recovered, err := w.Store.RecoverExpiredIngestionJobs(ctx)
	if err != nil {
		return fmt.Errorf("recover expired ingestion jobs: %w", err)
	}
	if recovered > 0 {
		w.logger().Info("recovered expired ingestion jobs", "count", recovered)
	}
	w.refreshQueueMetrics(ctx)
	w.running.Store(true)
	started := false
	w.startOnce.Do(func() {
		started = true
		go w.loop(ctx)
	})
	if !started {
		return fmt.Errorf("ingestion worker already started")
	}
	return nil
}

// Submit remains for API compatibility. PostgreSQL is the queue; the worker
// discovers newly committed jobs on its next poll.
func (w *IngestionWorker) Submit(string) {}

func (w *IngestionWorker) Wait() {
	<-w.done
}

func (w *IngestionWorker) Ready() bool { return w != nil && w.running.Load() }

func (w *IngestionWorker) loop(ctx context.Context) {
	defer func() { w.running.Store(false); close(w.done) }()
	lastRecovery := time.Now()
	lastMetrics := time.Time{}
	for {
		if ctx.Err() != nil {
			return
		}
		if time.Since(lastRecovery) >= w.RecoveryInterval {
			if recovered, err := w.Store.RecoverExpiredIngestionJobs(ctx); err != nil {
				w.logger().Error("recover expired ingestion jobs", "error", err)
			} else if recovered > 0 {
				w.logger().Info("recovered expired ingestion jobs", "count", recovered)
			}
			lastRecovery = time.Now()
		}

		acquireStarted := time.Now()
		job, err := w.Store.ClaimNextIngestionJob(ctx, w.WorkerID, w.LeaseDuration)
		if err != nil {
			_, acquireSpan := observability.Start(ctx, "Acquire Job", trace.WithTimestamp(acquireStarted))
			observability.End(acquireSpan, err)
			if w.Metrics != nil {
				w.Metrics.ObserveStage("queue", "acquire_job", acquireStarted)
			}
			w.logger().Error("claim ingestion job", "error", err)
			w.waitForNextPoll(ctx)
			continue
		}
		if job == nil {
			if time.Since(lastMetrics) >= 5*time.Second {
				w.refreshQueueMetrics(ctx)
				lastMetrics = time.Now()
			}
			w.waitForNextPoll(ctx)
			continue
		}
		acquireParent := observability.Extract(ctx, job.TraceContext)
		acquireCtx, acquireSpan := observability.Start(acquireParent, "Acquire Job", trace.WithTimestamp(acquireStarted))
		observability.End(acquireSpan, nil)
		if w.Metrics != nil {
			w.Metrics.ObserveStage("queue", "acquire_job", acquireStarted)
		}
		if w.Metrics != nil {
			w.Metrics.ObserveQueue("acquired")
		}
		w.process(acquireCtx, job)
	}
}

func (w *IngestionWorker) waitForNextPoll(ctx context.Context) {
	timer := time.NewTimer(w.PollInterval)
	defer timer.Stop()
	select {
	case <-ctx.Done():
	case <-timer.C:
	}
}

func (w *IngestionWorker) process(ctx context.Context, job *store.IngestionJob) {
	jobCtx, jobSpan := observability.Start(ctx, "Durable Queue Job")
	jobSpan.SetAttributes(attribute.String("job.id", job.ID), attribute.Int("job.attempt", job.Attempts))
	jobCtx, cancelJob := context.WithTimeout(jobCtx, w.JobTimeout)
	heartbeatCtx, stopHeartbeat := context.WithCancel(jobCtx)
	heartbeatDone := make(chan error, 1)
	go w.heartbeat(heartbeatCtx, job.ID, cancelJob, heartbeatDone)

	documentID, err := w.index(jobCtx, job)
	defer func() { observability.End(jobSpan, err) }()
	stopHeartbeat()
	heartbeatErr := <-heartbeatDone
	cancelJob()

	if ctx.Err() != nil {
		w.release(job.ID, "worker shutdown")
		return
	}
	if heartbeatErr != nil {
		err = heartbeatErr
		w.logger().Warn("ingestion job lease lost", "job_id", job.ID, "error", heartbeatErr)
		w.release(job.ID, "lease renewal failed")
		return
	}
	if err == nil {
		completeCtx, cancel := context.WithTimeout(context.WithoutCancel(jobCtx), 5*time.Second)
		defer cancel()
		completeCtx, completeSpan := observability.Start(completeCtx, "Complete")
		completeStarted := time.Now()
		completed, updateErr := w.Store.CompleteIngestionJob(completeCtx, job.ID, w.WorkerID, documentID)
		observability.End(completeSpan, updateErr)
		if w.Metrics != nil {
			w.Metrics.ObserveStage("queue", "complete", completeStarted)
		}
		if updateErr != nil {
			err = updateErr
			w.logger().Error("complete ingestion job", "job_id", job.ID, "error", updateErr)
			return
		}
		if !completed {
			err = errors.New("job completion rejected after lease changed")
			w.logger().Warn("ingestion job completion rejected after lease changed", "job_id", job.ID)
			return
		}
		w.logger().Info("ingestion job completed", "job_id", job.ID, "document_id", documentID, "reindex", job.Reindex)
		if w.Metrics != nil {
			w.Metrics.ObserveQueue("completed")
			w.refreshQueueMetrics(completeCtx)
		}
		return
	}

	retry := isRetryableIngestionError(err) && job.RetryCount < len(w.RetryDelays)
	delay := time.Duration(0)
	if retry {
		delay = w.RetryDelays[job.RetryCount]
	}
	statusCtx, cancel := context.WithTimeout(context.WithoutCancel(jobCtx), 5*time.Second)
	defer cancel()
	statusName := "Failed"
	if retry {
		statusName = "Retry"
	}
	statusCtx, statusSpan := observability.Start(statusCtx, statusName)
	statusStarted := time.Now()
	updated, updateErr := w.Store.FailIngestionJob(statusCtx, job.ID, w.WorkerID, truncateError(err.Error(), 4096), retry, delay)
	observability.End(statusSpan, updateErr)
	if w.Metrics != nil {
		w.Metrics.ObserveStage("queue", strings.ToLower(statusName), statusStarted)
	}
	if updateErr != nil {
		w.logger().Error("update ingestion job failure", "job_id", job.ID, "error", updateErr)
		return
	}
	if !updated {
		w.logger().Warn("ingestion job failure rejected after lease changed", "job_id", job.ID)
		return
	}
	if retry {
		if w.Metrics != nil {
			w.Metrics.ObserveQueue("retried")
			w.refreshQueueMetrics(statusCtx)
		}
		w.logger().Warn("ingestion job retry scheduled", "job_id", job.ID, "retry_count", job.RetryCount+1, "retry_in", delay, "error", err)
		return
	}
	if w.Metrics != nil {
		w.Metrics.ObserveQueue("failed")
		w.refreshQueueMetrics(statusCtx)
	}
	w.logger().Error("ingestion job failed", "job_id", job.ID, "attempts", job.Attempts, "error", err)
}

func (w *IngestionWorker) index(ctx context.Context, job *store.IngestionJob) (string, error) {
	if job.Reindex {
		if len(job.Payload) == 0 {
			return job.DocumentID, w.Service.ReindexStoredChunks(ctx, job.DocumentID)
		}
		return job.DocumentID, w.Service.ReindexDocument(ctx, job.DocumentID, job.Name, job.Payload)
	}
	document, err := w.Service.IndexDocument(ctx, job.Name, job.ContentType, job.Payload)
	return document.ID, err
}

func (w *IngestionWorker) heartbeat(ctx context.Context, jobID string, cancelJob context.CancelFunc, done chan<- error) {
	interval := w.LeaseDuration / 3
	if interval <= 0 {
		interval = time.Second
	}
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			done <- nil
			return
		case <-ticker.C:
			renewStarted := time.Now()
			renewParent, renewSpan := observability.Start(ctx, "Lease Renew")
			renewCtx, cancel := context.WithTimeout(renewParent, interval)
			renewed, err := w.Store.RenewIngestionJobLease(renewCtx, jobID, w.WorkerID, w.LeaseDuration)
			cancel()
			observability.End(renewSpan, err)
			if w.Metrics != nil {
				w.Metrics.ObserveStage("queue", "lease_renew", renewStarted)
			}
			if err != nil {
				if ctx.Err() != nil {
					done <- nil
					return
				}
				cancelJob()
				done <- fmt.Errorf("renew lease: %w", err)
				return
			}
			if !renewed {
				cancelJob()
				done <- errors.New("lease is no longer owned by this worker")
				return
			}
		}
	}
}

func (w *IngestionWorker) refreshQueueMetrics(ctx context.Context) {
	if w.Metrics == nil {
		return
	}
	queued, processing, err := w.Store.QueueState(ctx)
	if err == nil {
		w.Metrics.SetQueueState(queued, processing)
	}
}

func (w *IngestionWorker) release(jobID, reason string) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	released, err := w.Store.ReleaseIngestionJob(ctx, jobID, w.WorkerID)
	if err != nil {
		w.logger().Error("release ingestion job", "job_id", jobID, "reason", reason, "error", err)
		return
	}
	if released {
		w.logger().Info("released ingestion job", "job_id", jobID, "reason", reason)
	}
}

func isRetryableIngestionError(err error) bool {
	if err == nil || errors.Is(err, context.Canceled) {
		return false
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return true
	}
	var marked interface{ Retryable() bool }
	if errors.As(err, &marked) {
		return marked.Retryable()
	}
	var networkError net.Error
	if errors.As(err, &networkError) {
		return true
	}
	if pgconn.SafeToRetry(err) || pgconn.Timeout(err) {
		return true
	}
	var postgresError *pgconn.PgError
	if !errors.As(err, &postgresError) {
		return false
	}
	switch postgresError.Code {
	case "23505", "40001", "40P01", "55P03", "57P01", "57P02", "57P03":
		return true
	}
	return len(postgresError.Code) >= 2 && (postgresError.Code[:2] == "08" || postgresError.Code[:2] == "53")
}

func truncateError(message string, maxBytes int) string {
	if maxBytes < 1 || len(message) <= maxBytes {
		return message
	}
	return message[:maxBytes]
}

func (w *IngestionWorker) logger() *slog.Logger {
	if w.Logger != nil {
		return w.Logger
	}
	return slog.Default()
}
