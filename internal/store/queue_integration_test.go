package store

import (
	"context"
	"fmt"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

func TestQueueClaimSkipsLockedJob(t *testing.T) {
	s := newQueueIntegrationStore(t)
	ctx := context.Background()
	first := createQueueTestJob(t, s, "first.txt")
	second := createQueueTestJob(t, s, "second.txt")
	_, err := s.Pool.Exec(ctx, `
UPDATE ingestion_jobs
SET next_attempt_at=CASE id WHEN $1 THEN now()-interval '2 seconds' ELSE now()-interval '1 second' END
WHERE id IN ($1, $2)`, first.ID, second.ID)
	if err != nil {
		t.Fatal(err)
	}

	tx, err := s.Pool.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer tx.Rollback(ctx)
	if err := tx.QueryRow(ctx, `SELECT id FROM ingestion_jobs WHERE id=$1 FOR UPDATE`, first.ID).Scan(new(string)); err != nil {
		t.Fatal(err)
	}

	claimed, err := s.ClaimNextIngestionJob(ctx, "worker-skip-locked", time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	if claimed == nil || claimed.ID != second.ID {
		t.Fatalf("claimed %#v, want unlocked job %s", claimed, second.ID)
	}
}

func TestQueueTraceContextSurvivesClaim(t *testing.T) {
	s := newQueueIntegrationStore(t)
	traceContext := map[string]string{"traceparent": "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01"}
	created, err := s.CreateIngestionJobWithTrace(context.Background(), "trace.txt", "text/plain", []byte("trace payload"), traceContext)
	if err != nil {
		t.Fatal(err)
	}
	claimed, err := s.ClaimNextIngestionJob(context.Background(), "worker-trace", time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	if claimed == nil || claimed.ID != created.ID {
		t.Fatalf("claimed job = %#v, want %s", claimed, created.ID)
	}
	if claimed.TraceContext["traceparent"] != traceContext["traceparent"] {
		t.Fatalf("claimed trace context = %#v, want %#v", claimed.TraceContext, traceContext)
	}
}

func TestQueueMultipleWorkersClaimOnce(t *testing.T) {
	s := newQueueIntegrationStore(t)
	createQueueTestJob(t, s, "only.txt")
	ctx := context.Background()

	start := make(chan struct{})
	results := make(chan *IngestionJob, 2)
	errors := make(chan error, 2)
	var wg sync.WaitGroup
	for _, workerID := range []string{"worker-a", "worker-b"} {
		wg.Add(1)
		go func(id string) {
			defer wg.Done()
			<-start
			job, err := s.ClaimNextIngestionJob(ctx, id, time.Minute)
			results <- job
			errors <- err
		}(workerID)
	}
	close(start)
	wg.Wait()
	close(results)
	close(errors)

	for err := range errors {
		if err != nil {
			t.Fatal(err)
		}
	}
	claimed := 0
	for job := range results {
		if job != nil {
			claimed++
		}
	}
	if claimed != 1 {
		t.Fatalf("%d workers claimed one job, want exactly 1", claimed)
	}
}

func TestQueueServiceRestartRecoversExpiredLease(t *testing.T) {
	s := newQueueIntegrationStore(t)
	created := createQueueTestJob(t, s, "restart.txt")
	ctx := context.Background()
	claimed, err := s.ClaimNextIngestionJob(ctx, "worker-before-crash", time.Minute)
	if err != nil || claimed == nil {
		t.Fatalf("initial claim: %#v, %v", claimed, err)
	}
	if _, err := s.Pool.Exec(ctx, `UPDATE ingestion_jobs SET lease_until=now()-interval '1 second' WHERE id=$1`, created.ID); err != nil {
		t.Fatal(err)
	}

	recovered, err := s.RecoverExpiredIngestionJobs(ctx)
	if err != nil || recovered != 1 {
		t.Fatalf("recovered %d jobs: %v", recovered, err)
	}
	restartedClaim, err := s.ClaimNextIngestionJob(ctx, "worker-after-restart", time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	if restartedClaim == nil || restartedClaim.ID != created.ID || restartedClaim.Attempts != 2 {
		t.Fatalf("claim after restart = %#v", restartedClaim)
	}
}

func TestQueueRetryAndPermanentFailure(t *testing.T) {
	s := newQueueIntegrationStore(t)
	created := createQueueTestJob(t, s, "retry.txt")
	ctx := context.Background()
	claimed, err := s.ClaimNextIngestionJob(ctx, "worker-retry", time.Minute)
	if err != nil || claimed == nil {
		t.Fatalf("initial claim: %#v, %v", claimed, err)
	}
	updated, err := s.FailIngestionJob(ctx, created.ID, "worker-retry", "temporary failure", true, time.Minute)
	if err != nil || !updated {
		t.Fatalf("schedule retry: updated=%t err=%v", updated, err)
	}
	job, err := s.GetIngestionJob(ctx, created.ID)
	if err != nil {
		t.Fatal(err)
	}
	if job.Status != "queued" || job.RetryCount != 1 || job.ErrorMessage != "temporary failure" {
		t.Fatalf("retry state = %#v", job)
	}
	if next, err := s.ClaimNextIngestionJob(ctx, "worker-too-early", time.Minute); err != nil || next != nil {
		t.Fatalf("delayed job was claimable: %#v, %v", next, err)
	}
	if _, err := s.Pool.Exec(ctx, `UPDATE ingestion_jobs SET next_attempt_at=now() WHERE id=$1`, created.ID); err != nil {
		t.Fatal(err)
	}
	claimed, err = s.ClaimNextIngestionJob(ctx, "worker-final", time.Minute)
	if err != nil || claimed == nil {
		t.Fatalf("retry claim: %#v, %v", claimed, err)
	}
	updated, err = s.FailIngestionJob(ctx, created.ID, "worker-final", "invalid document", false, 0)
	if err != nil || !updated {
		t.Fatalf("permanent failure: updated=%t err=%v", updated, err)
	}
	job, err = s.GetIngestionJob(ctx, created.ID)
	if err != nil {
		t.Fatal(err)
	}
	if job.Status != "failed" || job.RetryCount != 1 || job.ErrorMessage != "invalid document" {
		t.Fatalf("failed state = %#v", job)
	}
}

func TestQueueExpiredOwnerCannotCompleteAfterRecovery(t *testing.T) {
	s := newQueueIntegrationStore(t)
	created := createQueueTestJob(t, s, "fenced.txt")
	ctx := context.Background()
	if _, err := s.Pool.Exec(ctx, `
INSERT INTO documents(id, name, content_type, size_bytes, checksum)
VALUES('doc-fenced', 'fenced.txt', 'text/plain', 1, 'checksum-fenced')`); err != nil {
		t.Fatal(err)
	}
	claimed, err := s.ClaimNextIngestionJob(ctx, "worker-old", time.Minute)
	if err != nil || claimed == nil {
		t.Fatalf("initial claim: %#v, %v", claimed, err)
	}
	if _, err := s.Pool.Exec(ctx, `UPDATE ingestion_jobs SET lease_until=now()-interval '1 second' WHERE id=$1`, created.ID); err != nil {
		t.Fatal(err)
	}
	if recovered, err := s.RecoverExpiredIngestionJobs(ctx); err != nil || recovered != 1 {
		t.Fatalf("recover: count=%d err=%v", recovered, err)
	}
	claimed, err = s.ClaimNextIngestionJob(ctx, "worker-new", time.Minute)
	if err != nil || claimed == nil {
		t.Fatalf("replacement claim: %#v, %v", claimed, err)
	}
	completed, err := s.CompleteIngestionJob(ctx, created.ID, "worker-old", "doc-fenced")
	if err != nil || completed {
		t.Fatalf("expired owner completed job: completed=%t err=%v", completed, err)
	}
	completed, err = s.CompleteIngestionJob(ctx, created.ID, "worker-new", "doc-fenced")
	if err != nil || !completed {
		t.Fatalf("current owner did not complete job: completed=%t err=%v", completed, err)
	}
}

func TestQueueMigrationUpgradesExistingJobs(t *testing.T) {
	s := newQueueIntegrationStoreWithoutTables(t)
	ctx := context.Background()
	if _, err := s.Pool.Exec(ctx, `
CREATE TABLE documents (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    content_type TEXT NOT NULL,
    size_bytes BIGINT NOT NULL,
    checksum TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE ingestion_jobs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    content_type TEXT NOT NULL,
    payload BYTEA NOT NULL,
    is_reindex BOOLEAN NOT NULL DEFAULT FALSE,
    status TEXT NOT NULL CHECK (status IN ('queued', 'processing', 'completed', 'failed')),
    attempts INT NOT NULL DEFAULT 0,
    error_message TEXT NOT NULL DEFAULT '',
    document_id TEXT REFERENCES documents(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
INSERT INTO ingestion_jobs(id, name, content_type, payload, status)
VALUES('job-before-upgrade', 'old.txt', 'text/plain', 'old payload', 'processing')`); err != nil {
		t.Fatal(err)
	}
	if err := s.Migrate(ctx); err != nil {
		t.Fatal(err)
	}
	recovered, err := s.RecoverExpiredIngestionJobs(ctx)
	if err != nil || recovered != 1 {
		t.Fatalf("recover upgraded job: count=%d err=%v", recovered, err)
	}
	job, err := s.GetIngestionJob(ctx, "job-before-upgrade")
	if err != nil {
		t.Fatal(err)
	}
	if job == nil || job.Status != "queued" || job.RetryCount != 0 || job.WorkerID != "" || job.LeaseUntil != nil {
		t.Fatalf("upgraded job = %#v", job)
	}
}

func newQueueIntegrationStore(t *testing.T) *Store {
	t.Helper()
	s := newQueueIntegrationStoreWithoutTables(t)
	ctx := context.Background()
	if _, err := s.Pool.Exec(ctx, `
CREATE TABLE documents (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    content_type TEXT NOT NULL,
    size_bytes BIGINT NOT NULL,
    checksum TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE ingestion_jobs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    content_type TEXT NOT NULL,
    payload BYTEA NOT NULL,
    is_reindex BOOLEAN NOT NULL DEFAULT FALSE,
    status TEXT NOT NULL CHECK (status IN ('queued', 'processing', 'completed', 'failed')),
    attempts INT NOT NULL DEFAULT 0,
    retry_count INT NOT NULL DEFAULT 0,
    next_attempt_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    lease_until TIMESTAMPTZ,
    worker_id TEXT NOT NULL DEFAULT '',
    trace_context JSONB NOT NULL DEFAULT '{}'::jsonb,
    error_message TEXT NOT NULL DEFAULT '',
    document_id TEXT REFERENCES documents(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX ingestion_jobs_available_idx ON ingestion_jobs(next_attempt_at, created_at) WHERE status='queued';
CREATE INDEX ingestion_jobs_lease_idx ON ingestion_jobs(lease_until) WHERE status='processing'`); err != nil {
		t.Fatal(err)
	}
	return s
}

func newQueueIntegrationStoreWithoutTables(t *testing.T) *Store {
	t.Helper()
	databaseURL := strings.TrimSpace(os.Getenv("MINIRAG_TEST_DATABASE_URL"))
	if databaseURL == "" {
		t.Skip("MINIRAG_TEST_DATABASE_URL is not set")
	}
	ctx := context.Background()
	admin, err := pgxpool.New(ctx, databaseURL)
	if err != nil {
		t.Fatal(err)
	}
	schema := fmt.Sprintf("minirag_queue_test_%d", time.Now().UnixNano())
	identifier := pgx.Identifier{schema}.Sanitize()
	if _, err := admin.Exec(ctx, "CREATE SCHEMA "+identifier); err != nil {
		admin.Close()
		t.Fatal(err)
	}
	t.Cleanup(func() {
		defer admin.Close()
		if _, err := admin.Exec(context.Background(), "DROP SCHEMA "+identifier+" CASCADE"); err != nil {
			t.Errorf("drop test schema: %v", err)
		}
	})

	config, err := pgxpool.ParseConfig(databaseURL)
	if err != nil {
		t.Fatal(err)
	}
	config.ConnConfig.RuntimeParams["search_path"] = schema + ",public"
	pool, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(pool.Close)
	return &Store{Pool: pool}
}

func createQueueTestJob(t *testing.T, s *Store, name string) IngestionJob {
	t.Helper()
	job, err := s.CreateIngestionJob(context.Background(), name, "text/plain", []byte("test payload"))
	if err != nil {
		t.Fatal(err)
	}
	return job
}
