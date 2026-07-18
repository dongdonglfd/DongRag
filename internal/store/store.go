package store

import (
	"context"
	"crypto/rand"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
)

type Store struct {
	Pool *pgxpool.Pool
}

type Document struct {
	ID          string    `json:"id"`
	Name        string    `json:"name"`
	ContentType string    `json:"content_type"`
	SizeBytes   int64     `json:"size_bytes"`
	Checksum    string    `json:"checksum"`
	CreatedAt   time.Time `json:"created_at"`
}

type Chunk struct {
	ID         string
	DocumentID string
	Ordinal    int
	Content    string
	Metadata   map[string]any
	Embedding  []float32
}

type Hit struct {
	ID           string         `json:"chunk_id"`
	DocumentID   string         `json:"document_id"`
	DocumentName string         `json:"document_name"`
	Content      string         `json:"content"`
	Metadata     map[string]any `json:"metadata"`
	Score        float64        `json:"score"`
	VectorRank   int            `json:"vector_rank,omitempty"`
	VectorScore  float64        `json:"vector_score,omitempty"`
	LexicalRank  int            `json:"lexical_rank,omitempty"`
	LexicalScore float64        `json:"lexical_score,omitempty"`
	RRFScore     float64        `json:"rrf_score,omitempty"`
	RerankRank   int            `json:"rerank_rank,omitempty"`
	RerankScore  float64        `json:"rerank_score,omitempty"`
	FinalRank    int            `json:"final_rank,omitempty"`
}

type SearchMode string

const (
	SearchModeVector   SearchMode = "vector"
	SearchModeLexical  SearchMode = "lexical"
	SearchModeWeighted SearchMode = "weighted"
	SearchModeHybrid   SearchMode = "hybrid"
	SearchModeRerank   SearchMode = "rerank"
)

type IngestionJob struct {
	ID            string            `json:"id"`
	Name          string            `json:"name"`
	ContentType   string            `json:"content_type"`
	Reindex       bool              `json:"reindex,omitempty"`
	Status        string            `json:"status"`
	Attempts      int               `json:"attempts"`
	RetryCount    int               `json:"retry_count"`
	ErrorMessage  string            `json:"error_message,omitempty"`
	DocumentID    string            `json:"document_id,omitempty"`
	NextAttemptAt time.Time         `json:"next_attempt_at"`
	LeaseUntil    *time.Time        `json:"lease_until,omitempty"`
	WorkerID      string            `json:"worker_id,omitempty"`
	CreatedAt     time.Time         `json:"created_at"`
	UpdatedAt     time.Time         `json:"updated_at"`
	Payload       []byte            `json:"-"`
	TraceContext  map[string]string `json:"-"`
}

type DocumentSource struct {
	Document
	Payload []byte `json:"-"`
}

func New(ctx context.Context, databaseURL string) (*Store, error) {
	pool, err := pgxpool.New(ctx, databaseURL)
	if err != nil {
		return nil, fmt.Errorf("create postgres pool: %w", err)
	}
	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		return nil, fmt.Errorf("ping postgres: %w", err)
	}
	return &Store{Pool: pool}, nil
}

func (s *Store) Close() { s.Pool.Close() }

func (s *Store) Migrate(ctx context.Context) error {
	_, err := s.Pool.Exec(ctx, `
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    content_type TEXT NOT NULL,
    size_bytes BIGINT NOT NULL,
    checksum TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal INT NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding vector(1024),
    search_vector tsvector GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED,
    UNIQUE(document_id, ordinal)
);
CREATE INDEX IF NOT EXISTS chunks_document_idx ON chunks(document_id);
CREATE INDEX IF NOT EXISTS chunks_search_idx ON chunks USING gin(search_vector);
CREATE INDEX IF NOT EXISTS chunks_content_trgm_idx ON chunks USING gin(content gin_trgm_ops);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE TABLE IF NOT EXISTS ingestion_jobs (
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
ALTER TABLE ingestion_jobs ADD COLUMN IF NOT EXISTS is_reindex BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE ingestion_jobs ADD COLUMN IF NOT EXISTS retry_count INT NOT NULL DEFAULT 0;
ALTER TABLE ingestion_jobs ADD COLUMN IF NOT EXISTS next_attempt_at TIMESTAMPTZ NOT NULL DEFAULT now();
ALTER TABLE ingestion_jobs ADD COLUMN IF NOT EXISTS lease_until TIMESTAMPTZ;
ALTER TABLE ingestion_jobs ADD COLUMN IF NOT EXISTS worker_id TEXT NOT NULL DEFAULT '';
ALTER TABLE ingestion_jobs ADD COLUMN IF NOT EXISTS trace_context JSONB NOT NULL DEFAULT '{}'::jsonb;
CREATE INDEX IF NOT EXISTS ingestion_jobs_status_idx ON ingestion_jobs(status, created_at);
CREATE INDEX IF NOT EXISTS ingestion_jobs_available_idx ON ingestion_jobs(next_attempt_at, created_at) WHERE status='queued';
CREATE INDEX IF NOT EXISTS ingestion_jobs_lease_idx ON ingestion_jobs(lease_until) WHERE status='processing';
`)
	if err != nil {
		return fmt.Errorf("migrate database: %w", err)
	}
	return nil
}

func (s *Store) FindDocumentByChecksum(ctx context.Context, checksum string) (*Document, error) {
	var d Document
	err := s.Pool.QueryRow(ctx, `SELECT id, name, content_type, size_bytes, checksum, created_at FROM documents WHERE checksum=$1`, checksum).
		Scan(&d.ID, &d.Name, &d.ContentType, &d.SizeBytes, &d.Checksum, &d.CreatedAt)
	if err == pgx.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return &d, nil
}

func (s *Store) CreateDocument(ctx context.Context, document Document, chunks []Chunk) error {
	tx, err := s.Pool.Begin(ctx)
	if err != nil {
		return err
	}
	defer tx.Rollback(ctx)
	_, err = tx.Exec(ctx, `INSERT INTO documents(id, name, content_type, size_bytes, checksum) VALUES($1,$2,$3,$4,$5)`, document.ID, document.Name, document.ContentType, document.SizeBytes, document.Checksum)
	if err != nil {
		return err
	}
	if err := insertChunks(ctx, tx, chunks); err != nil {
		return err
	}
	return tx.Commit(ctx)
}

func (s *Store) ReplaceDocumentChunks(ctx context.Context, documentID string, chunks []Chunk) error {
	tx, err := s.Pool.Begin(ctx)
	if err != nil {
		return err
	}
	defer tx.Rollback(ctx)
	if _, err := tx.Exec(ctx, `DELETE FROM chunks WHERE document_id=$1`, documentID); err != nil {
		return err
	}
	if err := insertChunks(ctx, tx, chunks); err != nil {
		return err
	}
	return tx.Commit(ctx)
}

func (s *Store) ListDocumentChunks(ctx context.Context, documentID string) ([]Chunk, error) {
	rows, err := s.Pool.Query(ctx, `
SELECT id, document_id, ordinal, content, metadata
FROM chunks
WHERE document_id=$1
ORDER BY ordinal`, documentID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var chunks []Chunk
	for rows.Next() {
		var chunk Chunk
		var rawMetadata []byte
		if err := rows.Scan(&chunk.ID, &chunk.DocumentID, &chunk.Ordinal, &chunk.Content, &rawMetadata); err != nil {
			return nil, err
		}
		if err := json.Unmarshal(rawMetadata, &chunk.Metadata); err != nil {
			return nil, err
		}
		chunks = append(chunks, chunk)
	}
	return chunks, rows.Err()
}

func insertChunks(ctx context.Context, tx pgx.Tx, chunks []Chunk) error {
	for _, chunk := range chunks {
		metadata, err := json.Marshal(chunk.Metadata)
		if err != nil {
			return err
		}
		if _, err := tx.Exec(ctx, `INSERT INTO chunks(id, document_id, ordinal, content, metadata, embedding) VALUES($1,$2,$3,$4,$5,$6::vector)`, chunk.ID, chunk.DocumentID, chunk.Ordinal, chunk.Content, metadata, vectorLiteral(chunk.Embedding)); err != nil {
			return err
		}
	}
	return nil
}

func (s *Store) ListDocuments(ctx context.Context) ([]Document, error) {
	rows, err := s.Pool.Query(ctx, `SELECT id, name, content_type, size_bytes, checksum, created_at FROM documents ORDER BY created_at DESC`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var documents []Document
	for rows.Next() {
		var d Document
		if err := rows.Scan(&d.ID, &d.Name, &d.ContentType, &d.SizeBytes, &d.Checksum, &d.CreatedAt); err != nil {
			return nil, err
		}
		documents = append(documents, d)
	}
	return documents, rows.Err()
}

func (s *Store) DeleteDocument(ctx context.Context, id string) error {
	_, err := s.Pool.Exec(ctx, `DELETE FROM documents WHERE id=$1`, id)
	return err
}

func (s *Store) CreateIngestionJob(ctx context.Context, name, contentType string, payload []byte) (IngestionJob, error) {
	return s.CreateIngestionJobWithTrace(ctx, name, contentType, payload, nil)
}

func (s *Store) CreateReindexJob(ctx context.Context, documentID, name, contentType string, payload []byte) (IngestionJob, error) {
	return s.CreateReindexJobWithTrace(ctx, documentID, name, contentType, payload, nil)
}

func (s *Store) CreateIngestionJobWithTrace(ctx context.Context, name, contentType string, payload []byte, traceContext map[string]string) (IngestionJob, error) {
	return s.createIngestionJob(ctx, name, contentType, payload, "", false, traceContext)
}

func (s *Store) CreateReindexJobWithTrace(ctx context.Context, documentID, name, contentType string, payload []byte, traceContext map[string]string) (IngestionJob, error) {
	return s.createIngestionJob(ctx, name, contentType, payload, documentID, true, traceContext)
}

func (s *Store) createIngestionJob(ctx context.Context, name, contentType string, payload []byte, documentID string, reindex bool, traceContext map[string]string) (IngestionJob, error) {
	job := IngestionJob{ID: newID("job"), Name: name, ContentType: contentType, Reindex: reindex, Status: "queued", DocumentID: documentID, Payload: payload, TraceContext: traceContext}
	tracePayload, err := json.Marshal(traceContext)
	if err != nil {
		return IngestionJob{}, err
	}
	err = s.Pool.QueryRow(ctx, `
INSERT INTO ingestion_jobs(id, name, content_type, payload, document_id, is_reindex, status, trace_context)
VALUES($1, $2, $3, $4, NULLIF($5, ''), $6, 'queued', $7)
RETURNING next_attempt_at, created_at, updated_at`, job.ID, job.Name, job.ContentType, job.Payload, job.DocumentID, job.Reindex, tracePayload).
		Scan(&job.NextAttemptAt, &job.CreatedAt, &job.UpdatedAt)
	return job, err
}

func (s *Store) GetDocumentSource(ctx context.Context, id string) (*DocumentSource, error) {
	var source DocumentSource
	err := s.Pool.QueryRow(ctx, `
SELECT d.id, d.name, d.content_type, d.size_bytes, d.checksum, d.created_at,
       COALESCE(j.payload, ''::bytea)
FROM documents d
LEFT JOIN LATERAL (
    SELECT payload
    FROM ingestion_jobs
    WHERE document_id=$1 AND status='completed'
    ORDER BY updated_at DESC, created_at DESC
    LIMIT 1
) j ON TRUE
WHERE d.id=$1`, id).
		Scan(&source.ID, &source.Name, &source.ContentType, &source.SizeBytes, &source.Checksum, &source.CreatedAt, &source.Payload)
	if err == pgx.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return &source, nil
}

func (s *Store) FindActiveReindexJob(ctx context.Context, documentID string) (*IngestionJob, error) {
	var job IngestionJob
	err := s.Pool.QueryRow(ctx, `
SELECT id, name, content_type, is_reindex, status, attempts, retry_count, error_message,
       COALESCE(document_id, ''), next_attempt_at, lease_until, worker_id, created_at, updated_at
FROM ingestion_jobs
WHERE document_id=$1 AND is_reindex=TRUE AND status IN ('queued', 'processing')
ORDER BY created_at DESC
LIMIT 1`, documentID).
		Scan(&job.ID, &job.Name, &job.ContentType, &job.Reindex, &job.Status, &job.Attempts, &job.RetryCount, &job.ErrorMessage, &job.DocumentID, &job.NextAttemptAt, &job.LeaseUntil, &job.WorkerID, &job.CreatedAt, &job.UpdatedAt)
	if err == pgx.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return &job, nil
}

func (s *Store) GetIngestionJob(ctx context.Context, id string) (*IngestionJob, error) {
	var job IngestionJob
	err := s.Pool.QueryRow(ctx, `
SELECT id, name, content_type, is_reindex, status, attempts, retry_count, error_message,
       COALESCE(document_id, ''), next_attempt_at, lease_until, worker_id, created_at, updated_at
FROM ingestion_jobs WHERE id=$1`, id).
		Scan(&job.ID, &job.Name, &job.ContentType, &job.Reindex, &job.Status, &job.Attempts, &job.RetryCount, &job.ErrorMessage, &job.DocumentID, &job.NextAttemptAt, &job.LeaseUntil, &job.WorkerID, &job.CreatedAt, &job.UpdatedAt)
	if err == pgx.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return &job, nil
}

func (s *Store) RecoverExpiredIngestionJobs(ctx context.Context) (int64, error) {
	result, err := s.Pool.Exec(ctx, `
UPDATE ingestion_jobs
SET status='queued', next_attempt_at=now(), lease_until=NULL, worker_id='', updated_at=now()
WHERE status='processing' AND (lease_until IS NULL OR lease_until < now())`)
	if err != nil {
		return 0, err
	}
	return result.RowsAffected(), nil
}

func (s *Store) ClaimNextIngestionJob(ctx context.Context, workerID string, leaseDuration time.Duration) (*IngestionJob, error) {
	tx, err := s.Pool.Begin(ctx)
	if err != nil {
		return nil, err
	}
	defer tx.Rollback(ctx)
	var job IngestionJob
	var tracePayload []byte
	err = tx.QueryRow(ctx, `
WITH next_job AS (
    SELECT id
    FROM ingestion_jobs
    WHERE status='queued' AND next_attempt_at <= now()
    ORDER BY next_attempt_at, created_at
    FOR UPDATE SKIP LOCKED
    LIMIT 1
)
UPDATE ingestion_jobs j
SET status='processing', attempts=j.attempts+1,
    lease_until=now() + ($2::double precision * interval '1 millisecond'), worker_id=$1, updated_at=now()
FROM next_job
WHERE j.id=next_job.id
RETURNING j.id, j.name, j.content_type, j.payload, j.is_reindex, j.status, j.attempts,
	          j.retry_count, j.error_message, COALESCE(j.document_id, ''), j.next_attempt_at,
	          j.lease_until, j.worker_id, j.trace_context, j.created_at, j.updated_at`, workerID, leaseDuration.Milliseconds()).
		Scan(&job.ID, &job.Name, &job.ContentType, &job.Payload, &job.Reindex, &job.Status, &job.Attempts, &job.RetryCount, &job.ErrorMessage, &job.DocumentID, &job.NextAttemptAt, &job.LeaseUntil, &job.WorkerID, &tracePayload, &job.CreatedAt, &job.UpdatedAt)
	if err == pgx.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(tracePayload, &job.TraceContext); err != nil {
		return nil, err
	}
	if err := tx.Commit(ctx); err != nil {
		return nil, err
	}
	return &job, nil
}

func (s *Store) RenewIngestionJobLease(ctx context.Context, id, workerID string, leaseDuration time.Duration) (bool, error) {
	result, err := s.Pool.Exec(ctx, `
UPDATE ingestion_jobs
SET lease_until=now() + ($3::double precision * interval '1 millisecond'), updated_at=now()
WHERE id=$1 AND status='processing' AND worker_id=$2 AND lease_until >= now()`, id, workerID, leaseDuration.Milliseconds())
	if err != nil {
		return false, err
	}
	return result.RowsAffected() == 1, nil
}

func (s *Store) CompleteIngestionJob(ctx context.Context, id, workerID, documentID string) (bool, error) {
	result, err := s.Pool.Exec(ctx, `
UPDATE ingestion_jobs
SET status='completed', document_id=$3, error_message='', lease_until=NULL, worker_id='', updated_at=now()
WHERE id=$1 AND status='processing' AND worker_id=$2 AND lease_until >= now()`, id, workerID, documentID)
	if err != nil {
		return false, err
	}
	return result.RowsAffected() == 1, nil
}

func (s *Store) FailIngestionJob(ctx context.Context, id, workerID, message string, retry bool, retryDelay time.Duration) (bool, error) {
	status := "failed"
	if retry {
		status = "queued"
	}
	result, err := s.Pool.Exec(ctx, `
UPDATE ingestion_jobs
SET status=$3, retry_count=retry_count + CASE WHEN $4 THEN 1 ELSE 0 END,
    next_attempt_at=CASE WHEN $4 THEN now() + ($5::double precision * interval '1 millisecond') ELSE next_attempt_at END,
    error_message=$6, lease_until=NULL, worker_id='', updated_at=now()
WHERE id=$1 AND status='processing' AND worker_id=$2 AND lease_until >= now()`, id, workerID, status, retry, retryDelay.Milliseconds(), message)
	if err != nil {
		return false, err
	}
	return result.RowsAffected() == 1, nil
}

func (s *Store) ReleaseIngestionJob(ctx context.Context, id, workerID string) (bool, error) {
	result, err := s.Pool.Exec(ctx, `
UPDATE ingestion_jobs
SET status='queued', next_attempt_at=now(), lease_until=NULL, worker_id='', updated_at=now()
WHERE id=$1 AND status='processing' AND worker_id=$2`, id, workerID)
	if err != nil {
		return false, err
	}
	return result.RowsAffected() == 1, nil
}

func (s *Store) Search(ctx context.Context, query string, embedding []float32, mode SearchMode, topK, candidateK int) ([]Hit, error) {
	topK, candidateK = normalizeSearchLimits(topK, candidateK)
	switch mode {
	case SearchModeVector:
		return s.searchVector(ctx, embedding, topK, candidateK)
	case SearchModeLexical:
		return s.searchLexical(ctx, query, topK, candidateK)
	case SearchModeWeighted:
		return s.searchWeighted(ctx, query, embedding, topK, candidateK)
	case SearchModeHybrid:
		return s.searchHybrid(ctx, query, embedding, topK, candidateK)
	default:
		return nil, fmt.Errorf("unsupported search mode %q", mode)
	}
}

func (s *Store) QueueState(ctx context.Context) (queued, processing int, err error) {
	err = s.Pool.QueryRow(ctx, `
SELECT COUNT(*) FILTER (WHERE status='queued'), COUNT(*) FILTER (WHERE status='processing')
FROM ingestion_jobs`).Scan(&queued, &processing)
	return
}

func (s *Store) searchWeighted(ctx context.Context, query string, embedding []float32, topK, candidateK int) ([]Hit, error) {
	rows, err := s.Pool.Query(ctx, `
WITH params AS (SELECT websearch_to_tsquery('simple', $2) AS ts_query),
semantic AS (
    SELECT c.id, c.document_id, c.content, c.metadata,
           ROW_NUMBER() OVER (ORDER BY c.embedding <=> $1::vector, c.id) AS vector_rank,
           1 - (c.embedding <=> $1::vector) AS vector_score
    FROM chunks c WHERE c.embedding IS NOT NULL
    ORDER BY c.embedding <=> $1::vector, c.id LIMIT $3
),
lexical_scored AS (
    SELECT c.id, c.document_id, c.content, c.metadata,
           GREATEST(ts_rank_cd(c.search_vector, p.ts_query)::double precision, word_similarity(lower($2), lower(c.content))::double precision) AS lexical_score
    FROM chunks c CROSS JOIN params p
),
lexical AS (
    SELECT l.*, ROW_NUMBER() OVER (ORDER BY l.lexical_score DESC, l.id) AS lexical_rank
    FROM lexical_scored l WHERE l.lexical_score > 0 ORDER BY l.lexical_score DESC, l.id LIMIT $3
),
combined AS (
    SELECT COALESCE(s.id,l.id) id, COALESCE(s.document_id,l.document_id) document_id, COALESCE(s.content,l.content) content, COALESCE(s.metadata,l.metadata) metadata,
           COALESCE(s.vector_rank,0) vector_rank, COALESCE(s.vector_score,0) vector_score, COALESCE(l.lexical_rank,0) lexical_rank, COALESCE(l.lexical_score,0) lexical_score
    FROM semantic s FULL OUTER JOIN lexical l ON s.id=l.id
), normalized AS (
    SELECT c.*,
      CASE WHEN MAX(vector_score) OVER ()=MIN(vector_score) OVER () THEN 0 ELSE (vector_score-MIN(vector_score) OVER ())/NULLIF(MAX(vector_score) OVER ()-MIN(vector_score) OVER (),0) END AS vector_norm,
      CASE WHEN MAX(lexical_score) OVER ()=MIN(lexical_score) OVER () THEN 0 ELSE (lexical_score-MIN(lexical_score) OVER ())/NULLIF(MAX(lexical_score) OVER ()-MIN(lexical_score) OVER (),0) END AS lexical_norm
    FROM combined c
)
SELECT n.id,n.document_id,d.name,n.content,n.metadata,(n.vector_norm+n.lexical_norm)/2 AS score,
       n.vector_rank,n.vector_score,n.lexical_rank,n.lexical_score,0::double precision AS rrf_score
FROM normalized n JOIN documents d ON d.id=n.document_id
ORDER BY score DESC,GREATEST(n.vector_score,n.lexical_score) DESC,n.id LIMIT $4`, vectorLiteral(embedding), query, candidateK, topK)
	if err != nil {
		return nil, err
	}
	return scanHits(rows)
}

func (s *Store) searchVector(ctx context.Context, embedding []float32, topK, candidateK int) ([]Hit, error) {
	rows, err := s.Pool.Query(ctx, `
WITH semantic AS (
    SELECT c.id, c.document_id, c.content, c.metadata,
           ROW_NUMBER() OVER (ORDER BY c.embedding <=> $1::vector, c.id) AS vector_rank,
           1 - (c.embedding <=> $1::vector) AS vector_score
    FROM chunks c
    WHERE c.embedding IS NOT NULL
    ORDER BY c.embedding <=> $1::vector, c.id
    LIMIT $2
)
SELECT s.id, s.document_id, d.name, s.content, s.metadata,
       s.vector_score AS score,
       s.vector_rank, s.vector_score,
       0::bigint AS lexical_rank, 0::double precision AS lexical_score,
       0::double precision AS rrf_score
FROM semantic s
JOIN documents d ON d.id=s.document_id
ORDER BY s.vector_rank
LIMIT $3`, vectorLiteral(embedding), candidateK, topK)
	if err != nil {
		return nil, err
	}
	return scanHits(rows)
}

func (s *Store) searchLexical(ctx context.Context, query string, topK, candidateK int) ([]Hit, error) {
	rows, err := s.Pool.Query(ctx, `
WITH params AS (
    SELECT websearch_to_tsquery('simple', $1) AS ts_query
), lexical_scored AS (
    SELECT c.id, c.document_id, c.content, c.metadata,
           GREATEST(
               ts_rank_cd(c.search_vector, p.ts_query)::double precision,
               word_similarity(lower($1), lower(c.content))::double precision
           ) AS lexical_score
    FROM chunks c
    CROSS JOIN params p
), lexical AS (
    SELECT l.*,
           ROW_NUMBER() OVER (ORDER BY l.lexical_score DESC, l.id) AS lexical_rank
    FROM lexical_scored l
    WHERE l.lexical_score > 0
    ORDER BY l.lexical_score DESC, l.id
    LIMIT $2
)
SELECT l.id, l.document_id, d.name, l.content, l.metadata,
       l.lexical_score AS score,
       0::bigint AS vector_rank, 0::double precision AS vector_score,
       l.lexical_rank, l.lexical_score,
       0::double precision AS rrf_score
FROM lexical l
JOIN documents d ON d.id=l.document_id
ORDER BY l.lexical_rank
LIMIT $3`, query, candidateK, topK)
	if err != nil {
		return nil, err
	}
	return scanHits(rows)
}

func (s *Store) searchHybrid(ctx context.Context, query string, embedding []float32, topK, candidateK int) ([]Hit, error) {
	rows, err := s.Pool.Query(ctx, `
WITH params AS (
    SELECT websearch_to_tsquery('simple', $2) AS ts_query
), semantic AS (
    SELECT c.id, c.document_id, c.content, c.metadata,
           ROW_NUMBER() OVER (ORDER BY c.embedding <=> $1::vector, c.id) AS vector_rank,
           1 - (c.embedding <=> $1::vector) AS vector_score
    FROM chunks c
    WHERE c.embedding IS NOT NULL
    ORDER BY c.embedding <=> $1::vector, c.id
    LIMIT $3
), lexical_scored AS (
    SELECT c.id, c.document_id, c.content, c.metadata,
           GREATEST(
               ts_rank_cd(c.search_vector, p.ts_query)::double precision,
               word_similarity(lower($2), lower(c.content))::double precision
           ) AS lexical_score
    FROM chunks c
    CROSS JOIN params p
), lexical AS (
    SELECT l.*,
           ROW_NUMBER() OVER (ORDER BY l.lexical_score DESC, l.id) AS lexical_rank
    FROM lexical_scored l
    WHERE l.lexical_score > 0
    ORDER BY l.lexical_score DESC, l.id
    LIMIT $3
), combined AS (
    SELECT COALESCE(s.id, l.id) AS id,
           COALESCE(s.document_id, l.document_id) AS document_id,
           COALESCE(s.content, l.content) AS content,
           COALESCE(s.metadata, l.metadata) AS metadata,
           COALESCE(s.vector_rank, 0) AS vector_rank,
           COALESCE(s.vector_score, 0) AS vector_score,
           COALESCE(l.lexical_rank, 0) AS lexical_rank,
           COALESCE(l.lexical_score, 0) AS lexical_score,
           COALESCE(1.0::double precision / (60 + s.vector_rank), 0) +
           COALESCE(1.0::double precision / (60 + l.lexical_rank), 0) AS rrf_score
    FROM semantic s
    FULL OUTER JOIN lexical l ON s.id=l.id
)
SELECT c.id, c.document_id, d.name, c.content, c.metadata,
       c.rrf_score AS score,
       c.vector_rank, c.vector_score,
       c.lexical_rank, c.lexical_score,
       c.rrf_score
FROM combined c
JOIN documents d ON d.id=c.document_id
ORDER BY c.rrf_score DESC, GREATEST(c.vector_score, c.lexical_score) DESC, c.id
LIMIT $4`, vectorLiteral(embedding), query, candidateK, topK)
	if err != nil {
		return nil, err
	}
	return scanHits(rows)
}

func normalizeSearchLimits(topK, candidateK int) (int, int) {
	if topK < 1 {
		topK = 5
	}
	if candidateK < topK {
		candidateK = topK
	}
	return topK, candidateK
}

func scanHits(rows pgx.Rows) ([]Hit, error) {
	defer rows.Close()
	var hits []Hit
	for rows.Next() {
		var hit Hit
		var rawMetadata []byte
		if err := rows.Scan(
			&hit.ID, &hit.DocumentID, &hit.DocumentName, &hit.Content, &rawMetadata,
			&hit.Score, &hit.VectorRank, &hit.VectorScore,
			&hit.LexicalRank, &hit.LexicalScore, &hit.RRFScore,
		); err != nil {
			return nil, err
		}
		if err := json.Unmarshal(rawMetadata, &hit.Metadata); err != nil {
			return nil, err
		}
		hits = append(hits, hit)
	}
	return hits, rows.Err()
}

func vectorLiteral(values []float32) string {
	if len(values) == 0 {
		return "[]"
	}
	parts := make([]string, len(values))
	for i, value := range values {
		parts[i] = fmt.Sprintf("%g", value)
	}
	return "[" + strings.Join(parts, ",") + "]"
}

func newID(prefix string) string {
	var random [12]byte
	if _, err := rand.Read(random[:]); err != nil {
		return fmt.Sprintf("%s-%d", prefix, time.Now().UnixNano())
	}
	return fmt.Sprintf("%s-%x", prefix, random)
}
