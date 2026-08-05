package web

import (
	"context"
	"embed"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"log/slog"
	"mime"
	"net/http"
	"path"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/lfd/minirag/internal/observability"
	"github.com/lfd/minirag/internal/rag"
	"github.com/lfd/minirag/internal/store"
)

//go:embed static/*
var staticFiles embed.FS

var staticFS, _ = fs.Sub(staticFiles, "static")

type Server struct {
	RAG             *rag.Service
	Worker          *rag.IngestionWorker
	Store           *store.Store
	MaxUploadSize   int64
	Logger          *slog.Logger
	Metrics         *observability.Metrics
	EmbeddingHealth func(context.Context) error
	LLMHealth       func(context.Context) error
	RerankerHealth  func(context.Context) error
	PostgresHealth  func(context.Context) error
	WorkerReady     func() bool
}

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", s.health)
	mux.HandleFunc("/readyz", s.ready)
	if s.Metrics != nil {
		mux.Handle("/metrics", s.Metrics.Handler())
	}
	mux.HandleFunc("/v1/documents", s.documents)
	mux.HandleFunc("/v1/documents/", s.documents)
	mux.HandleFunc("/v1/jobs/", s.job)
	mux.HandleFunc("/v1/query", s.query)
	mux.HandleFunc("/v1/chat/stream", s.stream)
	mux.Handle("/static/", http.StripPrefix("/static/", http.FileServer(http.FS(staticFS))))
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		data, err := staticFiles.ReadFile("static/index.html")
		if err != nil {
			writeError(w, http.StatusInternalServerError, err)
			return
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		_, _ = w.Write(data)
	})
	handler := logging(mux, s.Logger)
	if s.Metrics != nil {
		handler = s.Metrics.Middleware(handler, nil)
	}
	return handler
}

func (s *Server) health(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func (s *Server) ready(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), 3*time.Second)
	defer cancel()
	type checkResult struct {
		name string
		err  error
	}
	postgresHealth := s.PostgresHealth
	if postgresHealth == nil && s.Store != nil {
		postgresHealth = func(ctx context.Context) error { return s.Store.Pool.Ping(ctx) }
	}
	checks := []struct {
		name  string
		check func(context.Context) error
	}{
		{name: "postgres", check: postgresHealth},
		{name: "embedding", check: s.EmbeddingHealth},
		{name: "llm", check: s.LLMHealth},
	}
	if s.RerankerHealth != nil {
		checks = append(checks, struct {
			name  string
			check func(context.Context) error
		}{name: "reranker", check: s.RerankerHealth})
	}
	results := make(chan checkResult, len(checks))
	for _, item := range checks {
		go func() {
			if item.check == nil {
				results <- checkResult{name: item.name, err: fmt.Errorf("health check is not configured")}
				return
			}
			results <- checkResult{name: item.name, err: item.check(ctx)}
		}()
	}
	dependencies := map[string]string{"worker": "ready", "reranker": "disabled"}
	workerReady := s.WorkerReady
	if workerReady == nil && s.Worker != nil {
		workerReady = s.Worker.Ready
	}
	ready := workerReady != nil && workerReady()
	if !ready {
		dependencies["worker"] = "not_ready"
	}
	for range checks {
		result := <-results
		if result.err != nil {
			dependencies[result.name] = "not_ready"
			ready = false
		} else {
			dependencies[result.name] = "ready"
		}
	}
	status := "ready"
	code := http.StatusOK
	if !ready {
		status = "not_ready"
		code = http.StatusServiceUnavailable
	}
	writeJSON(w, code, map[string]any{"status": status, "dependencies": dependencies})
}

func (s *Server) documents(w http.ResponseWriter, r *http.Request) {
	if id, ok := documentReindexID(r.URL.Path); ok {
		if r.Method != http.MethodPost {
			writeError(w, http.StatusMethodNotAllowed, fmt.Errorf("method not allowed"))
			return
		}
		source, err := s.Store.GetDocumentSource(r.Context(), id)
		if err != nil {
			writeError(w, http.StatusInternalServerError, err)
			return
		}
		if source == nil {
			writeError(w, http.StatusNotFound, fmt.Errorf("document not found"))
			return
		}
		active, err := s.Store.FindActiveReindexJob(r.Context(), id)
		if err != nil {
			writeError(w, http.StatusInternalServerError, err)
			return
		}
		if active != nil {
			writeJSON(w, http.StatusConflict, map[string]any{"error": "reindex already in progress", "job": active})
			return
		}
		job, err := s.Store.CreateReindexJobWithTrace(r.Context(), source.ID, source.Name, source.ContentType, source.Payload, observability.Inject(r.Context()))
		if err != nil {
			writeError(w, http.StatusInternalServerError, err)
			return
		}
		if s.Metrics != nil {
			s.Metrics.ObserveQueue("created")
		}
		s.Worker.Submit(job.ID)
		writeJSON(w, http.StatusAccepted, map[string]any{"job": job})
		return
	}
	if r.Method == http.MethodGet {
		documents, err := s.Store.ListDocuments(r.Context())
		if err != nil {
			writeError(w, http.StatusInternalServerError, err)
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"documents": documents})
		return
	}
	if r.Method == http.MethodDelete {
		id := path.Base(r.URL.Path)
		if id == "documents" || id == "." || id == "/" {
			writeError(w, http.StatusBadRequest, fmt.Errorf("document id is required"))
			return
		}
		if err := s.Store.DeleteDocument(r.Context(), id); err != nil {
			writeError(w, http.StatusInternalServerError, err)
			return
		}
		w.WriteHeader(http.StatusNoContent)
		return
	}
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, fmt.Errorf("method not allowed"))
		return
	}
	if err := r.ParseMultipartForm(s.MaxUploadSize); err != nil {
		writeError(w, http.StatusBadRequest, fmt.Errorf("parse upload: %w", err))
		return
	}
	file, header, err := r.FormFile("file")
	if err != nil {
		writeError(w, http.StatusBadRequest, fmt.Errorf("field file is required"))
		return
	}
	defer file.Close()
	data, err := io.ReadAll(http.MaxBytesReader(w, file, s.MaxUploadSize))
	if err != nil {
		writeError(w, http.StatusRequestEntityTooLarge, fmt.Errorf("read upload: %w", err))
		return
	}
	contentType := header.Header.Get("Content-Type")
	if contentType == "" {
		contentType = mime.TypeByExtension(strings.ToLower(filepath.Ext(header.Filename)))
	}
	if contentType == "" {
		contentType = "text/plain"
	}
	job, err := s.Store.CreateIngestionJobWithTrace(r.Context(), filepath.Base(header.Filename), contentType, data, observability.Inject(r.Context()))
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	if s.Metrics != nil {
		s.Metrics.ObserveQueue("created")
	}
	s.Worker.Submit(job.ID)
	writeJSON(w, http.StatusAccepted, map[string]any{"job": job})
}

func documentReindexID(urlPath string) (string, bool) {
	parts := strings.Split(strings.Trim(path.Clean(urlPath), "/"), "/")
	if len(parts) != 4 || parts[0] != "v1" || parts[1] != "documents" || parts[3] != "reindex" || parts[2] == "" {
		return "", false
	}
	return parts[2], true
}

func (s *Server) job(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, fmt.Errorf("method not allowed"))
		return
	}
	id := path.Base(r.URL.Path)
	if id == "jobs" || id == "." || id == "/" {
		writeError(w, http.StatusBadRequest, fmt.Errorf("job id is required"))
		return
	}
	job, err := s.Store.GetIngestionJob(r.Context(), id)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	if job == nil {
		writeError(w, http.StatusNotFound, fmt.Errorf("job not found"))
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"job": job})
}

func (s *Server) query(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, fmt.Errorf("method not allowed"))
		return
	}
	var request struct {
		Question string `json:"question"`
		TopK     int    `json:"top_k"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&request); err != nil {
		writeError(w, http.StatusBadRequest, fmt.Errorf("invalid JSON: %w", err))
		return
	}
	result, err := s.RAG.Query(r.Context(), request.Question, request.TopK)
	if err != nil {
		writeError(w, http.StatusBadRequest, err)
		return
	}
	writeJSON(w, http.StatusOK, result)
}

func (s *Server) stream(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, fmt.Errorf("method not allowed"))
		return
	}
	var request struct {
		Question string `json:"question"`
		TopK     int    `json:"top_k"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&request); err != nil {
		writeError(w, http.StatusBadRequest, fmt.Errorf("invalid JSON: %w", err))
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	f, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, fmt.Errorf("streaming unsupported"))
		return
	}
	result, err := s.RAG.QueryStream(r.Context(), request.Question, request.TopK, func(token string) error {
		return writeSSE(w, f, "token", map[string]string{"content": token})
	})
	if err != nil {
		if r.Context().Err() != nil {
			return
		}
		_ = writeSSE(w, f, "error", map[string]string{"error": err.Error()})
		return
	}
	if err := writeSSE(w, f, "citations", map[string]any{"citations": result.Citations}); err != nil {
		return
	}
	_ = writeSSE(w, f, "done", map[string]any{
		"total_ms":        result.TotalMS,
		"retrieval_ms":    result.RetrievalMS,
		"candidate_k":     result.CandidateK,
		"reranked":        result.Reranked,
		"rerank_fallback": result.RerankFallback,
	})
}

func writeSSE(w http.ResponseWriter, f http.Flusher, event string, payload any) error {
	data, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	if _, err := fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, data); err != nil {
		return err
	}
	f.Flush()
	return nil
}

func logging(next http.Handler, logger *slog.Logger) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if logger == nil {
			logger = slog.Default()
		}
		logger.Info("http request", "method", r.Method, "path", r.URL.Path)
		next.ServeHTTP(w, r)
	})
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}

func writeError(w http.ResponseWriter, status int, err error) {
	writeJSON(w, status, map[string]string{"error": err.Error(), "status": strconv.Itoa(status)})
}
