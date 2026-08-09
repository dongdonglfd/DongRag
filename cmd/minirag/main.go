package main

import (
	"context"
	"log/slog"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/lfd/minirag/internal/config"
	"github.com/lfd/minirag/internal/observability"
	"github.com/lfd/minirag/internal/provider"
	"github.com/lfd/minirag/internal/rag"
	"github.com/lfd/minirag/internal/store"
	"github.com/lfd/minirag/internal/web"
)

type funcReranker struct{ client provider.LlamaCppReranker }

func (r funcReranker) Rerank(ctx context.Context, query string, texts []string) ([]rag.RerankResult, error) {
	items, err := r.client.Rerank(ctx, query, texts)
	if err != nil {
		return nil, err
	}
	result := make([]rag.RerankResult, len(items))
	for i, item := range items {
		result[i] = rag.RerankResult{Index: item.Index, Score: item.Score}
	}
	return result, nil
}

func main() {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))
	slog.SetDefault(logger)
	cfg, err := config.Load()
	if err != nil {
		logger.Error("invalid configuration", "error", err)
		os.Exit(1)
	}
	if len(os.Args) > 1 && os.Args[1] == "eval" {
		printCommandError(runEvaluation(cfg, os.Args[2:]))
		return
	}
	if err := cfg.ValidateChat(); err != nil {
		logger.Error("invalid chat configuration", "error", err)
		os.Exit(1)
	}
	telemetry, err := observability.New(context.Background(), observability.Config{ServiceName: cfg.OTELServiceName, Endpoint: cfg.OTELEndpoint, Insecure: cfg.OTELInsecure})
	if err != nil {
		logger.Error("initialize observability", "error", err)
		os.Exit(1)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	database, err := store.New(ctx, cfg.DatabaseURL)
	if err != nil {
		logger.Error("database unavailable", "error", err)
		os.Exit(1)
	}
	defer database.Close()
	if err := database.Migrate(ctx); err != nil {
		logger.Error("database migration failed", "error", err)
		os.Exit(1)
	}

	embedder := provider.OllamaEmbedder{BaseURL: cfg.OllamaBaseURL, Model: cfg.OllamaEmbedModel}
	chatger := provider.BailianChat{APIKey: cfg.BailianAPIKey, BaseURL: cfg.BailianBaseURL, Model: cfg.BailianChatModel, Temperature: 0.1}
	var reranker rag.Reranker
	var rerankerHealth func(context.Context) error
	if cfg.RerankEnabled {
		client := provider.LlamaCppReranker{BaseURL: cfg.RerankURL}
		reranker = funcReranker{client}
		rerankerHealth = client.Health
	}
	var metrics *observability.Metrics
	if cfg.MetricsEnabled {
		metrics = telemetry.Metrics
	}
	service := &rag.Service{Store: database, Embedder: embedder, Chatger: chatger, Reranker: reranker, RerankMaxPerDoc: cfg.RerankMaxPerDoc, RerankTimeout: time.Duration(cfg.RerankTimeoutMS) * time.Millisecond, ChunkSize: cfg.ChunkSize, ChunkOverlap: cfg.ChunkOverlap, EmbeddingDim: cfg.EmbeddingDim, CandidateK: cfg.RetrievalCandidateK, Metrics: metrics}
	worker := rag.NewIngestionWorker(database, service, logger)
	worker.Metrics = metrics
	runCtx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	if err := worker.Start(runCtx); err != nil {
		stop()
		logger.Error("start ingestion worker", "error", err)
		os.Exit(1)
	}
	server := &web.Server{RAG: service, Worker: worker, Store: database, MaxUploadSize: cfg.MaxUploadBytes, Logger: logger, Metrics: metrics, EmbeddingHealth: embedder.Health, LLMHealth: chatger.Health, RerankerHealth: rerankerHealth, PostgresHealth: database.Pool.Ping, WorkerReady: worker.Ready}

	logger.Info("minirag started", "addr", cfg.Addr, "chat_model", cfg.BailianChatModel, "embedding_model", cfg.OllamaEmbedModel)
	serverErr := httpServer(runCtx, cfg.Addr, server.Handler())
	stop()
	worker.Wait()
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 5*time.Second)
	if err := telemetry.Shutdown(shutdownCtx); err != nil {
		logger.Error("flush traces", "error", err)
	}
	shutdownCancel()
	if serverErr != nil {
		logger.Error("server stopped", "error", serverErr)
		os.Exit(1)
	}
}
