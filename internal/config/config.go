package config

import (
	"fmt"
	"os"
	"strconv"
	"strings"
)

type Config struct {
	Addr                string
	DatabaseURL         string
	BailianAPIKey       string
	BailianBaseURL      string
	BailianChatModel    string
	OllamaBaseURL       string
	OllamaEmbedModel    string
	EmbeddingDim        int
	ChunkSize           int
	ChunkOverlap        int
	RetrievalCandidateK int
	RerankURL           string
	RerankEnabled       bool
	RerankMaxPerDoc     int
	RerankTimeoutMS     int
	MaxUploadBytes      int64
	OTELServiceName     string
	OTELEndpoint        string
	OTELInsecure        bool
	MetricsEnabled      bool
}

func Load() (Config, error) {
	c := Config{
		Addr:                env("MINIRAG_ADDR", ":8080"),
		DatabaseURL:         env("DATABASE_URL", "postgres://minirag:minirag@localhost:5432/minirag?sslmode=disable"),
		BailianAPIKey:       strings.TrimSpace(os.Getenv("BAILIAN_API_KEY")),
		BailianBaseURL:      env("BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
		BailianChatModel:    env("BAILIAN_CHAT_MODEL", "glm-5.2"),
		OllamaBaseURL:       strings.TrimRight(env("OLLAMA_BASE_URL", "http://localhost:11434"), "/"),
		OllamaEmbedModel:    env("OLLAMA_EMBED_MODEL", "bge-m3"),
		EmbeddingDim:        envInt("EMBEDDING_DIM", 1024),
		ChunkSize:           envInt("CHUNK_SIZE", 1200),
		ChunkOverlap:        envInt("CHUNK_OVERLAP", 200),
		RetrievalCandidateK: envInt("RETRIEVAL_CANDIDATE_K", 50),
		RerankURL:           strings.TrimRight(env("RERANK_URL", "http://localhost:8081"), "/"),
		RerankEnabled:       envBool("RERANK_ENABLED", true),
		RerankMaxPerDoc:     envInt("RERANK_MAX_PER_DOC", 2),
		RerankTimeoutMS:     envInt("RERANK_TIMEOUT_MS", 30000),
		MaxUploadBytes:      int64(envInt("MAX_UPLOAD_MB", 10)) * 1024 * 1024,
		OTELServiceName:     env("OTEL_SERVICE_NAME", "minirag"),
		OTELEndpoint:        strings.TrimRight(env("OTEL_EXPORTER_OTLP_ENDPOINT", ""), "/"),
		OTELInsecure:        envBool("OTEL_EXPORTER_OTLP_INSECURE", false),
		MetricsEnabled:      envBool("METRICS_ENABLED", true),
	}
	if c.EmbeddingDim <= 0 || c.ChunkSize <= 0 || c.ChunkOverlap < 0 || c.ChunkOverlap >= c.ChunkSize || c.RetrievalCandidateK <= 0 || c.RerankMaxPerDoc <= 0 || c.RerankTimeoutMS <= 0 {
		return c, fmt.Errorf("invalid embedding, chunk, or retrieval configuration")
	}
	return c, nil
}

func envBool(key string, fallback bool) bool {
	value := strings.TrimSpace(os.Getenv(key))
	if value == "" {
		return fallback
	}
	parsed, err := strconv.ParseBool(value)
	if err != nil {
		return fallback
	}
	return parsed
}

func (c Config) ValidateChat() error {
	if c.BailianAPIKey == "" {
		return fmt.Errorf("BAILIAN_API_KEY is required")
	}
	return nil
}

func env(key, fallback string) string {
	if value := strings.TrimSpace(os.Getenv(key)); value != "" {
		return value
	}
	return fallback
}

func envInt(key string, fallback int) int {
	value := strings.TrimSpace(os.Getenv(key))
	if value == "" {
		return fallback
	}
	n, err := strconv.Atoi(value)
	if err != nil {
		return fallback
	}
	return n
}
