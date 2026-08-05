package rag

import (
	"context"
	"testing"

	"github.com/lfd/minirag/internal/store"
)

type fixedDimensionEmbedder struct {
	dimension int
}

func (e fixedDimensionEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	embeddings := make([][]float32, len(texts))
	for i := range texts {
		embeddings[i] = make([]float32, e.dimension)
	}
	return embeddings, nil
}

func TestRebuildStoredChunks(t *testing.T) {
	service := &Service{Embedder: fixedDimensionEmbedder{dimension: 3}, EmbeddingDim: 3}
	current := []store.Chunk{{
		ID:         "chunk-old",
		DocumentID: "doc-old",
		Ordinal:    4,
		Content:    "旧文档中保留下来的内容",
		Metadata:   map[string]any{"source": "legacy.pdf", "page": 2},
	}}

	rebuilt, err := service.rebuildStoredChunks(context.Background(), "doc-old", current)
	if err != nil {
		t.Fatal(err)
	}
	if len(rebuilt) != 1 {
		t.Fatalf("expected one rebuilt chunk, got %d", len(rebuilt))
	}
	chunk := rebuilt[0]
	if chunk.ID == current[0].ID || chunk.DocumentID != "doc-old" || chunk.Ordinal != 4 || chunk.Content != current[0].Content {
		t.Fatalf("unexpected rebuilt chunk: %#v", chunk)
	}
	if chunk.Metadata["page"] != 2 || chunk.Metadata["reindex_source"] != "stored_chunks" {
		t.Fatalf("metadata was not preserved: %#v", chunk.Metadata)
	}
	if len(chunk.Embedding) != 3 {
		t.Fatalf("unexpected embedding dimension: %d", len(chunk.Embedding))
	}
}
