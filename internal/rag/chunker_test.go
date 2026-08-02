package rag

import "testing"

func TestChunkTextSupportsOverlapAndUnicode(t *testing.T) {
	chunks := ChunkText("第一段内容。第二段内容。第三段内容。", 8, 2)
	if len(chunks) < 3 {
		t.Fatalf("expected multiple chunks, got %d", len(chunks))
	}
	for _, chunk := range chunks {
		if chunk == "" {
			t.Fatal("chunk must not be empty")
		}
	}
}

func TestChunkTextEmpty(t *testing.T) {
	if chunks := ChunkText("  ", 100, 10); chunks != nil {
		t.Fatalf("expected nil chunks, got %#v", chunks)
	}
}

func TestParseMarkdownBlocksPreservesHeadingAndCodeMetadata(t *testing.T) {
	blocks := ParseMarkdownBlocks("# Go\n\n## Context\n\n用于传递取消信号。\n\n```go\nctx, cancel := context.WithCancel(ctx)\n```")
	if len(blocks) != 2 {
		t.Fatalf("expected 2 blocks, got %d", len(blocks))
	}
	headingPath, ok := blocks[0].Metadata["heading_path"].([]string)
	if !ok || len(headingPath) != 2 || headingPath[1] != "Context" {
		t.Fatalf("unexpected heading path: %#v", blocks[0].Metadata["heading_path"])
	}
	if blocks[1].Metadata["block_type"] != "code" || blocks[1].Metadata["language"] != "go" {
		t.Fatalf("unexpected code metadata: %#v", blocks[1].Metadata)
	}
}

func TestChunkBlocksCarriesMetadata(t *testing.T) {
	parts := ChunkBlocks([]SourceBlock{{Text: "第一段。第二段。", Metadata: map[string]any{"page": 3}}}, 6, 1)
	if len(parts) == 0 {
		t.Fatal("expected chunks")
	}
	if parts[0].Metadata["page"] != 3 || parts[0].Metadata["chunk_index"] != 0 {
		t.Fatalf("metadata was not carried: %#v", parts[0].Metadata)
	}
}

func TestChunkBlocksMergesAdjacentParagraphsAndAddsHeadingContext(t *testing.T) {
	metadata := map[string]any{"block_type": "text", "heading_path": []string{"Go", "Context"}}
	parts := ChunkBlocks([]SourceBlock{
		{Text: "第一段。", Metadata: cloneMetadata(metadata)},
		{Text: "第二段。", Metadata: cloneMetadata(metadata)},
	}, 100, 10)
	if len(parts) != 1 {
		t.Fatalf("expected adjacent paragraphs to be merged, got %d chunks", len(parts))
	}
	if parts[0].Text != "Go > Context\n\n第一段。\n\n第二段。" {
		t.Fatalf("unexpected structured chunk: %q", parts[0].Text)
	}
	if parts[0].Metadata["block_end_index"] != 1 {
		t.Fatalf("unexpected merged block metadata: %#v", parts[0].Metadata)
	}
}
