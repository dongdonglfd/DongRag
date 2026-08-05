package rag

import (
	"strings"
	"testing"

	"github.com/lfd/minirag/internal/store"
)

func TestPrepareGenerationContextSortsDeduplicatesAndKeepsMetadata(t *testing.T) {
	hits := []store.Hit{
		{ID: "low", DocumentID: "b", DocumentName: "b.md", Content: "较低相关内容", RRFScore: .01},
		{ID: "high", DocumentID: "a", DocumentName: "a.md", Content: "最相关内容", RRFScore: .03, Metadata: map[string]any{"heading_path": []any{"一级", "二级"}, "page": 3}},
		{ID: "duplicate", DocumentID: "a", DocumentName: "a.md", Content: "  最相关内容  ", RRFScore: .02},
	}
	selected, contextText := PrepareGenerationContext(hits, 1000, 2)
	if len(selected) != 2 || selected[0].ID != "high" || selected[1].ID != "low" {
		t.Fatalf("unexpected selected hits: %#v", selected)
	}
	if strings.Count(contextText, "最相关内容") != 1 {
		t.Fatalf("duplicate content was not removed: %s", contextText)
	}
	if !strings.Contains(contextText, "章节：一级 > 二级 | 页码：3\n正文：最相关内容") {
		t.Fatalf("metadata is missing or misplaced: %s", contextText)
	}
}

func TestPrepareGenerationContextLimitsDocumentAndBudget(t *testing.T) {
	hits := []store.Hit{
		{ID: "a1", DocumentID: "a", DocumentName: "a.md", Content: "第一段", Score: 3},
		{ID: "a2", DocumentID: "a", DocumentName: "a.md", Content: "第二段", Score: 2},
		{ID: "b1", DocumentID: "b", DocumentName: "b.md", Content: "第三段", Score: 1},
	}
	selected, contextText := PrepareGenerationContext(hits, 100, 1)
	if len(selected) != 2 || selected[0].ID != "a1" || selected[1].ID != "b1" {
		t.Fatalf("unexpected selected hits: %#v", selected)
	}
	if len(contextText) > 100 {
		t.Fatalf("context exceeds budget: %d", len(contextText))
	}
}

func TestGenerationPromptsRequireGroundedRefusalAndCitations(t *testing.T) {
	systemPrompt, userPrompt := GenerationPrompts("问题", "[1] Context")
	combined := systemPrompt + userPrompt
	for _, required := range []string{"只能使用", "不得使用外部知识", "资料不足，我不知道", "[1]"} {
		if !strings.Contains(combined, required) {
			t.Fatalf("prompt does not contain %q: %s", required, combined)
		}
	}
}

func TestFuseRRFMatchesRankFormulaAndTieBreakers(t *testing.T) {
	vector := []store.Hit{{ID: "a", VectorScore: .9}, {ID: "b", VectorScore: .8}}
	lexical := []store.Hit{{ID: "b", LexicalScore: .7}, {ID: "c", LexicalScore: .6}}
	got := fuseRRF(vector, lexical, 3)
	if len(got) != 3 || got[0].ID != "b" || got[1].ID != "a" || got[2].ID != "c" {
		t.Fatalf("unexpected RRF order: %#v", got)
	}
	if got[0].VectorRank != 2 || got[0].LexicalRank != 1 || got[0].RRFScore <= got[1].RRFScore {
		t.Fatalf("unexpected fused hit: %#v", got[0])
	}
}
