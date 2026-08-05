package rag

import (
	"context"
	"testing"

	"github.com/lfd/minirag/internal/store"
)

type fakeReranker struct{}

func (fakeReranker) Rerank(context.Context, string, []string) ([]RerankResult, error) {
	return []RerankResult{{Index: 2, Score: .9}, {Index: 0, Score: .8}, {Index: 1, Score: .7}}, nil
}

func TestRerankHitsLimitsDocumentDuplicates(t *testing.T) {
	hits := []store.Hit{{ID: "a1", DocumentID: "a", Content: "a1"}, {ID: "b1", DocumentID: "b", Content: "b1"}, {ID: "a2", DocumentID: "a", Content: "a2"}}
	got, err := RerankHits(context.Background(), fakeReranker{}, "q", hits, 3, 1)
	if err != nil || len(got) != 2 || got[0].ID != "a2" || got[1].ID != "b1" || got[0].FinalRank != 1 {
		t.Fatalf("unexpected reranked hits: %#v, %v", got, err)
	}
}
