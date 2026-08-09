package main

import (
	"testing"

	"github.com/lfd/minirag/internal/evaluation"
)

func TestEvaluationModes(t *testing.T) {
	modes, err := evaluationModes("all")
	if err != nil {
		t.Fatal(err)
	}
	if len(modes) != 4 {
		t.Fatalf("expected four benchmark modes, got %d", len(modes))
	}
	if _, err := evaluationModes("unknown"); err == nil {
		t.Fatal("expected unsupported mode error")
	}
}

func TestRAGDatasetLoads(t *testing.T) {
	dataset, err := evaluation.LoadDataset("../../testdata/rag_eval.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := evaluation.ValidateRAGDataset(dataset); err != nil {
		t.Fatal(err)
	}
	if len(dataset.Cases) < 30 {
		t.Fatalf("rag benchmark has %d cases, want at least 30", len(dataset.Cases))
	}
	refusals := 0
	sources := make(map[string]struct{})
	for _, item := range dataset.Cases {
		if item.ShouldRefuse {
			refusals++
		}
		if item.SourceType != "" {
			sources[item.SourceType] = struct{}{}
		}
	}
	if refusals < 2 {
		t.Fatalf("rag benchmark has %d refusal cases, want at least 2", refusals)
	}
	for _, sourceType := range []string{"code", "pdf", "multi_document", "hard_negative"} {
		if _, ok := sources[sourceType]; !ok {
			t.Fatalf("rag benchmark missing source type %q", sourceType)
		}
	}
}
