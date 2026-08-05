package evaluation

import (
	"path/filepath"
	"testing"

	"github.com/lfd/minirag/internal/store"
)

func TestBaselineRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "baseline.json")
	want := BaselineFile{Version: 1, Dataset: "eval.json", Pipeline: "retrieval", TopK: 5, CandidateK: 50, Reports: []Report{{Mode: "vector", RecallAtK: 1}}}
	if err := SaveBaseline(path, want); err != nil {
		t.Fatal(err)
	}
	got, err := LoadBaseline(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Reports) != 1 || got.Reports[0].Mode != "vector" || got.Reports[0].RecallAtK != 1 {
		t.Fatalf("unexpected baseline: %#v", got)
	}
}

func TestFirstRelevantRankNormalizesPaths(t *testing.T) {
	// The evaluator stores document names, while datasets may use relative paths.
	got := firstRelevantRank([]store.Hit{{DocumentName: "rag-basics.md"}}, []string{"seeddata/rag-basics.md"})
	if got != 1 {
		t.Fatalf("expected rank 1, got %d", got)
	}
}

func TestPercentile95(t *testing.T) {
	got := percentile95([]int64{4, 1, 3, 2})
	if got != 4 {
		t.Fatalf("expected p95 4, got %d", got)
	}
}

func TestRecallAtKCountsEachRelevantDocumentOnce(t *testing.T) {
	hits := []store.Hit{{DocumentName: "a.md"}, {DocumentName: "a.md"}, {DocumentName: "b.md"}}
	if got := recallAtK(hits, []string{"a.md", "b.md"}, 2); got != 0.5 {
		t.Fatalf("expected recall 0.5, got %v", got)
	}
	if got := recallAtK(hits, []string{"a.md", "b.md"}, 3); got != 1 {
		t.Fatalf("expected recall 1, got %v", got)
	}
}

func TestNDCGAtKPenalizesLateRelevantDocuments(t *testing.T) {
	ideal := ndcgAtK([]store.Hit{{DocumentName: "a.md"}, {DocumentName: "b.md"}}, []string{"a.md", "b.md"}, 2)
	late := ndcgAtK([]store.Hit{{DocumentName: "noise.md"}, {DocumentName: "a.md"}}, []string{"a.md", "b.md"}, 2)
	if ideal != 1 || late <= 0 || late >= ideal {
		t.Fatalf("unexpected nDCG values: ideal=%v late=%v", ideal, late)
	}
}

func TestCompareBaselineDetectsQualityLatencyAndFailureRegressions(t *testing.T) {
	baseline := Report{Mode: "hybrid_rrf", Pipeline: "retrieval", RecallAtK: 1, MRR: 1, AverageLatencyMS: 100, FailedCases: 0}
	current := Report{Mode: "hybrid_rrf", Pipeline: "retrieval", RecallAtK: .8, MRR: .9, AverageLatencyMS: 130, FailedCases: 1}
	regressions := CompareBaseline([]Report{current}, []Report{baseline}, .05)
	if len(regressions) != 4 {
		t.Fatalf("expected four regressions, got %#v", regressions)
	}
}

func TestMergeBaselineKeepsBestValues(t *testing.T) {
	old := BaselineFile{Version: 1, Dataset: "testdata/eval.json", Pipeline: "retrieval", TopK: 5, CandidateK: 50, Reports: []Report{{Mode: "vector", Pipeline: "retrieval", RecallAtK: .9, AverageLatencyMS: 100}}}
	now := BaselineFile{Version: 1, Dataset: "testdata/eval.json", Pipeline: "retrieval", TopK: 5, CandidateK: 50, Reports: []Report{{Mode: "vector", Pipeline: "retrieval", RecallAtK: 1, AverageLatencyMS: 120}}}
	merged := MergeBaseline(old, now)
	if len(merged.Reports) != 1 || merged.Reports[0].RecallAtK != 1 || merged.Reports[0].AverageLatencyMS != 100 {
		t.Fatalf("unexpected merged baseline: %#v", merged)
	}
}

func TestValidateBaselineAndDatasetCoverage(t *testing.T) {
	dataset, err := LoadDataset("../../testdata/eval.json")
	if err != nil {
		t.Fatal(err)
	}
	if len(dataset.Cases) < 50 {
		t.Fatalf("retrieval benchmark has %d cases, want at least 50", len(dataset.Cases))
	}
	seen := make(map[string]struct{}, len(dataset.Cases))
	for _, item := range dataset.Cases {
		if _, ok := seen[item.ID]; ok {
			t.Fatalf("duplicate case id %q", item.ID)
		}
		seen[item.ID] = struct{}{}
	}
	if err := ValidateBaseline(BaselineFile{Dataset: "testdata/eval.json", Pipeline: "retrieval", TopK: 5, CandidateK: 50}, "testdata/eval.json", "retrieval", 5, 50); err != nil {
		t.Fatal(err)
	}
	if err := ValidateBaseline(BaselineFile{Dataset: "other.json", Pipeline: "retrieval", TopK: 5, CandidateK: 50}, "testdata/eval.json", "retrieval", 5, 50); err == nil {
		t.Fatal("expected dataset mismatch")
	}
}
