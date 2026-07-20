package store

import "testing"

func TestNormalizeSearchLimits(t *testing.T) {
	tests := []struct {
		topK, candidateK int
		wantTopK         int
		wantCandidateK   int
	}{
		{topK: 5, candidateK: 50, wantTopK: 5, wantCandidateK: 50},
		{topK: 10, candidateK: 3, wantTopK: 10, wantCandidateK: 10},
		{topK: 0, candidateK: 0, wantTopK: 5, wantCandidateK: 5},
	}
	for _, test := range tests {
		topK, candidateK := normalizeSearchLimits(test.topK, test.candidateK)
		if topK != test.wantTopK || candidateK != test.wantCandidateK {
			t.Fatalf("normalizeSearchLimits(%d, %d) = (%d, %d), want (%d, %d)", test.topK, test.candidateK, topK, candidateK, test.wantTopK, test.wantCandidateK)
		}
	}
}
