package evaluation

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/lfd/minirag/internal/rag"
	"github.com/lfd/minirag/internal/store"
)

type Embedder interface {
	Embed(context.Context, []string) ([][]float32, error)
}

type Case struct {
	ID                string   `json:"id"`
	Question          string   `json:"question"`
	RelevantDocuments []string `json:"relevant_documents"`
	Difficulty        string   `json:"difficulty,omitempty"`
	QueryType         string   `json:"query_type,omitempty"`
	ReferenceAnswer   string   `json:"reference_answer,omitempty"`
	RequiredCitations []string `json:"required_citations,omitempty"`
	ShouldRefuse      bool     `json:"should_refuse,omitempty"`
	SourceType        string   `json:"source_type,omitempty"`
}

type Dataset struct {
	Cases []Case `json:"cases"`
}

type CaseResult struct {
	ID                   string   `json:"id"`
	Question             string   `json:"question"`
	Difficulty           string   `json:"difficulty,omitempty"`
	QueryType            string   `json:"query_type,omitempty"`
	SourceType           string   `json:"source_type,omitempty"`
	TopDocuments         []string `json:"top_documents"`
	FirstRelevantRank    int      `json:"first_relevant_rank"`
	RecallAtK            float64  `json:"recall_at_k"`
	NDCGAtK              float64  `json:"ndcg_at_k"`
	LatencyMS            int64    `json:"latency_ms"`
	PipelineLatencyMS    int64    `json:"pipeline_latency_ms,omitempty"`
	Error                string   `json:"error,omitempty"`
	Answer               string   `json:"answer,omitempty"`
	Citations            []string `json:"citations,omitempty"`
	AnswerCorrectness    float64  `json:"answer_correctness"`
	CitationPrecision    float64  `json:"citation_precision"`
	CitationRecall       float64  `json:"citation_recall"`
	Faithfulness         float64  `json:"faithfulness"`
	UnsupportedClaimRate float64  `json:"unsupported_claim_rate"`
	RefusalAccuracy      float64  `json:"refusal_accuracy"`
}

type Report struct {
	Mode                     string       `json:"mode"`
	TopK                     int          `json:"top_k"`
	CandidateK               int          `json:"candidate_k"`
	TotalCases               int          `json:"total_cases"`
	EvaluatedCases           int          `json:"evaluated_cases"`
	FailedCases              int          `json:"failed_cases"`
	AnswerEvaluatedCases     int          `json:"answer_evaluated_cases"`
	AnswerFailedCases        int          `json:"answer_failed_cases"`
	RecallAt1                float64      `json:"recall_at_1"`
	RecallAtK                float64      `json:"recall_at_k"`
	MRR                      float64      `json:"mrr"`
	NDCGAtK                  float64      `json:"ndcg_at_k"`
	AverageLatencyMS         float64      `json:"average_latency_ms"`
	P95LatencyMS             int64        `json:"p95_latency_ms"`
	AveragePipelineLatencyMS float64      `json:"average_pipeline_latency_ms"`
	P95PipelineLatencyMS     int64        `json:"p95_pipeline_latency_ms"`
	Pipeline                 string       `json:"pipeline"`
	AnswerCorrectness        float64      `json:"answer_correctness"`
	CitationPrecision        float64      `json:"citation_precision"`
	CitationRecall           float64      `json:"citation_recall"`
	Faithfulness             float64      `json:"faithfulness"`
	UnsupportedClaimRate     float64      `json:"unsupported_claim_rate"`
	RefusalAccuracy          float64      `json:"refusal_accuracy"`
	Results                  []CaseResult `json:"results"`
}

type Chatger interface {
	Complete(context.Context, string, string) (string, error)
}
type Judge interface {
	Evaluate(Case, string, []store.Hit) AnswerScores
}
type AnswerScores struct{ AnswerCorrectness, CitationPrecision, CitationRecall, Faithfulness, UnsupportedClaimRate, RefusalAccuracy float64 }

// RuleJudge is deterministic and network-free; Judge can be replaced by an LLM judge later.
type RuleJudge struct{}

func (RuleJudge) Evaluate(item Case, answer string, hits []store.Hit) AnswerScores {
	s := AnswerScores{AnswerCorrectness: tokenF1(answer, item.ReferenceAnswer)}
	refused := looksRefused(answer)
	if item.ShouldRefuse && refused {
		s.AnswerCorrectness = 1
	}
	cited := citedDocuments(answer, hits)
	wanted := relevantDocumentSet(item.RequiredCitations)
	if len(cited) == 0 {
		if len(wanted) == 0 {
			s.CitationPrecision = 1
		}
	} else {
		correct := 0
		for name := range cited {
			if _, ok := wanted[name]; ok {
				correct++
			}
		}
		s.CitationPrecision = float64(correct) / float64(len(cited))
	}
	if len(wanted) > 0 {
		correct := 0
		for name := range wanted {
			if _, ok := cited[name]; ok {
				correct++
			}
		}
		s.CitationRecall = float64(correct) / float64(len(wanted))
	} else {
		s.CitationRecall = 1
	}
	unsupported, total := unsupportedClaims(answer, hits)
	if item.ShouldRefuse && refused {
		s.UnsupportedClaimRate = 0
		s.Faithfulness = 1
	} else if total > 0 {
		s.UnsupportedClaimRate = float64(unsupported) / float64(total)
		s.Faithfulness = 1 - s.UnsupportedClaimRate
	}
	if refused == item.ShouldRefuse {
		s.RefusalAccuracy = 1
	}
	return s
}

func RunRAG(ctx context.Context, database *store.Store, embedder Embedder, reranker rag.Reranker, chatger Chatger, judge Judge, dataset Dataset, mode store.SearchMode, topK, candidateK int) (Report, error) {
	if chatger == nil || judge == nil {
		return Report{}, fmt.Errorf("chatger and judge are required for rag evaluation")
	}
	report, err := Run(ctx, database, embedder, reranker, dataset, mode, topK, candidateK)
	if err != nil {
		return Report{}, err
	}
	report.Pipeline = "rag"
	var sums AnswerScores
	count := 0
	pipelineLatencies := make([]int64, 0, len(dataset.Cases))
	for i, item := range dataset.Cases {
		if report.Results[i].Error != "" {
			report.AnswerFailedCases++
			continue
		}
		pipelineStarted := time.Now()
		hits, err := retrieve(ctx, database, embedder, reranker, item.Question, mode, topK, candidateK)
		if err != nil {
			report.Results[i].Error = err.Error()
			report.AnswerFailedCases++
			continue
		}
		hits, contextText := rag.PrepareGenerationContext(hits, 6000, 2)
		systemPrompt, userPrompt := rag.GenerationPrompts(item.Question, contextText)
		answer, err := chatger.Complete(ctx, systemPrompt, userPrompt)
		if err != nil {
			report.Results[i].Error = err.Error()
			report.AnswerFailedCases++
			continue
		}
		report.Results[i].PipelineLatencyMS = time.Since(pipelineStarted).Milliseconds()
		pipelineLatencies = append(pipelineLatencies, report.Results[i].PipelineLatencyMS)
		scores := judge.Evaluate(item, answer, hits)
		report.Results[i].Answer = answer
		report.Results[i].AnswerCorrectness = scores.AnswerCorrectness
		report.Results[i].CitationPrecision = scores.CitationPrecision
		report.Results[i].CitationRecall = scores.CitationRecall
		report.Results[i].Faithfulness = scores.Faithfulness
		report.Results[i].UnsupportedClaimRate = scores.UnsupportedClaimRate
		report.Results[i].RefusalAccuracy = scores.RefusalAccuracy
		for _, hit := range hits {
			report.Results[i].Citations = append(report.Results[i].Citations, hit.DocumentName)
		}
		sums.AnswerCorrectness += scores.AnswerCorrectness
		sums.CitationPrecision += scores.CitationPrecision
		sums.CitationRecall += scores.CitationRecall
		sums.Faithfulness += scores.Faithfulness
		sums.UnsupportedClaimRate += scores.UnsupportedClaimRate
		sums.RefusalAccuracy += scores.RefusalAccuracy
		count++
	}
	if count > 0 {
		report.AnswerEvaluatedCases = count
		report.AnswerCorrectness = sums.AnswerCorrectness / float64(count)
		report.CitationPrecision = sums.CitationPrecision / float64(count)
		report.CitationRecall = sums.CitationRecall / float64(count)
		report.Faithfulness = sums.Faithfulness / float64(count)
		report.UnsupportedClaimRate = sums.UnsupportedClaimRate / float64(count)
		report.RefusalAccuracy = sums.RefusalAccuracy / float64(count)
	}
	report.AveragePipelineLatencyMS = average(pipelineLatencies)
	report.P95PipelineLatencyMS = percentile95(pipelineLatencies)
	return report, nil
}

type ComparisonReport struct {
	Reports     []Report      `json:"reports"`
	Baseline    *BaselineInfo `json:"baseline,omitempty"`
	Regressions []Regression  `json:"regressions,omitempty"`
}

// BaselineFile is intentionally plain JSON so it can be reviewed and stored as a CI artifact.
type BaselineFile struct {
	Version    int      `json:"version"`
	Dataset    string   `json:"dataset"`
	Pipeline   string   `json:"pipeline"`
	TopK       int      `json:"top_k"`
	CandidateK int      `json:"candidate_k"`
	Reports    []Report `json:"reports"`
}

func LoadBaseline(path string) (BaselineFile, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return BaselineFile{}, fmt.Errorf("read baseline: %w", err)
	}
	var baseline BaselineFile
	if err := json.Unmarshal(data, &baseline); err != nil {
		return BaselineFile{}, fmt.Errorf("decode baseline: %w", err)
	}
	if len(baseline.Reports) == 0 {
		return BaselineFile{}, fmt.Errorf("baseline has no reports")
	}
	return baseline, nil
}

func SaveBaseline(path string, baseline BaselineFile) error {
	data, err := json.MarshalIndent(baseline, "", "  ")
	if err != nil {
		return fmt.Errorf("encode baseline: %w", err)
	}
	data = append(data, '\n')
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("create baseline directory: %w", err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return fmt.Errorf("write baseline: %w", err)
	}
	return nil
}

type BaselineInfo struct {
	Path        string `json:"path"`
	Updated     bool   `json:"updated"`
	Regressions int    `json:"regressions"`
}

type Regression struct {
	Mode      string  `json:"mode"`
	Metric    string  `json:"metric"`
	Baseline  float64 `json:"baseline"`
	Current   float64 `json:"current"`
	Tolerance float64 `json:"tolerance"`
}

// CompareBaseline reports material regressions. Scores use higher-is-better semantics;
// latency metrics use lower-is-better semantics. Tolerance is a relative fraction.
func CompareBaseline(current, baseline []Report, tolerance float64) []Regression {
	if tolerance < 0 {
		tolerance = 0
	}
	previous := make(map[string]Report, len(baseline))
	for _, report := range baseline {
		previous[report.Mode] = report
	}
	var regressions []Regression
	for _, report := range current {
		old, ok := previous[report.Mode]
		if !ok {
			continue
		}
		for _, metric := range benchmarkMetrics(report.Pipeline) {
			now, before := metricValue(report, metric.name), metricValue(old, metric.name)
			if before <= 0 && !metric.zeroIsValid {
				continue
			}
			limit := before * (1 - tolerance)
			if metric.lowerIsBetter {
				limit = before * (1 + tolerance)
				if now > limit {
					regressions = append(regressions, Regression{Mode: report.Mode, Metric: metric.name, Baseline: before, Current: now, Tolerance: tolerance})
				}
			} else if now < limit {
				regressions = append(regressions, Regression{Mode: report.Mode, Metric: metric.name, Baseline: before, Current: now, Tolerance: tolerance})
			}
		}
	}
	return regressions
}

type benchmarkMetric struct {
	name          string
	lowerIsBetter bool
	zeroIsValid   bool
}

func benchmarkMetrics(pipeline string) []benchmarkMetric {
	metrics := []benchmarkMetric{
		{name: "recall_at_1"}, {name: "recall_at_k"}, {name: "mrr"}, {name: "ndcg_at_k"},
		{name: "average_latency_ms", lowerIsBetter: true}, {name: "p95_latency_ms", lowerIsBetter: true},
		{name: "failed_cases", lowerIsBetter: true, zeroIsValid: true},
	}
	if pipeline == "rag" {
		metrics = append(metrics,
			benchmarkMetric{name: "answer_correctness"},
			benchmarkMetric{name: "citation_precision"},
			benchmarkMetric{name: "citation_recall"},
			benchmarkMetric{name: "faithfulness"},
			benchmarkMetric{name: "unsupported_claim_rate", lowerIsBetter: true, zeroIsValid: true},
			benchmarkMetric{name: "refusal_accuracy"},
			benchmarkMetric{name: "average_pipeline_latency_ms", lowerIsBetter: true},
			benchmarkMetric{name: "p95_pipeline_latency_ms", lowerIsBetter: true},
			benchmarkMetric{name: "answer_failed_cases", lowerIsBetter: true, zeroIsValid: true},
		)
	}
	return metrics
}

func metricValue(report Report, name string) float64 {
	switch name {
	case "recall_at_1":
		return report.RecallAt1
	case "recall_at_k":
		return report.RecallAtK
	case "mrr":
		return report.MRR
	case "ndcg_at_k":
		return report.NDCGAtK
	case "average_latency_ms":
		return report.AverageLatencyMS
	case "p95_latency_ms":
		return float64(report.P95LatencyMS)
	case "failed_cases":
		return float64(report.FailedCases)
	case "answer_correctness":
		return report.AnswerCorrectness
	case "citation_precision":
		return report.CitationPrecision
	case "citation_recall":
		return report.CitationRecall
	case "faithfulness":
		return report.Faithfulness
	case "unsupported_claim_rate":
		return report.UnsupportedClaimRate
	case "refusal_accuracy":
		return report.RefusalAccuracy
	case "average_pipeline_latency_ms":
		return report.AveragePipelineLatencyMS
	case "p95_pipeline_latency_ms":
		return float64(report.P95PipelineLatencyMS)
	case "answer_failed_cases":
		return float64(report.AnswerFailedCases)
	default:
		return 0
	}
}

// MergeBaseline keeps the best observed value for each metric while preserving report details.
func MergeBaseline(existing, current BaselineFile) BaselineFile {
	byMode := make(map[string]Report, len(existing.Reports))
	for _, report := range existing.Reports {
		byMode[report.Mode] = report
	}
	for _, report := range current.Reports {
		best, ok := byMode[report.Mode]
		if !ok {
			byMode[report.Mode] = report
			continue
		}
		for _, metric := range benchmarkMetrics(report.Pipeline) {
			currentValue := metricValue(report, metric.name)
			bestValue := metricValue(best, metric.name)
			if metric.lowerIsBetter {
				if (!metric.zeroIsValid && bestValue <= 0) || currentValue < bestValue {
					setMetricValue(&best, metric.name, currentValue)
				}
			} else if currentValue > bestValue {
				setMetricValue(&best, metric.name, currentValue)
			}
		}
		byMode[report.Mode] = best
	}
	merged := current
	merged.Version = 1
	if existing.Dataset != "" {
		merged.Dataset = existing.Dataset
	}
	if existing.Pipeline != "" {
		merged.Pipeline = existing.Pipeline
	}
	merged.Reports = make([]Report, 0, len(byMode))
	for _, report := range current.Reports {
		if best, ok := byMode[report.Mode]; ok {
			merged.Reports = append(merged.Reports, best)
			delete(byMode, report.Mode)
		}
	}
	for _, report := range byMode {
		merged.Reports = append(merged.Reports, report)
	}
	return merged
}

func CompactReports(reports []Report) []Report {
	compact := make([]Report, len(reports))
	for i, report := range reports {
		report.Results = nil
		compact[i] = report
	}
	return compact
}

func ValidateBaseline(baseline BaselineFile, dataset, pipeline string, topK, candidateK int) error {
	if baseline.Dataset != dataset {
		return fmt.Errorf("baseline dataset mismatch: got %q, want %q", baseline.Dataset, dataset)
	}
	if baseline.Pipeline != pipeline {
		return fmt.Errorf("baseline pipeline mismatch: got %q, want %q", baseline.Pipeline, pipeline)
	}
	if baseline.TopK != topK || baseline.CandidateK != candidateK {
		return fmt.Errorf("baseline retrieval settings mismatch: got top-k=%d candidate-k=%d, want top-k=%d candidate-k=%d", baseline.TopK, baseline.CandidateK, topK, candidateK)
	}
	return nil
}

func setMetricValue(report *Report, name string, value float64) {
	switch name {
	case "recall_at_1":
		report.RecallAt1 = value
	case "recall_at_k":
		report.RecallAtK = value
	case "mrr":
		report.MRR = value
	case "ndcg_at_k":
		report.NDCGAtK = value
	case "average_latency_ms":
		report.AverageLatencyMS = value
	case "p95_latency_ms":
		report.P95LatencyMS = int64(value)
	case "failed_cases":
		report.FailedCases = int(value)
	case "answer_correctness":
		report.AnswerCorrectness = value
	case "citation_precision":
		report.CitationPrecision = value
	case "citation_recall":
		report.CitationRecall = value
	case "faithfulness":
		report.Faithfulness = value
	case "unsupported_claim_rate":
		report.UnsupportedClaimRate = value
	case "refusal_accuracy":
		report.RefusalAccuracy = value
	case "average_pipeline_latency_ms":
		report.AveragePipelineLatencyMS = value
	case "p95_pipeline_latency_ms":
		report.P95PipelineLatencyMS = int64(value)
	case "answer_failed_cases":
		report.AnswerFailedCases = int(value)
	}
}

func LoadDataset(path string) (Dataset, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Dataset{}, fmt.Errorf("read evaluation dataset: %w", err)
	}
	var dataset Dataset
	if err := json.Unmarshal(data, &dataset); err != nil {
		return Dataset{}, fmt.Errorf("decode evaluation dataset: %w", err)
	}
	if len(dataset.Cases) == 0 {
		return Dataset{}, fmt.Errorf("evaluation dataset has no cases")
	}
	seenIDs := make(map[string]struct{}, len(dataset.Cases))
	for i, item := range dataset.Cases {
		if strings.TrimSpace(item.ID) == "" {
			return Dataset{}, fmt.Errorf("case %d has no id", i)
		}
		if _, exists := seenIDs[item.ID]; exists {
			return Dataset{}, fmt.Errorf("duplicate case id %s", item.ID)
		}
		seenIDs[item.ID] = struct{}{}
		if strings.TrimSpace(item.Question) == "" {
			return Dataset{}, fmt.Errorf("case %s has no question", item.ID)
		}
		if len(item.RelevantDocuments) == 0 && !item.ShouldRefuse {
			return Dataset{}, fmt.Errorf("case %s has no relevant_documents", item.ID)
		}
	}
	return dataset, nil
}

func ValidateRAGDataset(dataset Dataset) error {
	for _, item := range dataset.Cases {
		if strings.TrimSpace(item.ReferenceAnswer) == "" && !item.ShouldRefuse {
			return fmt.Errorf("case %s has no reference_answer", item.ID)
		}
		if len(item.RequiredCitations) == 0 && !item.ShouldRefuse {
			return fmt.Errorf("case %s has no required_citations", item.ID)
		}
	}
	return nil
}

func Run(ctx context.Context, database *store.Store, embedder Embedder, reranker rag.Reranker, dataset Dataset, mode store.SearchMode, topK, candidateK int) (Report, error) {
	if mode != store.SearchModeVector && mode != store.SearchModeLexical && mode != store.SearchModeWeighted && mode != store.SearchModeHybrid && mode != store.SearchModeRerank {
		return Report{}, fmt.Errorf("unsupported search mode %q", mode)
	}
	if topK < 1 {
		topK = 5
	}
	if candidateK < topK {
		candidateK = topK
	}
	report := Report{Mode: string(mode), Pipeline: "retrieval", TopK: topK, CandidateK: candidateK, TotalCases: len(dataset.Cases), Results: make([]CaseResult, 0, len(dataset.Cases))}
	latencies := make([]int64, 0, len(dataset.Cases))
	var reciprocalRankSum float64
	var recallAt1Sum float64
	var recallAtKSum float64
	var ndcgAtKSum float64

	for _, item := range dataset.Cases {
		result := CaseResult{ID: item.ID, Question: item.Question, Difficulty: item.Difficulty, QueryType: item.QueryType, SourceType: item.SourceType}
		started := time.Now()
		var embedding []float32
		var found []store.Hit
		var err error
		searchMode := mode
		if mode == store.SearchModeRerank {
			searchMode = store.SearchModeHybrid
		}
		if searchMode != store.SearchModeLexical {
			var embeddings [][]float32
			embeddings, err = embedder.Embed(ctx, []string{item.Question})
			if err == nil && (len(embeddings) != 1 || len(embeddings[0]) == 0) {
				err = fmt.Errorf("invalid embedding response")
			}
			if err == nil {
				embedding = embeddings[0]
			}
		}
		if err == nil {
			searchTopK := topK
			if mode == store.SearchModeRerank {
				searchTopK = candidateK
			}
			found, err = database.Search(ctx, item.Question, embedding, searchMode, searchTopK, candidateK)
			if err == nil && mode == store.SearchModeRerank {
				if reranker == nil {
					err = fmt.Errorf("reranker is not configured")
				} else {
					found, err = rag.RerankHits(ctx, reranker, item.Question, found, topK, 2)
				}
			}
			for _, hit := range found {
				result.TopDocuments = append(result.TopDocuments, hit.DocumentName)
			}
			result.FirstRelevantRank = firstRelevantRank(found, item.RelevantDocuments)
		}
		result.LatencyMS = time.Since(started).Milliseconds()
		if err != nil {
			result.Error = err.Error()
			report.FailedCases++
			report.Results = append(report.Results, result)
			continue
		}
		latencies = append(latencies, result.LatencyMS)
		if item.ShouldRefuse {
			report.Results = append(report.Results, result)
			continue
		}
		report.EvaluatedCases++
		result.RecallAtK = recallAtK(found, item.RelevantDocuments, topK)
		result.NDCGAtK = ndcgAtK(found, item.RelevantDocuments, topK)
		recallAt1Sum += recallAtK(found, item.RelevantDocuments, 1)
		recallAtKSum += result.RecallAtK
		ndcgAtKSum += result.NDCGAtK
		if result.FirstRelevantRank > 0 {
			reciprocalRankSum += 1.0 / float64(result.FirstRelevantRank)
		}
		report.Results = append(report.Results, result)
	}

	if report.EvaluatedCases > 0 {
		report.RecallAt1 = recallAt1Sum / float64(report.EvaluatedCases)
		report.RecallAtK = recallAtKSum / float64(report.EvaluatedCases)
		report.MRR = reciprocalRankSum / float64(report.EvaluatedCases)
		report.NDCGAtK = ndcgAtKSum / float64(report.EvaluatedCases)
		report.AverageLatencyMS = average(latencies)
		report.P95LatencyMS = percentile95(latencies)
	}
	return report, nil
}

func retrieve(ctx context.Context, database *store.Store, embedder Embedder, reranker rag.Reranker, question string, mode store.SearchMode, topK, candidateK int) ([]store.Hit, error) {
	var embedding []float32
	searchMode := mode
	if mode == store.SearchModeRerank {
		searchMode = store.SearchModeHybrid
	}
	if searchMode != store.SearchModeLexical {
		embeddings, err := embedder.Embed(ctx, []string{question})
		if err != nil {
			return nil, err
		}
		if len(embeddings) != 1 || len(embeddings[0]) == 0 {
			return nil, fmt.Errorf("invalid embedding response")
		}
		embedding = embeddings[0]
	}
	searchTopK := topK
	if mode == store.SearchModeRerank {
		searchTopK = candidateK
	}
	hits, err := database.Search(ctx, question, embedding, searchMode, searchTopK, candidateK)
	if err != nil {
		return nil, err
	}
	if mode == store.SearchModeRerank {
		if reranker == nil {
			return nil, fmt.Errorf("reranker is not configured")
		}
		return rag.RerankHits(ctx, reranker, question, hits, topK, 2)
	}
	return hits, nil
}

func citedDocuments(answer string, hits []store.Hit) map[string]struct{} {
	result := make(map[string]struct{})
	for i := 0; i < len(answer); i++ {
		if answer[i] != '[' {
			continue
		}
		end := strings.IndexByte(answer[i:], ']')
		if end < 2 {
			continue
		}
		var n int
		if _, err := fmt.Sscanf(answer[i:i+end+1], "[%d]", &n); err == nil && n > 0 && n <= len(hits) {
			result[normalizeName(hits[n-1].DocumentName)] = struct{}{}
		}
		i += end
	}
	return result
}

func tokenF1(actual, expected string) float64 {
	a := textTokens(actual)
	e := textTokens(expected)
	if len(e) == 0 {
		if len(a) == 0 {
			return 1
		}
		return 0
	}
	if len(a) == 0 {
		return 0
	}
	ac := make(map[string]int)
	ec := make(map[string]int)
	for _, v := range a {
		ac[v]++
	}
	for _, v := range e {
		ec[v]++
	}
	common := 0
	for v, n := range ac {
		if ec[v] < n {
			n = ec[v]
		}
		common += n
	}
	if common == 0 {
		return 0
	}
	p := float64(common) / float64(len(a))
	r := float64(common) / float64(len(e))
	return 2 * p * r / (p + r)
}

func textTokens(value string) []string {
	value = strings.ToLower(value)
	var tokens []string
	var word []rune
	flush := func() {
		if len(word) > 0 {
			tokens = append(tokens, string(word))
			word = nil
		}
	}
	var han []rune
	flushHan := func() {
		if len(han) == 1 {
			tokens = append(tokens, string(han[0]))
		}
		for i := 0; i+1 < len(han); i++ {
			tokens = append(tokens, string(han[i:i+2]))
		}
		han = nil
	}
	for _, r := range value {
		if r >= '一' && r <= '鿿' {
			flush()
			han = append(han, r)
			continue
		}
		flushHan()
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') {
			word = append(word, r)
		} else {
			flush()
		}
	}
	flush()
	flushHan()
	return tokens
}

func unsupportedClaims(answer string, hits []store.Hit) (int, int) {
	evidence := ""
	for _, hit := range hits {
		evidence += " " + hit.Content
	}
	evidenceTokens := make(map[string]struct{})
	for _, t := range textTokens(evidence) {
		evidenceTokens[t] = struct{}{}
	}
	clean := answer
	for i := 1; i <= len(hits); i++ {
		clean = strings.ReplaceAll(clean, fmt.Sprintf("[%d]", i), "")
	}
	claims := strings.FieldsFunc(clean, func(r rune) bool {
		return r == '。' || r == '！' || r == '？' || r == '.' || r == '!' || r == '?' || r == '\n'
	})
	unsupported := 0
	total := 0
	for _, claim := range claims {
		tokens := textTokens(claim)
		if len(tokens) < 2 {
			continue
		}
		total++
		matched := 0
		for _, t := range tokens {
			if _, ok := evidenceTokens[t]; ok {
				matched++
			}
		}
		if float64(matched)/float64(len(tokens)) < 0.5 {
			unsupported++
		}
	}
	return unsupported, total
}

func looksRefused(answer string) bool {
	v := strings.ToLower(strings.TrimSpace(answer))
	for _, phrase := range []string{"不知道", "无法回答", "资料不足", "没有足够", "无法确定", "not enough information", "cannot answer", "don't know"} {
		if strings.Contains(v, phrase) {
			return true
		}
	}
	return false
}

func recallAtK(hits []store.Hit, relevantDocuments []string, k int) float64 {
	wanted := relevantDocumentSet(relevantDocuments)
	if len(wanted) == 0 || k < 1 {
		return 0
	}
	seen := make(map[string]struct{}, len(wanted))
	for i := 0; i < len(hits) && i < k; i++ {
		name := normalizeName(hits[i].DocumentName)
		if _, ok := wanted[name]; ok {
			seen[name] = struct{}{}
		}
	}
	return float64(len(seen)) / float64(len(wanted))
}

func ndcgAtK(hits []store.Hit, relevantDocuments []string, k int) float64 {
	wanted := relevantDocumentSet(relevantDocuments)
	if len(wanted) == 0 || k < 1 {
		return 0
	}
	seen := make(map[string]struct{}, len(wanted))
	var dcg float64
	for i := 0; i < len(hits) && i < k; i++ {
		name := normalizeName(hits[i].DocumentName)
		if _, relevant := wanted[name]; !relevant {
			continue
		}
		if _, duplicate := seen[name]; duplicate {
			continue
		}
		seen[name] = struct{}{}
		dcg += 1 / math.Log2(float64(i+2))
	}
	idealCount := len(wanted)
	if idealCount > k {
		idealCount = k
	}
	var idealDCG float64
	for i := 0; i < idealCount; i++ {
		idealDCG += 1 / math.Log2(float64(i+2))
	}
	return dcg / idealDCG
}

func relevantDocumentSet(relevantDocuments []string) map[string]struct{} {
	wanted := make(map[string]struct{}, len(relevantDocuments))
	for _, name := range relevantDocuments {
		wanted[normalizeName(name)] = struct{}{}
	}
	return wanted
}

func firstRelevantRank(hits []store.Hit, relevantDocuments []string) int {
	wanted := relevantDocumentSet(relevantDocuments)
	for index, hit := range hits {
		if _, ok := wanted[normalizeName(hit.DocumentName)]; ok {
			return index + 1
		}
	}
	return 0
}

func normalizeName(name string) string {
	return strings.ToLower(filepath.Base(strings.TrimSpace(name)))
}

func average(values []int64) float64 {
	if len(values) == 0 {
		return 0
	}
	var total int64
	for _, value := range values {
		total += value
	}
	return float64(total) / float64(len(values))
}

func percentile95(values []int64) int64 {
	if len(values) == 0 {
		return 0
	}
	sorted := append([]int64(nil), values...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
	index := (len(sorted)*95 + 99) / 100
	if index < 1 {
		index = 1
	}
	return sorted[index-1]
}
