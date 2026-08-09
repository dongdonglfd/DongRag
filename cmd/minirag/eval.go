package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/lfd/minirag/internal/config"
	"github.com/lfd/minirag/internal/evaluation"
	"github.com/lfd/minirag/internal/provider"
	"github.com/lfd/minirag/internal/rag"
	"github.com/lfd/minirag/internal/store"
)

func runEvaluation(cfg config.Config, args []string) error {
	flags := flag.NewFlagSet("eval", flag.ContinueOnError)
	datasetPath := flags.String("dataset", "testdata/eval.json", "evaluation dataset path")
	topK := flags.Int("top-k", 5, "number of retrieved chunks")
	candidateK := flags.Int("candidate-k", cfg.RetrievalCandidateK, "number of candidates per retrieval branch")
	mode := flags.String("mode", "all", "search mode: all, vector, lexical, weighted, hybrid, or rerank")
	pipeline := flags.String("pipeline", "retrieval", "evaluation pipeline: retrieval or rag")
	baselinePath := flags.String("baseline", "", "optional baseline JSON to compare against")
	updateBaseline := flags.Bool("update-baseline", false, "save current results as the historical-best baseline")
	failOnRegression := flags.Bool("fail-on-regression", false, "return an error when baseline metrics regress")
	tolerance := flags.Float64("regression-tolerance", 0.05, "relative tolerance for baseline regression checks")
	if err := flags.Parse(args); err != nil {
		return err
	}
	dataset, err := evaluation.LoadDataset(*datasetPath)
	if err != nil {
		return err
	}
	if *pipeline != "retrieval" && *pipeline != "rag" {
		return fmt.Errorf("unsupported evaluation pipeline %q", *pipeline)
	}
	if *pipeline == "rag" {
		if err := cfg.ValidateChat(); err != nil {
			return err
		}
		if err := evaluation.ValidateRAGDataset(dataset); err != nil {
			return err
		}
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()
	database, err := store.New(ctx, cfg.DatabaseURL)
	if err != nil {
		return err
	}
	defer database.Close()
	if err := database.Migrate(ctx); err != nil {
		return err
	}

	embedder := provider.OllamaEmbedder{BaseURL: cfg.OllamaBaseURL, Model: cfg.OllamaEmbedModel}
	var reranker rag.Reranker
	if cfg.RerankEnabled {
		reranker = funcReranker{client: provider.LlamaCppReranker{BaseURL: cfg.RerankURL}}
	}
	modes, err := evaluationModes(*mode)
	if err != nil {
		return err
	}
	comparison := evaluation.ComparisonReport{Reports: make([]evaluation.Report, 0, len(modes))}
	chatger := provider.BailianChat{APIKey: cfg.BailianAPIKey, BaseURL: cfg.BailianBaseURL, Model: cfg.BailianChatModel, Temperature: 0}
	judge := evaluation.RuleJudge{}
	for _, searchMode := range modes {
		var report evaluation.Report
		if *pipeline == "rag" {
			report, err = evaluation.RunRAG(ctx, database, embedder, reranker, chatger, judge, dataset, searchMode, *topK, *candidateK)
		} else {
			report, err = evaluation.Run(ctx, database, embedder, reranker, dataset, searchMode, *topK, *candidateK)
		}
		if err != nil {
			return err
		}
		if searchMode == store.SearchModeHybrid {
			report.Mode = "hybrid_rrf"
		} else if searchMode == store.SearchModeRerank {
			report.Mode = "hybrid_rerank"
		}
		comparison.Reports = append(comparison.Reports, report)
	}
	if *baselinePath != "" {
		regressionFailure := false
		baseline, loadErr := evaluation.LoadBaseline(*baselinePath)
		baselineLoaded := loadErr == nil
		if loadErr != nil {
			_, statErr := os.Stat(*baselinePath)
			if !*updateBaseline || !os.IsNotExist(statErr) {
				return loadErr
			}
		} else {
			if err := evaluation.ValidateBaseline(baseline, *datasetPath, *pipeline, *topK, *candidateK); err != nil {
				return err
			}
			regressions := evaluation.CompareBaseline(comparison.Reports, baseline.Reports, *tolerance)
			comparison.Regressions = regressions
			comparison.Baseline = &evaluation.BaselineInfo{Path: *baselinePath, Regressions: len(regressions)}
			regressionFailure = *failOnRegression && len(regressions) > 0 && !*updateBaseline
		}
		if *updateBaseline {
			current := evaluation.BaselineFile{Version: 1, Dataset: *datasetPath, Pipeline: *pipeline, TopK: *topK, CandidateK: *candidateK, Reports: evaluation.CompactReports(comparison.Reports)}
			if baselineLoaded {
				current = evaluation.MergeBaseline(baseline, current)
			}
			if err := evaluation.SaveBaseline(*baselinePath, current); err != nil {
				return err
			}
			if comparison.Baseline == nil {
				comparison.Baseline = &evaluation.BaselineInfo{Path: *baselinePath}
			}
			comparison.Baseline.Updated = true
		}
		if regressionFailure {
			encoder := json.NewEncoder(os.Stdout)
			encoder.SetIndent("", "  ")
			if err := encoder.Encode(comparison); err != nil {
				return err
			}
			return fmt.Errorf("evaluation regressed against baseline: %d metric(s)", len(comparison.Regressions))
		}
	}
	encoder := json.NewEncoder(os.Stdout)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(comparison); err != nil {
		return err
	}
	return nil
}

func evaluationModes(value string) ([]store.SearchMode, error) {
	switch value {
	case "all":
		return []store.SearchMode{store.SearchModeVector, store.SearchModeLexical, store.SearchModeHybrid, store.SearchModeRerank}, nil
	case string(store.SearchModeVector):
		return []store.SearchMode{store.SearchModeVector}, nil
	case string(store.SearchModeLexical):
		return []store.SearchMode{store.SearchModeLexical}, nil
	case string(store.SearchModeWeighted):
		return []store.SearchMode{store.SearchModeWeighted}, nil
	case string(store.SearchModeHybrid):
		return []store.SearchMode{store.SearchModeHybrid}, nil
	case string(store.SearchModeRerank):
		return []store.SearchMode{store.SearchModeRerank}, nil
	default:
		return nil, fmt.Errorf("unsupported evaluation mode %q", value)
	}
}

func printCommandError(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
