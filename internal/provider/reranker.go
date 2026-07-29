package provider

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
)

type RerankResult struct {
	Index int     `json:"index"`
	Score float64 `json:"score"`
}

type LlamaCppReranker struct {
	BaseURL string
	Client  *http.Client
}

func (r LlamaCppReranker) Rerank(ctx context.Context, query string, texts []string) ([]RerankResult, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	body, err := json.Marshal(map[string]any{"query": query, "documents": texts, "top_n": len(texts)})
	if err != nil {
		return nil, err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(r.BaseURL, "/")+"/v1/rerank", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	request.Header.Set("Content-Type", "application/json")
	response, err := r.client().Do(request)
	if err != nil {
		return nil, fmt.Errorf("reranker request: %w", err)
	}
	defer response.Body.Close()
	responseBody, _ := io.ReadAll(io.LimitReader(response.Body, 4<<20))
	if response.StatusCode/100 != 2 {
		return nil, fmt.Errorf("reranker: %s: %s", response.Status, strings.TrimSpace(string(responseBody)))
	}
	var responseResult struct {
		Results []struct {
			Index          int     `json:"index"`
			RelevanceScore float64 `json:"relevance_score"`
		} `json:"results"`
	}
	if err := json.Unmarshal(responseBody, &responseResult); err != nil {
		return nil, fmt.Errorf("decode reranker response: %w", err)
	}
	responseItems := responseResult.Results
	if len(responseItems) != len(texts) {
		return nil, fmt.Errorf("reranker returned %d results for %d documents", len(responseItems), len(texts))
	}
	result := make([]RerankResult, len(responseItems))
	for i, item := range responseItems {
		if item.Index < 0 || item.Index >= len(texts) {
			return nil, fmt.Errorf("reranker returned invalid index %d", item.Index)
		}
		result[i] = RerankResult{Index: item.Index, Score: item.RelevanceScore}
	}
	sort.SliceStable(result, func(i, j int) bool { return result[i].Score > result[j].Score })
	return result, nil
}

func (r LlamaCppReranker) Health(ctx context.Context) error {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimRight(r.BaseURL, "/")+"/health", nil)
	if err != nil {
		return err
	}
	response, err := r.client().Do(request)
	if err != nil {
		return fmt.Errorf("reranker health request: %w", err)
	}
	defer response.Body.Close()
	body, _ := io.ReadAll(io.LimitReader(response.Body, 1<<20))
	if response.StatusCode/100 != 2 {
		return fmt.Errorf("reranker health: %s: %s", response.Status, strings.TrimSpace(string(body)))
	}
	return nil
}

func (r LlamaCppReranker) client() *http.Client {
	if r.Client != nil {
		return r.Client
	}
	return http.DefaultClient
}
