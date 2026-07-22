package provider

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

type OllamaEmbedder struct {
	BaseURL string
	Model   string
	Client  *http.Client
}

func (o OllamaEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	body, err := json.Marshal(map[string]any{"model": o.Model, "input": texts})
	if err != nil {
		return nil, err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(o.BaseURL, "/")+"/api/embed", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	request.Header.Set("Content-Type", "application/json")
	response, err := o.client().Do(request)
	if err != nil {
		return nil, fmt.Errorf("ollama embed request: %w", err)
	}
	defer response.Body.Close()
	responseBody, _ := io.ReadAll(io.LimitReader(response.Body, 4<<20))
	if response.StatusCode/100 != 2 {
		return nil, &HTTPError{Service: "ollama embed", Status: response.Status, StatusCode: response.StatusCode, Body: strings.TrimSpace(string(responseBody))}
	}
	var result struct {
		Embeddings [][]float32 `json:"embeddings"`
	}
	if err := json.Unmarshal(responseBody, &result); err != nil {
		return nil, fmt.Errorf("decode ollama embedding response: %w", err)
	}
	if len(result.Embeddings) != len(texts) {
		return nil, fmt.Errorf("ollama returned %d embeddings for %d inputs", len(result.Embeddings), len(texts))
	}
	return result.Embeddings, nil
}

func (o OllamaEmbedder) Health(ctx context.Context) error {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimRight(o.BaseURL, "/")+"/api/tags", nil)
	if err != nil {
		return err
	}
	response, err := o.client().Do(request)
	if err != nil {
		return fmt.Errorf("ollama health request: %w", err)
	}
	defer response.Body.Close()
	body, _ := io.ReadAll(io.LimitReader(response.Body, 1<<20))
	if response.StatusCode/100 != 2 {
		return &HTTPError{Service: "ollama health", Status: response.Status, StatusCode: response.StatusCode, Body: strings.TrimSpace(string(body))}
	}
	var result struct {
		Models []struct {
			Name  string `json:"name"`
			Model string `json:"model"`
		} `json:"models"`
	}
	if err := json.Unmarshal(body, &result); err != nil {
		return fmt.Errorf("decode ollama health response: %w", err)
	}
	for _, model := range result.Models {
		if model.Name == o.Model || model.Model == o.Model || strings.HasPrefix(model.Name, o.Model+":") {
			return nil
		}
	}
	return fmt.Errorf("ollama embedding model %q is not installed", o.Model)
}

func (o OllamaEmbedder) client() *http.Client {
	if o.Client != nil {
		return o.Client
	}
	return http.DefaultClient
}
