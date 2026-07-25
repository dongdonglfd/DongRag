package provider

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

type BailianChat struct {
	APIKey      string
	BaseURL     string
	Model       string
	Temperature float64
	Client      *http.Client
}

// Stream sends an OpenAI-compatible streaming completion and invokes onChunk
// for each non-empty text delta. The request context owns the upstream HTTP
// connection, so client cancellation stops model generation as well.
func (b BailianChat) Stream(ctx context.Context, systemPrompt, userPrompt string, onChunk func(string) error) error {
	if onChunk == nil {
		return fmt.Errorf("bailian stream callback is required")
	}
	body, err := json.Marshal(map[string]any{
		"model": b.Model,
		"messages": []map[string]string{
			{"role": "system", "content": systemPrompt},
			{"role": "user", "content": userPrompt},
		},
		"temperature": b.Temperature,
		"stream":      true,
	})
	if err != nil {
		return err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(b.BaseURL, "/")+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return err
	}
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Authorization", "Bearer "+b.APIKey)
	response, err := b.client().Do(request)
	if err != nil {
		return fmt.Errorf("bailian stream request: %w", err)
	}
	defer response.Body.Close()
	if response.StatusCode/100 != 2 {
		responseBody, _ := io.ReadAll(io.LimitReader(response.Body, 1<<20))
		return fmt.Errorf("bailian stream: %s: %s", response.Status, strings.TrimSpace(string(responseBody)))
	}

	scanner := bufio.NewScanner(response.Body)
	scanner.Buffer(make([]byte, 4<<10), 1<<20)
	emitted := false
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "[DONE]" {
			if !emitted {
				return fmt.Errorf("bailian returned no answer")
			}
			return nil
		}
		var chunk struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			return fmt.Errorf("decode bailian stream response: %w", err)
		}
		for _, choice := range chunk.Choices {
			if choice.Delta.Content == "" {
				continue
			}
			if err := onChunk(choice.Delta.Content); err != nil {
				return err
			}
			emitted = true
		}
	}
	if err := scanner.Err(); err != nil {
		return fmt.Errorf("read bailian stream: %w", err)
	}
	if !emitted {
		return fmt.Errorf("bailian returned no answer")
	}
	return nil
}

func (b BailianChat) Complete(ctx context.Context, systemPrompt, userPrompt string) (string, error) {
	body, err := json.Marshal(map[string]any{
		"model": b.Model,
		"messages": []map[string]string{
			{"role": "system", "content": systemPrompt},
			{"role": "user", "content": userPrompt},
		},
		"temperature": b.Temperature,
	})
	if err != nil {
		return "", err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(b.BaseURL, "/")+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	request.Header.Set("Content-Type", "application/json")
	request.Header.Set("Authorization", "Bearer "+b.APIKey)
	response, err := b.client().Do(request)
	if err != nil {
		return "", fmt.Errorf("bailian chat request: %w", err)
	}
	defer response.Body.Close()
	responseBody, _ := io.ReadAll(io.LimitReader(response.Body, 8<<20))
	if response.StatusCode/100 != 2 {
		return "", fmt.Errorf("bailian chat: %s: %s", response.Status, strings.TrimSpace(string(responseBody)))
	}
	var result struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(responseBody, &result); err != nil {
		return "", fmt.Errorf("decode bailian response: %w", err)
	}
	if len(result.Choices) == 0 || strings.TrimSpace(result.Choices[0].Message.Content) == "" {
		return "", fmt.Errorf("bailian returned no answer")
	}
	return strings.TrimSpace(result.Choices[0].Message.Content), nil
}

func (b BailianChat) Health(ctx context.Context) error {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimRight(b.BaseURL, "/")+"/models", nil)
	if err != nil {
		return err
	}
	request.Header.Set("Authorization", "Bearer "+b.APIKey)
	response, err := b.client().Do(request)
	if err != nil {
		return fmt.Errorf("bailian health request: %w", err)
	}
	defer response.Body.Close()
	body, _ := io.ReadAll(io.LimitReader(response.Body, 1<<20))
	if response.StatusCode/100 != 2 {
		return fmt.Errorf("bailian health: %s: %s", response.Status, strings.TrimSpace(string(body)))
	}
	return nil
}

func (b BailianChat) client() *http.Client {
	if b.Client != nil {
		return b.Client
	}
	return http.DefaultClient
}
