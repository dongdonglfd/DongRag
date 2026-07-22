package provider

import "fmt"

// HTTPError preserves an upstream status code so callers can distinguish
// transient provider failures from permanent configuration or request errors.
type HTTPError struct {
	Service    string
	Status     string
	StatusCode int
	Body       string
}

func (e *HTTPError) Error() string {
	return fmt.Sprintf("%s: %s: %s", e.Service, e.Status, e.Body)
}

func (e *HTTPError) Retryable() bool {
	return e.StatusCode == 408 || e.StatusCode == 429 || e.StatusCode >= 500
}
