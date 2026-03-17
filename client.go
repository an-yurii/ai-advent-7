package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"

	"github.com/google/uuid"
)

// LLMClient represents a client for an LLM API.
type LLMClient struct {
	APIKey     string
	BaseURL    string
	Model      string
	HTTPClient *http.Client
}

// NewLLMClient creates a new client with default values.
func NewLLMClient(apiKey, baseURL, model string) *LLMClient {
	if baseURL == "" {
		baseURL = "https://api.openai.com/v1/chat/completions"
	}
	if model == "" {
		model = "gpt-3.5-turbo"
	}
	return &LLMClient{
		APIKey:     apiKey,
		BaseURL:    baseURL,
		Model:      model,
		HTTPClient: &http.Client{},
	}
}

// ChatRequest represents the request payload.
type ChatRequest struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
}

// Message represents a single message in the conversation.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ChatResponse represents the response from the API.
type ChatResponse struct {
	ID      string   `json:"id"`
	Choices []Choice `json:"choices"`
	Error   *APIError `json:"error,omitempty"`
}

// Choice represents a choice in the response.
type Choice struct {
	Message Message `json:"message"`
}

// APIError represents an error from the API.
type APIError struct {
	Message string `json:"message"`
}

// SendRequest sends a chat request to the LLM API and returns the response.
func (c *LLMClient) SendRequest(prompt string) (string, error) {
	reqBody := ChatRequest{
		Model: c.Model,
		Messages: []Message{
			{Role: "user", Content: prompt},
		},
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequest("POST", c.BaseURL, bytes.NewBuffer(jsonData))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.APIKey)

	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		var apiErr APIError
		if err := json.Unmarshal(body, &apiErr); err != nil {
			return "", fmt.Errorf("API error (status %d): %s", resp.StatusCode, string(body))
		}
		return "", fmt.Errorf("API error: %s", apiErr.Message)
	}

	var chatResp ChatResponse
	if err := json.Unmarshal(body, &chatResp); err != nil {
		return "", fmt.Errorf("failed to unmarshal response: %w", err)
	}

	if len(chatResp.Choices) == 0 {
		return "", fmt.Errorf("no choices in response")
	}

	return chatResp.Choices[0].Message.Content, nil
}

// GetAPIKeyFromEnv retrieves the API key from environment variable.
// It tries OPENAI_API_KEY, then GIGACHAT_API_KEY.
// Deprecated: use getAPIKeyFromEnv in main.go instead.
func GetAPIKeyFromEnv() string {
	if key := os.Getenv("OPENAI_API_KEY"); key != "" {
		return key
	}
	if key := os.Getenv("GIGACHAT_API_KEY"); key != "" {
		return key
	}
	return ""
}

// GigaChatTokenResponse represents the response from GigaChat OAuth endpoint.
type GigaChatTokenResponse struct {
	AccessToken string `json:"access_token"`
	ExpiresIn   int    `json:"expires_in"`
	TokenType   string `json:"token_type"`
}

// GetGigaChatAccessToken retrieves an access token using client credentials.
func GetGigaChatAccessToken(clientID, clientSecret, tokenURL string) (string, error) {
	if tokenURL == "" {
		tokenURL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
	}
	// Prepare form data
	form := url.Values{}
	form.Set("scope", "GIGACHAT_API_PERS")
	reqBody := form.Encode()

	req, err := http.NewRequest("POST", tokenURL, bytes.NewBufferString(reqBody))
	if err != nil {
		return "", fmt.Errorf("failed to create token request: %w", err)
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("Accept", "application/json")
	// Basic Auth
	req.Header.Set("Authorization", "Bearer "+clientSecret)
	// RqUID header (UUID v4)
	rqUID := uuid.New().String()
	req.Header.Set("RqUID", rqUID)

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to send token request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read token response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		// Include response body in error for debugging
		errMsg := fmt.Sprintf("token request failed (status %d)", resp.StatusCode)
		if len(body) > 0 {
			errMsg += fmt.Sprintf(": %s", string(body))
		} else {
			errMsg += " (empty response body)"
		}
		// Also log to stderr for immediate visibility
		fmt.Fprintf(os.Stderr, "Error details: URL=%s, client_id=%s, RqUID=%s\n", tokenURL, clientID, rqUID)
		return "", fmt.Errorf(errMsg)
	}

	var tokenResp GigaChatTokenResponse
	if err := json.Unmarshal(body, &tokenResp); err != nil {
		return "", fmt.Errorf("failed to unmarshal token response: %w", err)
	}

	if tokenResp.AccessToken == "" {
		return "", fmt.Errorf("empty access token in response")
	}
	return tokenResp.AccessToken, nil
}