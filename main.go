package main

import (
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"strings"
	"time"
)

func main() {
	// Common flags
	prompt := flag.String("p", "", "Prompt to send to LLM")
	model := flag.String("m", "", "Model to use (default: gpt-3.5-turbo for OpenAI, GigaChat for GigaChat). Supported GigaChat models: GigaChat, GigaChat‑Pro, GigaChat‑Max (check documentation)")
	apiKey := flag.String("k", "", "API key (default from environment)")
	baseURL := flag.String("u", "", "API base URL (default based on API)")
	apiProvider := flag.String("a", "openai", "API provider: openai, gigachat")
	listModels := flag.Bool("list-models", false, "List known models for each provider and exit")

	// Formatting flags
	systemPrompt := flag.String("system", "", "System prompt to set behavior/format")
	systemFile := flag.String("system-file", "", "Path to a file containing system prompt (overrides -system)")
	responseFormat := flag.String("format", "", "Response format: json, text (default)")
	temperature := flag.Float64("temp", 0.0, "Temperature for generation (0.0 to 2.0, 0.0 means default)")
	maxTokens := flag.Int("max-tokens", 0, "Maximum tokens in response (0 means default)")
	topP := flag.Float64("top-p", 0.0, "Top-p sampling (0.0 to 1.0, 0.0 means default)")
	numChoices := flag.Int("n", 1, "Number of response choices (default 1)")
	repetitionPenalty := flag.Float64("repetition-penalty", 0.0, "Repetition penalty (1.0 means no penalty, >1.0 reduces repetition)")
	stopSequences := flag.String("stop", "", "Stop sequences (comma-separated, e.g. 'END,STOP')")

	// Token retrieval flags
	getToken := flag.Bool("get-token", false, "Retrieve access token for GigaChat (requires client-id and client-secret)")
	clientID := flag.String("client-id", "", "Client ID for OAuth2 token retrieval")
	clientSecret := flag.String("client-secret", "", "Client secret for OAuth2 token retrieval")
	tokenURL := flag.String("token-url", "https://ngw.devices.sberbank.ru:9443/api/v2/oauth", "OAuth2 token endpoint URL")

	flag.Parse()

	// If list-models mode, print known models and exit
	if *listModels {
		printKnownModels()
		return
	}

	// If get-token mode, retrieve token and exit
	if *getToken {
		if *clientID == "" || *clientSecret == "" {
			log.Fatal("Client ID and client secret are required for token retrieval. Use -client-id and -client-secret.")
		}
		token, err := GetGigaChatAccessToken(*clientID, *clientSecret, *tokenURL)
		if err != nil {
			log.Fatalf("Failed to retrieve token: %v", err)
		}
		fmt.Println(token)
		return
	}

	// Normal chat mode
	// Set defaults based on provider
	if *baseURL == "" {
		switch *apiProvider {
		case "openai":
			*baseURL = "https://api.openai.com/v1/chat/completions"
		case "gigachat":
			*baseURL = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
		default:
			log.Fatalf("Unknown API provider: %s", *apiProvider)
		}
	}
	if *model == "" {
		switch *apiProvider {
		case "openai":
			*model = "gpt-3.5-turbo"
		case "gigachat":
			*model = "GigaChat"
		}
	}

	if *prompt == "" {
		// If no flag, try to read from stdin
		stat, _ := os.Stdin.Stat()
		if (stat.Mode() & os.ModeCharDevice) == 0 {
			// stdin is piped
			data, err := io.ReadAll(os.Stdin)
			if err == nil && len(data) > 0 {
				*prompt = string(data)
			}
		}
		if *prompt == "" {
			printUsage()
			os.Exit(1)
		}
	}

	key := *apiKey
	if key == "" {
		key = getAPIKeyFromEnv(*apiProvider)
		if key == "" {
			log.Fatal("API key not provided. Set appropriate environment variable or use -k flag.")
		}
	}

	client := NewLLMClient(key, *baseURL, *model)
	fmt.Printf("Sending request to %s with model %s...\n", *baseURL, *model)

	// Build request options
	var opts []RequestOption
	// Determine system prompt: file overrides direct string
	systemContent := ""
	if *systemFile != "" {
		content, err := readFileContent(*systemFile)
		if err != nil {
			log.Fatalf("Failed to read system prompt file: %v", err)
		}
		systemContent = content
	} else if *systemPrompt != "" {
		systemContent = *systemPrompt
	}
	if systemContent != "" {
		opts = append(opts, WithSystemMessage(systemContent))
	}
	if *responseFormat == "json" {
		opts = append(opts, WithJSONResponseFormat())
	}
	if *temperature > 0.0 {
		opts = append(opts, WithTemperature(*temperature))
	}
	if *maxTokens > 0 {
		opts = append(opts, WithMaxTokens(*maxTokens))
	}
	if *topP > 0.0 {
		opts = append(opts, WithTopP(*topP))
	}
	if *numChoices > 1 {
		opts = append(opts, WithNumChoices(*numChoices))
	}
	if *repetitionPenalty > 0.0 {
		opts = append(opts, WithRepetitionPenalty(*repetitionPenalty))
	}
	if *stopSequences != "" {
		// Split by comma and trim spaces
		parts := strings.Split(*stopSequences, ",")
		stops := make([]string, 0, len(parts))
		for _, p := range parts {
			trimmed := strings.TrimSpace(p)
			if trimmed != "" {
				stops = append(stops, trimmed)
			}
		}
		if len(stops) > 0 {
			opts = append(opts, WithStop(stops))
		}
	}

	start := time.Now()
	response, err := client.SendRequest(*prompt, opts...)
	duration := time.Since(start)
	if err != nil {
		log.Fatalf("Error: %v\n", err)
	}

	fmt.Printf("\n--- Response --- (%d мс)\n", duration.Milliseconds())
	fmt.Println(response)
}

func printUsage() {
	fmt.Println("Usage: llm-api-client -p \"Your prompt\"")
	fmt.Println("Or pipe prompt via stdin: echo \"Hello\" | llm-api-client")
	fmt.Println("\nOptions:")
	flag.PrintDefaults()
	fmt.Println("\nEnvironment variables:")
	fmt.Println("  OPENAI_API_KEY    API key for OpenAI")
	fmt.Println("  GIGACHAT_API_KEY  API key for GigaChat")
	fmt.Println("  GIGACHAT_CLIENT_ID     Client ID for GigaChat OAuth (optional)")
	fmt.Println("  GIGACHAT_CLIENT_SECRET Client secret for GigaChat OAuth (optional)")
	fmt.Println("\nToken retrieval:")
	fmt.Println("  llm-api-client -get-token -client-id <id> -client-secret <secret>")
}

// printKnownModels prints known models for each supported API provider.
func printKnownModels() {
	fmt.Println("Known models per provider:")
	fmt.Println("OpenAI:")
	fmt.Println("  gpt-3.5-turbo, gpt-4, gpt-4-turbo, gpt-4o, etc.")
	fmt.Println("GigaChat:")
	fmt.Println("  GigaChat (default), GigaChat‑Pro, GigaChat‑Max")
	fmt.Println("Note: Model availability may depend on your API access.")
}

// readFileContent reads the entire content of a file and returns it as a string.
func readFileContent(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("failed to read file %s: %w", path, err)
	}
	return string(data), nil
}

// getAPIKeyFromEnv returns API key based on provider.
func getAPIKeyFromEnv(provider string) string {
	switch strings.ToLower(provider) {
	case "openai":
		return os.Getenv("OPENAI_API_KEY")
	case "gigachat":
		// First try direct API key
		if key := os.Getenv("GIGACHAT_API_KEY"); key != "" {
			return key
		}
		// Otherwise try to get token via client credentials
		clientID := os.Getenv("GIGACHAT_CLIENT_ID")
		clientSecret := os.Getenv("GIGACHAT_CLIENT_SECRET")
		if clientID != "" && clientSecret != "" {
			token, err := GetGigaChatAccessToken(clientID, clientSecret, "")
			if err != nil {
				log.Printf("Warning: failed to obtain token: %v", err)
				return ""
			}
			return token
		}
		return ""
	default:
		return ""
	}
}
