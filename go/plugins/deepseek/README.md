# DeepSeek Plugin

This plugin provides access to [DeepSeek](https://deepseek.com) models via their OpenAI-compatible API.

## Supported Models

| Model ID             | Label           | Description                                                        |
|----------------------|-----------------|--------------------------------------------------------------------|
| `deepseek-v4-flash`  | DeepSeek-V4-Flash | Fast model, 1M context, thinking mode (default), tool calling     |
| `deepseek-v4-pro`    | DeepSeek-V4-Pro   | Pro model, 1M context, thinking mode (default), tool calling      |

> **Deprecated aliases:** `deepseek-chat` → `deepseek-v4-flash` (non-thinking), `deepseek-reasoner` → `deepseek-v4-flash` (thinking).
> These aliases will be removed in a future version.

## Setup

1. Get an API key from [DeepSeek Platform](https://platform.deepseek.com/api_keys)
2. Set the environment variable:

```bash
export DEEPSEEK_API_KEY=sk-your-api-key
```

## Usage

```go
package main

import (
	"context"
	"log"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/deepseek"
)

func main() {
	ctx := context.Background()

	g, err := genkit.New(ctx, genkit.WithPlugins(&deepseek.DeepSeek{}))
	if err != nil {
		log.Fatal(err)
	}

	// Use DeepSeek-V4-Flash for general chat
	resp, err := genkit.Generate(ctx, g,
		ai.WithModel("deepseek/deepseek-v4-flash"),
		ai.WithSimplePrompt("Hello!"),
	)
	if err != nil {
		log.Fatal(err)
	}
	log.Println(resp.Message.Content[0].Text)
}
```

## Thinking Mode

Both DeepSeek-V4-Flash and DeepSeek-V4-Pro support thinking mode (enabled by default).
The thinking content is automatically extracted and available as reasoning parts:

```go
resp, err := genkit.Generate(ctx, g,
	ai.WithModel("deepseek/deepseek-v4-pro"),
	ai.WithSimplePrompt("Solve this equation: 2x + 5 = 13"),
)
if err != nil {
	log.Fatal(err)
}

// The response includes both reasoning and answer content
for _, part := range resp.Message.Content {
	if part.IsReasoning() {
		log.Printf("Thinking: %s", part.Text)
	} else if part.IsText() {
		log.Printf("Answer: %s", part.Text)
	}
}
```

## Tool Calling

Both models support tool calling via the OpenAI-compatible API:

```go
type WeatherInput struct {
	City string `json:"city"`
}

weatherTool := ai.DefineTool("getWeather", "Get the current weather", func(ctx context.Context, input WeatherInput) (string, error) {
	return "Sunny, 22°C in " + input.City, nil
})

resp, err := genkit.Generate(ctx, g,
	ai.WithModel("deepseek/deepseek-v4-flash"),
	ai.WithSimplePrompt("What's the weather in Tokyo?"),
	ai.WithTools(weatherTool),
)
```

## Streaming

```go
resp, err := genkit.Generate(ctx, g,
	ai.WithModel("deepseek/deepseek-v4-flash"),
	ai.WithSimplePrompt("Write a short poem about coding"),
	ai.WithStreaming(func(ctx context.Context, chunk *ai.ModelResponseChunk) error {
		for _, part := range chunk.Content {
			if part.IsText() {
				fmt.Print(part.Text)
			}
		}
		return nil
	}),
)
```

## Running Tests

Set your API key and run tests:

```bash
export DEEPSEEK_API_KEY=sk-your-api-key
go test -v ./...
```

Note: Tests will be skipped if `DEEPSEEK_API_KEY` is not set.
