// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package deepseek

import (
	"context"
	"encoding/json"
	"os"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

const (
	provider = "deepseek"
	baseURL  = "https://api.deepseek.com"
)

// Supported models: https://api-docs.deepseek.com/quick_start/pricing
//
// deepseek-chat and deepseek-reasoner are deprecated aliases for
// deepseek-v4-flash (non-thinking and thinking modes respectively).
var supportedModels = map[string]ai.ModelOptions{
	"deepseek-v4-flash": {
		Label: "DeepSeek-V4-Flash",
		Supports: &ai.ModelSupports{
			Multiturn:  true,
			Tools:      true,
			ToolChoice: true,
			SystemRole: true,
			Media:      false,
		},
		Versions: []string{"deepseek-v4-flash", "deepseek-chat"},
	},
	"deepseek-v4-pro": {
		Label: "DeepSeek-V4-Pro",
		Supports: &ai.ModelSupports{
			Multiturn:  true,
			Tools:      true,
			ToolChoice: true,
			SystemRole: true,
			Media:      false,
		},
		Versions: []string{"deepseek-v4-pro"},
	},
}

// DeepSeek is a plugin that provides access to DeepSeek models via their OpenAI-compatible API.
// It wraps the compat_oai.OpenAICompatible base plugin.
type DeepSeek struct {
	// APIKey is the API key for the DeepSeek API.
	// If empty, the value of the environment variable "DEEPSEEK_API_KEY" will be consulted.
	// Request a key at https://platform.deepseek.com/api_keys
	APIKey string

	// Opts are additional options for the OpenAI client.
	// Can include other options like WithBaseURL, etc.
	Opts []option.RequestOption

	openAICompatible *compat_oai.OpenAICompatible
}

// Name implements genkit.Plugin.
func (d *DeepSeek) Name() string {
	return provider
}

// Init implements genkit.Plugin.
func (d *DeepSeek) Init(ctx context.Context) []api.Action {
	apiKey := d.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("DEEPSEEK_API_KEY")
	}
	if apiKey == "" {
		panic("deepseek plugin initialization failed: DEEPSEEK_API_KEY is required")
	}

	if d.openAICompatible == nil {
		d.openAICompatible = &compat_oai.OpenAICompatible{}
	}

	// Set the options with DeepSeek base URL and API key
	d.openAICompatible.Opts = []option.RequestOption{
		option.WithBaseURL(baseURL),
		option.WithAPIKey(apiKey),
	}
	if len(d.Opts) > 0 {
		d.openAICompatible.Opts = append(d.openAICompatible.Opts, d.Opts...)
	}

	d.openAICompatible.Provider = provider

	// DeepSeek-specific hooks for reasoning_content handling.
	// DeepSeek requires reasoning_content to be passed back in multi-round
	// conversations when the model performed a tool call in the previous turn.
	// Reference: https://api-docs.deepseek.com/guides/reasoning_model
	d.openAICompatible.OnAssistantMessage = deepseekOnAssistantMessage
	d.openAICompatible.OnStreamChunk = deepseekOnStreamChunk
	d.openAICompatible.OnResponse = deepseekOnResponse

	compatActions := d.openAICompatible.Init(ctx)

	var actions []api.Action
	actions = append(actions, compatActions...)

	// Define default models
	for model, opts := range supportedModels {
		actions = append(actions, d.DefineModel(model, opts).(api.Action))
	}

	return actions
}

// Model returns the ai.Model with the given name.
func (d *DeepSeek) Model(g *genkit.Genkit, name string) ai.Model {
	return d.openAICompatible.Model(g, api.NewName(provider, name))
}

// DefineModel defines a model in the registry.
func (d *DeepSeek) DefineModel(id string, opts ai.ModelOptions) ai.Model {
	return d.openAICompatible.DefineModel(provider, id, opts)
}

// ListActions implements genkit.Plugin.
func (d *DeepSeek) ListActions(ctx context.Context) []api.ActionDesc {
	return d.openAICompatible.ListActions(ctx)
}

// ResolveAction implements genkit.Plugin.
func (d *DeepSeek) ResolveAction(atype api.ActionType, name string) api.Action {
	return d.openAICompatible.ResolveAction(atype, name)
}

// DeepSeek-specific hooks for reasoning_content handling.
// DeepSeek uses reasoning_content as a non-standard field in the OpenAI-compatible
// API to support thinking mode (chain-of-thought reasoning).
//
// In multi-round conversations, reasoning_content must be passed back to the API
// when the model performed a tool call in the previous turn. These hooks handle:
//   1. Sending: injecting reasoning_content into assistant messages
//   2. Receiving (stream): extracting reasoning_content from streaming deltas
//   3. Receiving (complete): extracting reasoning_content from final responses

// deepseekOnAssistantMessage injects reasoning_content into assistant messages
// for multi-round conversations. This is required by DeepSeek when the model
// performed a tool call in the previous turn.
func deepseekOnAssistantMessage(msg *ai.Message, am *openai.ChatCompletionAssistantMessageParam) {
	reasoning := extractReasoningFromParts(msg.Content)
	if reasoning != "" {
		am.SetExtraFields(map[string]any{"reasoning_content": reasoning})
	}
}

// deepseekOnStreamChunk extracts reasoning_content from streaming chunk deltas.
// DeepSeek delivers reasoning content incrementally via the reasoning_content
// extra field on the delta object during streaming.
func deepseekOnStreamChunk(ctx context.Context, chunk *openai.ChatCompletionChunk) []*ai.Part {
	if len(chunk.Choices) == 0 {
		return nil
	}
	delta := chunk.Choices[0].Delta
	if field, ok := delta.JSON.ExtraFields["reasoning_content"]; ok && field.Valid() {
		if reasoning := unquoteJSONString(field.Raw()); reasoning != "" {
			return []*ai.Part{ai.NewReasoningPart(reasoning, nil)}
		}
	}
	return nil
}

// deepseekOnResponse extracts reasoning_content from the complete ChatCompletion response.
// DeepSeek returns reasoning_content as an extra field on the message object
// alongside the normal content field.
func deepseekOnResponse(ctx context.Context, completion *openai.ChatCompletion) []*ai.Part {
	if len(completion.Choices) == 0 {
		return nil
	}
	msg := completion.Choices[0].Message
	if field, ok := msg.JSON.ExtraFields["reasoning_content"]; ok && field.Valid() {
		if reasoning := unquoteJSONString(field.Raw()); reasoning != "" {
			return []*ai.Part{ai.NewReasoningPart(reasoning, nil)}
		}
	}
	return nil
}

// extractReasoningFromParts extracts reasoning text from ai.PartReasoning parts.
func extractReasoningFromParts(parts []*ai.Part) string {
	var reasoning string
	for _, part := range parts {
		if part.IsReasoning() {
			reasoning += part.Text
		}
	}
	return reasoning
}

// unquoteJSONString removes surrounding JSON quotes from a raw JSON string value.
func unquoteJSONString(raw string) string {
	var s string
	if err := json.Unmarshal([]byte(raw), &s); err != nil {
		return raw
	}
	return s
}
