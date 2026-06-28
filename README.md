# agent-from-scratch

A coding agent built from scratch that supports multiple LLM providers.

## Provider Configuration

Configure the provider and API keys via environment variables (copy `.env.example` to `.env`).

| Variable | Description |
|---|---|
| `AGENT_PROVIDER` | Provider to use: `anthropic`, `deepseek`, or `openai` |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `ANTHROPIC_MODEL` | Anthropic model (default: `claude-haiku-4-5`) |
| `DEEPSEEK_API_KEY` | DeepSeek API key |
| `DEEPSEEK_MODEL` | DeepSeek model (default: `deepseek-v4-flash`) |
| `DEEPSEEK_BASE_URL` | DeepSeek Chat Completions base URL (default: `https://api.deepseek.com`) |
| `OPENAI_API_KEY` | OpenAI API key |
| `OPENAI_MODEL` | OpenAI model (default: `gpt-4o-mini`) |
| `OPENAI_BASE_URL` | OpenAI API base URL (default: `https://api.openai.com/v1`) |

Set `AGENT_PROVIDER` to your chosen provider and populate the corresponding variables.
