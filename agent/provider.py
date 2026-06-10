import os
from typing import Literal

from anthropic import AsyncAnthropic
from pydantic import BaseModel

ProviderName = Literal["anthropic", "deepseek"]

DEFAULT_MODELS: dict[ProviderName, str] = {
    "anthropic": "claude-haiku-4-5",
    "deepseek": "deepseek-v4-flash",
}

DEFAULT_BASE_URLS: dict[ProviderName, str | None] = {
    "anthropic": None,
    "deepseek": "https://api.deepseek.com/anthropic",
}


class ProviderConfig(BaseModel):
    provider: ProviderName
    model: str
    api_key: str
    base_url: str | None = None


def load_provider_config(
    provider: str | None = None,
    model: str | None = None,
) -> ProviderConfig:
    provider_name = provider or os.getenv("AGENT_PROVIDER", "anthropic")
    if provider_name not in DEFAULT_MODELS:
        raise ValueError(
            f"Unknown provider: {provider_name}. "
            f"Available: {list(DEFAULT_MODELS)}"
        )

    typed_provider: ProviderName = provider_name
    prefix = typed_provider.upper()
    api_key = os.getenv(f"{prefix}_API_KEY", "")
    if not api_key:
        raise ValueError(f"{prefix}_API_KEY is not set")

    configured_model = (
        model
        or os.getenv(f"{prefix}_MODEL")
        or DEFAULT_MODELS[typed_provider]
    )
    base_url = (
        os.getenv(f"{prefix}_BASE_URL")
        or DEFAULT_BASE_URLS[typed_provider]
    )
    return ProviderConfig(
        provider=typed_provider,
        model=configured_model,
        api_key=api_key,
        base_url=base_url,
    )


def create_client(config: ProviderConfig) -> AsyncAnthropic:
    return AsyncAnthropic(
        api_key=config.api_key,
        base_url=config.base_url,
    )
