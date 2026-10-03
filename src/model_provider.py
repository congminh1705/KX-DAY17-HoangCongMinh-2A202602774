from __future__ import annotations
from dataclasses import dataclass, field
from importlib import import_module

@dataclass
class ProviderConfig:
    provider: str = 'openai'
    model_name: str = 'gpt-4o-mini'
    temperature: float = 0.0
    api_key: str | None = field(default=None, repr=False)
    base_url: str | None = None

def normalize_provider(value: str) -> str:
    value = value.strip().lower()
    value = {'anthorpic': 'anthropic', 'google': 'gemini', 'openai-compatible': 'custom'}.get(value, value)
    if value not in {'openai', 'custom', 'gemini', 'anthropic', 'ollama', 'openrouter'}:
        raise ValueError(f'Unsupported provider: {value}')
    return value

def build_chat_model(config: ProviderConfig):
    provider = normalize_provider(config.provider)
    modules = {'openai': ('langchain_openai', 'ChatOpenAI'), 'custom': ('langchain_openai', 'ChatOpenAI'), 'gemini': ('langchain_google_genai', 'ChatGoogleGenerativeAI'), 'anthropic': ('langchain_anthropic', 'ChatAnthropic'), 'ollama': ('langchain_ollama', 'ChatOllama'), 'openrouter': ('langchain_openrouter', 'ChatOpenRouter')}
    module, name = modules[provider]
    kwargs = {'model': config.model_name, 'temperature': config.temperature}
    if config.api_key and provider != 'ollama':
        kwargs['api_key'] = config.api_key
    if config.base_url:
        kwargs['base_url'] = config.base_url
    if provider == 'custom' and not config.base_url:
        raise ValueError('CUSTOM_BASE_URL is required')
    return getattr(import_module(module), name)(**kwargs)
