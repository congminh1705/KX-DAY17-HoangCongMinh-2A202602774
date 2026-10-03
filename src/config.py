from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from model_provider import ProviderConfig, normalize_provider

@dataclass
class LabConfig:
    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig
    live: bool = False
    profile_confidence_threshold: float = 0.85

def load_config(base_dir: Path | None = None) -> LabConfig:
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    try:
        from dotenv import load_dotenv
        load_dotenv(root / '.env', override=False)
    except ImportError:
        pass
    def provider_config(prefix, fallback=None):
        provider = normalize_provider(os.getenv(prefix + '_PROVIDER', fallback.provider if fallback else 'openai'))
        defaults = {'openai': 'gpt-4o-mini', 'custom': 'local-model', 'gemini': 'gemini-2.5-flash', 'anthropic': 'claude-sonnet-4-5', 'ollama': 'llama3.2', 'openrouter': 'openai/gpt-4o-mini'}
        key = os.getenv(prefix + '_API_KEY') or os.getenv(provider.upper() + '_API_KEY')
        if provider == 'gemini':
            key = key or os.getenv('GOOGLE_API_KEY')
        url = os.getenv(prefix + '_BASE_URL') or os.getenv(provider.upper() + '_BASE_URL')
        return ProviderConfig(provider, os.getenv(prefix + '_MODEL', fallback.model_name if fallback and fallback.provider == provider else defaults[provider]), float(os.getenv(prefix + '_TEMPERATURE', '0')), key, url)
    threshold = int(os.getenv('COMPACT_THRESHOLD_TOKENS', '1200'))
    keep = int(os.getenv('COMPACT_KEEP_MESSAGES', '4'))
    confidence = float(os.getenv('PROFILE_CONFIDENCE_THRESHOLD', '0.85'))
    if threshold <= 0 or keep < 1 or not 0 <= confidence <= 1:
        raise ValueError('Invalid compact or confidence settings')
    state = Path(os.getenv('STATE_DIR', str(root / 'state')))
    if not state.is_absolute():
        state = root / state
    state.mkdir(parents=True, exist_ok=True)
    model = provider_config('LLM')
    return LabConfig(root, root / 'data', state, threshold, keep, model, provider_config('JUDGE', model), os.getenv('LAB_LIVE', '0').lower() in {'1', 'true'}, confidence)
