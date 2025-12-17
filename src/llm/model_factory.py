from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    temperature: float
    site_url: Optional[str]
    app_name: Optional[str]


def load_openrouter_config() -> LLMConfig:
    load_dotenv()
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip()
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini").strip()
    temperature = float(os.getenv("OPENROUTER_TEMPERATURE", "0").strip() or "0")
    site_url = os.getenv("OPENROUTER_SITE_URL", "").strip() or None
    app_name = os.getenv("OPENROUTER_APP_NAME", "").strip() or None
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY fehlt in .env")
    return LLMConfig(
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=temperature,
        site_url=site_url,
        app_name=app_name,
    )


def build_openrouter_client(cfg: LLMConfig) -> OpenAI:
    # OpenRouter ist OpenAI-kompatibel. Zusätzliche Header sind optional.
    default_headers = {}
    if cfg.site_url:
        default_headers["HTTP-Referer"] = cfg.site_url
    if cfg.app_name:
        default_headers["X-Title"] = cfg.app_name
    return OpenAI(
        base_url=cfg.base_url,
        api_key=cfg.api_key,
        default_headers=default_headers or None,
    )
