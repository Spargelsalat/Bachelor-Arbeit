from __future__ import annotations
import argparse
import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from dotenv import load_dotenv
from openai import OpenAI
from config.settings import load_settings

logger = logging.getLogger(__name__)


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _clean_env(value: str | None) -> str:
    if value is None:
        return ""
    return value.split("#", 1)[0].strip()


def _to_int(value: str | None, default: int) -> int:
    v = _clean_env(value)
    if v == "":
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _to_bool(value: str | None, default: bool) -> bool:
    v = _clean_env(value).lower()
    if v == "":
        return default
    return v in {"1", "true", "yes", "y", "on"}


def _to_float(value: str | None, default: float) -> float:
    v = _clean_env(value)
    if v == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


def extract_first_json(text: str) -> Optional[Any]:
    s = (text or "").strip()
    if not s:
        return None
    import re

    m = re.search(r"(\{.*\}|\[.*\])", s, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


@dataclass(frozen=True)
class SchemaConfig:
    out_dir: Path
    alias_name: str | None
    backend: str
    temperature: float
    sample_docs: int
    max_chars_per_doc: int
    overwrite: bool
    model_override: str | None


def load_schema_config() -> SchemaConfig:
    load_dotenv()
    return SchemaConfig(
        out_dir=Path(_clean_env(os.getenv("SCHEMA_OUTPUT_DIR", "data/schemas"))),
        alias_name=(_clean_env(os.getenv("SCHEMA_OUTPUT_ALIAS")) or None),
        backend=_clean_env(os.getenv("SCHEMA_LLM_BACKEND", "openrouter")).lower(),
        temperature=_to_float(os.getenv("SCHEMA_TEMPERATURE"), 0.0),
        sample_docs=_to_int(os.getenv("SCHEMA_SAMPLE_DOCS"), 3),
        max_chars_per_doc=_to_int(os.getenv("SCHEMA_MAX_CHARS_PER_DOC"), 12000),
        overwrite=_to_bool(os.getenv("SCHEMA_OVERWRITE"), False),
        model_override=(_clean_env(os.getenv("SCHEMA_MODEL")) or None),
    )


@dataclass(frozen=True)
class LLMRuntime:
    backend: str
    model: str
    temperature: float
    client: OpenAI


def build_llm_runtime(
    backend: str, temperature: float, model_override: str | None
) -> LLMRuntime:
    if backend == "openrouter":
        api_key = _clean_env(os.getenv("OPENROUTER_API_KEY"))
        base_url = _clean_env(
            os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        )
        model = model_override or _clean_env(os.getenv("OPENROUTER_MODEL"))
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY fehlt.")
        if not model:
            raise ValueError("OPENROUTER_MODEL fehlt (oder SCHEMA_MODEL setzen).")
        site_url = _clean_env(os.getenv("OPENROUTER_SITE_URL"))
        app_name = _clean_env(os.getenv("OPENROUTER_APP_NAME"))
        headers: dict[str, str] = {}
        if site_url:
            headers["HTTP-Referer"] = site_url
        if app_name:
            headers["X-Title"] = app_name
        client = OpenAI(
            base_url=base_url, api_key=api_key, default_headers=headers or None
        )
        return LLMRuntime(
            backend=backend, model=model, temperature=temperature, client=client
        )
    if backend == "ollama":
        base_url = _clean_env(
            os.getenv("OLLAMA_OPENAI_BASE_URL", "http://localhost:11434/v1")
        )
        model = model_override or _clean_env(os.getenv("OLLAMA_MODEL"))
        if not model:
            raise ValueError("OLLAMA_MODEL fehlt (oder SCHEMA_MODEL setzen).")
        client = OpenAI(base_url=base_url, api_key="ollama")
        return LLMRuntime(
            backend=backend, model=model, temperature=temperature, client=client
        )
    raise ValueError(f"Unbekanntes SCHEMA_LLM_BACKEND: {backend}")


def load_sample_documents(
    processed_dir: Path, sample_docs: int, max_chars_per_doc: int
) -> list[dict[str, str]]:
    txt_files = sorted(processed_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"Keine .txt Dateien in: {processed_dir.resolve()}")
    selected = txt_files[: max(1, sample_docs)]
    docs: list[dict[str, str]] = []
    for p in selected:
        text = p.read_text(encoding="utf-8", errors="replace").strip()
        if max_chars_per_doc and len(text) > max_chars_per_doc:
            text = text[:max_chars_per_doc]
        docs.append({"title": p.stem, "text": text})
    return docs


def build_schema_prompt(docs: list[dict[str, str]]) -> str:
    return (
        "Analysiere die folgenden Textauszüge aus einer technischen Unternehmensdokumentation.\n"
        "Ziel: Definiere ein domänenspezifisches Wissensgraph-Schema für Informationsextraktion.\n\n"
        "Gib ausschließlich JSON im folgenden Format zurück:\n"
        "{\n"
        '  "domain": "kurzer Domänenname",\n'
        '  "entity_types": [\n'
        "    {\n"
        '      "name": "EntityTypeName",\n'
        '      "description": "kurze Beschreibung",\n'
        '      "attributes": ["attr1", "attr2"],\n'
        '      "examples": ["Beispiel", "Beispiel"]\n'
        "    }\n"
        "  ],\n"
        '  "relation_types": [\n'
        "    {\n"
        '      "name": "RELATION_NAME",\n'
        '      "description": "kurze Beschreibung",\n'
        '      "source_type": "EntityTypeName",\n'
        '      "target_type": "EntityTypeName",\n'
        '      "examples": ["Quelle --RELATION--> Ziel"]\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Regeln:\n"
        "- Nutze 8 bis 15 Entity Types.\n"
        "- Nutze 10 bis 25 Relation Types.\n"
        "- CamelCase für EntityTypes, UPPER_SNAKE_CASE für Relations.\n"
        "- Keine Texte außerhalb des JSON.\n\n"
        f"Dokumentauszüge:\n{json.dumps(docs, ensure_ascii=False, indent=2)}"
    )


def validate_schema(schema: dict[str, Any]) -> bool:
    if not isinstance(schema, dict):
        return False
    if "entity_types" not in schema or "relation_types" not in schema:
        return False
    if not isinstance(schema["entity_types"], list) or not isinstance(
        schema["relation_types"], list
    ):
        return False
    return True


def write_schema(
    out_dir: Path, schema: dict[str, Any], rt: LLMRuntime, overwrite: bool
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    domain = str(schema.get("domain", "schema")).strip() or "schema"
    safe_domain = "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in domain
    )
    out_path = out_dir / f"{safe_domain}.schema.json"
    if out_path.exists() and not overwrite:
        logger.info("Skip (schema exists): %s", out_path.name)
        return out_path
    final = {
        "_meta": {
            "llm_backend": rt.backend,
            "llm_model": rt.model,
            "llm_temperature": rt.temperature,
        },
        **schema,
    }
    out_path.write_text(
        json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Wrote schema: %s", out_path.name)
    return out_path


def write_alias(schema_path: Path, alias_name: str) -> None:
    alias_path = schema_path.parent / alias_name
    shutil.copyfile(schema_path, alias_path)
    logger.info("Wrote schema alias: %s -> %s", schema_path.name, alias_path.name)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate a domain schema (entity/relation types) using an LLM."
    )
    p.add_argument("--overwrite", action="store_true")
    return p


def run() -> None:
    settings = load_settings()
    cfg = load_schema_config()
    _setup_logging(settings.log_level)
    args = build_argparser().parse_args()
    overwrite = True if args.overwrite else cfg.overwrite
    docs = load_sample_documents(
        settings.data_processed_dir, cfg.sample_docs, cfg.max_chars_per_doc
    )
    rt = build_llm_runtime(cfg.backend, cfg.temperature, cfg.model_override)
    prompt = build_schema_prompt(docs)
    resp = rt.client.chat.completions.create(
        model=rt.model,
        temperature=rt.temperature,
        messages=[
            {"role": "system", "content": "Du gibst nur JSON aus. Keine Erklärungen."},
            {"role": "user", "content": prompt},
        ],
    )
    content = (resp.choices[0].message.content or "").strip()
    parsed = extract_first_json(content)
    if not isinstance(parsed, dict) or not validate_schema(parsed):
        raise RuntimeError("Schema-Generierung: Ungültiges JSON.")
    schema_path = write_schema(cfg.out_dir, parsed, rt, overwrite=overwrite)
    if cfg.alias_name:
        write_alias(schema_path, cfg.alias_name)


if __name__ == "__main__":
    run()
