from __future__ import annotations
import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from dotenv import load_dotenv  # noqa: E402
from openai import OpenAI  # noqa: E402
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)  # noqa: E402
from config.settings import load_settings  # noqa: E402

logger = logging.getLogger(__name__)


# -----------------------------
# Helpers
# -----------------------------
def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _clean_env(value: str | None) -> str:
    if value is None:
        return ""
    # inline comments kill model ids -> strip everything after #
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


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def extract_first_json(text: str) -> Optional[Any]:
    s = (text or "").strip()
    if not s:
        return None
    m = re.search(r"(\{.*\}|\[.*\])", s, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class ExtractConfig:
    input_dir: Path
    output_dir: Path
    schema_path: Path
    backend: str  # openrouter | ollama
    temperature: float
    model: str | None
    overwrite: bool
    max_chunks: int
    sleep_sec: float
    error_dir: Path
    schema_strict: bool
    repair_enable: bool
    repair_model: str | None
    max_retries: int
    max_tokens: int
    parallelism: int


def load_extract_config() -> ExtractConfig:
    load_dotenv()
    return ExtractConfig(
        input_dir=Path(
            _clean_env(os.getenv("EXTRACT_INPUT_DIR", "data/processed/chunks"))
        ),
        output_dir=Path(
            _clean_env(os.getenv("EXTRACT_OUTPUT_DIR", "output/extractions"))
        ),
        schema_path=Path(_clean_env(os.getenv("EXTRACT_SCHEMA_PATH", ""))),
        backend=_clean_env(os.getenv("EXTRACT_LLM_BACKEND", "openrouter")).lower(),
        temperature=_to_float(os.getenv("EXTRACT_TEMPERATURE"), 0.0),
        model=(_clean_env(os.getenv("EXTRACT_MODEL")) or None),
        overwrite=_to_bool(os.getenv("EXTRACT_OVERWRITE"), False),
        max_chunks=_to_int(os.getenv("EXTRACT_MAX_CHUNKS"), 0),
        sleep_sec=_to_float(os.getenv("EXTRACT_SLEEP_SEC"), 0.0),
        error_dir=Path(_clean_env(os.getenv("EXTRACT_ERROR_DIR", "output/errors"))),
        schema_strict=_to_bool(os.getenv("EXTRACT_SCHEMA_STRICT"), True),
        repair_enable=_to_bool(os.getenv("EXTRACT_REPAIR_ENABLE"), True),
        repair_model=(_clean_env(os.getenv("EXTRACT_REPAIR_MODEL")) or None),
        max_retries=_to_int(os.getenv("EXTRACT_MAX_RETRIES"), 2),
        max_tokens=_to_int(os.getenv("EXTRACT_MAX_TOKENS"), 900),
        parallelism=max(1, _to_int(os.getenv("EXTRACT_PARALLELISM"), 1)),
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
            raise ValueError("OPENROUTER_MODEL fehlt (oder EXTRACT_MODEL setzen).")
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
            raise ValueError("OLLAMA_MODEL fehlt (oder EXTRACT_MODEL setzen).")
        client = OpenAI(base_url=base_url, api_key="ollama")
        return LLMRuntime(
            backend=backend, model=model, temperature=temperature, client=client
        )
    raise ValueError(f"Unbekanntes EXTRACT_LLM_BACKEND: {backend}")


# -----------------------------
# IO
# -----------------------------
def list_chunk_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Chunk input dir not found: {input_dir.resolve()}")
    return [p.resolve() for p in sorted(input_dir.glob("*.chunks.jsonl"))]


def iter_jsonl(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_schema(schema_path: Path) -> dict[str, Any]:
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path.resolve()}")
    return json.loads(schema_path.read_text(encoding="utf-8"))


def ensure_dirs(cfg: ExtractConfig) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.error_dir.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Schema constraints
# -----------------------------
def schema_allowed_types(schema: dict[str, Any]) -> tuple[set[str], set[str]]:
    ent = set()
    rel = set()
    for e in schema.get("entity_types", []):
        n = e.get("name")
        if isinstance(n, str) and n.strip():
            ent.add(n.strip())
    for r in schema.get("relation_types", []):
        n = r.get("name")
        if isinstance(n, str) and n.strip():
            rel.add(n.strip())
    return ent, rel


def filter_extraction_to_schema(
    extraction: dict[str, Any],
    allowed_entities: set[str],
    allowed_relations: set[str],
) -> dict[str, Any]:
    entities_in = extraction.get("entities", [])
    rels_in = extraction.get("relations", [])
    entities_out: list[dict[str, Any]] = []
    kept_ids: set[str] = set()
    for e in entities_in if isinstance(entities_in, list) else []:
        if not isinstance(e, dict):
            continue
        et = str(e.get("type", "")).strip()
        eid = str(e.get("id", "")).strip()
        name = str(e.get("name", "")).strip()
        if not eid or not name:
            continue
        if et not in allowed_entities:
            continue
        attrs = e.get("attributes", {})
        if attrs is None or not isinstance(attrs, dict):
            attrs = {}
        entities_out.append(
            {
                "id": eid,
                "type": et,
                "name": name,
                "attributes": attrs,
                "evidence": (
                    e.get("evidence") if isinstance(e.get("evidence"), str) else None
                ),
            }
        )
        kept_ids.add(eid)
    rels_out: list[dict[str, Any]] = []
    for r in rels_in if isinstance(rels_in, list) else []:
        if not isinstance(r, dict):
            continue
        rt = str(r.get("type", "")).strip()
        s = str(r.get("source_id", "")).strip()
        t = str(r.get("target_id", "")).strip()
        if rt not in allowed_relations:
            continue
        if s not in kept_ids or t not in kept_ids:
            continue
        rels_out.append(
            {
                "source_id": s,
                "type": rt,
                "target_id": t,
                "evidence": (
                    r.get("evidence") if isinstance(r.get("evidence"), str) else None
                ),
            }
        )
    return {"entities": entities_out, "relations": rels_out}


# -----------------------------
# Prompting
# -----------------------------
def schema_compact(schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "domain": schema.get("domain", "Domain"),
        "entity_types": schema.get("entity_types", []),
        "relation_types": schema.get("relation_types", []),
    }


def build_extraction_prompt(schema: dict[str, Any], chunk: dict[str, Any]) -> str:
    sc = schema_compact(schema)
    allowed_entities = [
        e["name"] for e in sc["entity_types"] if isinstance(e, dict) and "name" in e
    ]
    allowed_relations = [
        r["name"] for r in sc["relation_types"] if isinstance(r, dict) and "name" in r
    ]
    text = chunk.get("text", "")
    topic = chunk.get("topic")
    document_title = chunk.get("document_title")
    return (
        "Du extrahierst strukturierte Informationen aus einem Textabschnitt einer technischen Dokumentation.\n"
        "WICHTIG: Verwende ausschließlich die folgenden Entity- und Relation-Typen.\n"
        "Wenn ein Typ nicht passt, lasse die Entität/Relation weg.\n"
        "Erfinde keine Fakten. Nutze als evidence ein kurzes Zitat aus dem Text.\n"
        "Gib ausschließlich JSON aus.\n\n"
        f"Allowed Entity Types: {allowed_entities}\n"
        f"Allowed Relation Types: {allowed_relations}\n\n"
        "Schema:\n"
        f"{json.dumps(sc, ensure_ascii=False, indent=2)}\n\n"
        "Kontext:\n"
        f"- document_title: {document_title}\n"
        f"- topic: {topic}\n\n"
        "Text:\n"
        f"{text}\n\n"
        "Ausgabeformat (JSON):\n"
        "{\n"
        '  "entities": [\n'
        "    {\n"
        '      "id": "E1",\n'
        '      "type": "EntityTypeName",\n'
        '      "name": "kanonischer Name",\n'
        '      "attributes": { "attr": "value" },\n'
        '      "evidence": "Zitat"\n'
        "    }\n"
        "  ],\n"
        '  "relations": [\n'
        "    {\n"
        '      "source_id": "E1",\n'
        '      "type": "RELATION_NAME",\n'
        '      "target_id": "E2",\n'
        '      "evidence": "Zitat"\n'
        "    }\n"
        "  ]\n"
        "}\n"
    )


def build_repair_prompt(
    schema: dict[str, Any], chunk: dict[str, Any], bad_output: str
) -> str:
    sc = schema_compact(schema)
    text = chunk.get("text", "")
    return (
        "Die folgende JSON-Ausgabe ist nicht schema-konform.\n"
        "Repariere sie so, dass:\n"
        "- nur Entity/Relation-Typen aus dem Schema vorkommen\n"
        "- relations nur ids referenzieren, die in entities existieren\n"
        "- Ausgabe exakt dem JSON-Format entspricht\n"
        "Gib ausschließlich JSON aus.\n\n"
        "Schema:\n"
        f"{json.dumps(sc, ensure_ascii=False, indent=2)}\n\n"
        "Text:\n"
        f"{text}\n\n"
        "Fehlerhafte Ausgabe:\n"
        f"{bad_output}\n"
    )


def validate_extraction_shape(obj: dict[str, Any]) -> bool:
    if not isinstance(obj, dict):
        return False
    if "entities" not in obj or "relations" not in obj:
        return False
    if not isinstance(obj["entities"], list) or not isinstance(obj["relations"], list):
        return False
    return True


# -----------------------------
# LLM calls
# -----------------------------
class LLMCallError(RuntimeError):
    pass


def _llm_chat(rt: LLMRuntime, prompt: str, max_tokens: int) -> str:
    resp = rt.client.chat.completions.create(
        model=rt.model,
        temperature=rt.temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": "Du gibst nur JSON aus. Keine Erklärungen."},
            {"role": "user", "content": prompt},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


def make_llm_call_with_retry(max_attempts: int):
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        retry=retry_if_exception_type(LLMCallError),
        reraise=True,
    )


# -----------------------------
# Execution
# -----------------------------
def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="LLM-based information extraction from chunks + schema."
    )
    p.add_argument(
        "--chunks",
        type=str,
        default="",
        help="Specific chunks file (default: all in EXTRACT_INPUT_DIR)",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite output files")
    return p


def run() -> None:
    settings = load_settings()
    cfg = load_extract_config()
    _setup_logging(settings.log_level)
    ensure_dirs(cfg)
    if not str(cfg.schema_path).strip():
        raise ValueError("EXTRACT_SCHEMA_PATH fehlt in .env")
    schema = load_schema(cfg.schema_path)
    allowed_entity_types, allowed_relation_types = schema_allowed_types(schema)
    rt = build_llm_runtime(cfg.backend, cfg.temperature, cfg.model)
    repair_rt: Optional[LLMRuntime] = None
    if cfg.repair_enable:
        repair_model = cfg.repair_model or rt.model
        repair_rt = build_llm_runtime(cfg.backend, cfg.temperature, repair_model)
    args = build_argparser().parse_args()
    overwrite = True if args.overwrite else cfg.overwrite
    if args.chunks:
        chunk_files = [Path(args.chunks)]
    else:
        chunk_files = list_chunk_files(cfg.input_dir)
    if not chunk_files:
        logger.warning("No chunk files found in: %s", cfg.input_dir.resolve())
        return
    for chunk_path in chunk_files:
        if not chunk_path.is_absolute():
            chunk_path = (cfg.input_dir / chunk_path).resolve()
        if not chunk_path.exists():
            logger.error("Chunks file not found: %s", chunk_path)
            continue
        base_name = chunk_path.stem.replace(".chunks", "")
        out_path = cfg.output_dir / f"{base_name}.extractions.jsonl"
        err_path = cfg.error_dir / f"{base_name}.errors.jsonl"
        if out_path.exists() and not overwrite:
            logger.info("Skip (extractions exist): %s", out_path.name)
            continue
        # overwrite => Datei leeren
        if overwrite and out_path.exists():
            out_path.unlink(missing_ok=True)
        logger.info("Processing chunks: %s", chunk_path.name)
        processed = 0
        for chunk in iter_jsonl(chunk_path):
            if cfg.max_chunks and processed >= cfg.max_chunks:
                break
            processed += 1
            chunk_id = chunk.get("chunk_id")
            prompt = build_extraction_prompt(schema, chunk)

            @make_llm_call_with_retry(cfg.max_retries)
            def _call() -> dict[str, Any]:
                try:
                    content = _llm_chat(rt, prompt, max_tokens=cfg.max_tokens)
                    parsed = extract_first_json(content)
                    if not isinstance(parsed, dict) or not validate_extraction_shape(
                        parsed
                    ):
                        raise LLMCallError("invalid_json_shape")
                    if cfg.schema_strict:
                        parsed = filter_extraction_to_schema(
                            parsed, allowed_entity_types, allowed_relation_types
                        )
                    return parsed
                except LLMCallError:
                    raise
                except Exception as e:
                    raise LLMCallError(str(e))

            try:
                extraction = _call()
            except Exception as e:
                # optional repair attempt (1x)
                if cfg.repair_enable and repair_rt is not None:
                    try:
                        bad_out = str(e)
                        repair_prompt = build_repair_prompt(schema, chunk, bad_out)
                        content = _llm_chat(
                            repair_rt, repair_prompt, max_tokens=cfg.max_tokens
                        )
                        parsed = extract_first_json(content)
                        if not isinstance(
                            parsed, dict
                        ) or not validate_extraction_shape(parsed):
                            raise RuntimeError("repair_invalid_json_shape")
                        if cfg.schema_strict:
                            parsed = filter_extraction_to_schema(
                                parsed, allowed_entity_types, allowed_relation_types
                            )
                        extraction = parsed
                    except Exception as e2:
                        append_jsonl(
                            err_path,
                            {
                                "chunk_id": chunk_id,
                                "document_title": chunk.get("document_title"),
                                "error": str(e),
                                "repair_error": str(e2),
                                "llm_backend": rt.backend,
                                "llm_model": rt.model,
                                "repaired_with": (
                                    repair_rt.model if repair_rt else None
                                ),
                                "extracted_at": utc_now_iso(),
                            },
                        )
                        logger.warning(
                            "Extraction+repair failed for chunk_id=%s", str(chunk_id)
                        )
                        continue
                else:
                    append_jsonl(
                        err_path,
                        {
                            "chunk_id": chunk_id,
                            "document_title": chunk.get("document_title"),
                            "error": str(e),
                            "llm_backend": rt.backend,
                            "llm_model": rt.model,
                            "extracted_at": utc_now_iso(),
                        },
                    )
                    logger.warning("Extraction failed for chunk_id=%s", str(chunk_id))
                    continue
            row = {
                "document_title": chunk.get("document_title"),
                "chunk_id": chunk_id,
                "char_start": chunk.get("char_start"),
                "char_end": chunk.get("char_end"),
                "chunk_method": chunk.get("chunk_method"),
                "topic": chunk.get("topic"),
                "source_text_file": chunk.get("source_text_file"),
                "llm_backend": rt.backend,
                "llm_model": rt.model,
                "llm_temperature": rt.temperature,
                "extracted_at": utc_now_iso(),
                "entities": extraction.get("entities", []),
                "relations": extraction.get("relations", []),
            }
            append_jsonl(out_path, row)
            if cfg.sleep_sec and cfg.sleep_sec > 0:
                time.sleep(cfg.sleep_sec)
        logger.info("Wrote extractions: %s", out_path.name)


if __name__ == "__main__":
    run()
