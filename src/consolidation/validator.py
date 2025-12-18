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


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict[str, Any]], overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        logger.info("Skip (exists): %s", path.name)
        return
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("Wrote: %s (%d rows)", path.name, len(rows))


def normalize_for_match(text: str) -> str:
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"\s+", " ", t).strip()
    return t


@dataclass(frozen=True)
class ValidateConfig:
    input_dir: Path
    chunk_dir: Path
    output_dir: Path
    overwrite: bool
    case_insensitive: bool
    max_relations: int
    max_provenances: int
    use_llm: bool
    llm_backend: str
    llm_model: str | None
    llm_temperature: float
    llm_max_tokens: int
    sleep_sec: float


def load_validate_config() -> ValidateConfig:
    load_dotenv()
    return ValidateConfig(
        input_dir=Path(_clean_env(os.getenv("VALIDATE_INPUT_DIR", "output/resolved"))),
        chunk_dir=Path(
            _clean_env(os.getenv("VALIDATE_CHUNK_DIR", "data/processed/chunks"))
        ),
        output_dir=Path(
            _clean_env(os.getenv("VALIDATE_OUTPUT_DIR", "output/validated"))
        ),
        overwrite=_to_bool(os.getenv("VALIDATE_OVERWRITE"), False),
        case_insensitive=_to_bool(os.getenv("VALIDATE_CASE_INSENSITIVE"), True),
        max_relations=_to_int(os.getenv("VALIDATE_MAX_RELATIONS"), 0),
        max_provenances=_to_int(os.getenv("VALIDATE_MAX_PROVENANCES"), 0),
        use_llm=_to_bool(os.getenv("VALIDATE_USE_LLM"), False),
        llm_backend=_clean_env(os.getenv("VALIDATE_LLM_BACKEND", "openrouter")).lower(),
        llm_model=(_clean_env(os.getenv("VALIDATE_MODEL")) or None),
        llm_temperature=_to_float(os.getenv("VALIDATE_TEMPERATURE"), 0.0),
        llm_max_tokens=_to_int(os.getenv("VALIDATE_MAX_TOKENS"), 120),
        sleep_sec=_to_float(os.getenv("VALIDATE_SLEEP_SEC"), 0.0),
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
            raise ValueError("OPENROUTER_MODEL fehlt (oder VALIDATE_MODEL setzen).")
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
            raise ValueError("OLLAMA_MODEL fehlt (oder VALIDATE_MODEL setzen).")
        client = OpenAI(base_url=base_url, api_key="ollama")
        return LLMRuntime(
            backend=backend, model=model, temperature=temperature, client=client
        )
    raise ValueError(f"Unbekanntes VALIDATE_LLM_BACKEND: {backend}")


def llm_validate_relation(
    rt: LLMRuntime, chunk_text: str, triple: tuple[str, str, str]
) -> str:
    """
    Return: SUPPORTED | UNSUPPORTED | UNSURE
    """
    s, rel, t = triple
    prompt = (
        "Prüfe, ob die folgende Relation durch den Text belegt ist.\n"
        'Antworte ausschließlich mit JSON: {"verdict": "SUPPORTED|UNSUPPORTED|UNSURE"}\n\n'
        f"Relation: ({s}) -[{rel}]-> ({t})\n\n"
        f"Text:\n{chunk_text}\n"
    )
    resp = rt.client.chat.completions.create(
        model=rt.model,
        temperature=rt.temperature,
        max_tokens=rt.temperature and 120 or 120,
        messages=[
            {"role": "system", "content": "Du gibst nur JSON aus."},
            {"role": "user", "content": prompt},
        ],
    )
    content = (resp.choices[0].message.content or "").strip()
    m = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if not m:
        return "UNSURE"
    try:
        obj = json.loads(m.group(0))
        v = str(obj.get("verdict", "UNSURE")).strip().upper()
        if v in {"SUPPORTED", "UNSUPPORTED", "UNSURE"}:
            return v
        return "UNSURE"
    except Exception:
        return "UNSURE"


def infer_chunk_file_for_relations(rel_file: Path, chunk_dir: Path) -> Path:
    """
    rel_file:   <base>.relations.resolved.jsonl
    chunk file: <base>.chunks.jsonl
    """
    base = rel_file.stem.replace(".relations.resolved", "")
    candidate = chunk_dir / f"{base}.chunks.jsonl"
    if not candidate.exists():
        raise FileNotFoundError(
            f"Chunk file not found for {rel_file.name}: {candidate.resolve()}"
        )
    return candidate


def load_chunk_text_map(chunk_file: Path) -> dict[int, str]:
    m: dict[int, str] = {}
    for row in iter_jsonl(chunk_file):
        try:
            cid = int(row.get("chunk_id"))
            txt = row.get("text", "")
            if isinstance(txt, str):
                m[cid] = txt
        except Exception:
            continue
    return m


def validate_relations_file(rel_file: Path, cfg: ValidateConfig) -> Path:
    chunk_file = infer_chunk_file_for_relations(rel_file, cfg.chunk_dir)
    chunk_map = load_chunk_text_map(chunk_file)
    rt: Optional[LLMRuntime] = None
    if cfg.use_llm:
        rt = build_llm_runtime(cfg.llm_backend, cfg.llm_temperature, cfg.llm_model)
        logger.info("Validator LLM enabled: %s / %s", rt.backend, rt.model)
    out_path = cfg.output_dir / rel_file.name.replace(
        ".relations.resolved.jsonl", ".relations.validated.jsonl"
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    out_rows: list[dict[str, Any]] = []
    rel_count = 0
    for row in iter_jsonl(rel_file):
        # meta passthrough
        if isinstance(row, dict) and "_meta" in row:
            out_rows.append(row)
            continue
        rel_count += 1
        if cfg.max_relations and rel_count > cfg.max_relations:
            break
        source = row.get("source")
        rtype = row.get("type")
        target = row.get("target")
        prov_list = row.get("provenance", [])
        supported = 0
        unsupported = 0
        unknown = 0
        new_prov: list[dict[str, Any]] = []
        if not isinstance(prov_list, list):
            prov_list = []
        prov_iter = prov_list
        if cfg.max_provenances and len(prov_iter) > cfg.max_provenances:
            prov_iter = prov_iter[: cfg.max_provenances]
        for prov in prov_iter:
            if not isinstance(prov, dict):
                continue
            chunk_id = prov.get("chunk_id")
            ev = prov.get("evidence_relation")
            chunk_text = chunk_map.get(int(chunk_id)) if chunk_id is not None else None
            status = "UNKNOWN"
            if chunk_text is None:
                status = "UNKNOWN"
                unknown += 1
            else:
                if isinstance(ev, str) and ev.strip():
                    a = normalize_for_match(ev)
                    b = normalize_for_match(chunk_text)
                    if cfg.case_insensitive:
                        a = a.casefold()
                        b = b.casefold()
                    status = "SUPPORTED" if a in b else "UNSUPPORTED"
                    if status == "SUPPORTED":
                        supported += 1
                    else:
                        unsupported += 1
                else:
                    status = "UNKNOWN"
                    unknown += 1
            #  LLM escalation
            llm_status = None
            if (
                cfg.use_llm
                and rt is not None
                and chunk_text is not None
                and status in {"UNSUPPORTED", "UNKNOWN"}
            ):
                llm_status = llm_validate_relation(
                    rt, chunk_text, (str(source), str(rtype), str(target))
                )
                if cfg.sleep_sec > 0:
                    time.sleep(cfg.sleep_sec)
            new_prov.append(
                {**prov, "support_status": status, "llm_support_status": llm_status}
            )
        total = max(1, (supported + unsupported + unknown))
        support_ratio = supported / total
        out_rows.append(
            {
                **row,
                "validation": {
                    "supported": supported,
                    "unsupported": unsupported,
                    "unknown": unknown,
                    "support_ratio": round(support_ratio, 4),
                    "validated_at": utc_now_iso(),
                    "chunk_file": str(chunk_file.resolve()),
                    "case_insensitive": cfg.case_insensitive,
                    "use_llm": cfg.use_llm,
                    "llm_backend": (rt.backend if rt else None),
                    "llm_model": (rt.model if rt else None),
                },
                "provenance": new_prov,
            }
        )
    write_jsonl(out_path, out_rows, overwrite=cfg.overwrite)
    return out_path


def list_relation_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Validate input dir not found: {input_dir.resolve()}")
    return sorted(input_dir.glob("*.relations.resolved.jsonl"))


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Validate resolved relations against chunk evidence."
    )
    p.add_argument(
        "--file", type=str, default="", help="Specific relations.resolved.jsonl file"
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite outputs")
    return p


def run() -> None:
    settings = load_settings()
    cfg = load_validate_config()
    _setup_logging(settings.log_level)
    args = build_argparser().parse_args()
    overwrite = args.overwrite or cfg.overwrite
    cfg = ValidateConfig(**{**cfg.__dict__, "overwrite": overwrite})
    if args.file:
        files = [Path(args.file)]
        if not files[0].is_absolute():
            files = [(cfg.input_dir / files[0]).resolve()]
    else:
        files = list_relation_files(cfg.input_dir)
    if not files:
        logger.warning(
            "No relations.resolved.jsonl files found in: %s", cfg.input_dir.resolve()
        )
        return
    for f in files:
        if not f.exists():
            logger.error("File not found: %s", str(f))
            continue
        logger.info("Validating: %s", f.name)
        validate_relations_file(f, cfg)


if __name__ == "__main__":
    run()
