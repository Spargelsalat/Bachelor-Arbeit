from __future__ import annotations
import argparse
import difflib
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


def _to_float(value: str | None, default: float) -> float:
    v = _clean_env(value)
    if v == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _to_bool(value: str | None, default: bool) -> bool:
    v = _clean_env(value).lower()
    if v == "":
        return default
    return v in {"1", "true", "yes", "y", "on"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def slugify(value: str) -> str:
    v = value.strip().lower()
    v = re.sub(r"\s+", " ", v)
    v = re.sub(r"[^a-z0-9 _\-\.]", "", v)
    v = v.replace(" ", "_")
    v = re.sub(r"_+", "_", v)
    return v.strip("_") or "x"


def normalize_name(name: str) -> str:
    n = name.strip().lower()
    n = n.replace("\u00ad", "")  #  hyphen
    n = re.sub(r"\s+", " ", n)
    n = re.sub(r"[“”\"']", "", n)
    n = n.strip()
    return n


def similarity(a: str, b: str) -> float:
    # deterministisch
    return difflib.SequenceMatcher(None, a, b).ratio()


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
            raise ValueError("OPENROUTER_MODEL fehlt (oder RESOLVE_MODEL setzen).")
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
            raise ValueError("OLLAMA_MODEL fehlt (oder RESOLVE_MODEL setzen).")
        client = OpenAI(base_url=base_url, api_key="ollama")
        return LLMRuntime(
            backend=backend, model=model, temperature=temperature, client=client
        )
    raise ValueError(f"Unbekanntes RESOLVE_LLM_BACKEND: {backend}")


def llm_is_same_entity(
    rt: LLMRuntime,
    ent_type: str,
    a_name: str,
    a_ev: str | None,
    b_name: str,
    b_ev: str | None,
) -> bool:
    prompt = (
        "Entscheide, ob zwei Erwähnungen dieselbe Entität bezeichnen.\n"
        'Antworte ausschließlich mit JSON: {"same": true/false}\n\n'
        f"Entitätstyp: {ent_type}\n"
        f"A: name={a_name}\n"
        f"A evidence={a_ev}\n"
        f"B: name={b_name}\n"
        f"B evidence={b_ev}\n"
    )
    resp = rt.client.chat.completions.create(
        model=rt.model,
        temperature=rt.temperature,
        max_tokens=60,
        messages=[
            {"role": "system", "content": "Du gibst nur JSON aus."},
            {"role": "user", "content": prompt},
        ],
    )
    content = (resp.choices[0].message.content or "").strip()
    m = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if not m:
        return False
    try:
        obj = json.loads(m.group(0))
        return bool(obj.get("same", False))
    except Exception:
        return False


@dataclass
class CanonicalEntity:
    global_id: str
    type: str
    name: str
    norm_name: str
    attributes: dict[str, Any]
    mentions: list[dict[str, Any]]  # provenance


def pick_best_name(current: str, candidate: str) -> str:
    # bevorzugt "schöneren" Namen
    if len(candidate) > len(current):
        return candidate
    return current


def merge_attributes(base: dict[str, Any], add: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in add.items():
        if k not in out or out[k] in (None, "", []):
            out[k] = v
    return out


@dataclass(frozen=True)
class ResolveConfig:
    input_dir: Path
    output_dir: Path
    overwrite: bool
    sim_high: float
    sim_low: float
    max_candidates: int
    use_llm: bool
    llm_backend: str
    llm_model: str | None
    llm_temperature: float
    sleep_sec: float


def load_resolve_config() -> ResolveConfig:
    load_dotenv()
    return ResolveConfig(
        input_dir=Path(
            _clean_env(os.getenv("RESOLVE_INPUT_DIR", "output/extractions"))
        ),
        output_dir=Path(_clean_env(os.getenv("RESOLVE_OUTPUT_DIR", "output/resolved"))),
        overwrite=_to_bool(os.getenv("RESOLVE_OVERWRITE"), False),
        sim_high=_to_float(os.getenv("RESOLVE_SIM_HIGH"), 0.92),
        sim_low=_to_float(os.getenv("RESOLVE_SIM_LOW"), 0.80),
        max_candidates=_to_int(os.getenv("RESOLVE_MAX_CANDIDATES"), 25),
        use_llm=_to_bool(os.getenv("RESOLVE_USE_LLM"), False),
        llm_backend=_clean_env(os.getenv("RESOLVE_LLM_BACKEND", "openrouter")).lower(),
        llm_model=(_clean_env(os.getenv("RESOLVE_MODEL")) or None),
        llm_temperature=_to_float(os.getenv("RESOLVE_TEMPERATURE"), 0.0),
        sleep_sec=_to_float(os.getenv("RESOLVE_SLEEP_SEC"), 0.0),
    )


def list_extraction_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Resolve input dir not found: {input_dir.resolve()}")
    return sorted(input_dir.glob("*.extractions.jsonl"))


def resolve_entities_and_relations(
    extraction_file: Path,
    cfg: ResolveConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rt: Optional[LLMRuntime] = None
    if cfg.use_llm:
        rt = build_llm_runtime(cfg.llm_backend, cfg.llm_temperature, cfg.llm_model)
        logger.info("Resolver LLM enabled: %s / %s", rt.backend, rt.model)
    entities_by_type: dict[str, list[CanonicalEntity]] = {}
    # relation key: (src_global_id, rel_type, tgt_global_id)
    rel_map: dict[tuple[str, str, str], dict[str, Any]] = {}

    def find_or_create(ent: dict[str, Any], ctx: dict[str, Any]) -> CanonicalEntity:
        et = str(ent.get("type", "")).strip()
        name = str(ent.get("name", "")).strip()
        if not et or not name:
            raise ValueError("entity missing type/name")
        norm = normalize_name(name)
        candidates = entities_by_type.get(et, [])
        # exact match
        for c in candidates:
            if c.norm_name == norm:
                c.name = pick_best_name(c.name, name)
                c.attributes = merge_attributes(
                    c.attributes, ent.get("attributes") or {}
                )
                c.mentions.append(ctx)
                return c
        #  fuzzy match
        scored: list[tuple[float, CanonicalEntity]] = []
        for c in candidates:
            scored.append((similarity(norm, c.norm_name), c))
        scored.sort(key=lambda x: x[0], reverse=True)
        scored = scored[: cfg.max_candidates]
        if scored:
            best_score, best = scored[0]
            if best_score >= cfg.sim_high:
                best.name = pick_best_name(best.name, name)
                best.attributes = merge_attributes(
                    best.attributes, ent.get("attributes") or {}
                )
                best.mentions.append(ctx)
                return best
            if (
                cfg.use_llm
                and rt is not None
                and cfg.sim_low <= best_score < cfg.sim_high
            ):
                a_ev = ctx.get("evidence_entity")
                b_ev = (
                    best.mentions[-1].get("evidence_entity") if best.mentions else None
                )
                same = llm_is_same_entity(rt, et, name, a_ev, best.name, b_ev)
                if cfg.sleep_sec > 0:
                    time.sleep(cfg.sleep_sec)
                if same:
                    best.name = pick_best_name(best.name, name)
                    best.attributes = merge_attributes(
                        best.attributes, ent.get("attributes") or {}
                    )
                    best.mentions.append(ctx)
                    return best
        # create new canonical entity
        gid = f"{et}:{slugify(name)}"
        ce = CanonicalEntity(
            global_id=gid,
            type=et,
            name=name,
            norm_name=norm,
            attributes=(
                (ent.get("attributes") or {})
                if isinstance(ent.get("attributes"), dict)
                else {}
            ),
            mentions=[ctx],
        )
        entities_by_type.setdefault(et, []).append(ce)
        return ce

    for row in iter_jsonl(extraction_file):
        doc_title = row.get("document_title")
        chunk_id = row.get("chunk_id")
        source_file = row.get("source_text_file")
        local_entities = row.get("entities", [])
        local_relations = row.get("relations", [])
        local_id_to_global: dict[str, str] = {}
        # entities
        if isinstance(local_entities, list):
            for e in local_entities:
                if not isinstance(e, dict):
                    continue
                local_id = str(e.get("id", "")).strip()
                if not local_id:
                    continue
                ctx = {
                    "document_title": doc_title,
                    "chunk_id": chunk_id,
                    "source_text_file": source_file,
                    "local_entity_id": local_id,
                    "evidence_entity": (
                        e.get("evidence")
                        if isinstance(e.get("evidence"), str)
                        else None
                    ),
                }
                ce = find_or_create(e, ctx)
                local_id_to_global[local_id] = ce.global_id
        # relations
        if isinstance(local_relations, list):
            for r in local_relations:
                if not isinstance(r, dict):
                    continue
                rt_name = str(r.get("type", "")).strip()
                s_local = str(r.get("source_id", "")).strip()
                t_local = str(r.get("target_id", "")).strip()
                if not rt_name or not s_local or not t_local:
                    continue
                s_gid = local_id_to_global.get(s_local)
                t_gid = local_id_to_global.get(t_local)
                if not s_gid or not t_gid:
                    continue
                key = (s_gid, rt_name, t_gid)
                ev = r.get("evidence") if isinstance(r.get("evidence"), str) else None
                prov = {
                    "document_title": doc_title,
                    "chunk_id": chunk_id,
                    "source_text_file": source_file,
                    "evidence_relation": ev,
                }
                if key not in rel_map:
                    rel_map[key] = {
                        "source": s_gid,
                        "type": rt_name,
                        "target": t_gid,
                        "provenance": [prov],
                    }
                else:
                    rel_map[key]["provenance"].append(prov)
    # output objects
    out_entities: list[dict[str, Any]] = []
    for et, ents in entities_by_type.items():
        for e in ents:
            out_entities.append(
                {
                    "global_id": e.global_id,
                    "type": e.type,
                    "name": e.name,
                    "attributes": e.attributes,
                    "mentions": e.mentions,
                }
            )
    out_relations: list[dict[str, Any]] = list(rel_map.values())
    return out_entities, out_relations


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Entity resolution / consolidation over extractions.jsonl"
    )
    p.add_argument(
        "--file",
        type=str,
        default="",
        help="Specific extraction file (default: all in RESOLVE_INPUT_DIR)",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite outputs")
    return p


def run() -> None:
    settings = load_settings()
    cfg = load_resolve_config()
    _setup_logging(settings.log_level)
    files = [Path(cfg.input_dir / cfg_file) for cfg_file in []]  # placeholder
    args = build_argparser().parse_args()
    overwrite = True if args.overwrite else cfg.overwrite
    if args.file:
        in_files = [Path(args.file)]
        if not in_files[0].is_absolute():
            in_files = [(cfg.input_dir / in_files[0]).resolve()]
    else:
        in_files = list_extraction_files(cfg.input_dir)
    if not in_files:
        logger.warning("No extraction files found in: %s", cfg.input_dir.resolve())
        return
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    for f in in_files:
        if not f.exists():
            logger.error("File not found: %s", f)
            continue
        base = f.stem.replace(".extractions", "")
        out_entities_path = cfg.output_dir / f"{base}.entities.resolved.jsonl"
        out_relations_path = cfg.output_dir / f"{base}.relations.resolved.jsonl"
        if (
            out_entities_path.exists() or out_relations_path.exists()
        ) and not overwrite:
            logger.info("Skip (outputs exist): %s", base)
            continue
        logger.info("Resolving: %s", f.name)
        entities, relations = resolve_entities_and_relations(f, cfg)
        # meta header
        meta = {
            "_meta": {
                "resolved_at": utc_now_iso(),
                "sim_high": cfg.sim_high,
                "sim_low": cfg.sim_low,
                "use_llm": cfg.use_llm,
                "llm_backend": cfg.llm_backend if cfg.use_llm else None,
                "llm_model": cfg.llm_model if cfg.use_llm else None,
            }
        }
        write_jsonl(out_entities_path, [meta] + entities, overwrite=True)
        write_jsonl(out_relations_path, [meta] + relations, overwrite=True)


if __name__ == "__main__":
    run()
