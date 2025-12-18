from __future__ import annotations
import argparse
import json
import logging
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

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


def _to_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    v = value.strip()
    if v == "":
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _to_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _to_float(value: str | None, default: float) -> float:
    if value is None:
        return default
    v = value.strip()
    if v == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


def normalize_text_minimal(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()


def list_processed_texts(processed_dir: Path) -> list[Path]:
    processed_dir = processed_dir.resolve()
    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed directory not found: {processed_dir}")
    return sorted(processed_dir.glob("*.txt"))


def extract_first_json(text: str) -> Optional[Any]:
    s = (text or "").strip()
    if not s:
        return None
    m = re.search(r"(\{.*\}|\[.*\])", s, flags=re.DOTALL)
    if not m:
        return None
    candidate = m.group(0)
    try:
        return json.loads(candidate)
    except Exception:
        return None


@dataclass(frozen=True)
class ChunkConfig:
    mode: str  # rule | llm
    out_dir: Path
    overwrite: bool
    # rule baseline 
    size_chars: int
    overlap_chars: int
    max_doc_chars: int
    max_chunks: int
    # LLM chunking
    llm_backend: str  # openrouter | ollama
    llm_max_paragraphs: int
    llm_preview_chars: int
    llm_temperature: float
    llm_subchunk_enable: bool
    llm_subchunk_size_chars: int
    llm_subchunk_overlap_chars: int
    #  suffix
    suffix_rule: str
    suffix_llm: str
    suffix_llm_fallback: str


def load_chunk_config() -> ChunkConfig:
    load_dotenv()
    return ChunkConfig(
        mode=os.getenv("CHUNK_MODE", "rule").strip().lower(),
        out_dir=Path(os.getenv("CHUNK_OUTPUT_DIR", "data/processed/chunks")),
        overwrite=_to_bool(os.getenv("CHUNK_OVERWRITE"), False),
        size_chars=_to_int(os.getenv("CHUNK_SIZE_CHARS"), 4000),
        overlap_chars=_to_int(os.getenv("CHUNK_OVERLAP_CHARS"), 300),
        max_doc_chars=_to_int(os.getenv("CHUNK_MAX_DOC_CHARS"), 0),
        max_chunks=_to_int(os.getenv("CHUNK_MAX_CHUNKS"), 5000),
        llm_backend=os.getenv("CHUNK_LLM_BACKEND", "openrouter").strip().lower(),
        llm_max_paragraphs=_to_int(os.getenv("CHUNK_LLM_MAX_PARAGRAPHS"), 120),
        llm_preview_chars=_to_int(os.getenv("CHUNK_LLM_PREVIEW_CHARS"), 160),
        llm_temperature=_to_float(os.getenv("CHUNK_LLM_TEMPERATURE"), 0.0),
        llm_subchunk_enable=_to_bool(os.getenv("CHUNK_LLM_SUBCHUNK_ENABLE"), True),
        llm_subchunk_size_chars=_to_int(
            os.getenv("CHUNK_LLM_SUBCHUNK_SIZE_CHARS"), 4000
        ),
        llm_subchunk_overlap_chars=_to_int(
            os.getenv("CHUNK_LLM_SUBCHUNK_OVERLAP_CHARS"), 300
        ),
        suffix_rule=os.getenv("CHUNK_OUTPUT_SUFFIX_RULE", "rule").strip(),
        suffix_llm=os.getenv("CHUNK_OUTPUT_SUFFIX_LLM", "llm").strip(),
        suffix_llm_fallback=os.getenv(
            "CHUNK_OUTPUT_SUFFIX_LLM_FALLBACK", "llm_fallback_rule"
        ).strip(),
    )


def validate_rule_cfg(size_chars: int, overlap_chars: int) -> None:
    if size_chars <= 0:
        raise ValueError(f"CHUNK_SIZE_CHARS muss > 0 sein, ist aber: {size_chars}")
    if overlap_chars < 0:
        raise ValueError(
            f"CHUNK_OVERLAP_CHARS muss >= 0 sein, ist aber: {overlap_chars}"
        )
    if overlap_chars >= size_chars:
        raise ValueError(
            f"Ungültig: CHUNK_OVERLAP_CHARS ({overlap_chars}) muss kleiner sein "
            f"als CHUNK_SIZE_CHARS ({size_chars})."
        )


def estimate_expected_chunks(text_len: int, size_chars: int, overlap_chars: int) -> int:
    step = size_chars - overlap_chars
    if step <= 0:
        return 10**9
    return max(1, math.ceil(text_len / step))


def iter_rule_spans_abs(
    text: str,
    abs_start: int,
    abs_end: int,
    size_chars: int,
    overlap_chars: int,
) -> Iterator[dict[str, int]]:
    n = len(text)
    abs_start = max(0, min(abs_start, n))
    abs_end = max(0, min(abs_end, n))
    if abs_end <= abs_start:
        return
    start = abs_start
    sub_idx = 0
    while start < abs_end:
        target_end = min(start + size_chars, abs_end)
        end = target_end
        window_left = max(start, target_end - 800)
        window = text[window_left:target_end]
        idx = window.rfind("\n\n")
        if idx != -1:
            candidate_end = window_left + idx
            if candidate_end > start + 200:
                end = candidate_end
        if end <= start:
            end = target_end
        yield {"char_start": start, "char_end": end, "subchunk_index": sub_idx}
        sub_idx += 1
        if end >= abs_end:
            break
        next_start = end - overlap_chars
        if next_start <= start:
            next_start = end
        start = next_start


def iter_rule_records(text: str, cfg: ChunkConfig) -> Iterator[dict[str, Any]]:
    i = 0
    for s in iter_rule_spans_abs(text, 0, len(text), cfg.size_chars, cfg.overlap_chars):
        yield {
            "chunk_id": i,
            "char_start": s["char_start"],
            "char_end": s["char_end"],
            "chunk_method": "rule",
        }
        i += 1


@dataclass(frozen=True)
class Paragraph:
    idx: int
    char_start: int
    char_end: int
    text: str


def split_into_paragraphs(text: str) -> list[Paragraph]:
    text = normalize_text_minimal(text)
    if not text:
        return []
    parts = re.split(r"\n\s*\n", text)
    paragraphs: list[Paragraph] = []
    cursor = 0
    idx = 0
    for p in parts:
        p = p.strip()
        if not p:
            continue
        start = text.find(p, cursor)
        if start == -1:
            start = cursor
        end = start + len(p)
        paragraphs.append(Paragraph(idx=idx, char_start=start, char_end=end, text=p))
        cursor = end
        idx += 1
    return paragraphs


@dataclass(frozen=True)
class LLMRuntime:
    backend: str
    model: str
    temperature: float
    client: OpenAI


def build_llm_runtime(cfg: ChunkConfig) -> LLMRuntime:
    backend = cfg.llm_backend
    if backend == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        base_url = os.getenv(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        ).strip()
        model = os.getenv("OPENROUTER_MODEL", "").strip()
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY fehlt.")
        if not model:
            raise ValueError("OPENROUTER_MODEL fehlt.")
        site_url = os.getenv("OPENROUTER_SITE_URL", "").strip()
        app_name = os.getenv("OPENROUTER_APP_NAME", "").strip()
        headers: dict[str, str] = {}
        if site_url:
            headers["HTTP-Referer"] = site_url
        if app_name:
            headers["X-Title"] = app_name
        client = OpenAI(
            base_url=base_url, api_key=api_key, default_headers=headers or None
        )
        return LLMRuntime(
            backend=backend, model=model, temperature=cfg.llm_temperature, client=client
        )
    if backend == "ollama":
        base_url = os.getenv(
            "OLLAMA_OPENAI_BASE_URL", "http://localhost:11434/v1"
        ).strip()
        model = os.getenv("OLLAMA_MODEL", "").strip()
        if not model:
            raise ValueError("OLLAMA_MODEL fehlt.")
        client = OpenAI(base_url=base_url, api_key="ollama")
        return LLMRuntime(
            backend=backend, model=model, temperature=cfg.llm_temperature, client=client
        )
    raise ValueError(f"Unbekanntes CHUNK_LLM_BACKEND: {backend}")


def llm_semantic_segments(
    paragraphs: list[Paragraph],
    rt: LLMRuntime,
    preview_chars: int,
) -> Optional[list[dict[str, Any]]]:
    previews = []
    for p in paragraphs:
        prev = p.text.replace("\n", " ").strip()[:preview_chars]
        previews.append({"idx": p.idx, "preview": prev})
    prompt = (
        "Du bekommst Absätze (Paragraphen) aus einer technischen Dokumentation.\n"
        "Aufgabe: Segmentiere die Absätze in semantisch kohärente Abschnitte.\n"
        "Jeder Abschnitt besteht aus zusammenhängenden Absatz-Indizes.\n"
        "Erzeuge zusätzlich pro Abschnitt ein kurzes Thema (2-6 Wörter).\n\n"
        "Regeln:\n"
        "- start_paragraph beginnt bei 0.\n"
        "- end_paragraph ist exklusiv.\n"
        "- Keine Lücken, keine Überschneidungen.\n"
        "- Letzter Abschnitt muss end_paragraph = Anzahl_Paragraphen haben.\n"
        "- Gib ausschließlich JSON zurück (Liste von Objekten).\n\n"
        f"Anzahl_Paragraphen: {len(paragraphs)}\n"
        f"Paragraph-Previews:\n{json.dumps(previews, ensure_ascii=False, indent=2)}\n"
    )
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
    if not isinstance(parsed, list):
        return None
    return parsed


def validate_segments(segments: list[dict[str, Any]], n_paragraphs: int) -> bool:
    if not segments:
        return False
    norm: list[tuple[int, int, str]] = []
    for s in segments:
        try:
            a = int(s.get("start_paragraph"))
            b = int(s.get("end_paragraph"))
            t = str(s.get("topic", "")).strip()
            norm.append((a, b, t))
        except Exception:
            return False
    norm.sort(key=lambda x: x[0])
    if norm[0][0] != 0:
        return False
    if norm[-1][1] != n_paragraphs:
        return False
    prev_end = 0
    for a, b, _ in norm:
        if a != prev_end:
            return False
        if b <= a:
            return False
        if a < 0 or b > n_paragraphs:
            return False
        prev_end = b
    return True



def write_chunks_jsonl(
    out_path: Path,
    source_text_path: Path,
    document_title: str,
    text: str,
    records: Iterator[dict[str, Any]],
    overwrite: bool,
    max_chunks_hard: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        logger.info("Skip (chunks exist): %s", out_path.name)
        return
    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            cid = int(rec["chunk_id"])
            if max_chunks_hard and cid >= max_chunks_hard:
                raise RuntimeError(
                    f"Abbruch: CHUNK_MAX_CHUNKS={max_chunks_hard} erreicht (Schutzmechanismus)."
                )
            start = int(rec["char_start"])
            end = int(rec["char_end"])
            chunk_text = text[start:end].strip()
            out: dict[str, Any] = {
                "chunk_id": cid,
                "char_start": start,
                "char_end": end,
                "text": chunk_text,
                "document_title": document_title,
                "source_text_file": str(source_text_path.resolve()),
            }
            for k, v in rec.items():
                if k in {"chunk_id", "char_start", "char_end"}:
                    continue
                out[k] = v
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            written += 1
    logger.info("Wrote %d chunks -> %s", written, out_path.name)



def iter_llm_hybrid_records(
    text: str,
    paragraphs: list[Paragraph],
    segments: list[dict[str, Any]],
    rt: LLMRuntime,
    cfg: ChunkConfig,
) -> Iterator[dict[str, Any]]:
    chunk_id = 0
    validate_rule_cfg(cfg.llm_subchunk_size_chars, cfg.llm_subchunk_overlap_chars)
    for segment_id, seg in enumerate(segments):
        para_start = int(seg["start_paragraph"])
        para_end = int(seg["end_paragraph"])
        topic = str(seg.get("topic", "")).strip() or None
        seg_char_start = paragraphs[para_start].char_start
        seg_char_end = paragraphs[para_end - 1].char_end
        seg_meta = {
            "chunk_method": "llm_semantic",
            "segment_id": segment_id,
            "para_start": para_start,
            "para_end": para_end,
            "topic": topic,
            "llm_backend": rt.backend,
            "llm_model": rt.model,
            "llm_temperature": rt.temperature,
        }
        if not cfg.llm_subchunk_enable:
            yield {
                "chunk_id": chunk_id,
                "char_start": seg_char_start,
                "char_end": seg_char_end,
                **seg_meta,
                "chunk_granularity": "segment",
            }
            chunk_id += 1
            continue
        for sub in iter_rule_spans_abs(
            text=text,
            abs_start=seg_char_start,
            abs_end=seg_char_end,
            size_chars=cfg.llm_subchunk_size_chars,
            overlap_chars=cfg.llm_subchunk_overlap_chars,
        ):
            yield {
                "chunk_id": chunk_id,
                "char_start": sub["char_start"],
                "char_end": sub["char_end"],
                **seg_meta,
                "chunk_granularity": "subchunk",
                "subchunk_index": sub["subchunk_index"],
                "subchunk_method": "rule",
                "subchunk_size_chars": cfg.llm_subchunk_size_chars,
                "subchunk_overlap_chars": cfg.llm_subchunk_overlap_chars,
            }
            chunk_id += 1


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Chunk extracted documents into semantic units (rule or LLM hybrid)."
    )
    p.add_argument(
        "--file",
        type=str,
        default="",
        help="Optional: only process one .txt file (name or path)",
    )
    p.add_argument("--mode", type=str, default="", help="Override CHUNK_MODE: rule|llm")
    p.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing outputs"
    )
    return p


def out_path_for(document_title: str, cfg: ChunkConfig, variant: str) -> Path:
    # variant: rule | llm | llm_fallback_rule
    if variant == "rule":
        suffix = cfg.suffix_rule
    elif variant == "llm":
        suffix = cfg.suffix_llm
    else:
        suffix = cfg.suffix_llm_fallback
    return cfg.out_dir / f"{document_title}.{suffix}.chunks.jsonl"


def run() -> None:
    settings = load_settings()
    cfg = load_chunk_config()
    _setup_logging(settings.log_level)
    validate_rule_cfg(cfg.size_chars, cfg.overlap_chars)
    mode = cfg.mode
    args = build_argparser().parse_args()
    if args.mode.strip():
        mode = args.mode.strip().lower()
    processed_dir = settings.data_processed_dir
    txt_files = list_processed_texts(processed_dir)
    if args.file:
        one = Path(args.file)
        if not one.is_absolute():
            one = (processed_dir / one).resolve()
        txt_files = [one]
    if not txt_files:
        logger.warning("No .txt files found in: %s", processed_dir.resolve())
        return
    logger.info(
        "Chunk config: mode=%s out=%s size=%d overlap=%d backend=%s subchunk=%s",
        mode,
        str(cfg.out_dir),
        cfg.size_chars,
        cfg.overlap_chars,
        cfg.llm_backend,
        str(cfg.llm_subchunk_enable),
    )
    for txt_path in txt_files:
        if not txt_path.exists():
            logger.error("File not found: %s", str(txt_path))
            continue
        file_size_mb = txt_path.stat().st_size / (1024 * 1024)
        logger.info("Processing: %s (%.2f MB)", txt_path.name, file_size_mb)
        text = txt_path.read_text(encoding="utf-8", errors="replace")
        text = normalize_text_minimal(text)
        if cfg.max_doc_chars and len(text) > cfg.max_doc_chars:
            text = text[: cfg.max_doc_chars]
            logger.info("Truncated doc to %d chars (CHUNK_MAX_DOC_CHARS).", len(text))
        if not text:
            logger.warning("Empty text: %s", txt_path.name)
            continue
        document_title = txt_path.stem
        expected_rule = estimate_expected_chunks(
            len(text), cfg.size_chars, cfg.overlap_chars
        )
        max_chunks_hard = (
            cfg.max_chunks if cfg.max_chunks > 0 else max(2000, expected_rule * 10)
        )

        if mode == "llm":
            paragraphs = split_into_paragraphs(text)
            logger.info("Paragraphs detected: %d", len(paragraphs))
            if len(paragraphs) == 0:
                logger.warning("No paragraphs found -> fallback to rule.")
            elif len(paragraphs) > cfg.llm_max_paragraphs:
                logger.warning(
                    "Too many paragraphs for LLM (%d > %d) -> fallback to rule.",
                    len(paragraphs),
                    cfg.llm_max_paragraphs,
                )
            else:
                try:
                    rt = build_llm_runtime(cfg)
                    segments = llm_semantic_segments(
                        paragraphs, rt, cfg.llm_preview_chars
                    )
                    if segments is None:
                        logger.warning(
                            "LLM segmentation returned no JSON -> fallback to rule."
                        )
                    elif not validate_segments(segments, len(paragraphs)):
                        logger.warning("LLM segmentation invalid -> fallback to rule.")
                    else:
                        out_path = out_path_for(document_title, cfg, "llm")
                        records = iter_llm_hybrid_records(
                            text, paragraphs, segments, rt, cfg
                        )
                        write_chunks_jsonl(
                            out_path=out_path,
                            source_text_path=txt_path,
                            document_title=document_title,
                            text=text,
                            records=records,
                            overwrite=(args.overwrite or cfg.overwrite),
                            max_chunks_hard=max_chunks_hard,
                        )
                        continue
                except Exception as e:
                    logger.exception(
                        "LLM chunking failed (%s) -> fallback to rule.", str(e)
                    )
            # LLM Mode, aber fallback genutzt
            out_path = out_path_for(document_title, cfg, "llm_fallback_rule")
            records = iter_rule_records(text, cfg)
            write_chunks_jsonl(
                out_path=out_path,
                source_text_path=txt_path,
                document_title=document_title,
                text=text,
                records=records,
                overwrite=(args.overwrite or cfg.overwrite),
                max_chunks_hard=max_chunks_hard,
            )
            continue

        logger.info("Rule-based chunking: expected ~%d chunks", expected_rule)
        out_path = out_path_for(document_title, cfg, "rule")
        records = iter_rule_records(text, cfg)
        write_chunks_jsonl(
            out_path=out_path,
            source_text_path=txt_path,
            document_title=document_title,
            text=text,
            records=records,
            overwrite=(args.overwrite or cfg.overwrite),
            max_chunks_hard=max_chunks_hard,
        )


if __name__ == "__main__":
    run()
