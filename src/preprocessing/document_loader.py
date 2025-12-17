from __future__ import annotations
import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

# Falls du das Skript direkt ausführst: python src/preprocessing/document_loader.py
# Dann muss der Projektroot in den Path, damit "config.settings" gefunden wird.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from config.settings import load_settings  # noqa: E402

logger = logging.getLogger(__name__)


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def list_pdfs(raw_dir: Path) -> list[Path]:
    raw_dir = raw_dir.resolve()
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")
    pdfs = sorted(raw_dir.glob("**/*.pdf"))
    pdfs += sorted(raw_dir.glob("**/*.PDF"))
    # Duplikate entfernen (falls beides matched)
    pdfs = sorted(set(pdfs))
    return pdfs


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_text_pdfplumber(
    pdf_path: Path, max_pages: int = 0, min_chars: int = 0
) -> tuple[str, dict]:
    import pdfplumber  # local import, damit das Modul nicht crasht wenn dependency fehlt

    pages_used = 0
    pages_total = 0
    parts: list[str] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        pages_total = len(pdf.pages)
        limit = pages_total if max_pages <= 0 else min(max_pages, pages_total)
        for i in range(limit):
            page = pdf.pages[i]
            text = page.extract_text() or ""
            text = text.strip()
            if min_chars > 0 and len(text) < min_chars:
                continue
            parts.append(text)
            pages_used += 1
    meta = {
        "engine": "pdfplumber",
        "pages_total": pages_total,
        "pages_used": pages_used,
        "max_pages": max_pages,
        "min_chars": min_chars,
    }
    return "\n\n".join(parts).strip(), meta


def extract_text_pypdf(
    pdf_path: Path, max_pages: int = 0, min_chars: int = 0
) -> tuple[str, dict]:
    from pypdf import PdfReader  # local import

    reader = PdfReader(str(pdf_path))
    pages_total = len(reader.pages)
    limit = pages_total if max_pages <= 0 else min(max_pages, pages_total)
    pages_used = 0
    parts: list[str] = []
    for i in range(limit):
        page = reader.pages[i]
        text = page.extract_text() or ""
        text = text.strip()
        if min_chars > 0 and len(text) < min_chars:
            continue
        parts.append(text)
        pages_used += 1
    meta = {
        "engine": "pypdf",
        "pages_total": pages_total,
        "pages_used": pages_used,
        "max_pages": max_pages,
        "min_chars": min_chars,
    }
    return "\n\n".join(parts).strip(), meta


def extract_text(
    pdf_path: Path, engine: str, max_pages: int, min_chars: int
) -> tuple[str, dict]:
    engine = engine.lower().strip()
    if engine == "pdfplumber":
        try:
            return extract_text_pdfplumber(
                pdf_path, max_pages=max_pages, min_chars=min_chars
            )
        except Exception as e:
            logger.warning(
                "pdfplumber failed for %s (%s). Fallback to pypdf.",
                pdf_path.name,
                str(e),
            )
            return extract_text_pypdf(
                pdf_path, max_pages=max_pages, min_chars=min_chars
            )
    if engine == "pypdf":
        try:
            return extract_text_pypdf(
                pdf_path, max_pages=max_pages, min_chars=min_chars
            )
        except Exception as e:
            logger.warning(
                "pypdf failed for %s (%s). Fallback to pdfplumber.",
                pdf_path.name,
                str(e),
            )
            return extract_text_pdfplumber(
                pdf_path, max_pages=max_pages, min_chars=min_chars
            )
    raise ValueError(f"Unknown PDF_EXTRACT_ENGINE: {engine}")


def safe_stem(path: Path) -> str:
    # Dateiname stabil und windows-safe
    stem = path.stem
    keep = []
    for ch in stem:
        if ch.isalnum() or ch in {"-", "_", ".", " "}:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip().replace("  ", " ")


from datetime import datetime, timezone

...


def write_outputs(
    pdf_path: Path, processed_dir: Path, text: str, meta: dict, overwrite: bool
) -> tuple[Path, Path]:
    processed_dir.mkdir(parents=True, exist_ok=True)
    base = safe_stem(pdf_path)
    out_txt = processed_dir / f"{base}.txt"
    out_meta = processed_dir / f"{base}.meta.json"
    if not overwrite and out_txt.exists() and out_meta.exists():
        logger.info("Skip (already processed): %s", pdf_path.name)
        return out_txt, out_meta
    extracted_at = (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )
    meta_full = {
        "source_file": str(pdf_path.resolve()),
        "source_file_name": pdf_path.name,
        "source_sha256": file_sha256(pdf_path),
        "extracted_at": extracted_at,
        **meta,
        "text_chars": len(text),
    }
    out_txt.write_text(text, encoding="utf-8", errors="replace")
    out_meta.write_text(
        json.dumps(meta_full, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Processed: %s -> %s", pdf_path.name, out_txt.name)
    return out_txt, out_meta


def process_all_pdfs(
    raw_dir: Path,
    processed_dir: Path,
    engine: str,
    max_pages: int,
    min_chars: int,
    overwrite: bool,
) -> None:
    pdfs = list_pdfs(raw_dir)
    if not pdfs:
        logger.warning("No PDFs found in: %s", raw_dir.resolve())
        return
    logger.info("Found %d PDF(s) in %s", len(pdfs), raw_dir.resolve())
    for pdf_path in pdfs:
        try:
            text, meta = extract_text(
                pdf_path, engine=engine, max_pages=max_pages, min_chars=min_chars
            )
            if not text:
                logger.warning("Empty text after extraction: %s", pdf_path.name)
            write_outputs(pdf_path, processed_dir, text, meta, overwrite=overwrite)
        except Exception as e:
            logger.exception("Failed to process %s (%s)", pdf_path.name, str(e))


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract text from all PDFs in DATA_RAW_DIR."
    )
    p.add_argument(
        "--raw", type=str, default="", help="Override raw dir (default: from .env)"
    )
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="Override processed dir (default: from .env)",
    )
    p.add_argument(
        "--engine",
        type=str,
        default="",
        help="Override engine: pdfplumber|pypdf (default: from .env)",
    )
    p.add_argument(
        "--max-pages",
        type=int,
        default=-1,
        help="Override max pages (0=all). -1 uses .env",
    )
    p.add_argument(
        "--min-chars",
        type=int,
        default=-1,
        help="Override min chars per page. -1 uses .env",
    )
    p.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing outputs"
    )
    return p


if __name__ == "__main__":
    settings = load_settings()
    _setup_logging(settings.log_level)
    args = build_argparser().parse_args()
    raw_dir = Path(args.raw) if args.raw else settings.data_raw_dir
    out_dir = Path(args.out) if args.out else settings.data_processed_dir
    engine = args.engine if args.engine else settings.pdf_extract_engine
    max_pages = settings.pdf_max_pages if args.max_pages == -1 else args.max_pages
    min_chars = settings.pdf_min_chars if args.min_chars == -1 else args.min_chars
    overwrite = True if args.overwrite else settings.pdf_overwrite
    process_all_pdfs(
        raw_dir=raw_dir,
        processed_dir=out_dir,
        engine=engine,
        max_pages=max_pages,
        min_chars=min_chars,
        overwrite=overwrite,
    )
