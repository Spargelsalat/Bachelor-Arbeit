from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv


def _to_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    value = value.strip()
    if value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _to_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    value = value.strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Settings:
    data_raw_dir: Path
    data_processed_dir: Path
    pdf_extract_engine: str
    pdf_max_pages: int
    pdf_min_chars: int
    pdf_overwrite: bool
    log_level: str


def load_settings() -> Settings:
    # .env wird im Projektroot erwartet
    load_dotenv()
    raw_dir = Path(os.getenv("DATA_RAW_DIR", "data/raw"))
    processed_dir = Path(os.getenv("DATA_PROCESSED_DIR", "data/processed"))
    engine = os.getenv("PDF_EXTRACT_ENGINE", "pdfplumber").strip().lower()
    max_pages = _to_int(os.getenv("PDF_MAX_PAGES"), 0)
    min_chars = _to_int(os.getenv("PDF_MIN_CHARS"), 0)
    overwrite = _to_bool(os.getenv("PDF_OVERWRITE"), False)
    log_level = os.getenv("LOG_LEVEL", "INFO").strip().upper()
    return Settings(
        data_raw_dir=raw_dir,
        data_processed_dir=processed_dir,
        pdf_extract_engine=engine,
        pdf_max_pages=max_pages,
        pdf_min_chars=min_chars,
        pdf_overwrite=overwrite,
        log_level=log_level,
    )
