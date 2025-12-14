"""
Lädt PDFs und extrahiert den Text daraus.
Klingt simpel, aber PDFs können echt fies sein mit ihrem Layout.
"""

from pathlib import Path
from typing import List, Dict
import pdfplumber
from loguru import logger


def load_pdf(file_path: Path) -> Dict:
    """
    Extrahiert Text aus einer PDF-Datei.
    Gibt ein Dict mit Metadaten und Text zurück.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"PDF nicht gefunden: {file_path}")

    logger.info(f"Lade PDF: {file_path.name}")

    pages = []
    full_text = []

    with pdfplumber.open(file_path) as pdf:
        total_pages = len(pdf.pages)

        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""

            # Grundlegende Bereinigung
            text = clean_text(text)

            if text.strip():  # Nur nicht-leere Seiten
                pages.append({"page_number": i + 1, "text": text})
                full_text.append(text)

            # Fortschritt loggen bei großen PDFs
            if (i + 1) % 50 == 0:
                logger.info(f"  Seite {i + 1}/{total_pages} verarbeitet")

    logger.info(f"  {len(pages)} Seiten mit Text extrahiert")

    return {
        "filename": file_path.name,
        "filepath": str(file_path),
        "total_pages": total_pages,
        "pages_with_text": len(pages),
        "pages": pages,
        "full_text": "\n\n".join(full_text),
    }


def clean_text(text: str) -> str:
    """
    Bereinigt den extrahierten Text.
    PDFs haben oft komische Zeilenumbrüche und Artefakte.
    """
    # Mehrfache Leerzeichen reduzieren
    import re

    text = re.sub(r" +", " ", text)

    # Mehrfache Zeilenumbrüche reduzieren
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Bindestriche am Zeilenende (Silbentrennung) zusammenführen
    text = re.sub(r"-\n", "", text)

    return text.strip()


def load_all_pdfs(directory: Path) -> List[Dict]:
    """
    Lädt alle PDFs aus einem Verzeichnis.
    """
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Verzeichnis nicht gefunden: {directory}")

    # Case-insensitive PDF-Suche (Windows-kompatibel)
    pdf_files = [f for f in directory.iterdir() if f.suffix.lower() == ".pdf"]

    if not pdf_files:
        logger.warning(f"Keine PDFs gefunden in: {directory}")
        return []

    logger.info(f"Gefunden: {len(pdf_files)} PDF(s)")

    documents = []
    for pdf_path in pdf_files:
        try:
            doc = load_pdf(pdf_path)
            documents.append(doc)
        except Exception as e:
            logger.error(f"Fehler bei {pdf_path.name}: {e}")

    return documents


# Test
if __name__ == "__main__":
    import sys

    sys.path.append(str(Path(__file__).parent.parent.parent))
    from config.settings import RAW_DIR

    docs = load_all_pdfs(RAW_DIR)

    for doc in docs:
        print(f"\n{doc['filename']}:")
        print(f"  Seiten: {doc['total_pages']}")
        print(f"  Mit Text: {doc['pages_with_text']}")
        print(f"  Zeichen gesamt: {len(doc['full_text'])}")
