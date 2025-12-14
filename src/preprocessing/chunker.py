from typing import List, Dict
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
import json
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP
from src.llm.model_factory import get_llm


def chunk_document_simple(
    document: Dict, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> List[Dict]:
    """
    Regelbasiertes Chunking - schnell und zuverlaessig.
    Splitet nach Absaetzen, dann nach Saetzen wenn noetig.
    """
    text = document["full_text"]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    text_chunks = splitter.split_text(text)
    chunks = []
    for i, chunk_text in enumerate(text_chunks):
        chunks.append(
            {
                "chunk_id": i,
                "text": chunk_text,
                "source_document": document["filename"],
                "method": "rule-based",
            }
        )
    logger.info("  " + str(len(chunks)) + " Chunks erstellt (regelbasiert)")
    return chunks


def chunk_document_semantic(document: Dict, model_name: str) -> List[Dict]:
    """
    LLM-gestuetztes semantisches Chunking.
    Das Modell entscheidet wo thematische Grenzen sind.
    Fallback auf regelbasiert wenn was schiefgeht.
    """
    text = document["full_text"]
    if len(text) < 3000:
        logger.info("  Text zu kurz fuer semantisches Chunking, nutze regelbasiert")
        return chunk_document_simple(document)
    try:
        llm = get_llm(model_name, temperature=0.1)
        sample_text = text[:8000] if len(text) > 8000 else text
        prompt = (
            """Analysiere diesen Text und identifiziere die logischen Abschnittsgrenzen.
Text:
"""
            + sample_text
            + """
Aufgabe: Finde die Stellen wo ein neuer thematischer Abschnitt beginnt.
Gib fuer jeden Abschnitt zurueck:
- Die ersten 5-10 Worte des Abschnitts (damit ich die Stelle finden kann)
- Eine kurze thematische Beschreibung (max 10 Worte)
Antwort als JSON-Array:
[
    {"start_text": "Die ersten Worte...", "topic": "Kurze Beschreibung"},
    ...
]
Nur das JSON, keine Erklaerung."""
        )
        response = llm.invoke(prompt)
        response_text = response.content.strip()
        if "```" in response_text:
            parts = response_text.split("```")
            for part in parts:
                part_stripped = part.strip()
                if part_stripped.startswith("json"):
                    response_text = part_stripped[4:]
                    break
                elif part_stripped.startswith("["):
                    response_text = part_stripped
                    break
        boundaries = json.loads(response_text)
        if not boundaries or len(boundaries) < 2:
            logger.warning("  LLM hat zu wenige Grenzen gefunden, nutze regelbasiert")
            return chunk_document_simple(document)
        chunks = []
        current_pos = 0
        for i, boundary in enumerate(boundaries):
            start_text = boundary.get("start_text", "")
            topic = boundary.get("topic", "")
            pos = text.find(start_text[:30], current_pos)
            if pos == -1 and i > 0:
                continue
            if pos > current_pos and i > 0:
                chunk_text = text[current_pos:pos].strip()
                if chunk_text:
                    prev_topic = boundaries[i - 1].get("topic", "") if i > 0 else ""
                    chunks.append(
                        {
                            "chunk_id": len(chunks),
                            "text": chunk_text,
                            "topic": prev_topic,
                            "source_document": document["filename"],
                            "method": "semantic",
                        }
                    )
            current_pos = pos if pos != -1 else current_pos
        if current_pos < len(text):
            remaining = text[current_pos:].strip()
            if remaining:
                last_topic = boundaries[-1].get("topic", "") if boundaries else ""
                chunks.append(
                    {
                        "chunk_id": len(chunks),
                        "text": remaining,
                        "topic": last_topic,
                        "source_document": document["filename"],
                        "method": "semantic",
                    }
                )
        if len(chunks) < 2:
            logger.warning("  Semantisches Chunking fehlgeschlagen, nutze regelbasiert")
            return chunk_document_simple(document)
        logger.info("  " + str(len(chunks)) + " Chunks erstellt (semantisch)")
        return chunks
    except Exception as e:
        logger.warning("  Fehler beim semantischen Chunking: " + str(e))
        logger.info("  Fallback auf regelbasiertes Chunking")
        return chunk_document_simple(document)


def chunk_document(
    document: Dict, model_name: str = None, use_semantic: bool = True
) -> List[Dict]:
    """
    Hauptfunktion zum Chunking.
    """
    logger.info("Chunke: " + document["filename"])
    if use_semantic and model_name:
        return chunk_document_semantic(document, model_name)
    else:
        return chunk_document_simple(document)


if __name__ == "__main__":
    from config.settings import RAW_DIR, DEFAULT_MODEL
    from src.preprocessing.document_loader import load_all_pdfs

    docs = load_all_pdfs(RAW_DIR)
    if docs:
        doc = docs[0]
        print("\nRegelbasiertes Chunking:")
        chunks_simple = chunk_document(doc, use_semantic=False)
        print("  Chunks: " + str(len(chunks_simple)))
        print("\nErster Chunk:")
        print(chunks_simple[0]["text"][:300] + "...")
