"""
Hauptpipeline: Von PDF zu Wissensgraph
"""

import json
from pathlib import Path
from datetime import datetime
from loguru import logger
from config.settings import (
    RAW_DIR,
    PROCESSED_DIR,
    OUTPUT_DIR,
    DEFAULT_MODEL,
    LOCAL_MODEL,
    CONFIDENCE_THRESHOLD,
)
from src.preprocessing.document_loader import load_all_pdfs
from src.preprocessing.chunker import chunk_document
from src.schema.schema_generator import generate_schema, save_schema, load_schema
from src.extraction.extractor import extract_from_document
from src.consolidation.entity_resolver import resolve_entities
from src.consolidation.validator import filter_by_confidence
from src.graph.neo4j_handler import Neo4jHandler


def run_pipeline(
    model_name: str = None,
    use_semantic_chunking: bool = False,
    use_entity_resolution: bool = True,
    max_chunks: int = None,
    clear_graph: bool = True,
):
    """
    Fuehrt die komplette Pipeline aus.

    Args:
        model_name: LLM zu verwenden (default aus config)
        use_semantic_chunking: LLM-basiertes Chunking statt regelbasiert
        use_entity_resolution: Duplikate zusammenfuehren
        max_chunks: Limit fuer Tests (None = alle)
        clear_graph: Datenbank vor Import leeren
    """
    model = model_name or DEFAULT_MODEL
    logger.info("=== Pipeline Start ===")
    logger.info("Modell: " + model)
    # 1. Dokumente laden
    logger.info("\n--- Schritt 1: Dokumente laden ---")
    documents = load_all_pdfs(RAW_DIR)
    if not documents:
        logger.error("Keine Dokumente gefunden in " + str(RAW_DIR))
        return None
    # 2. Schema laden oder generieren
    logger.info("\n--- Schritt 2: Schema ---")
    schema = load_schema()
    if not schema:
        logger.info("Generiere neues Schema...")
        schema = generate_schema(documents, model)
        save_schema(schema)
    else:
        logger.info("Bestehendes Schema geladen")
    # 3. Chunking
    logger.info("\n--- Schritt 3: Chunking ---")
    all_chunks = []
    for doc in documents:
        if use_semantic_chunking:
            chunks = chunk_document(doc, model_name=model, use_semantic=True)
        else:
            chunks = chunk_document(doc, use_semantic=False)
        all_chunks.extend(chunks)
    if max_chunks:
        logger.info("Limitiere auf " + str(max_chunks) + " Chunks (Test-Modus)")
        all_chunks = all_chunks[:max_chunks]
    logger.info("Gesamt: " + str(len(all_chunks)) + " Chunks")
    # 4. Extraktion
    logger.info("\n--- Schritt 4: Extraktion ---")
    extractions = extract_from_document(all_chunks, schema, model)
    # 5. Konsolidierung
    logger.info("\n--- Schritt 5: Konsolidierung ---")
    # Konfidenz-Filter
    extractions = filter_by_confidence(extractions, CONFIDENCE_THRESHOLD)
    # Entity Resolution
    if use_entity_resolution:
        extractions["entities"] = resolve_entities(
            extractions["entities"],
            model_name=model,
            use_llm=False,  # Erstmal ohne LLM fuer Speed
        )
    logger.info(
        "Nach Konsolidierung: "
        + str(len(extractions["entities"]))
        + " Entitaeten, "
        + str(len(extractions["relations"]))
        + " Relationen"
    )
    # 6. Graph Import
    logger.info("\n--- Schritt 6: Graph Import ---")
    handler = Neo4jHandler()
    if handler.connect():
        result = handler.import_extractions(extractions, clear_first=clear_graph)
        stats = handler.get_stats()
        logger.info("Graph-Statistiken:")
        logger.info("  Knoten: " + str(stats["total_nodes"]))
        logger.info("  Relationen: " + str(stats["total_relations"]))
        handler.close()
    else:
        logger.error("Konnte nicht mit Neo4j verbinden!")
        result = None
    # 7. Ergebnisse speichern
    logger.info("\n--- Schritt 7: Ergebnisse speichern ---")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / ("extraction_" + timestamp + ".json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(extractions, f, indent=2, ensure_ascii=False)
    logger.info("Ergebnisse gespeichert: " + str(output_file))
    logger.info("\n=== Pipeline Ende ===")
    return extractions


def quick_test(num_chunks: int = 5):
    """Schneller Test mit wenigen Chunks."""
    logger.info("Quick Test mit " + str(num_chunks) + " Chunks")
    return run_pipeline(
        max_chunks=num_chunks,
        use_semantic_chunking=False,
        use_entity_resolution=True,
        clear_graph=True,
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Quick Test
        result = quick_test(5)
    else:
        # Volle Pipeline
        result = run_pipeline(
            use_semantic_chunking=False,
            use_entity_resolution=True,
            max_chunks=20,  # Erstmal limitiert
            clear_graph=True,
        )
    if result:
        print("\nFertig!")
        print("Entitaeten: " + str(len(result["entities"])))
        print("Relationen: " + str(len(result["relations"])))
