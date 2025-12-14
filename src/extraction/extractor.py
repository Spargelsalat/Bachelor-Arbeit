import json
from pathlib import Path
from typing import Dict, List
from loguru import logger
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.llm.model_factory import get_llm
from src.extraction.prompts import EXTRACTION_PROMPT
from config.settings import EXTRACTION_TEMPERATURE


def format_schema_for_prompt(schema: Dict) -> tuple:
    entity_types = ", ".join([et["name"] for et in schema["entity_types"]])
    relation_types = ", ".join([rt["name"] for rt in schema["relation_types"]])
    return entity_types, relation_types


def extract_from_chunk(chunk: Dict, schema: Dict, model_name: str) -> Dict:
    entity_types, relation_types = format_schema_for_prompt(schema)
    prompt = EXTRACTION_PROMPT.format(
        entity_types=entity_types, relation_types=relation_types, text=chunk["text"]
    )
    llm = get_llm(model_name, temperature=EXTRACTION_TEMPERATURE)
    try:
        response = llm.invoke(prompt)
        response_text = response.content.strip()
        # JSON extrahieren falls Markdown drumrum ist
        if "```" in response_text:
            parts = response_text.split("```")
            for part in parts:
                part_stripped = part.strip()
                if part_stripped.startswith("json"):
                    response_text = part_stripped[4:]
                    break
                elif part_stripped.startswith("{"):
                    response_text = part_stripped
                    break
        result = json.loads(response_text)
        # Metadaten hinzufuegen
        for entity in result.get("entities", []):
            entity["source_chunk"] = chunk["chunk_id"]
            entity["source_document"] = chunk["source_document"]
        for relation in result.get("relations", []):
            relation["source_chunk"] = chunk["chunk_id"]
            relation["source_document"] = chunk["source_document"]
        return result
    except json.JSONDecodeError as e:
        logger.warning(
            "JSON-Parsing fehlgeschlagen fuer Chunk "
            + str(chunk["chunk_id"])
            + ": "
            + str(e)
        )
        return {"entities": [], "relations": []}
    except Exception as e:
        logger.error(
            "Extraktion fehlgeschlagen fuer Chunk "
            + str(chunk["chunk_id"])
            + ": "
            + str(e)
        )
        return {"entities": [], "relations": []}


def extract_from_document(
    chunks: List[Dict], schema: Dict, model_name: str, show_progress: bool = True
) -> Dict:
    all_entities = []
    all_relations = []
    iterator = tqdm(chunks, desc="Extrahiere") if show_progress else chunks
    for chunk in iterator:
        result = extract_from_chunk(chunk, schema, model_name)
        all_entities.extend(result.get("entities", []))
        all_relations.extend(result.get("relations", []))
    logger.info(
        "Extraktion abgeschlossen: "
        + str(len(all_entities))
        + " Entitaeten, "
        + str(len(all_relations))
        + " Relationen"
    )
    return {"entities": all_entities, "relations": all_relations}


if __name__ == "__main__":
    from config.settings import RAW_DIR, DEFAULT_MODEL
    from src.preprocessing.document_loader import load_all_pdfs
    from src.preprocessing.chunker import chunk_document
    from src.schema.schema_generator import load_schema

    docs = load_all_pdfs(RAW_DIR)
    if not docs:
        print("Keine Dokumente gefunden!")
        exit()
    doc = docs[0]
    schema = load_schema()
    if not schema:
        print("Kein Schema gefunden! Erst schema_generator.py ausfuehren.")
        exit()
    # Chunken
    chunks = chunk_document(doc, use_semantic=False)
    # Nur erste 3 Chunks fuer Test
    test_chunks = chunks[:3]
    print("Teste Extraktion mit " + str(len(test_chunks)) + " Chunks...")
    result = extract_from_document(
        test_chunks, schema, DEFAULT_MODEL, show_progress=True
    )
    print("\nErgebnisse:")
    print("  Entitaeten: " + str(len(result["entities"])))
    print("  Relationen: " + str(len(result["relations"])))
    if result["entities"]:
        print("\nBeispiel-Entitaeten:")
        for e in result["entities"][:5]:
            print("  - " + e["name"] + " (" + e["type"] + ")")
    if result["relations"]:
        print("\nBeispiel-Relationen:")
        for r in result["relations"][:5]:
            print(
                "  - " + r["subject"] + " --[" + r["predicate"] + "]--> " + r["object"]
            )
