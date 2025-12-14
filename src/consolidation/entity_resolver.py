import json
from pathlib import Path
from typing import Dict, List, Tuple
from loguru import logger
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.llm.model_factory import get_llm
from src.extraction.prompts import ENTITY_RESOLUTION_PROMPT


def simple_match(name1: str, name2: str) -> bool:
    """
    Einfacher String-Vergleich fuer offensichtliche Matches.
    Schneller als LLM und fuer die meisten Faelle ausreichend.
    """
    n1 = name1.lower().strip()
    n2 = name2.lower().strip()
    # Exakter Match
    if n1 == n2:
        return True
    # Einer ist Substring des anderen
    if n1 in n2 or n2 in n1:
        return True
    return False


def llm_resolve(entity1: Dict, entity2: Dict, model_name: str) -> Tuple[bool, float]:
    """
    LLM-basierte Entscheidung ob zwei Entitaeten dieselbe sind.
    Wird nur aufgerufen wenn simple_match nicht eindeutig ist.
    """
    prompt = ENTITY_RESOLUTION_PROMPT.format(
        entity1_name=entity1["name"],
        entity1_type=entity1["type"],
        entity1_context=entity1.get("text_evidence", ""),
        entity2_name=entity2["name"],
        entity2_type=entity2["type"],
        entity2_context=entity2.get("text_evidence", ""),
    )
    llm = get_llm(model_name, temperature=0.1)
    try:
        response = llm.invoke(prompt)
        response_text = response.content.strip()
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
        return result.get("same_entity", False), result.get("confidence", 0.5)
    except Exception as e:
        logger.warning("LLM Resolution fehlgeschlagen: " + str(e))
        return False, 0.0


def resolve_entities(
    entities: List[Dict], model_name: str = None, use_llm: bool = True
) -> List[Dict]:
    """
    Fuehrt Entity Resolution durch - findet und merged doppelte Entitaeten.
    Strategie:
    1. Gruppiere nach Typ (nur gleiche Typen koennen Duplikate sein)
    2. Innerhalb jeder Gruppe: simple_match fuer offensichtliche Faelle
    3. Bei Unsicherheit: LLM fragen (wenn use_llm=True)
    """
    if not entities:
        return []
    logger.info("Starte Entity Resolution fuer " + str(len(entities)) + " Entitaeten")
    # Nach Typ gruppieren
    by_type = {}
    for entity in entities:
        entity_type = entity.get("type", "unknown")
        if entity_type not in by_type:
            by_type[entity_type] = []
        by_type[entity_type].append(entity)
    resolved = []
    merge_count = 0
    for entity_type, type_entities in by_type.items():
        # Cluster von zusammengehoerigen Entitaeten bilden
        clusters = []
        for entity in type_entities:
            matched_cluster = None
            for cluster in clusters:
                representative = cluster[0]
                # Erst einfachen Match versuchen
                if simple_match(entity["name"], representative["name"]):
                    matched_cluster = cluster
                    break
                # Bei Bedarf LLM fragen
                if use_llm and model_name:
                    is_same, confidence = llm_resolve(
                        entity, representative, model_name
                    )
                    if is_same and confidence > 0.7:
                        matched_cluster = cluster
                        break
            if matched_cluster:
                matched_cluster.append(entity)
                merge_count += 1
            else:
                clusters.append([entity])
        # Cluster zu einzelnen Entitaeten mergen
        for cluster in clusters:
            if len(cluster) == 1:
                resolved.append(cluster[0])
            else:
                merged = merge_entities(cluster)
                resolved.append(merged)
    logger.info(
        "Entity Resolution fertig: "
        + str(len(entities))
        + " -> "
        + str(len(resolved))
        + " Entitaeten ("
        + str(merge_count)
        + " gemerged)"
    )
    return resolved


def merge_entities(entities: List[Dict]) -> Dict:
    """
    Merged mehrere Entitaeten zu einer.
    Kombiniert Aliase, Quellen etc.
    """
    if len(entities) == 1:
        return entities[0]
    # Nimm den laengsten Namen als Hauptnamen (oft am spezifischsten)
    main_entity = max(entities, key=lambda e: len(e.get("name", "")))
    # Sammle alle Aliase
    all_names = set()
    all_mentions = []
    all_sources = set()
    for entity in entities:
        all_names.add(entity.get("name", ""))
        all_mentions.extend(entity.get("mentions", []))
        if "source_document" in entity:
            all_sources.add(entity["source_document"])
    merged = {
        "name": main_entity["name"],
        "type": main_entity["type"],
        "aliases": list(all_names - {main_entity["name"]}),
        "mentions": list(set(all_mentions)),
        "source_documents": list(all_sources),
        "merged_from": len(entities),
    }
    if "text_evidence" in main_entity:
        merged["text_evidence"] = main_entity["text_evidence"]
    return merged


if __name__ == "__main__":
    # Kleiner Test
    test_entities = [
        {
            "name": "Assembly Control",
            "type": "module",
            "text_evidence": "The Assembly Control module...",
        },
        {
            "name": "assembly control",
            "type": "module",
            "text_evidence": "In assembly control you can...",
        },
        {"name": "ASC", "type": "module", "text_evidence": "ASC handles..."},
        {
            "name": "Production Planning",
            "type": "module",
            "text_evidence": "Production Planning is...",
        },
    ]
    print("Vor Resolution: " + str(len(test_entities)) + " Entitaeten")
    resolved = resolve_entities(test_entities, use_llm=False)
    print("Nach Resolution: " + str(len(resolved)) + " Entitaeten")
    for e in resolved:
        print("  - " + e["name"] + " (aliases: " + str(e.get("aliases", [])) + ")")
