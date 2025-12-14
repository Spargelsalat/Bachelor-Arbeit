import json
from pathlib import Path
from typing import Dict, List
from loguru import logger
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.llm.model_factory import get_llm
from src.extraction.prompts import VALIDATION_PROMPT
from config.settings import CONFIDENCE_THRESHOLD


def validate_relation(relation: Dict, model_name: str) -> Dict:
    """
    Validiert eine einzelne Relation mit dem LLM.
    Prueft ob das Tripel tatsaechlich im Text belegt ist.
    """
    prompt = VALIDATION_PROMPT.format(
        subject=relation.get("subject", ""),
        predicate=relation.get("predicate", ""),
        object=relation.get("object", ""),
        text_evidence=relation.get("text_evidence", "Kein Textbeleg vorhanden"),
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
        relation["validated"] = result.get("valid", False)
        relation["validation_confidence"] = result.get("confidence", 0.5)
        relation["validation_issues"] = result.get("issues", [])
        return relation
    except Exception as e:
        logger.warning("Validierung fehlgeschlagen: " + str(e))
        relation["validated"] = None
        relation["validation_confidence"] = 0.0
        relation["validation_issues"] = ["Validierung fehlgeschlagen: " + str(e)]
        return relation


def validate_relations(
    relations: List[Dict], model_name: str, threshold: float = CONFIDENCE_THRESHOLD
) -> Dict:
    """
    Validiert alle Relationen und filtert nach Konfidenz.
    Returns:
        Dict mit 'valid', 'invalid' und 'uncertain' Listen
    """
    if not relations:
        return {"valid": [], "invalid": [], "uncertain": []}
    logger.info("Validiere " + str(len(relations)) + " Relationen...")
    valid = []
    invalid = []
    uncertain = []
    for i, relation in enumerate(relations):
        validated = validate_relation(relation, model_name)
        if (
            validated.get("validated") is True
            and validated.get("validation_confidence", 0) >= threshold
        ):
            valid.append(validated)
        elif validated.get("validated") is False:
            invalid.append(validated)
        else:
            uncertain.append(validated)
        # Fortschritt loggen
        if (i + 1) % 10 == 0:
            logger.info("  " + str(i + 1) + "/" + str(len(relations)) + " validiert")
    logger.info(
        "Validierung abgeschlossen: "
        + str(len(valid))
        + " valid, "
        + str(len(invalid))
        + " invalid, "
        + str(len(uncertain))
        + " uncertain"
    )
    return {"valid": valid, "invalid": invalid, "uncertain": uncertain}


def filter_by_confidence(
    extractions: Dict, threshold: float = CONFIDENCE_THRESHOLD
) -> Dict:
    """
    Einfache Filterung nach Konfidenz ohne LLM-Validierung.
    Schneller aber weniger genau.
    """
    entities = extractions.get("entities", [])
    relations = extractions.get("relations", [])
    # Relationen nach Konfidenz filtern
    filtered_relations = [r for r in relations if r.get("confidence", 0) >= threshold]
    removed = len(relations) - len(filtered_relations)
    if removed > 0:
        logger.info(
            "Gefiltert: "
            + str(removed)
            + " Relationen unter Konfidenz-Threshold "
            + str(threshold)
        )
    return {"entities": entities, "relations": filtered_relations}


if __name__ == "__main__":
    # Kleiner Test mit Dummy-Daten
    test_relations = [
        {
            "subject": "Assembly Control",
            "predicate": "uses_session",
            "object": "tiasc1100m000",
            "confidence": 0.9,
            "text_evidence": "In Assembly Control, use session tiasc1100m000 to manage orders.",
        },
        {
            "subject": "Production",
            "predicate": "depends_on",
            "object": "Magic Module",
            "confidence": 0.3,
            "text_evidence": "Production handles manufacturing.",
        },
    ]
    print("Test ohne LLM (nur Konfidenz-Filter):")
    filtered = filter_by_confidence({"relations": test_relations}, threshold=0.5)
    print("  Uebrig: " + str(len(filtered["relations"])) + " Relationen")
