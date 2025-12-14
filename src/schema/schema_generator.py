import json
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.llm.model_factory import get_llm
from src.extraction.prompts import SCHEMA_GENERATION_PROMPT
from config.settings import SCHEMAS_DIR


def extract_text_samples(
    documents: List[Dict], num_samples: int = 5, sample_length: int = 1500
) -> str:
    all_text = "\n\n".join([doc["full_text"] for doc in documents])
    text_length = len(all_text)
    samples = []

    for i in range(num_samples):
        start = int((text_length / num_samples) * i)
        end = start + sample_length
        sample = all_text[start:end]

        first_space = sample.find(" ")
        last_period = sample.rfind(".")

        if first_space > 0 and last_period > first_space:
            sample = sample[first_space + 1 : last_period + 1]

        samples.append("--- Ausschnitt " + str(i + 1) + " ---\n" + sample)

    return "\n\n".join(samples)


def generate_schema(documents: List[Dict], model_name: str) -> Dict:
    logger.info("Generiere Schema mit " + model_name)

    text_samples = extract_text_samples(documents)
    llm = get_llm(model_name, temperature=0.2)
    prompt = SCHEMA_GENERATION_PROMPT.format(text_samples=text_samples)

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

        schema = json.loads(response_text)

        if "entity_types" not in schema or "relation_types" not in schema:
            raise ValueError("Schema unvollstaendig")

        entity_count = len(schema["entity_types"])
        relation_count = len(schema["relation_types"])
        logger.info("  " + str(entity_count) + " Entitaetstypen gefunden")
        logger.info("  " + str(relation_count) + " Relationstypen gefunden")

        return schema

    except json.JSONDecodeError as e:
        logger.error("JSON-Parsing fehlgeschlagen: " + str(e))
        return get_default_schema()
    except Exception as e:
        logger.error("Schema-Generierung fehlgeschlagen: " + str(e))
        return get_default_schema()


def get_default_schema() -> Dict:
    logger.warning("Nutze Default-Schema als Fallback")

    return {
        "entity_types": [
            {
                "name": "module",
                "description": "Software module or component",
                "examples": [],
            },
            {"name": "function", "description": "Function or feature", "examples": []},
            {
                "name": "parameter",
                "description": "Configuration parameter or setting",
                "examples": [],
            },
            {
                "name": "process",
                "description": "Business process or workflow",
                "examples": [],
            },
            {
                "name": "data_object",
                "description": "Data entity or object type",
                "examples": [],
            },
            {
                "name": "user_role",
                "description": "User role or permission group",
                "examples": [],
            },
        ],
        "relation_types": [
            {"name": "contains", "description": "X contains Y", "example_triple": []},
            {"name": "uses", "description": "X uses Y", "example_triple": []},
            {
                "name": "depends_on",
                "description": "X depends on Y",
                "example_triple": [],
            },
            {
                "name": "configures",
                "description": "X configures Y",
                "example_triple": [],
            },
            {"name": "part_of", "description": "X is part of Y", "example_triple": []},
            {"name": "triggers", "description": "X triggers Y", "example_triple": []},
        ],
    }


def save_schema(schema: Dict, name: str = "schema") -> Path:
    filepath = SCHEMAS_DIR / (name + ".json")

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)

    logger.info("Schema gespeichert: " + str(filepath))
    return filepath


def load_schema(name: str = "schema") -> Optional[Dict]:
    filepath = SCHEMAS_DIR / (name + ".json")

    if not filepath.exists():
        return None

    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    from config.settings import RAW_DIR, DEFAULT_MODEL
    from src.preprocessing.document_loader import load_all_pdfs

    docs = load_all_pdfs(RAW_DIR)

    if docs:
        schema = generate_schema(docs, DEFAULT_MODEL)

        print("\nGeneriertes Schema:")
        print("\nEntitaetstypen:")
        for et in schema["entity_types"]:
            print("  - " + et["name"] + ": " + et["description"])

        print("\nRelationstypen:")
        for rt in schema["relation_types"]:
            print("  - " + rt["name"] + ": " + rt["description"])

        save_schema(schema)
