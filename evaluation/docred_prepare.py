from __future__ import annotations
import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def _safe_filename(name: str) -> str:
    name = name.strip()
    name = re.sub(r"\s+", " ", name)
    name = re.sub(r"[^A-Za-z0-9 _\.-]", "_", name)
    name = name.replace(" ", "_")
    name = re.sub(r"_+", "_", name)
    return name[:120] or "doc"


def docred_doc_to_text(doc: dict[str, Any]) -> str:
    sents = doc.get("sents", [])
    lines: list[str] = []
    if isinstance(sents, list):
        for s in sents:
            if isinstance(s, list):
                lines.append(" ".join([str(tok) for tok in s]).strip())
    return "\n".join([ln for ln in lines if ln]).strip()


def pick_canonical_name(mentions: list[dict[str, Any]]) -> str:
    names = [m.get("name", "") for m in mentions if isinstance(m, dict)]
    names = [str(n).strip() for n in names if str(n).strip()]
    if not names:
        return ""
    c = Counter(names)
    best = max(names, key=lambda x: (c[x], -names.index(x)))
    return best


def build_subset_schema(relation_names: list[str], out_path: Path) -> None:
    relation_names = sorted(set([r.strip() for r in relation_names if r.strip()]))
    schema = {
        "_meta": {
            "note": "Schema aus DocRED-Subset abgeleitet (Relationstypen aus Labels).",
            "entity_types_note": "DocRED liefert viele Entitäten; hier wird nur ein generischer Typ verwendet.",
        },
        "domain": "DocRED_Subset",
        "entity_types": [
            {
                "name": "Entity",
                "description": "Generischer Entitätstyp für DocRED-Evaluation",
                "attributes": [],
                "examples": [],
            }
        ],
        "relation_types": [
            {
                "name": r,
                "description": "DocRED Relationstyp",
                "source_type": "Entity",
                "target_type": "Entity",
                "examples": [],
            }
            for r in relation_names
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Pfad zu DocRED dev.json")
    ap.add_argument("--n", type=int, default=100, help="Anzahl Dokumente im Subset")
    ap.add_argument(
        "--processed-dir",
        default="data/processed/docred",
        help="Zielordner für .txt Dateien",
    )
    ap.add_argument(
        "--gold-out",
        default="data/docred/subset/docred_100.gold.jsonl",
        help="Gold-Ausgabe (jsonl)",
    )
    ap.add_argument(
        "--schema-out",
        default="data/schemas/docred_100.schema.json",
        help="Schema-Datei für Extraktion",
    )
    args = ap.parse_args()
    input_path = Path(args.input)
    processed_dir = Path(args.processed_dir)
    gold_out = Path(args.gold_out)
    schema_out = Path(args.schema_out)
    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("DocRED input muss eine JSON-Liste von Dokumenten sein.")
    subset = data[: max(1, args.n)]
    processed_dir.mkdir(parents=True, exist_ok=True)
    gold_out.parent.mkdir(parents=True, exist_ok=True)
    rel_names_all: list[str] = []
    with gold_out.open("w", encoding="utf-8") as f_gold:
        for i, doc in enumerate(subset):
            title = str(doc.get("title", f"doc_{i}"))
            doc_id = f"docred_{i:04d}_{_safe_filename(title)}"
            text = docred_doc_to_text(doc)
            (processed_dir / f"{doc_id}.txt").write_text(
                text, encoding="utf-8", errors="replace"
            )
            vertex_set = doc.get("vertexSet", [])
            entities: list[dict[str, Any]] = []
            if isinstance(vertex_set, list):
                for e_idx, mentions in enumerate(vertex_set):
                    if not isinstance(mentions, list):
                        continue
                    cname = pick_canonical_name(mentions)
                    if cname:
                        entities.append(
                            {"eid": f"E{e_idx}", "name": cname, "type": "Entity"}
                        )
            labels = doc.get("labels", [])
            relations: list[dict[str, Any]] = []
            if isinstance(labels, list):
                for lab in labels:
                    if not isinstance(lab, dict):
                        continue
                    r = str(lab.get("r", "")).strip()
                    h = lab.get("h", None)
                    t = lab.get("t", None)
                    if not r or h is None or t is None:
                        continue
                    if not (isinstance(h, int) and isinstance(t, int)):
                        continue
                    if h < 0 or t < 0 or h >= len(entities) or t >= len(entities):
                        continue
                    relations.append(
                        {
                            "source_name": entities[h]["name"],
                            "type": r,
                            "target_name": entities[t]["name"],
                        }
                    )
                    rel_names_all.append(r)
            f_gold.write(
                json.dumps(
                    {
                        "doc_id": doc_id,
                        "title": title,
                        "entities": entities,
                        "relations": relations,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    build_subset_schema(rel_names_all, schema_out)
    print(f"[OK] Subset .txt Dateien: {processed_dir.resolve()}")
    print(f"[OK] Gold-Datei:        {gold_out.resolve()}")
    print(f"[OK] Schema-Datei:      {schema_out.resolve()}")


if __name__ == "__main__":
    main()
