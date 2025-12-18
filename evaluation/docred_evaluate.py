from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


def _norm(s: str) -> str:
    """
    Normalisiert Namen
    """
    s = (s or "").strip().lower()
    s = s.replace("\u00ad", "")  #  Hyphen
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[\"“”'`]", "", s)
    return s


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def doc_id_from_filename(stem: str) -> str:
    """
    doc_id ist alles vor dem ersten Punkt.
    """
    return stem.split(".", 1)[0]


def load_gold(gold_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    - entities: Liste mit {eid, name, type}
    - relations: Liste mit {source_name, type, target_name}
    """
    gold: Dict[str, Dict[str, Any]] = {}
    for row in iter_jsonl(gold_path):
        doc_id = row["doc_id"]
        ents = row.get("entities", [])
        rels = row.get("relations", [])
        gold[doc_id] = {"entities": ents, "relations": rels}
    return gold


def load_pred_entities(entities_file: Path) -> Dict[str, str]:
    """
    - global_id -> name
    """
    m: Dict[str, str] = {}
    for row in iter_jsonl(entities_file):
        if isinstance(row, dict) and "_meta" in row:
            continue
        gid = row.get("global_id")
        name = row.get("name")
        if (
            isinstance(gid, str)
            and isinstance(name, str)
            and gid.strip()
            and name.strip()
        ):
            m[gid.strip()] = name.strip()
    return m


def pred_sets_from_resolved(
    entities_file: Path,
    relations_file: Path,
) -> Tuple[Set[str], Set[Tuple[str, str, str]]]:
    gid_to_name = load_pred_entities(entities_file)
    ent_set: Set[str] = set(_norm(n) for n in gid_to_name.values() if _norm(n))
    rel_set: Set[Tuple[str, str, str]] = set()
    for row in iter_jsonl(relations_file):
        if isinstance(row, dict) and "_meta" in row:
            continue
        src = row.get("source")
        rtype = row.get("type")
        tgt = row.get("target")
        if not (
            isinstance(src, str) and isinstance(rtype, str) and isinstance(tgt, str)
        ):
            continue
        s_name = gid_to_name.get(src.strip(), "")
        t_name = gid_to_name.get(tgt.strip(), "")
        s_norm = _norm(s_name)
        t_norm = _norm(t_name)
        r_norm = rtype.strip()
        if s_norm and t_norm and r_norm:
            rel_set.add((s_norm, r_norm, t_norm))
    return ent_set, rel_set


def gold_sets_from_doc(g: Dict[str, Any]) -> Tuple[Set[str], Set[Tuple[str, str, str]]]:
    ents = g.get("entities", [])
    rels = g.get("relations", [])
    ent_set: Set[str] = set()
    rel_set: Set[Tuple[str, str, str]] = set()
    if isinstance(ents, list):
        for e in ents:
            if not isinstance(e, dict):
                continue
            name = _norm(str(e.get("name", "")))
            if name:
                ent_set.add(name)
    if isinstance(rels, list):
        for r in rels:
            if not isinstance(r, dict):
                continue
            s = _norm(str(r.get("source_name", "")))
            t = _norm(str(r.get("target_name", "")))
            ty = str(r.get("type", "")).strip()
            if s and t and ty:
                rel_set.add((s, ty, t))
    return ent_set, rel_set


def prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="DocRED-Auswertung (name-basiertes Matching)."
    )
    ap.add_argument("--gold", required=True, help="Pfad zu docred_100.gold.jsonl")
    ap.add_argument(
        "--resolved-dir",
        required=True,
        help="Ordner mit *.entities.resolved.jsonl / *.relations.resolved.jsonl",
    )
    ap.add_argument("--out", required=True, help="Ausgabe JSON-Datei")
    args = ap.parse_args()
    gold_path = Path(args.gold).resolve()
    resolved_dir = Path(args.resolved_dir).resolve()
    out_path = Path(args.out).resolve()
    gold = load_gold(gold_path)
    rel_files = sorted(resolved_dir.glob("*.relations.resolved.jsonl"))
    if not rel_files:
        raise FileNotFoundError(f"Keine relations.resolved.jsonl in: {resolved_dir}")
    ner_tp = ner_fp = ner_fn = 0
    re_tp = re_fp = re_fn = 0
    evaluated_docs = 0
    missing_gold = 0
    for rel_file in rel_files:
        stem = rel_file.stem.replace(".relations.resolved", "")
        doc_id = doc_id_from_filename(stem)
        ent_file = resolved_dir / f"{stem}.entities.resolved.jsonl"
        if not ent_file.exists():
            continue
        g = gold.get(doc_id)
        if g is None:
            missing_gold += 1
            continue
        pred_ent_set, pred_rel_set = pred_sets_from_resolved(ent_file, rel_file)
        gold_ent_set, gold_rel_set = gold_sets_from_doc(g)
        # NER
        ner_tp += len(pred_ent_set & gold_ent_set)
        ner_fp += len(pred_ent_set - gold_ent_set)
        ner_fn += len(gold_ent_set - pred_ent_set)
        # RE
        re_tp += len(pred_rel_set & gold_rel_set)
        re_fp += len(pred_rel_set - gold_rel_set)
        re_fn += len(gold_rel_set - pred_rel_set)
        evaluated_docs += 1
    result = {
        "meta": {
            "gold": str(gold_path),
            "resolved_dir": str(resolved_dir),
            "evaluated_docs": evaluated_docs,
            "missing_gold_docs": missing_gold,
            "note": "Name-basiertes Matching für Entitäten und Relationen.",
        },
        "ner_counts": {"tp": ner_tp, "fp": ner_fp, "fn": ner_fn},
        "re_counts": {"tp": re_tp, "fp": re_fp, "fn": re_fn},
        "ner": prf(ner_tp, ner_fp, ner_fn),
        "relation_extraction": prf(re_tp, re_fp, re_fn),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[OK] Ergebnis geschrieben:", out_path)
    print("NER:", result["ner"])
    print("RE :", result["relation_extraction"])


if __name__ == "__main__":
    main()
