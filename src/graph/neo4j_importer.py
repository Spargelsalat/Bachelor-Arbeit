from __future__ import annotations
import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from dotenv import load_dotenv  # noqa: E402
from neo4j import GraphDatabase  # noqa: E402
from config.settings import load_settings  # noqa: E402

logger = logging.getLogger(__name__)


# -----------------------------
# Helpers
# -----------------------------
def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _clean_env(value: str | None) -> str:
    if value is None:
        return ""
    return value.split("#", 1)[0].strip()


def _to_bool(value: str | None, default: bool) -> bool:
    v = _clean_env(value).lower()
    if v == "":
        return default
    return v in {"1", "true", "yes", "y", "on"}


def _to_int(value: str | None, default: int) -> int:
    v = _clean_env(value)
    if v == "":
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _to_float(value: str | None, default: float) -> float:
    v = _clean_env(value)
    if v == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def sanitize_rel_type(rel: str) -> str:
    r = (rel or "").strip().upper()
    r = re.sub(r"[^A-Z0-9_]", "_", r)
    r = re.sub(r"_+", "_", r).strip("_")
    if not r:
        r = "REL"
    if r[0].isdigit():
        r = "R_" + r
    return r


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class Neo4jConfig:
    uri: str
    user: str
    password: str
    database: str


@dataclass(frozen=True)
class ImportConfig:
    entities_file: Optional[Path]
    relations_file: Optional[Path]
    entities_dir: Optional[Path]
    relations_dir: Optional[Path]
    min_support_ratio: float
    accept_llm_supported: bool
    max_provenance: int
    max_mention_samples: int
    overwrite_rel_properties: bool
    clear_db: bool


def load_configs() -> tuple[Neo4jConfig, ImportConfig]:
    load_dotenv()
    neo = Neo4jConfig(
        uri=_clean_env(os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")),
        user=_clean_env(os.getenv("NEO4J_USER", "neo4j")),
        password=_clean_env(os.getenv("NEO4J_PASSWORD", "")),
        database=_clean_env(os.getenv("NEO4J_DATABASE", "neo4j")),
    )
    if not neo.password:
        raise ValueError("NEO4J_PASSWORD fehlt in .env")
    entities_file = _clean_env(os.getenv("GRAPH_ENTITIES_FILE"))
    relations_file = _clean_env(os.getenv("GRAPH_RELATIONS_FILE"))
    entities_dir = _clean_env(os.getenv("GRAPH_ENTITIES_DIR"))
    relations_dir = _clean_env(os.getenv("GRAPH_RELATIONS_DIR"))
    imp = ImportConfig(
        entities_file=Path(entities_file) if entities_file else None,
        relations_file=Path(relations_file) if relations_file else None,
        entities_dir=Path(entities_dir) if entities_dir else None,
        relations_dir=Path(relations_dir) if relations_dir else None,
        min_support_ratio=_to_float(os.getenv("GRAPH_MIN_SUPPORT_RATIO"), 0.6),
        accept_llm_supported=_to_bool(os.getenv("GRAPH_ACCEPT_LLM_SUPPORTED"), True),
        max_provenance=_to_int(os.getenv("GRAPH_MAX_PROVENANCE"), 5),
        max_mention_samples=_to_int(os.getenv("GRAPH_MAX_MENTION_SAMPLES"), 3),
        overwrite_rel_properties=_to_bool(
            os.getenv("GRAPH_OVERWRITE_REL_PROPERTIES"), True
        ),
        clear_db=_to_bool(os.getenv("GRAPH_CLEAR_DB"), False),
    )
    return neo, imp


def list_entity_files(cfg: ImportConfig) -> list[Path]:
    if cfg.entities_dir and cfg.entities_dir.exists():
        return sorted(cfg.entities_dir.glob("*.entities.resolved.jsonl"))
    if cfg.entities_file and cfg.entities_file.exists():
        return [cfg.entities_file]
    raise FileNotFoundError(
        "Keine Entities-Quelle gefunden (GRAPH_ENTITIES_DIR oder GRAPH_ENTITIES_FILE)."
    )


def list_relation_files(cfg: ImportConfig) -> list[Path]:
    if cfg.relations_dir and cfg.relations_dir.exists():
        return sorted(cfg.relations_dir.glob("*.relations.validated.jsonl"))
    if cfg.relations_file and cfg.relations_file.exists():
        return [cfg.relations_file]
    raise FileNotFoundError(
        "Keine Relations-Quelle gefunden (GRAPH_RELATIONS_DIR oder GRAPH_RELATIONS_FILE)."
    )


# -----------------------------
# Neo4j helpers
# -----------------------------
def ensure_constraints(driver, database: str) -> None:
    cypher = "CREATE CONSTRAINT entity_global_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.global_id IS UNIQUE"
    with driver.session(database=database) as session:
        session.run(cypher)
    logger.info("Neo4j constraint ensured: :Entity(global_id) unique")


def clear_database(driver, database: str) -> None:
    with driver.session(database=database) as session:
        session.run("MATCH (n) DETACH DELETE n")
    logger.warning("Neo4j database cleared (MATCH (n) DETACH DELETE n).")


# -----------------------------
# Import: Entities
# -----------------------------
def import_entities(
    driver, database: str, entity_files: list[Path], max_mention_samples: int
) -> int:
    count = 0
    with driver.session(database=database) as session:
        for file_path in entity_files:
            logger.info("Import entities file: %s", file_path.name)
            for row in iter_jsonl(file_path):
                if isinstance(row, dict) and "_meta" in row:
                    continue
                gid = row.get("global_id")
                etype = row.get("type")
                name = row.get("name")
                attrs = (
                    row.get("attributes")
                    if isinstance(row.get("attributes"), dict)
                    else {}
                )
                mentions = (
                    row.get("mentions") if isinstance(row.get("mentions"), list) else []
                )
                if not gid or not etype or not name:
                    continue
                # Neo4j Properties dürfen keine Maps sein -> JSON-String speichern
                attrs_json = json.dumps(attrs, ensure_ascii=False)
                mention_samples = (
                    mentions[:max_mention_samples] if max_mention_samples > 0 else []
                )
                mention_samples_json = json.dumps(mention_samples, ensure_ascii=False)
                session.run(
                    """
                    MERGE (e:Entity {global_id: $gid})
                    SET e.type = $etype,
                        e.name = $name,
                        e.attributes_json = $attrs_json,
                        e.mention_count = $mention_count,
                        e.mention_samples_json = $mention_samples_json
                    """,
                    gid=str(gid),
                    etype=str(etype),
                    name=str(name),
                    attrs_json=attrs_json,
                    mention_count=len(mentions),
                    mention_samples_json=mention_samples_json,
                )
                count += 1
    logger.info("Imported entities total: %d", count)
    return count


# -----------------------------
# Import: Relations
# -----------------------------
def effective_supported(
    prov_list: list[dict[str, Any]], accept_llm_supported: bool
) -> bool:
    for p in prov_list:
        if not isinstance(p, dict):
            continue
        if str(p.get("support_status", "")).upper() == "SUPPORTED":
            return True
        if (
            accept_llm_supported
            and str(p.get("llm_support_status", "")).upper() == "SUPPORTED"
        ):
            return True
    return False


def import_relations(
    driver,
    database: str,
    relation_files: list[Path],
    min_support_ratio: float,
    accept_llm_supported: bool,
    max_provenance: int,
    overwrite_rel_properties: bool,
) -> int:
    count = 0
    skipped_missing_nodes = 0
    skipped_support = 0
    with driver.session(database=database) as session:
        for file_path in relation_files:
            logger.info("Import relations file: %s", file_path.name)
            for row in iter_jsonl(file_path):
                if isinstance(row, dict) and "_meta" in row:
                    continue
                src = row.get("source")
                rtype_raw = row.get("type")
                tgt = row.get("target")
                prov = (
                    row.get("provenance")
                    if isinstance(row.get("provenance"), list)
                    else []
                )
                val = (
                    row.get("validation")
                    if isinstance(row.get("validation"), dict)
                    else {}
                )
                if not src or not rtype_raw or not tgt:
                    continue
                support_ratio = float(val.get("support_ratio", 0.0) or 0.0)
                is_supported = effective_supported(prov, accept_llm_supported)
                if (support_ratio < min_support_ratio) and (not is_supported):
                    skipped_support += 1
                    continue
                rel_type = sanitize_rel_type(str(rtype_raw))
                evidences: list[str] = []
                for p in prov[:max_provenance]:
                    if isinstance(p, dict) and isinstance(
                        p.get("evidence_relation"), str
                    ):
                        evidences.append(p["evidence_relation"])
                # Relationship type kann nicht parametrisiert werden -> sanitized string in Query
                if overwrite_rel_properties:
                    cypher = f"""
                    MATCH (s:Entity {{global_id: $src}})
                    MATCH (t:Entity {{global_id: $tgt}})
                    MERGE (s)-[r:{rel_type}]->(t)
                    SET r.type = $rtype_raw,
                        r.support_ratio = $support_ratio,
                        r.supported = $supported,
                        r.evidence_samples = $evidences,
                        r.provenance_count = $prov_count
                    RETURN count(r) AS c
                    """
                else:
                    cypher = f"""
                    MATCH (s:Entity {{global_id: $src}})
                    MATCH (t:Entity {{global_id: $tgt}})
                    MERGE (s)-[r:{rel_type}]->(t)
                    SET r.type = coalesce(r.type, $rtype_raw),
                        r.support_ratio = coalesce(r.support_ratio, $support_ratio),
                        r.supported = coalesce(r.supported, $supported),
                        r.evidence_samples = coalesce(r.evidence_samples, $evidences),
                        r.provenance_count = coalesce(r.provenance_count, $prov_count)
                    RETURN count(r) AS c
                    """
                result = session.run(
                    cypher,
                    src=str(src),
                    tgt=str(tgt),
                    rtype_raw=str(rtype_raw),
                    support_ratio=support_ratio,
                    supported=is_supported,
                    evidences=evidences,
                    prov_count=len(prov),
                )
                rec = result.single()
                if not rec or rec.get("c", 0) == 0:
                    skipped_missing_nodes += 1
                    continue
                count += 1
    logger.info("Imported relations total: %d", count)
    logger.info("Skipped relations (support filter): %d", skipped_support)
    logger.info("Skipped relations (missing nodes): %d", skipped_missing_nodes)
    return count


# -----------------------------
# CLI / Runner
# -----------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Import resolved+validated KG data into Neo4j (single-file or directory mode)."
    )
    p.add_argument(
        "--clear-db", action="store_true", help="Clear DB before import (dangerous)."
    )
    return p


def run() -> None:
    settings = load_settings()
    _setup_logging(settings.log_level)
    neo_cfg, imp_cfg = load_configs()
    args = build_argparser().parse_args()
    driver = GraphDatabase.driver(neo_cfg.uri, auth=(neo_cfg.user, neo_cfg.password))
    try:
        if args.clear_db or imp_cfg.clear_db:
            clear_database(driver, neo_cfg.database)
        ensure_constraints(driver, neo_cfg.database)
        entity_files = list_entity_files(imp_cfg)
        relation_files = list_relation_files(imp_cfg)
        if not entity_files:
            raise RuntimeError("Keine Entity-Files gefunden.")
        if not relation_files:
            raise RuntimeError("Keine Relation-Files gefunden.")
        logger.info(
            "Entity files: %d | Relation files: %d",
            len(entity_files),
            len(relation_files),
        )
        import_entities(
            driver, neo_cfg.database, entity_files, imp_cfg.max_mention_samples
        )
        import_relations(
            driver,
            neo_cfg.database,
            relation_files,
            imp_cfg.min_support_ratio,
            imp_cfg.accept_llm_supported,
            imp_cfg.max_provenance,
            imp_cfg.overwrite_rel_properties,
        )
    finally:
        driver.close()


if __name__ == "__main__":
    run()
