from typing import Dict, List, Optional
from pathlib import Path
from neo4j import GraphDatabase
from loguru import logger
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD


class Neo4jHandler:
    def __init__(self, uri: str = None, user: str = None, password: str = None):
        self.uri = uri or NEO4J_URI
        self.user = user or NEO4J_USER
        self.password = password or NEO4J_PASSWORD
        self.driver = None

    def connect(self):
        if not self.password:
            raise ValueError("NEO4J_PASSWORD nicht in .env gesetzt!")
        try:
            self.driver = GraphDatabase.driver(
                self.uri, auth=(self.user, self.password)
            )
            self.driver.verify_connectivity()
            logger.info("Verbunden mit Neo4j: " + self.uri)
            return True
        except Exception as e:
            logger.error("Neo4j Verbindung fehlgeschlagen: " + str(e))
            return False

    def close(self):
        if self.driver:
            self.driver.close()
            logger.info("Neo4j Verbindung geschlossen")

    def clear_database(self):
        """Loescht alle Knoten und Kanten - Vorsicht!"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.warning("Datenbank geleert")

    def create_entity(self, entity: Dict) -> Optional[str]:
        """Erstellt einen Knoten fuer eine Entitaet."""
        query = """
        MERGE (e:Entity {name: $name})
        SET e.type = $type,
            e.source_document = $source_document,
            e.source_chunk = $source_chunk
        RETURN elementId(e) as id
        """
        try:
            with self.driver.session() as session:
                result = session.run(
                    query,
                    name=entity.get("name", ""),
                    type=entity.get("type", "unknown"),
                    source_document=entity.get("source_document", ""),
                    source_chunk=entity.get("source_chunk", 0),
                )
                record = result.single()
                return record["id"] if record else None
        except Exception as e:
            logger.error("Fehler beim Erstellen der Entitaet: " + str(e))
            return None

    def create_relation(self, relation: Dict) -> bool:
        """Erstellt eine Kante zwischen zwei Entitaeten."""
        # Dynamischer Relationstyp - Neo4j braucht das in der Query selbst
        rel_type = relation.get("predicate", "RELATED_TO").upper().replace(" ", "_")
        query = (
            """
        MATCH (a:Entity {name: $subject})
        MATCH (b:Entity {name: $object})
        MERGE (a)-[r:"""
            + rel_type
            + """]->(b)
        SET r.confidence = $confidence,
            r.source_document = $source_document,
            r.source_chunk = $source_chunk,
            r.text_evidence = $text_evidence
        RETURN r
        """
        )
        try:
            with self.driver.session() as session:
                result = session.run(
                    query,
                    subject=relation.get("subject", ""),
                    object=relation.get("object", ""),
                    confidence=relation.get("confidence", 0.5),
                    source_document=relation.get("source_document", ""),
                    source_chunk=relation.get("source_chunk", 0),
                    text_evidence=relation.get("text_evidence", "")[
                        :500
                    ],  # Limit fuer Neo4j
                )
                return result.single() is not None
        except Exception as e:
            logger.error("Fehler beim Erstellen der Relation: " + str(e))
            return False

    def import_extractions(self, extractions: Dict, clear_first: bool = False):
        """Importiert alle Entitaeten und Relationen in den Graphen."""
        if clear_first:
            self.clear_database()
        entities = extractions.get("entities", [])
        relations = extractions.get("relations", [])
        logger.info(
            "Importiere "
            + str(len(entities))
            + " Entitaeten und "
            + str(len(relations))
            + " Relationen"
        )
        # Erst Entitaeten erstellen
        entity_count = 0
        for entity in entities:
            if self.create_entity(entity):
                entity_count += 1
        logger.info("  " + str(entity_count) + " Entitaeten erstellt")
        # Dann Relationen
        relation_count = 0
        for relation in relations:
            if self.create_relation(relation):
                relation_count += 1
        logger.info("  " + str(relation_count) + " Relationen erstellt")
        return {"entities": entity_count, "relations": relation_count}

    def get_stats(self) -> Dict:
        """Gibt Statistiken zum Graphen zurueck."""
        with self.driver.session() as session:
            # Anzahl Knoten
            node_result = session.run("MATCH (n) RETURN count(n) as count")
            node_count = node_result.single()["count"]
            # Anzahl Kanten
            rel_result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            rel_count = rel_result.single()["count"]
            # Knoten nach Typ
            type_result = session.run(
                "MATCH (n:Entity) RETURN n.type as type, count(*) as count ORDER BY count DESC"
            )
            types = {record["type"]: record["count"] for record in type_result}
        return {
            "total_nodes": node_count,
            "total_relations": rel_count,
            "nodes_by_type": types,
        }

    def get_all_entities(self) -> List[Dict]:
        """Holt alle Entitaeten aus dem Graphen."""
        query = "MATCH (e:Entity) RETURN e.name as name, e.type as type ORDER BY e.type, e.name"
        with self.driver.session() as session:
            result = session.run(query)
            return [{"name": r["name"], "type": r["type"]} for r in result]

    def get_relations_for_entity(self, entity_name: str) -> List[Dict]:
        """Holt alle Relationen fuer eine Entitaet."""
        query = """
        MATCH (a:Entity {name: $name})-[r]->(b:Entity)
        RETURN a.name as subject, type(r) as predicate, b.name as object
        UNION
        MATCH (a:Entity)-[r]->(b:Entity {name: $name})
        RETURN a.name as subject, type(r) as predicate, b.name as object
        """
        with self.driver.session() as session:
            result = session.run(query, name=entity_name)
            return [dict(r) for r in result]


if __name__ == "__main__":
    handler = Neo4jHandler()
    if handler.connect():
        print("\nDatenbank-Statistiken:")
        stats = handler.get_stats()
        print("  Knoten: " + str(stats["total_nodes"]))
        print("  Relationen: " + str(stats["total_relations"]))
        if stats["nodes_by_type"]:
            print("\n  Nach Typ:")
            for t, c in stats["nodes_by_type"].items():
                print("    " + str(t) + ": " + str(c))
        handler.close()
