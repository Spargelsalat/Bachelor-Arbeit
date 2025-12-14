"""
Alle Prompt-Templates an einem Ort.
So können wir leicht verschiedene Prompts ausprobieren ohne überall im Code rumzusuchen.
"""

# Schema-Generierung: LLM analysiert Dokumente und schlägt Entitäts-/Relationstypen vor
SCHEMA_GENERATION_PROMPT = """Analysiere diese Textausschnitte aus einer technischen Dokumentation.
Textausschnitte:
{text_samples}
Aufgabe: Identifiziere die wichtigsten Konzepte und Beziehungen in dieser Dokumentation.
Gib zurück:
1. entity_types: Liste von Entitätstypen die vorkommen (z.B. Module, Function, Parameter, Process)
2. relation_types: Liste von Beziehungstypen zwischen Entitäten (z.B. contains, uses, depends_on)
Für jeden Typ gib an:
- name: Technischer Name (englisch, snake_case)
- description: Kurze Beschreibung was dieser Typ repräsentiert
- examples: 2-3 Beispiele aus dem Text
Antwort als JSON:
{{
    "entity_types": [
        {{"name": "...", "description": "...", "examples": ["...", "..."]}}
    ],
    "relation_types": [
        {{"name": "...", "description": "...", "example_triple": ["subject", "relation", "object"]}}
    ]
}}
Nur das JSON, keine Erklärung davor oder danach."""
# Entitäts- und Relationsextraktion
EXTRACTION_PROMPT = """Extrahiere alle Entitäten und Beziehungen aus diesem Text.
Schema:
- Entitätstypen: {entity_types}
- Relationstypen: {relation_types}
Text:
{text}
Aufgabe:
1. Finde alle Entitäten im Text und ordne sie einem Typ zu
2. Finde alle Beziehungen zwischen den Entitäten
3. Gib für jede Extraktion die Textstelle an (als Beleg)
Antwort als JSON:
{{
    "entities": [
        {{
            "name": "Kanonischer Name der Entität",
            "type": "Entitätstyp aus Schema",
            "mentions": ["Wie im Text erwähnt"],
            "text_evidence": "Relevanter Textausschnitt"
        }}
    ],
    "relations": [
        {{
            "subject": "Name der Quell-Entität",
            "predicate": "Relationstyp aus Schema",
            "object": "Name der Ziel-Entität",
            "confidence": 0.0 bis 1.0,
            "text_evidence": "Relevanter Textausschnitt"
        }}
    ]
}}
Wichtig:
- Nur Entitäten und Relationen die wirklich im Text stehen
- Bei Unsicherheit: confidence niedriger setzen
- Keine Vermutungen oder Erfindungen
Nur das JSON, keine Erklärung."""
# Entity Resolution: Prüft ob zwei Entitäten dieselbe sind
ENTITY_RESOLUTION_PROMPT = """Sind diese beiden Entitäten dieselbe Sache?
Entität 1:
- Name: {entity1_name}
- Typ: {entity1_type}
- Kontext: {entity1_context}
Entität 2:
- Name: {entity2_name}
- Typ: {entity2_type}
- Kontext: {entity2_context}
Antwort als JSON:
{{
    "same_entity": true oder false,
    "confidence": 0.0 bis 1.0,
    "reasoning": "Kurze Begründung"
}}
Nur das JSON."""
# Validierung: Prüft ob ein extrahiertes Tripel korrekt ist
VALIDATION_PROMPT = """Prüfe ob diese Beziehung korrekt aus dem Text extrahiert wurde.
Extrahiertes Tripel:
- Subjekt: {subject}
- Beziehung: {predicate}
- Objekt: {object}
Originaltext:
{text_evidence}
Fragen:
1. Wird diese Beziehung tatsächlich im Text ausgesagt?
2. Sind Subjekt und Objekt korrekt identifiziert?
3. Ist der Beziehungstyp passend?
Antwort als JSON:
{{
    "valid": true oder false,
    "confidence": 0.0 bis 1.0,
    "issues": ["Liste von Problemen falls vorhanden"]
}}
Nur das JSON."""
# Semantisches Chunking: Identifiziert thematische Grenzen
SEMANTIC_CHUNKING_PROMPT = """Analysiere diesen Text und identifiziere die logischen Abschnittsgrenzen.
Text:
{text}
Aufgabe: Finde die Stellen wo ein neuer thematischer Abschnitt beginnt.
Gib für jeden Abschnitt zurück:
- start_text: Die ersten 5-10 Worte des Abschnitts
- topic: Kurze thematische Beschreibung (max 10 Worte)
Antwort als JSON-Array:
[
    {{"start_text": "Die ersten Worte...", "topic": "Kurze Beschreibung"}},
    ...
]
Nur das JSON, keine Erklärung."""
