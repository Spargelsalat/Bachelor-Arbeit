"""
Zentrale Konfiguration für die Pipeline.
Hier landen alle Einstellungen damit wir nicht überall hardcoded Werte haben.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# .env Datei laden
load_dotenv()
# Projektpfade
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SCHEMAS_DIR = DATA_DIR / "schemas"
OUTPUT_DIR = PROJECT_ROOT / "output"
# Ordner erstellen falls sie nicht existieren
for dir_path in [RAW_DIR, PROCESSED_DIR, SCHEMAS_DIR, OUTPUT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)
# API Keys
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# Modell-Konfiguration
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "openrouter/openai/gpt-4o")
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "ollama/deepseek-r1:14b")
# Chunking Einstellungen
CHUNK_SIZE = 2000  # Tokens pro Chunk (für regelbasiertes Fallback)
CHUNK_OVERLAP = 200  # Überlappung zwischen Chunks
# Extraktions-Einstellungen
EXTRACTION_TEMPERATURE = 0.1  # Niedrig für konsistente Ergebnisse
MAX_RETRIES = 3  # Wie oft bei Fehlern wiederholen
# Validierungs-Einstellungen
CONFIDENCE_THRESHOLD = 0.7  # Minimum Konfidenz für Graphaufnahme
# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
