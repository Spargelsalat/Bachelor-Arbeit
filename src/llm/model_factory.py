import subprocess
import time
import requests
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from loguru import logger
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import OPENROUTER_API_KEY, OLLAMA_BASE_URL

# Cache fuer LLM Instanzen - einmal erstellen, wiederverwenden
_llm_cache = {}
_ollama_started = False


def check_ollama_running():
    try:
        response = requests.get(OLLAMA_BASE_URL + "/api/tags", timeout=2)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def start_ollama():
    global _ollama_started
    if _ollama_started and check_ollama_running():
        return True
    if check_ollama_running():
        _ollama_started = True
        logger.info("Ollama laeuft bereits")
        return True
    logger.info("Starte Ollama...")
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        for i in range(30):
            time.sleep(1)
            if check_ollama_running():
                logger.info("Ollama erfolgreich gestartet")
                _ollama_started = True
                return True
        logger.error("Ollama konnte nicht gestartet werden")
        return False
    except FileNotFoundError:
        logger.error("Ollama nicht gefunden. Ist es installiert?")
        return False


def get_llm(model_name: str, temperature: float = 0.1):
    """
    Gibt das passende LLM-Objekt zurueck.
    Cached die Instanz damit nicht jedes Mal neu erstellt wird.
    """
    cache_key = model_name + "_" + str(temperature)
    if cache_key in _llm_cache:
        return _llm_cache[cache_key]
    parts = model_name.split("/", 1)
    provider = parts[0].lower()
    model = parts[1] if len(parts) > 1 else model_name
    if provider == "openrouter":
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY nicht in .env gesetzt!")
        llm = ChatOpenAI(
            model=model,
            openai_api_key=OPENROUTER_API_KEY,
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=temperature,
            max_retries=3,
        )
    elif provider == "ollama":
        if not start_ollama():
            raise ConnectionError("Konnte Ollama nicht starten")
        llm = ChatOllama(model=model, base_url=OLLAMA_BASE_URL, temperature=temperature)
    else:
        raise ValueError("Unbekannter Provider: " + provider)
    _llm_cache[cache_key] = llm
    return llm


def clear_cache():
    """Cache leeren falls man Modell wechseln will."""
    global _llm_cache
    _llm_cache = {}


if __name__ == "__main__":
    print("Teste LLM Setup...")
    print("\nTeste Ollama:")
    try:
        llm = get_llm("ollama/deepseek-r1:32b")
        response = llm.invoke("Sag nur: Test erfolgreich")
        print("  " + response.content[:100])
    except Exception as e:
        print("  Fehler: " + str(e))
    print("\nTeste Cache (sollte schnell sein):")
    llm2 = get_llm("ollama/deepseek-r1:32b")
    print("  Gleiche Instanz: " + str(llm is llm2))
