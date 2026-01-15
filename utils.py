import os
import json
import re
import logging
from pathlib import Path

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
TEMP_DIR = BASE_DIR / "temp"
HISTORY_DIR = BASE_DIR / "history"
IMG_DIR = BASE_DIR / "persona_images"
PERSONA_FILE = BASE_DIR / "personas.json"
CONFIG_FILE = BASE_DIR / "config.json"
HISTORY_FILE = HISTORY_DIR / "history.json"
STYLES_FILE = BASE_DIR / "styles.json"

# Ensure directories exist
TEMP_DIR.mkdir(exist_ok=True)
HISTORY_DIR.mkdir(exist_ok=True)
IMG_DIR.mkdir(exist_ok=True)

# Ensure critical files exist
def ensure_files():
    if not PERSONA_FILE.exists():
        with open(PERSONA_FILE, "w", encoding="utf-8") as f: json.dump({}, f)
    if not HISTORY_FILE.exists():
        with open(HISTORY_FILE, "w", encoding="utf-8") as f: json.dump([], f)
    if not STYLES_FILE.exists():
        default_styles = {
            "Standard": "Write a natural, descriptive sentence.",
            "Booru Tags": "Use comma-separated tags. Start with character count (e.g. 1girl), then character traits, then clothing, then environment. No sentences.",
            "Cinematic": "Artistic, poetic description. Use /imagine style parameters. Focus on lighting and atmosphere."
        }
        with open(STYLES_FILE, "w", encoding="utf-8") as f: json.dump(default_styles, f)

# --- UTILITY FUNCTIONS ---
def safe_file_cleanup(path: Path):
    try:
        if path and path.exists():
            os.remove(path)
            logger.info(f"Cleaned up: {path.name}")
    except Exception as e:
        logger.error(f"Cleanup failed for {path}: {e}")

def load_json_file(path: Path, default=None):
    if default is None: default = {}
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return default
    return default

def save_json_file(path: Path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def extract_json_from_text(text: str) -> dict:
    try:
        text = re.sub(r'```json\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'```\s*', '', text)
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx == -1 or end_idx == -1: raise ValueError("No JSON brackets found")
        json_str = text[start_idx:end_idx + 1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Loose cleanup for common LLM JSON errors
            json_str = re.sub(r'(?<=[}\]"0-9e])\s+(?=")', ', ', json_str)
            json_str = re.sub(r'(?<=true)\s+(?=")', ', ', json_str)
            json_str = re.sub(r'(?<=false)\s+(?=")', ', ', json_str)
            json_str = re.sub(r'(?<=null)\s+(?=")', ', ', json_str)
            json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
            return json.loads(json_str)
    except Exception as e:
        logger.error(f"JSON extraction failed: {e}")
        return {}

def save_history(entry):
    history = load_json_file(HISTORY_FILE, [])
    if not isinstance(history, list): history = []
    history.insert(0, entry)
    save_json_file(HISTORY_FILE, history[:50])