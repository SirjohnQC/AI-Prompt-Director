import shutil
import json
import time
import re
import logging
import os
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
import ollama

# Import shared utilities (assumes utils.py exists, otherwise copy helper functions here)
try:
    from utils import BASE_DIR, load_json_file, save_json_file, extract_json_from_text, safe_file_cleanup
except ImportError:
    # Fallback if utils.py is missing
    BASE_DIR = Path(__file__).parent
    def load_json_file(path, default=None):
        if default is None: default = {}
        try: 
            with open(path, "r", encoding="utf-8") as f: return json.load(f)
        except: return default
    def save_json_file(path, data):
        with open(path, "w", encoding="utf-8") as f: json.dump(data, f, indent=2)
    def extract_json_from_text(text): return {} # Simplified
    def safe_file_cleanup(path): 
        try: os.remove(path)
        except: pass

logger = logging.getLogger(__name__)

# Constants
WARDROBE_FILE = BASE_DIR / "wardrobe.json"
WARD_IMG_DIR = BASE_DIR / "wardrobe_images"

# Ensure directories exist
WARD_IMG_DIR.mkdir(exist_ok=True)
if not WARDROBE_FILE.exists():
    with open(WARDROBE_FILE, "w", encoding="utf-8") as f:
        json.dump({}, f)

router = APIRouter()

# --- HELPER: Apply Wardrobe to Analysis Data ---
def apply_wardrobe_to_data(data: dict, wardrobe_id: str):
    """Injects wardrobe description and cleans up old conflicting data."""
    if not wardrobe_id or wardrobe_id == "none": return data
    
    wardrobe = load_json_file(WARDROBE_FILE)
    if wardrobe_id not in wardrobe: return data
    
    item = wardrobe[wardrobe_id]
    
    # Ensure clothing dict exists
    if "clothing" not in data or not isinstance(data["clothing"], dict):
        data["clothing"] = {}

    # 1. Overwrite the outfit description
    desc = item.get("description", "")
    name = item.get("name", "")
    full_outfit = f"{name}, {desc}".strip(", ")
    
    data["clothing"]["outfit"] = full_outfit
    
    # 2. CRITICAL FIX: Wipe old clothing details that might conflict
    # The old 'fit' likely describes the old clothes (e.g. "Loose sweater")
    data["clothing"]["fit"] = "" 
    
    # 3. Wipe old accessories to prevent clashes (optional, but recommended)
    data["clothing"]["accessories"] = ""

    # 4. CRITICAL FIX: Remove meta_tokens from the original image analysis
    # These tokens (e.g. "red sweater", "blue jeans") come from the original image.
    # If we are changing the outfit, these tokens are now WRONG and will confuse the generator.
    if "meta_tokens" in data:
        del data["meta_tokens"]
    
    return data

# --- ROUTES ---

@router.get("/wardrobe")
async def list_wardrobe():
    try:
        data = load_json_file(WARDROBE_FILE, {})
        clean = []
        for k, v in data.items():
            clean.append({"id": k, "name": v.get("name", "Unknown"), "image": f"/wardrobe-image/{k}"})
        return clean
    except: return []

@router.get("/wardrobe-image/{wid}")
async def get_wardrobe_image(wid: str):
    p = WARD_IMG_DIR / f"{wid}.jpg"
    return FileResponse(p) if p.exists() else JSONResponse(status_code=404, content={})

@router.post("/wardrobe/create")
async def create_wardrobe(
    name: str = Form(...),
    file: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None),
    model: str = Form("qwen2.5-vl"), 
):
    temp_path = None
    try:
        # Import main processing logic specifically to avoid circular imports
        from main import process_uploaded_image
        temp_path = process_uploaded_image(file, image_url)
        
        # Analyze Clothing
        system_prompt = """You are a Fashion Designer. Analyze this clothing item.
        Describe it for an AI Image Prompt. Focus on: Material, Cut, Color, Texture.
        Return JSON: { "description": "...", "type": "..." }"""
        
        response = ollama.chat(
            model=model, 
            messages=[{'role': 'user', 'content': system_prompt, 'images': [str(temp_path)]}], 
            options={"temperature": 0.2}
        )
        # Import extraction locally
        from main import extract_json_from_text
        ai_data = extract_json_from_text(response['message']['content'])
        
        # Save
        safe_id = re.sub(r'[^a-zA-Z0-9]', '_', name.lower()) + "_" + str(int(time.time()))
        shutil.copy(temp_path, WARD_IMG_DIR / f"{safe_id}.jpg")
        
        all_w = load_json_file(WARDROBE_FILE, {})
        all_w[safe_id] = { "name": name, "description": ai_data.get("description", ""), "type": ai_data.get("type", "") }
        save_json_file(WARDROBE_FILE, all_w)
        
        return {"status": "success", "id": safe_id}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally: 
        if temp_path: 
            try: os.remove(temp_path)
            except: pass

@router.delete("/wardrobe/{wid}")
async def delete_wardrobe(wid: str):
    all_w = load_json_file(WARDROBE_FILE)
    if wid in all_w:
        del all_w[wid]
        save_json_file(WARDROBE_FILE, all_w)
        try: os.remove(WARD_IMG_DIR / f"{wid}.jpg")
        except: pass
    return {"status": "deleted"}