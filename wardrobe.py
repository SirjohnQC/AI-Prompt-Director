import shutil
import json
import time
import re
import os
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
import ollama

BASE_DIR = Path(__file__).parent
WARDROBE_FILE = BASE_DIR / "wardrobe.json"
WARD_IMG_DIR = BASE_DIR / "wardrobe_images"
WARD_IMG_DIR.mkdir(exist_ok=True)
if not WARDROBE_FILE.exists(): json.dump({}, open(WARDROBE_FILE, "w"))

router = APIRouter()

# --- HELPER FUNCTIONS (Self-contained to avoid import errors) ---
def load_json_file(path, default=None):
    if default is None: default = {}
    try: return json.load(open(path, "r", encoding="utf-8"))
    except: return default

def save_json_file(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def extract_json_from_text(text):
    try:
        text = re.sub(r'```json\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'```\s*', '', text)
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            return json.loads(text[start:end+1])
    except: pass
    return {}

def safe_file_cleanup(path):
    try: os.remove(path)
    except: pass

# --- CORE LOGIC ---
def apply_wardrobe_to_data(data: dict, wardrobe_id: str):
    """Injects wardrobe into the NEW primary_subject schema."""
    if not wardrobe_id or wardrobe_id == "none": return data
    
    wardrobe = load_json_file(WARDROBE_FILE)
    if wardrobe_id not in wardrobe: return data
    item = wardrobe[wardrobe_id]
    
    # FIND TARGET: Handle both old and new schemas
    target = None
    if "primary_subject" in data:
        if "clothing" not in data["primary_subject"]: data["primary_subject"]["clothing"] = {}
        target = data["primary_subject"]["clothing"]
    elif "clothing" in data: # Fallback to old schema
        target = data["clothing"]
    else:
        # Create structure if missing
        data["primary_subject"] = {"clothing": {}}
        target = data["primary_subject"]["clothing"]

    # INJECT
    target["outfit_type"] = item.get("name", "Outfit")
    target["description"] = item.get("description", "")
    
    # CLEANUP CONFLICTS
    # We remove specific 'fit' or 'material' details from the original image 
    # because the new wardrobe item has its own physics.
    if "fit" in target: target["fit"] = "natural fit"
    if "material" in target: del target["material"] 
    
    return data

# --- ROUTES ---

@router.get("/wardrobe")
async def list_wardrobe():
    return [{"id":k, "name":v["name"], "image":f"/wardrobe-image/{k}"} for k,v in load_json_file(WARDROBE_FILE).items()]

@router.get("/wardrobe-image/{wid}")
async def get_w_img(wid):
    p = WARD_IMG_DIR/f"{wid}.jpg"
    return FileResponse(p) if p.exists() else JSONResponse(status_code=404, content={})

@router.post("/wardrobe/create")
async def create_wardrobe(
    name: str = Form(...), 
    file: Optional[UploadFile] = File(None), 
    image_url: Optional[str] = Form(None),
    model: str = Form("qwen2.5-vl")
):
    # Import from main HERE to avoid circular import errors at startup
    from main import process_uploaded_image
    
    if not file and not image_url:
        return {"status": "error", "message": "No image provided"}
    
    temp = None
    try:
        temp = process_uploaded_image(file=file, image_url=image_url)
        resp = ollama.chat(model=model, messages=[{'role':'user', 'content':'Describe this clothing (material, cut, color). JSON: {"description": "...", "type": "..."}', 'images':[str(temp)]}])
        data = extract_json_from_text(resp['message']['content'])
        
        safe_id = f"{int(time.time())}_{re.sub(r'[^a-z0-9]','',name.lower())}"
        shutil.copy(temp, WARD_IMG_DIR/f"{safe_id}.jpg")
        
        w = load_json_file(WARDROBE_FILE)
        w[safe_id] = {"name": name, "description": data.get("description"), "type": data.get("type")}
        save_json_file(WARDROBE_FILE, w)
        return {"status": "success", "id": safe_id}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}
    finally: 
        if temp:
            safe_file_cleanup(temp)

@router.delete("/wardrobe/{wid}")
async def del_wardrobe(wid: str):
    w = load_json_file(WARDROBE_FILE)
    if wid in w: del w[wid]; save_json_file(WARDROBE_FILE, w)
    return {"status": "deleted"}