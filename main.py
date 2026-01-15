import os
import shutil
import json
import logging
import time
import re
import requests
import signal
import base64
import subprocess
import zipfile
import io
import gc
from PIL import Image, ImageOps
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import ollama
import uvicorn

# --- CONFIGURATION ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
BASE_DIR = Path(__file__).parent
TEMP_DIR = BASE_DIR / "temp"
HISTORY_DIR = BASE_DIR / "history"
IMG_DIR = BASE_DIR / "persona_images"
PERSONA_FILE = BASE_DIR / "personas.json"
CONFIG_FILE = BASE_DIR / "config.json"
HISTORY_FILE = HISTORY_DIR / "history.json"

# Ensure directories exist
TEMP_DIR.mkdir(exist_ok=True)
HISTORY_DIR.mkdir(exist_ok=True)
IMG_DIR.mkdir(exist_ok=True)

# Ensure critical files exist
if not PERSONA_FILE.exists():
    with open(PERSONA_FILE, "w", encoding="utf-8") as f: json.dump({}, f)
if not HISTORY_FILE.exists():
    with open(HISTORY_FILE, "w", encoding="utf-8") as f: json.dump([], f)

# Templates & Static
templates = Jinja2Templates(directory=BASE_DIR) 
app.mount("/static", StaticFiles(directory=BASE_DIR), name="static")

# Constants
DEFAULT_MODELS = ["minicpm-v", "llava:v1.6", "qwen3-vl", "llama3.2"]
GITHUB_VERSION_URL = "https://raw.githubusercontent.com/SirjohnQC/AI-Prompt-Director/main/version.txt"

# API Keys
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
FAL_KEY = os.getenv("FAL_KEY")
XAI_KEY = os.getenv("XAI_API_KEY")

class CloudGenRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    model_provider: str 
    api_key: Optional[str] = None

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

def extract_json_from_text(text: str) -> Dict[str, Any]:
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
            pass 
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

def process_uploaded_image(file: Optional[UploadFile] = None, image_url: Optional[str] = None) -> Path:
    if not file and not image_url: raise HTTPException(status_code=400, detail="No image provided.")
    try:
        # 1. Get Byte Stream
        if image_url:
            response = requests.get(image_url, stream=True, timeout=15)
            response.raise_for_status()
            img_bytes = io.BytesIO(response.content)
            filename = f"url_img_{int(time.time())}.jpg"
        elif file:
            img_bytes = io.BytesIO(file.file.read())
            filename = file.filename

        # 2. Open & Optimize with Pillow (RAM SAVER)
        img = Image.open(img_bytes)
        img = ImageOps.exif_transpose(img) # Fix rotation
        
        # 3. Resize if too big (Cap at 1536px)
        max_size = 1536 
        if img.width > max_size or img.height > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # 4. Convert to RGB (Strip Alpha channel)
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')

        # 5. Save Optimized File
        temp_path = TEMP_DIR / f"opt_{int(time.time())}_{re.sub(r'[^a-zA-Z0-9.]', '_', filename)}"
        img.save(temp_path, "JPEG", quality=85, optimize=True)
        
        img.close()
        del img_bytes
        
        return temp_path

    except Exception as e:
        logger.error(f"Image processing failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# --- ROBUST TWO-PASS ANALYSIS (V3: BODY PROPORTIONS) ---
async def enhanced_qwen_analysis(temp_path: Path, model: str) -> dict:
    """Two-pass analysis with Micro-Detail and Structured Body Proportions"""
    
    # PASS 1: Base Structure
    complex_prompt = """Analyze this image. Return JSON: 
    { 
      "subject": { 
        "description": "...", 
        "age": "...",
        "ethnicity": "...",
        "face": { "expression": "...", "eyes": "...", "skin": "..." }, 
        "hair": { "color": "...", "style": "..." },
        "body_type": "..."
      }, 
      "clothing": { "outfit": "...", "fit": "..." }, 
      "environment": { "location": "...", "lighting": "..." }, 
      "style": { "aesthetic": "...", "vibe": "..." }, 
      "meta_tokens": [...] 
    }"""
    
    simple_prompt = """Describe this image in JSON format. Keys: subject, clothing, environment, style."""

    logger.info(f"Pass 1: Trying Complex Analysis with {model}...")
    try:
        response1 = ollama.chat(model=model, messages=[{'role': 'user', 'content': complex_prompt, 'images': [str(temp_path)]}], options={"temperature": 0.1, "num_predict": 3072})
        data = extract_json_from_text(response1['message']['content'])
    except: data = {}

    if not data or not data.get("subject"):
        logger.warning("⚠️ Complex analysis failed. Switching to Simple Mode.")
        try:
            response_fallback = ollama.chat(model=model, messages=[{'role': 'user', 'content': simple_prompt, 'images': [str(temp_path)]}], options={"temperature": 0.2})
            data = extract_json_from_text(response_fallback['message']['content'])
        except: data = {}

    # Safety Normalization
    if not isinstance(data, dict): data = {}
    data.setdefault("subject", {})
    data.setdefault("clothing", {})
    data.setdefault("environment", {})
    if not isinstance(data["subject"].get("face"), dict): data["subject"]["face"] = {}

    # PASS 2: MICRO-DETAILS & STRUCTURED PROPORTIONS
    if data.get("subject"):
        detail_prompt = """Scan for HIGH-FIDELITY details. Return JSON:
        {
            "skin_imperfections": "Freckles, pores, moles, texture...",
            "body_proportions": {
                 "build": "e.g. Voluminous, curvy...",
                 "chest": "e.g. Prominent projection, cleavage...",
                 "shoulders": "e.g. Soft, rounded...",
                 "waist_ratio": "e.g. Hourglass..."
            },
            "material_physics": "Fabric tension, stretch, drape...",
            "camera_tech": "Focal length, depth of field...",
            "lighting_nuance": "Shadow placement, hardness...",
            "small_accessories": "Jewelry, buttons..."
        }"""
        
        try:
            response2 = ollama.chat(model=model, messages=[{'role': 'user', 'content': detail_prompt, 'images': [str(temp_path)]}], options={"temperature": 0.1})
            details = extract_json_from_text(response2['message']['content'])
            
            # 1. Merge Skin
            if details.get("skin_imperfections"):
                current_skin = data["subject"]["face"].get("skin", "")
                imp = details["skin_imperfections"]
                if isinstance(imp, list): imp = ", ".join(imp)
                data["subject"]["face"]["skin"] = f"{current_skin}, {imp}".strip(", ")

            # 2. Merge Structured Body Proportions
            if details.get("body_proportions"):
                data["subject"]["body_proportions"] = details["body_proportions"]
                # Sync generic key for compatibility
                if details["body_proportions"].get("build"):
                    data["subject"]["body_type"] = details["body_proportions"]["build"]

            # 3. Merge Cloth Physics
            if details.get("material_physics"):
                current_fit = data["clothing"].get("fit", "")
                phys = details["material_physics"]
                if isinstance(phys, list): phys = ", ".join(phys)
                data["clothing"]["fit"] = f"{current_fit}, {phys}".strip(", ")

            # 4. Merge Camera
            if details.get("camera_tech"):
                data["camera"] = details["camera_tech"] if isinstance(details["camera_tech"], dict) else {"details": str(details["camera_tech"])}

            # 5. Merge Accessories
            if details.get("small_accessories"):
                current_acc = str(data["clothing"].get("accessories", ""))
                new_acc = details["small_accessories"]
                if isinstance(new_acc, list): new_acc = ", ".join(new_acc)
                data["clothing"]["accessories"] = f"{current_acc}, {new_acc}".strip(", ")
                
        except Exception as e: 
            logger.error(f"Pass 2 failed: {e}")
    
    return data

# ==========================================
# --- SHARED PERSONA LOGIC ---
# ==========================================
def apply_persona_to_data(data: dict, persona_id: str, use_ref_mode: bool = False):
    if persona_id == "none":
        return data

    all_personas = load_json_file(PERSONA_FILE)
    if persona_id not in all_personas:
        return data

    persona = all_personas[persona_id]
    if not persona or not isinstance(persona, dict):
        return data

    # SAFETY CHECK
    if not isinstance(data.get("subject"), dict): data["subject"] = {}
    
    p_subject = persona.get("subject", {})
    
    # 1. Inject Identity
    data["subject"]["name"] = persona.get("name", "Character")
    
    # 2. Overwrite physical traits
    for key in ["age", "ethnicity", "body_type"]:
        if p_subject.get(key): 
            data["subject"][key] = p_subject[key]
    
    if p_subject.get("body_proportions"):
        data["subject"]["body_proportions"] = p_subject["body_proportions"]
        # Also sync generic body_type for backwards compatibility
        if p_subject["body_proportions"].get("build"):
             data["subject"]["body_type"] = p_subject["body_proportions"]["build"]
    elif p_subject.get("body_type"):
        data["subject"]["physique"] = p_subject["body_type"]

    # 3. Smart Hair Merge
    if "hair" in p_subject:
        if not isinstance(data["subject"].get("hair"), dict): data["subject"]["hair"] = {}
        
        if isinstance(p_subject["hair"], dict):
            current_cond = data["subject"]["hair"].get("condition")
            data["subject"]["hair"].update(p_subject["hair"])
            if current_cond and "condition" not in p_subject["hair"]:
                data["subject"]["hair"]["condition"] = current_cond
        else:
            current_cond = data["subject"]["hair"].get("condition", "healthy")
            data["subject"]["hair"]["style"] = str(p_subject["hair"])
            data["subject"]["hair"]["condition"] = current_cond

    # 4. Face attributes & Skin Cleanup
    if not isinstance(data["subject"].get("face"), dict): 
        data["subject"]["face"] = {}
        
    for k in ["face_structure", "eyes", "nose", "lips", "skin", "makeup"]:
        val = p_subject.get(k)
        if val and str(val).lower() not in ["none", "detected", ""]:
            data["subject"]["face"][k] = val

    # Conflict Resolution for Skin
    if p_subject.get("skin"):
        conflicting_keys = ["skin_tone", "complexion", "skin_color"]
        for bad_key in conflicting_keys:
            if bad_key in data["subject"]["face"]:
                del data["subject"]["face"][bad_key]
        
        skin_banned = ["dark skin", "pale skin", "tan skin", "olive skin", "fair skin"]
        if isinstance(data.get("meta_tokens"), list):
            data["meta_tokens"] = [t for t in data["meta_tokens"] if not any(b in t.lower() for b in skin_banned)]
    
    if p_subject.get("tattoos"): data["subject"]["tattoos"] = p_subject.get("tattoos")

    # 5. Regenerate Description
    p_desc_parts = [data["subject"]["name"]]
    if data["subject"].get("age"): p_desc_parts.append(f"{data['subject']['age']} years old")
    if data["subject"].get("ethnicity"): p_desc_parts.append(data['subject']['ethnicity'])
    if data["subject"].get("body_type"): p_desc_parts.append(data['subject']['body_type'])
    p_desc_parts.append("woman")
    data["subject"]["description"] = ", ".join(p_desc_parts)

    # 6. Meta Tokens Scrub & Inject
    if "meta_tokens" not in data or not isinstance(data["meta_tokens"], list): data["meta_tokens"] = []
    
    banned_words = ["hair", "blonde", "blond", "brunette", "redhead", "ginger", "cut", "style"]
    data["meta_tokens"] = [t for t in data["meta_tokens"] if not any(b in t.lower() for b in banned_words)]
    
    if data["subject"]["name"] not in str(data["meta_tokens"]):
        data["meta_tokens"].insert(0, data["subject"]["name"])
        
    p_hair = data["subject"].get("hair", {})
    if isinstance(p_hair, dict):
        if p_hair.get("style"): data["meta_tokens"].insert(1, p_hair.get("style"))
        if p_hair.get("color"): data["meta_tokens"].insert(1, f"{p_hair.get('color')} hair")

    # 7. Reference Mode Instruction
    if use_ref_mode:
        data["reference_image_instruction"] = "Keep the same person from the reference image."
    elif "reference_image_instruction" in data:
        del data["reference_image_instruction"]

    return data

# --- ENDPOINTS ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    if (BASE_DIR / "index.html").exists():
        return templates.TemplateResponse("index.html", {"request": request})
    return {"error": "index.html not found"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/version")
async def check_version():
    try:
        local_v = load_json_file(BASE_DIR / "version.txt", "1.0")
        if isinstance(local_v, dict): local_v = "1.0"
        r = requests.get(GITHUB_VERSION_URL, timeout=2)
        remote_v = r.text.strip() if r.status_code == 200 else local_v
        return {"local": str(local_v).strip(), "remote": str(remote_v).strip(), "update_available": str(local_v).strip() != str(remote_v).strip()}
    except Exception as e:
        return {"local": "1.0", "remote": "Unknown", "error": str(e)}

@app.post("/trigger-update")
async def trigger_update():
    try:
        if os.name == 'nt': subprocess.Popen("start cmd /c UPDATE.bat", shell=True)
        else: subprocess.Popen(["sh", "UPDATE.sh"])
        os.kill(os.getpid(), signal.SIGTERM)
        return {"status": "updating"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/generate-image-cloud")
async def generate_image_cloud(req: CloudGenRequest):
    logger.info(f"Cloud Gen Request: {req.model_provider}")
    
    if req.model_provider == "nanobana":
        key = req.api_key or GOOGLE_KEY
        if not key: return {"status": "error", "message": "Missing Google API Key"}
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={key}"
        payload = {"contents": [{"parts": [{"text": "Generate a high quality image: " + req.prompt}]}], "generationConfig": {"sampleCount": 1}}
        try:
            r = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=45)
            if r.status_code != 200: return {"status": "error", "message": f"Google API Error: {r.text}"}
            data = r.json()
            try:
                for candidate in data.get("candidates", []):
                    for part in candidate.get("content", {}).get("parts", []):
                        if "inlineData" in part: return {"status": "success", "image": part["inlineData"]["data"], "provider": "nanobana"}
            except: pass
            return {"status": "error", "message": "No image found in Google response."}
        except Exception as e: return {"status": "error", "message": str(e)}

    elif req.model_provider == "grok":
        key = req.api_key or XAI_KEY
        if not key: return {"status": "error", "message": "Missing xAI API Key"}
        url = "https://api.x.ai/v1/images/generations"
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        final_prompt = req.prompt + (f" --no {req.negative_prompt}" if req.negative_prompt else "")
        payload = {"model": "grok-2-image", "prompt": final_prompt, "n": 1, "size": "1024x1024", "response_format": "b64_json"}
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=60)
            if r.status_code != 200: return {"status": "error", "message": f"Grok API Error: {r.text}"}
            data = r.json()
            if "data" in data and len(data["data"]) > 0: return {"status": "success", "image": data["data"][0]["b64_json"], "provider": "grok"}
            return {"status": "error", "message": "No image returned from Grok"}
        except Exception as e: return {"status": "error", "message": str(e)}

    elif req.model_provider in ["flux", "illustrious"]:
        key = req.api_key or FAL_KEY
        if not key: return {"status": "error", "message": "Missing Fal.ai API Key"}
        endpoint = "fal-ai/flux/dev" if req.model_provider == "flux" else "fal-ai/illustrious-xl"
        headers = {"Authorization": f"Key {key}", "Content-Type": "application/json"}
        payload = {"prompt": req.prompt, "image_size": {"width": req.width, "height": req.height}, "num_inference_steps": 28, "guidance_scale": 3.5, "enable_safety_checker": False}
        if req.negative_prompt and req.model_provider == "illustrious": payload["negative_prompt"] = req.negative_prompt
        try:
            r = requests.post(f"https://queue.fal.run/{endpoint}", json=payload, headers=headers)
            if r.status_code != 200: return {"status": "error", "message": f"Fal Error: {r.text}"}
            request_id = r.json()['request_id']
            for _ in range(60):
                time.sleep(1)
                status_r = requests.get(f"https://queue.fal.run/{endpoint}/requests/{request_id}", headers=headers)
                status_data = status_r.json()
                if "images" in status_data and status_data["images"]:
                    img_content = requests.get(status_data["images"][0]["url"]).content
                    b64 = base64.b64encode(img_content).decode('utf-8')
                    return {"status": "success", "image": b64, "provider": req.model_provider}
            return {"status": "error", "message": "Timeout waiting for Fal.ai"}
        except Exception as e: return {"status": "error", "message": str(e)}

    return {"status": "error", "message": "Unknown provider"}

# --- MAIN ANALYZE ENDPOINT ---
@app.post("/analyze")
async def analyze_image(
    file: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None),
    text_prompt: Optional[str] = Form(None), 
    model: str = Form("qwen3-vl"),
    persona_id: str = Form("none"),
    time_override: str = Form("auto"),
    expr_override: str = Form("auto"),
    ratio_override: str = Form("auto"),
    style_override: str = Form("auto"),
    quality_override: str = Form("auto"),
    hair_style_override: str = Form("auto"),
    hair_color_override: str = Form("auto"),
    makeup_override: str = Form("auto"),
    glasses_override: str = Form("auto"),
    reference_mode: bool = Form(False) 
):
    temp_path = None
    try:
        data = {}
        
        # PATH A: TEXT
        if text_prompt and text_prompt.strip():
            logger.info("Parsing Text Prompt...")
            filename = "text_import.json"
            response = ollama.chat(model="llama3.2", messages=[{'role': 'system', 'content': 'Convert text to JSON keys: subject, clothing, environment, style.'}, {'role': 'user', 'content': text_prompt}], format="json")
            data = extract_json_from_text(response['message']['content'])
        # PATH B: IMAGE
        else:
            temp_path = process_uploaded_image(file, image_url)
            filename = file.filename if file else "url_image.jpg"
            data = await enhanced_qwen_analysis(temp_path, model)

        # --- CRITICAL SAFETY NORMALIZATION ---
        if not isinstance(data, dict): data = {}
        
        # Force essential keys to be dicts
        for key in ["subject", "clothing", "environment", "style"]:
            if not isinstance(data.get(key), dict): data[key] = {}
            
        # Force nested essential keys
        if not isinstance(data["subject"].get("hair"), dict): data["subject"]["hair"] = {}
        if not isinstance(data["subject"].get("face"), dict): data["subject"]["face"] = {}

        # --- APPLY PERSONA & REFERENCE MODE ---
        data = apply_persona_to_data(data, persona_id, reference_mode)

        # --- OVERRIDES ---
        if hair_style_override != "auto": data["subject"]["hair"]["style"] = hair_style_override
        if hair_color_override != "auto": data["subject"]["hair"]["color"] = hair_color_override
        if makeup_override != "auto": data["subject"]["face"]["makeup"] = makeup_override
        
        if glasses_override != "auto":
            outfit_key = "clothing" if "clothing" in data else "outfit"
            if not isinstance(data.get(outfit_key), dict): data[outfit_key] = {}
            
            acc = str(data[outfit_key].get("accessories", ""))
            clean_acc = re.sub(r"(?i)(,\s*)?(no\s+)?(reading\s+|sun)?glasses(,\s*)?", "", acc).strip(", ")
            
            eyes = str(data["subject"]["face"].get("eyes", ""))
            clean_eyes = re.sub(r"(?i)(,\s*)?(no\s+)?(reading\s+|sun)?glasses(,\s*)?", "", eyes).strip(", ")
            
            if glasses_override == "none": 
                data[outfit_key]["accessories"] = (clean_acc + ", no glasses").strip(", ")
                data["subject"]["face"]["eyes"] = clean_eyes
            else: 
                data[outfit_key]["accessories"] = f"{clean_acc}, {glasses_override}".strip(", ")
                data["subject"]["face"]["eyes"] = f"{clean_eyes}, wearing {glasses_override}".strip(", ")

        if time_override != "auto": data["environment"]["time_indicator"] = time_override
        if expr_override != "auto": data["subject"]["face"]["expression"] = expr_override
        if ratio_override != "auto": data["aspect_ratio"] = ratio_override
        if style_override != "auto": data["style"]["aesthetic"] = style_override

        if quality_override != "auto":
            if not isinstance(data.get("meta_tokens"), list): data["meta_tokens"] = []
            if quality_override == "Best": data["meta_tokens"].extend(["8k", "best quality"])
            elif quality_override == "Raw": data["meta_tokens"].extend(["raw photo", "film grain"])
            elif quality_override == "Phone": data["meta_tokens"].extend(["iphone photo", "candid"])

        data["negative_prompt"] = ["lowres", "bad anatomy", "text", "error", "missing fingers", "cropped", "worst quality", "jpeg artifacts"]

        save_history({"filename": filename, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "model": model, "persona": persona_id, "json": data})
        return data

    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        safe_file_cleanup(temp_path)
        gc.collect()

# --- QUICK PERSONA SWAP ---
@app.post("/inject-persona")
async def inject_persona_endpoint(request: Request):
    """Updates JSON with Persona + Reference Mode without re-analysis."""
    try:
        req_data = await request.json()
        data = req_data.get("json", {})
        persona_id = req_data.get("persona_id")
        use_ref_mode = req_data.get("reference_mode", False) 
        
        if not data or not persona_id: return {"status": "error", "message": "Missing Data"}

        # Safety Normalization
        if not isinstance(data, dict): data = {}
        if not isinstance(data.get("subject"), dict): data["subject"] = {}
        if not isinstance(data["subject"].get("hair"), dict): data["subject"]["hair"] = {}
        if not isinstance(data["subject"].get("face"), dict): data["subject"]["face"] = {}

        data = apply_persona_to_data(data, persona_id, use_ref_mode)

        return {"status": "success", "json": data}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# --- PERSONA CREATION ---
@app.post("/personas/create")
async def create_persona(
    name: str = Form(...),
    file: Optional[UploadFile] = File(None),
    model: str = Form("qwen3-vl"),
    mode: str = Form("create"),
    pid: Optional[str] = Form(None)
):
    temp_path = None
    try:
        if not file: raise HTTPException(status_code=400, detail="File required")
        safe_id = pid if (mode == "edit" and pid) else re.sub(r'[^a-zA-Z0-9]', '_', name.lower())
        temp_path = TEMP_DIR / f"scan_{file.filename}"
        with open(temp_path, "wb") as buffer: shutil.copyfileobj(file.file, buffer)
        shutil.copy(temp_path, IMG_DIR / f"{safe_id}.jpg")
        
        system_prompt = f"""Analyze character profile. Return JSON: 
        {{ 
          "age": "...", 
          "ethnicity": "...", 
          "face_structure": "...", 
          "skin": "...", 
          "eyes": "...", 
          "nose": "...", 
          "lips": "...", 
          "hair": {{ "color": "...", "style": "..." }}, 
          "makeup": "...", 
          "tattoos": "...",
          "body_proportions": {{
             "build": "...",
             "chest": "...",
             "waist_to_chest_ratio": "...",
             "shoulders": "...",
             "dominance": "..."
          }}
        }}"""
        
        response = ollama.chat(model=model, messages=[{'role': 'user', 'content': system_prompt, 'images': [str(temp_path)]}], options={"temperature": 0.1})
        ai_data = extract_json_from_text(response['message']['content'])
        
        if not isinstance(ai_data, dict): ai_data = {}
        if "hair" in ai_data and isinstance(ai_data["hair"], str): ai_data["hair"] = {"style": ai_data["hair"], "color": "Unknown"}
        
        all_p = load_json_file(PERSONA_FILE, {})
        all_p[safe_id] = { "name": name, "subject": ai_data }
        save_json_file(PERSONA_FILE, all_p)
        return {"status": "success", "id": safe_id, "data": all_p[safe_id]}
    finally: safe_file_cleanup(temp_path)

@app.post("/generate-prompt")
async def generate_natural_prompt(request: Request):
    try:
        from prompt_generator import generate_narrative_prompt
        data = await request.json()
        return {"status": "success", "prompt": generate_narrative_prompt(data.get("json"), preferred_model=data.get("model", "llama3.2"))}
    except Exception as e: return {"status": "error", "error": str(e)}

@app.post("/generate-tags")
async def generate_tags(file: Optional[UploadFile] = File(None), image_url: Optional[str] = Form(None), model: str = Form("qwen3-vl"), persona_id: str = Form("none"), reference_mode: bool = Form(False)):
    temp_path = None
    try:
        temp_path = process_uploaded_image(file, image_url)
        resp = ollama.chat(model=model, messages=[{'role': 'user', 'content': "List booru tags.", 'images': [str(temp_path)]}])
        return {"positive_tags": resp['message']['content'], "negative_prompt": "lowres"}
    finally: safe_file_cleanup(temp_path)

@app.post("/refine")
async def refine_json(request: Request):
    try:
        data = await request.json()
        resp = ollama.chat(model=data.get("model", "llama3.2"), messages=[{'role': 'system', 'content': f"Update JSON: {data.get('instruction')}"}, {'role': 'user', 'content': json.dumps(data.get("current_json"))}], format="json")
        return {"status": "success", "json": extract_json_from_text(resp['message']['content'])}
    except Exception as e: return {"status": "error", "error": str(e)}

@app.post("/create-batch-zip")
async def create_batch_zip(request: Request):
    try:
        data = await request.json()
        mem_zip = io.BytesIO()
        with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for f in data.get("files", []): zf.writestr(f["name"], f["content"])
        mem_zip.seek(0)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return StreamingResponse(mem_zip, media_type="application/zip", headers={"Content-Disposition": f"attachment; filename=batch_{timestamp}.zip"})
    except Exception as e: return JSONResponse(status_code=500, content={"error": str(e)})

# --- ROBUST GETTERS (FIXES STUCK LOADING) ---
@app.get("/personas")
async def list_personas():
    try:
        data = load_json_file(PERSONA_FILE, {})
        if not isinstance(data, dict): return []
        clean = []
        for k, v in data.items():
            if isinstance(v, dict):
                clean.append({
                    "id": k, 
                    "name": v.get("name", "Unknown"), 
                    "subject": v.get("subject", {})
                })
        return clean
    except: return []

@app.get("/persona-image/{pid}")
async def get_persona_image(pid: str):
    p = IMG_DIR / f"{pid}.jpg"
    return FileResponse(p) if p.exists() else JSONResponse(status_code=404, content={})

@app.put("/personas/{pid}")
async def update_persona(pid: str, request: Request):
    data = await request.json()
    all_p = load_json_file(PERSONA_FILE)
    if pid in all_p:
        all_p[pid].update(data)
        save_json_file(PERSONA_FILE, all_p)
    return {"status": "updated"}

@app.delete("/personas/{pid}")
async def delete_persona(pid: str):
    all_p = load_json_file(PERSONA_FILE)
    if pid in all_p:
        del all_p[pid]
        save_json_file(PERSONA_FILE, all_p)
        safe_file_cleanup(IMG_DIR / f"{pid}.jpg")
    return {"status": "deleted"}

@app.get("/models")
async def get_models():
    try:
        models = [m['model'] for m in ollama.list().get('models', [])]
        return {"models": models}
    except:
        return {"models": DEFAULT_MODELS}

@app.get("/history")
async def get_history(): return load_json_file(HISTORY_FILE, [])

@app.delete("/history")
async def clear_history(): save_json_file(HISTORY_FILE, []); return {"status": "cleared"}

@app.post("/system/free-vram")
async def free_vram():
    try:
        gc.collect()
        return {"status": "success", "message": "VRAM cleanup triggered"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/shutdown")
async def shutdown():
    import threading
    def stop(): time.sleep(1); os.kill(os.getpid(), signal.SIGTERM)
    threading.Thread(target=stop, daemon=True).start()
    return {"message": "Bye"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")