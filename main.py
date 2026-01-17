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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include batch router
try:
    from batch import router as batch_router
    app.include_router(batch_router)
    logger.info("✓ Batch analyzer module loaded")
except ImportError as e:
    logger.warning(f"⚠️ Batch module not loaded: {e}")
except Exception as e:
    logger.error(f"❌ Batch module error: {e}")
    import traceback
    logger.error(traceback.format_exc())

# Directories
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
if not PERSONA_FILE.exists():
    with open(PERSONA_FILE, "w", encoding="utf-8") as f: json.dump({}, f)
if not HISTORY_FILE.exists():
    with open(HISTORY_FILE, "w", encoding="utf-8") as f: json.dump([], f)
if not STYLES_FILE.exists():
    with open(STYLES_FILE, "w", encoding="utf-8") as f: 
        json.dump({"Standard": "Write a natural, descriptive sentence."}, f)

# Templates & Static
templates = Jinja2Templates(directory=BASE_DIR) 
app.mount("/static", StaticFiles(directory=BASE_DIR), name="static")

# Constants
DEFAULT_MODELS = ["minicpm-v", "llava:v1.6", "qwen2.5-vl", "llama3.2"]
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

def cleanup_temp_files():
    """Deletes files in temp directory older than 1 hour on startup"""
    try:
        now = time.time()
        count = 0
        if not TEMP_DIR.exists(): return
        
        # Safer iteration using pathlib (Fixed crash on locked files)
        for f in TEMP_DIR.iterdir():
            try:
                if f.is_file() and f.stat().st_mtime < now - 3600: # 1 hour
                    f.unlink()
                    count += 1
            except Exception: pass
            
        if count > 0:
            logger.info(f"🧹 Startup: Cleaned {count} old temporary files.")
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")

# Run cleanup immediately
cleanup_temp_files()

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
    img = None
    try:
        if image_url:
            response = requests.get(image_url, stream=True, timeout=15)
            response.raise_for_status()
            img_bytes = io.BytesIO(response.content)
            filename = f"url_img_{int(time.time())}.jpg"
        elif file:
            img_bytes = io.BytesIO(file.file.read())
            filename = file.filename

        img = Image.open(img_bytes)
        img = ImageOps.exif_transpose(img) 
        
        max_size = 1536 
        if img.width > max_size or img.height > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')

        temp_path = TEMP_DIR / f"opt_{int(time.time())}_{re.sub(r'[^a-zA-Z0-9.]', '_', filename)}"
        img.save(temp_path, "JPEG", quality=85, optimize=True)
        
        img.close()
        del img_bytes
        return temp_path

    except Exception as e:
        if img: img.close()
        logger.error(f"Image processing failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# --- OLLAMA HEALTH CHECK ---
def check_ollama_running() -> dict:
    """Check if Ollama service is running and accessible"""
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        if r.status_code == 200:
            return {"running": True, "status": "ok"}
        return {"running": False, "status": "error", "message": f"Ollama returned status {r.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"running": False, "status": "error", "message": "Ollama is not running. Start it with 'ollama serve'"}
    except requests.exceptions.Timeout:
        return {"running": False, "status": "error", "message": "Ollama connection timed out"}
    except Exception as e:
        return {"running": False, "status": "error", "message": str(e)}

# --- MODEL VALIDATION HELPER ---
def get_installed_models() -> List[str]:
    """Robustly fetch models from Ollama regardless of version"""
    try:
        response = ollama.list()
        
        # Handle various response structures
        raw_models = []
        if isinstance(response, list):
            raw_models = response
        elif hasattr(response, 'models'): # Object style (new)
            raw_models = list(response.models)
        elif isinstance(response, dict): # Dictionary style (old)
            raw_models = response.get('models', [])

        clean_models = []
        for m in raw_models:
            name = None
            # Extract name safely from Dict, Object, or String
            if isinstance(m, dict):
                name = m.get('name') or m.get('model') or m.get('id')
            elif isinstance(m, str):
                name = m
            elif hasattr(m, 'model'): # Pydantic object
                name = m.model
            elif hasattr(m, 'name'):
                name = m.name
                
            if name:
                clean_models.append(str(name))
        
        logger.info(f"Loaded models: {clean_models}")
        return clean_models
    except Exception as e:
        logger.error(f"Ollama list error: {e}")
        return []  # Return empty list instead of defaults when Ollama is down

def validate_vision_model(model: str) -> str:
    """Checks if model is vision-capable. Returns a safe model if not."""
    VISION_KEYWORDS = ["vl", "vision", "llava", "moondream", "bakllava", "minicpm"]
    
    available_models = get_installed_models()
    
    # Fuzzy match logic
    found_model = next((m for m in available_models if model.lower() in m.lower()), None)
    
    if not found_model:
        # Try to find ANY vision model
        found_model = next((m for m in available_models if any(k in m.lower() for k in VISION_KEYWORDS)), None)
        
    if not found_model:
        return model # Return original and hope for best
        
    # Check if it's actually a vision model
    is_vision = any(k in found_model.lower() for k in VISION_KEYWORDS)
    if not is_vision:
        # User selected a text model for vision task
        better_model = next((m for m in available_models if any(k in m.lower() for k in VISION_KEYWORDS)), None)
        if better_model:
            logger.info(f"🔄 Auto-switched from text model '{found_model}' to vision model '{better_model}'")
            return better_model
            
    return found_model

# --- MEMORY OPTIMIZED AI CALLS ---
async def enhanced_qwen_analysis(temp_path: Path, model: str) -> dict:
    # 1. Validate/Switch Model
    effective_model = validate_vision_model(model)
    logger.info(f"✓ Using vision model: {effective_model}")
    
    # PASS 1: Scene Analysis (Structure & Camera)
    complex_prompt = """Analyze this image. Return JSON: 
    { 
      "subject": { 
        "description": "A 1-sentence summary of the subject's ACTION and CONTEXT only. Do NOT describe physical features here.",
        "age": "Estimate",
        "ethnicity": "Estimate",
        "face": { "expression": "Visible emotion", "eyes": "Color/Shape", "skin": "Texture" }, 
        "hair": { "color": "Color", "style": "Style" },
        "body_type": "SHORT TAG (e.g. Curvy, Athletic)"
      }, 
      "pose": { "type": "Stance", "orientation": "Angle", "limbs": "Positions", "head": "Tilt" },
      "clothing": { "outfit": "Description", "fit": "Fit details", "accessories": "Items" }, 
      "environment": { "location": "Setting", "lighting": "Light source/mood" }, 
      "camera": { 
         "angle": "Shot angle (Eye-level/Low/High)", 
         "lens": "Lens type (Wide/Telephoto/Macro)", 
         "quality": "Image quality (Film grain/Sharp/Bokeh)" 
      },
      "style": { "aesthetic": "Vibe", "vibe": "Mood" }, 
      "meta_tokens": ["tag1", "tag2", "tag3"]
    }"""
    
    simple_prompt = "Describe this image in JSON format: subject, clothing, environment."
    
    data = {}
    logger.info(f"Pass 1: Analysis with {effective_model}...")
    
    try:
        response1 = ollama.chat(
            model=effective_model, 
            messages=[{'role': 'user', 'content': complex_prompt, 'images': [str(temp_path)]}], 
            options={"temperature": 0.1, "num_predict": 4096, "num_ctx": 8192} 
        )
        raw_response = response1['message']['content']
        logger.info(f"Pass 1 raw response (first 500 chars): {raw_response[:500]}")
        data = extract_json_from_text(raw_response)
        logger.info(f"Pass 1 parsed keys: {list(data.keys()) if data else 'None'}")
    except Exception as e: 
        logger.error(f"Pass 1 error: {e}")
        import traceback
        logger.error(traceback.format_exc())

    if not data or not data.get("subject"):
        logger.warning("Pass 1 failed or empty, trying fallback...")
        try:
            response_fallback = ollama.chat(
                model=effective_model, 
                messages=[{'role': 'user', 'content': simple_prompt, 'images': [str(temp_path)]}], 
                options={"temperature": 0.2, "num_ctx": 8192}
            )
            raw_fallback = response_fallback['message']['content']
            logger.info(f"Fallback raw response (first 500 chars): {raw_fallback[:500]}")
            data = extract_json_from_text(raw_fallback)
            logger.info(f"Fallback parsed keys: {list(data.keys()) if data else 'None'}")
        except Exception as e: 
            logger.error(f"Fallback error: {e}")
            data = {}

    # --- CRITICAL FIX: Ensure dictionary structure ---
    if not isinstance(data, dict): data = {}
    
    if "subject" not in data:
        data["subject"] = {}
    elif isinstance(data["subject"], str):
        # FIX: Convert string subject to dictionary to prevent crashes
        data["subject"] = {"description": data["subject"]}
    elif not isinstance(data["subject"], dict):
        data["subject"] = {}

    # Normalize other fields
    for key in ["pose", "clothing", "environment", "style", "camera"]:
        if key not in data or not isinstance(data[key], dict):
            data[key] = {}
            
    if "face" not in data["subject"] or not isinstance(data["subject"]["face"], dict):
        data["subject"]["face"] = {}
    if "hair" not in data["subject"] or not isinstance(data["subject"]["hair"], dict):
        data["subject"]["hair"] = {}

    # PASS 2: Detail Scan
    if data.get("subject"):
        detail_prompt = """Scan for details. Return JSON:
        {
            "skin_imperfections": "freckles, pores...",
            "body_proportions": { "build": "...", "chest": "...", "shoulders": "...", "waist_ratio": "..." },
            "pose_dynamics": "weight distribution...",
            "material_physics": "fabric tension, drape...",
            "camera_tech": "lens type, depth of field, framing...",
            "small_accessories": "jewelry details..."
        }"""
        
        try:
            response2 = ollama.chat(
                model=effective_model, 
                messages=[{'role': 'user', 'content': detail_prompt, 'images': [str(temp_path)]}], 
                options={"temperature": 0.1, "num_ctx": 8192} 
            )
            details = extract_json_from_text(response2['message']['content'])
            
            if details.get("skin_imperfections"):
                data["subject"]["face"]["skin"] = str(details["skin_imperfections"])
            
            if details.get("body_proportions"):
                data["subject"]["body_proportions"] = details["body_proportions"]

            if details.get("pose_dynamics"):
                if "details" not in data["pose"]: data["pose"]["details"] = ""
                data["pose"]["details"] = str(details["pose_dynamics"])
            
            if details.get("material_physics"):
                current = data["clothing"].get("fit", "")
                data["clothing"]["fit"] = f"{current}, {details['material_physics']}".strip(", ")
            
            # Merge camera details
            if details.get("camera_tech"):
                 if "details" not in data["camera"]: data["camera"]["details"] = ""
                 data["camera"]["details"] = str(details["camera_tech"])

        except Exception as e: 
            logger.error(f"Pass 2 failed: {e}")
    
    return data

# --- SHARED PERSONA LOGIC (FIXED) ---
def apply_persona_to_data(data: dict, persona_id: str, use_ref_mode: bool = False):
    """Injects persona data into the analysis JSON."""
    if persona_id == "none": return data
    all_personas = load_json_file(PERSONA_FILE)
    if persona_id not in all_personas: return data
    persona = all_personas[persona_id]
    
    p_data = persona.get("profile") or persona.get("subject") or {}
    
    if not isinstance(data.get("subject"), dict): data["subject"] = {}
    data["subject"]["name"] = persona.get("name", "Character")
    
    if p_data.get("age"): data["subject"]["age"] = p_data["age"]
    if p_data.get("ethnicity"): data["subject"]["ethnicity"] = p_data["ethnicity"]
    
    # Ensure face dict exists
    if "face" not in data["subject"] or not isinstance(data["subject"]["face"], dict):
        data["subject"]["face"] = {}
    
    # Ensure hair dict exists
    if "hair" not in data["subject"] or not isinstance(data["subject"]["hair"], dict):
        data["subject"]["hair"] = {}
        
    p_face = p_data.get("facial_features", p_data)
    
    # Inject facial features
    for key in ["eyes", "skin", "expression", "face_structure", "nose", "lips"]:
        val = p_face.get(key)
        if val: data["subject"]["face"][key] = val
    
    # Inject hair - check multiple possible locations
    hair_data = p_data.get("hair") or p_face.get("hair")
    if hair_data:
        if isinstance(hair_data, dict):
            if hair_data.get("color"): data["subject"]["hair"]["color"] = hair_data["color"]
            if hair_data.get("style"): data["subject"]["hair"]["style"] = hair_data["style"]
        else:
            data["subject"]["hair"]["style"] = str(hair_data)

    if p_data.get("body_type"): data["subject"]["body_type"] = p_data["body_type"]
    if p_data.get("body_proportions"): data["subject"]["body_proportions"] = p_data["body_proportions"]
    
    # Inject eyewear if present
    if p_data.get("eyewear") and str(p_data["eyewear"]).lower() not in ["none", "no", ""]:
        if not isinstance(data.get("clothing"), dict): data["clothing"] = {}
        current_acc = data["clothing"].get("accessories", "")
        data["clothing"]["accessories"] = f"{current_acc}, {p_data['eyewear']}".strip(", ")
    
    # Inject makeup if present
    if p_data.get("makeup"):
        data["subject"]["face"]["makeup"] = p_data["makeup"]

    if use_ref_mode:
        ref_text = f"Maintain consistent appearance of {persona.get('name', 'Character')} from reference image."
        # Force ref instruction to top
        if "reference_image_instruction" in data: del data["reference_image_instruction"]
        data = {"reference_image_instruction": ref_text, **data}
    elif "reference_image_instruction" in data:
        del data["reference_image_instruction"]
    
    return data

# --- ENDPOINTS ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    if (BASE_DIR / "index.html").exists():
        return templates.TemplateResponse("index.html", {"request": request})
    return {"error": "index.html not found"}

@app.get("/batch-modal", response_class=HTMLResponse)
async def get_batch_modal():
    """Serve the batch analyzer modal HTML"""
    batch_file = BASE_DIR / "batch.html"
    if batch_file.exists():
        with open(batch_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<!-- Batch modal not found -->")

@app.get("/health")
async def health_check(): return {"status": "healthy"}

@app.get("/version")
async def check_version():
    try:
        v_file = BASE_DIR / "version.txt"
        if v_file.exists():
            with open(v_file, "r", encoding="utf-8") as f:
                local_v = f.read().strip()
        else:
            local_v = "1.0"
        r = requests.get(GITHUB_VERSION_URL, timeout=2)
        remote_v = r.text.strip() if r.status_code == 200 else local_v
        return {"local": local_v, "remote": remote_v, "update_available": local_v != remote_v}
    except Exception as e:
        return {"local": "1.0", "remote": "Unknown", "error": str(e)}

@app.post("/trigger-update")
async def trigger_update():
    try:
        if os.name == 'nt': subprocess.Popen("start cmd /c UPDATE.bat", shell=True)
        else: subprocess.Popen(["sh", "UPDATE.sh"])
        os.kill(os.getpid(), signal.SIGTERM)
        return {"status": "updating"}
    except Exception as e: return {"status": "error", "message": str(e)}

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
            data = r.json()
            try:
                for candidate in data.get("candidates", []):
                    for part in candidate.get("content", {}).get("parts", []):
                        if "inlineData" in part: return {"status": "success", "image": part["inlineData"]["data"], "provider": "nanobana"}
            except: pass
            return {"status": "error", "message": "No image found"}
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

# --- SYSTEM ENDPOINTS (VRAM & SHUTDOWN) ---
@app.get("/system/memory-stats")
async def memory_stats():
    """Returns system memory statistics for the UI dashboard"""
    try:
        import psutil
        mem = psutil.virtual_memory()
        
        # Try to get GPU memory if available
        gpu_info = None
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(", ")
                if len(parts) == 2:
                    gpu_info = {
                        "used_mb": int(parts[0]),
                        "total_mb": int(parts[1]),
                        "percent": round(int(parts[0]) / int(parts[1]) * 100, 1)
                    }
        except:
            pass
        
        return {
            "status": "success",
            "ram": {
                "used_mb": round(mem.used / 1024 / 1024),
                "total_mb": round(mem.total / 1024 / 1024),
                "percent": mem.percent
            },
            "gpu": gpu_info
        }
    except ImportError:
        return {"status": "error", "message": "psutil not installed. Run: pip install psutil"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/system/free-vram")
async def free_vram():
    """Forces Ollama model unload and Python GC"""
    try:
        # Get installed models to know what to unload
        installed_names = get_installed_models()
        for model_name in installed_names:
            try:
                requests.post("http://localhost:11434/api/generate", json={"model": model_name, "keep_alive": 0}, timeout=1)
            except: pass
        
        gc.collect()
        return {"status": "success", "message": "VRAM cleanup triggered"}
    except Exception as e: return {"status": "error", "message": str(e)}

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
        if text_prompt and text_prompt.strip():
            filename = "text_import.json"
            response = ollama.chat(model="llama3.2", messages=[{'role': 'system', 'content': 'Convert text to JSON keys: subject, pose, clothing, environment, style.'}, {'role': 'user', 'content': text_prompt}], format="json", options={"num_ctx": 4096})
            data = extract_json_from_text(response['message']['content'])
        else:
            temp_path = process_uploaded_image(file, image_url)
            filename = file.filename if file else "url_image.jpg"
            data = await enhanced_qwen_analysis(temp_path, model)

            # Check if analysis returned an error (model validation failed)
            if data.get("error"):
                return JSONResponse(status_code=400, content={"error": data["error"]})

        if not isinstance(data, dict): data = {}
        
        # Ensure Keys Exist
        for key in ["subject", "environment", "style", "clothing", "pose", "camera"]:
            if not isinstance(data.get(key), dict): data[key] = {}

        data = apply_persona_to_data(data, persona_id, reference_mode)

        # Apply Overrides
        if hair_style_override != "auto": 
             if "face" in data["subject"]: data["subject"]["face"]["hair"] = hair_style_override
        if time_override != "auto": data["environment"]["time"] = time_override
        if style_override != "auto": data["style"]["aesthetic"] = style_override
        
        # Meta Tokens
        if quality_override != "auto":
            if "meta_tokens" not in data: data["meta_tokens"] = []
            if quality_override == "Best": data["meta_tokens"].extend(["8k", "best quality"])
            elif quality_override == "Raw": data["meta_tokens"].extend(["raw photo", "film grain"])

        data["negative_prompt"] = ["lowres", "bad anatomy", "text", "error", "missing fingers", "cropped", "worst quality", "jpeg artifacts"]
        save_history({"filename": filename, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "model": model, "persona": persona_id, "json": data})
        return data
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        safe_file_cleanup(temp_path)
        gc.collect()
        try:
            requests.post("http://localhost:11434/api/generate", json={"model": model, "keep_alive": 0}, timeout=1)
        except: pass

@app.post("/inject-persona")
async def inject_persona_endpoint(request: Request):
    try:
        req_data = await request.json()
        data = req_data.get("json", {})
        persona_id = req_data.get("persona_id")
        use_ref_mode = req_data.get("reference_mode", False) 
        if not data or not persona_id: return {"status": "error", "message": "Missing Data"}
        data = apply_persona_to_data(data, persona_id, use_ref_mode)
        return {"status": "success", "json": data}
    except Exception as e: return {"status": "error", "message": str(e)}

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
        
        system_prompt = """You are an expert Character Designer. Analyze with EXTREME detail.
        Use rich vocabulary (e.g. "piercing crystalline blue"). Describe skin texture, pores, freckles.
        Return strictly valid JSON: { 
          "age": "...", "ethnicity": "...", "face": {"eyes": "...", "skin": "...", "hair": "...", "structure": "..."},
          "body": {"build": "...", "type": "..."}
        }"""
        
        response = ollama.chat(
            model=model, 
            messages=[{'role': 'user', 'content': system_prompt, 'images': [str(temp_path)]}], 
            options={"temperature": 0.2, "num_ctx": 4096}
        )
        ai_data = extract_json_from_text(response['message']['content'])
        
        all_p = load_json_file(PERSONA_FILE, {})
        all_p[safe_id] = { "name": name, "profile": ai_data }
        save_json_file(PERSONA_FILE, all_p)
        return {"status": "success", "id": safe_id, "data": all_p[safe_id]}
    finally: safe_file_cleanup(temp_path)

# --- STYLE MANAGER ENDPOINTS ---
@app.get("/styles")
async def list_styles():
    return load_json_file(STYLES_FILE, {})

@app.post("/styles/analyze")
async def analyze_style_structure(request: Request):
    try:
        data = await request.json()
        raw_prompt = data.get("prompt")
        model = data.get("model", "llama3.2")
        system_prompt = """Analyze the PROMPT STRUCTURE. Format? Order? Vocabulary? 
        Return a SHORT, precise instruction to replicate it (e.g. "Comma-separated tags. Subject first. No verbs.")"""
        response = ollama.chat(
            model=model, 
            messages=[{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': f"Analyze this:\n\n{raw_prompt}"}],
            options={"num_ctx": 4096}
        )
        return {"status": "success", "instruction": response['message']['content']}
    except Exception as e: return {"status": "error", "message": str(e)}

@app.post("/styles/save")
async def save_style(request: Request):
    try:
        data = await request.json()
        styles = load_json_file(STYLES_FILE, {})
        styles[data.get("name")] = data.get("instruction")
        save_json_file(STYLES_FILE, styles)
        return {"status": "success"}
    except Exception as e: return {"status": "error", "message": str(e)}

@app.delete("/styles/{name}")
async def delete_style(name: str):
    styles = load_json_file(STYLES_FILE)
    if name in styles:
        del styles[name]
        save_json_file(STYLES_FILE, styles)
    return {"status": "deleted"}

@app.post("/generate-prompt")
async def generate_natural_prompt(request: Request):
    try:
        data = await request.json()
        json_data = data.get("json")
        model = data.get("model", "llama3.2")
        style_instruction = data.get("style_instruction", "Write a detailed, natural description.")
        sys_prompt = f"""You are an AI Image Prompt Writer.
        INPUT: JSON object describing an image.
        TASK: Write a text prompt text.
        STYLE RULES: {style_instruction}
        IMPORTANT: Only output the prompt. No intro/outro."""
        
        response = ollama.chat(
            model=model, 
            messages=[{'role': 'system', 'content': sys_prompt}, {'role': 'user', 'content': json.dumps(json_data)}],
            options={"num_ctx": 8192, "temperature": 0.6}
        )
        return {"status": "success", "prompt": response['message']['content']}
    except Exception as e: return {"status": "error", "error": str(e)}

@app.post("/generate-tags")
async def generate_tags(file: Optional[UploadFile] = File(None), image_url: Optional[str] = Form(None), model: str = Form("qwen3-vl"), persona_id: str = Form("none"), reference_mode: bool = Form(False)):
    temp_path = None
    try:
        temp_path = process_uploaded_image(file, image_url)
        resp = ollama.chat(
            model=model, 
            messages=[{'role': 'user', 'content': "List concise, comma-separated booru tags. Do not use sentences.", 'images': [str(temp_path)]}],
            options={"num_ctx": 4096}
        )
        return {"positive_tags": resp['message']['content'], "negative_prompt": "lowres"}
    finally: 
        safe_file_cleanup(temp_path)
        gc.collect()
        try:
            requests.post("http://localhost:11434/api/generate", json={"model": model, "keep_alive": 0}, timeout=1)
        except: pass

@app.post("/refine")
async def refine_json(request: Request):
    try:
        data = await request.json()
        resp = ollama.chat(model=data.get("model", "llama3.2"), messages=[{'role': 'system', 'content': f"Update JSON: {data.get('instruction')}"}, {'role': 'user', 'content': json.dumps(data.get("current_json"))}], format="json", options={"num_ctx": 4096})
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

@app.get("/personas")
async def list_personas():
    try:
        data = load_json_file(PERSONA_FILE, {})
        clean = []
        for k, v in data.items():
            if isinstance(v, dict): clean.append({ "id": k, "name": v.get("name", "Unknown"), "subject": v.get("subject", {}) or v.get("profile", {}) })
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
async def list_models():
    """List all installed Ollama models and identify vision-capable ones"""
    VISION_KEYWORDS = ["vl", "vision", "llava", "moondream", "bakllava", "minicpm"]
    try:
        installed_names = get_installed_models()
        models = []
        vision_models = []
        text_models = []
        
        for name in installed_names:
            is_vision = any(kw in name.lower() for kw in VISION_KEYWORDS)
            models.append({"name": name, "is_vision": is_vision, "size": 0})
            if is_vision: vision_models.append(name)
            else: text_models.append(name)
        
        return {
            "status": "success",
            "ollama_running": True,
            "models": models,
            "all_model_names": installed_names,
            "vision_models": vision_models,
            "text_models": text_models,
            "has_vision_model": len(vision_models) > 0,
            "recommended_vision": vision_models[0] if vision_models else None,
            "recommended_text": text_models[0] if text_models else None
        }
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        ollama_status = check_ollama_running()
        return {
            "status": "error", 
            "ollama_running": ollama_status["running"],
            "message": ollama_status.get("message", str(e)), 
            "models": [], 
            "vision_models": [],
            "text_models": [],
            "all_model_names": []
        }

@app.get("/system/ollama-status")
async def ollama_status():
    """Check if Ollama is running"""
    status = check_ollama_running()
    if status["running"]:
        models = get_installed_models()
        status["model_count"] = len(models)
        status["models"] = models
    return status

@app.get("/system/memory-stats")
async def memory_stats():
    """Get current RAM and GPU memory usage"""
    try:
        import psutil
        mem = psutil.virtual_memory()
        
        # Try to get GPU info
        gpu_info = {"available": False}
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(', ')
                if len(parts) >= 2:
                    gpu_info = {
                        "available": True,
                        "used_mb": int(parts[0]),
                        "total_mb": int(parts[1]),
                        "percent": round(int(parts[0]) / int(parts[1]) * 100, 1)
                    }
        except: pass
        
        return {
            "status": "success",
            "ram": {
                "used_mb": round(mem.used / 1024 / 1024),
                "total_mb": round(mem.total / 1024 / 1024),
                "percent": mem.percent
            },
            "gpu": gpu_info
        }
    except ImportError:
        return {"status": "error", "message": "psutil not installed"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/system/stats")
async def system_stats():
    """Get comprehensive system stats for the monitor widget"""
    result = {
        "cpu": {"percent": 0},
        "ram": {"percent": 0, "used_mb": 0, "total_mb": 0},
        "gpu": {"available": False}
    }
    
    try:
        import psutil
        
        # CPU
        result["cpu"]["percent"] = psutil.cpu_percent(interval=0.1)
        
        # RAM
        mem = psutil.virtual_memory()
        result["ram"] = {
            "percent": round(mem.percent, 1),
            "used_mb": round(mem.used / 1024 / 1024),
            "total_mb": round(mem.total / 1024 / 1024)
        }
    except ImportError:
        pass
    except Exception as e:
        logger.error(f"CPU/RAM stats error: {e}")
    
    # GPU (NVIDIA)
    try:
        gpu_result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=2
        )
        if gpu_result.returncode == 0:
            parts = [p.strip() for p in gpu_result.stdout.strip().split(',')]
            if len(parts) >= 4:
                vram_used = int(parts[1])
                vram_total = int(parts[2])
                result["gpu"] = {
                    "available": True,
                    "utilization": int(parts[0]),
                    "vram_used": vram_used,
                    "vram_total": vram_total,
                    "vram_percent": round(vram_used / vram_total * 100, 1) if vram_total > 0 else 0,
                    "temperature": int(parts[3])
                }
    except FileNotFoundError:
        # nvidia-smi not found (no NVIDIA GPU or drivers)
        pass
    except Exception as e:
        logger.error(f"GPU stats error: {e}")
    
    return result

@app.post("/system/free-vram")
async def free_vram():
    """Forces Ollama model unload and Python GC"""
    unloaded = []
    try:
        installed_names = get_installed_models()
        
        for model_name in installed_names:
            try:
                requests.post("http://localhost:11434/api/generate", 
                            json={"model": model_name, "keep_alive": 0}, timeout=2)
                unloaded.append(model_name)
            except: pass
        
        gc.collect()
        logger.info(f"🧹 Unloaded models: {unloaded}")
        return {"status": "success", "message": f"Unloaded {len(unloaded)} models", "unloaded": unloaded}
    except Exception as e: 
        return {"status": "error", "message": str(e)}

@app.get("/history")
async def get_history(): return load_json_file(HISTORY_FILE, [])

@app.delete("/history")
async def clear_history(): save_json_file(HISTORY_FILE, []); return {"status": "cleared"}

@app.post("/shutdown")
async def shutdown():
    import threading
    def stop(): time.sleep(1); os.kill(os.getpid(), signal.SIGTERM)
    threading.Thread(target=stop, daemon=True).start()
    return {"message": "Bye"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")