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
import glob
from PIL import Image, ImageOps
from pathlib import Path
from datetime import datetime, timedelta
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
        for f in glob.glob(str(TEMP_DIR / "*")):
            try:
                if os.stat(f).st_mtime < now - 3600:  # 1 hour
                    os.remove(f)
                    count += 1
            except (OSError, FileNotFoundError):
                pass  # File may have been deleted between glob and stat/remove
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
        
        # 1. OPTIMIZE: Resize extremely large images to save VRAM
        max_size = 1536 
        if img.width > max_size or img.height > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # 2. OPTIMIZE: Strip Alpha channel
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')

        temp_path = TEMP_DIR / f"opt_{int(time.time())}_{re.sub(r'[^a-zA-Z0-9.]', '_', filename)}"
        img.save(temp_path, "JPEG", quality=85, optimize=True)
        
        # 3. OPTIMIZE: Close pointers immediately
        img.close()
        del img_bytes
        return temp_path

    except Exception as e:
        if img: img.close()
        logger.error(f"Image processing failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# --- VISION MODEL VALIDATION ---
def get_installed_models() -> list[str]:
    """Get list of installed model names from Ollama"""
    try:
        installed = ollama.list()
        logger.info(f"Raw ollama.list() type: {type(installed)}")
        
        # Try to log the actual content
        try:
            if hasattr(installed, '__dict__'):
                logger.info(f"ollama.list() __dict__: {installed.__dict__}")
            else:
                logger.info(f"ollama.list() content: {installed}")
        except:
            pass
        
        # Handle different response formats
        models_list = []
        
        if isinstance(installed, dict):
            models_list = installed.get('models', []) or installed.get('Models', [])
        elif isinstance(installed, list):
            models_list = installed
        elif hasattr(installed, 'models'):
            # Object with models attribute (newer ollama python library)
            models_list = list(installed.models) if installed.models else []
        
        logger.info(f"models_list type: {type(models_list)}, length: {len(models_list) if models_list else 0}")
        
        # Extract model names
        installed_names = []
        for i, m in enumerate(models_list):
            logger.info(f"Model {i}: type={type(m)}, value={m}")
            name = None
            if isinstance(m, dict):
                name = m.get('name') or m.get('model') or m.get('id')
            elif isinstance(m, str):
                name = m
            elif hasattr(m, 'model'):
                name = m.model
            elif hasattr(m, 'name'):
                name = m.name
            
            if name:
                installed_names.append(str(name))
        
        logger.info(f"Extracted model names: {installed_names}")
        return installed_names
        
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return []


def validate_vision_model(model: str) -> tuple[bool, str]:
    """
    Validates that the specified model exists and supports vision.
    Returns (is_valid, error_message or validated_model_name)
    """
    VISION_KEYWORDS = ["vl", "vision", "llava", "moondream", "bakllava", "minicpm-v"]
    
    try:
        installed_names = get_installed_models()
        
        if not installed_names:
            logger.warning("No models found via API, attempting to use requested model directly")
            return True, model
        
        model_base = model.split(':')[0].lower()
        
        # Find exact match or partial match
        found_model = None
        for installed_name in installed_names:
            installed_base = installed_name.split(':')[0].lower()
            if model.lower() == installed_name.lower() or model_base == installed_base:
                found_model = installed_name
                break
        
        if not found_model:
            # Try to find any vision model
            for installed_name in installed_names:
                if any(kw in installed_name.lower() for kw in VISION_KEYWORDS):
                    logger.warning(f"⚠️ '{model}' not found. Using '{installed_name}' instead.")
                    return True, installed_name
            
            # Use first available
            if installed_names:
                logger.warning(f"⚠️ '{model}' not found. Using '{installed_names[0]}' instead.")
                return True, installed_names[0]
            return True, model
        
        # Check if it's a vision model
        is_vision = any(kw in found_model.lower() for kw in VISION_KEYWORDS)
        
        if not is_vision:
            for installed_name in installed_names:
                if any(kw in installed_name.lower() for kw in VISION_KEYWORDS):
                    logger.warning(f"⚠️ '{model}' is not a vision model. Using '{installed_name}' instead.")
                    return True, installed_name
            logger.warning(f"⚠️ No vision models installed. Using '{found_model}' anyway.")
        
        return True, found_model
        
    except Exception as e:
        logger.error(f"Error validating model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.warning(f"⚠️ Could not validate model, attempting to use '{model}' directly")
        return True, model


# --- MEMORY OPTIMIZED AI CALLS ---
async def enhanced_qwen_analysis(temp_path: Path, model: str) -> dict:
    """
    Two-pass analysis - uses simplified prompts that force actual image observation.
    """
    
    # Validate that we have a working vision model
    is_valid, result = validate_vision_model(model)
    if not is_valid:
        logger.error(f"❌ Vision model validation failed: {result}")
        return {"error": result, "subject": {}, "pose": {}, "clothing": {}, "environment": {}, "style": {}}
    
    # Use the validated model name (might be corrected)
    model = result
    logger.info(f"✓ Using vision model: {model}")
    
    # PASS 1: Direct observation prompt - NO template, just questions
    observation_prompt = """Analyze this image and return a JSON object describing what you see.

Describe:
1. The person: age, gender, ethnicity, hair color and style, eye color, skin tone, body type
2. Their pose: how are they positioned, what are they doing
3. Their clothing: what exactly are they wearing, colors, materials
4. The environment: where is this, what's in the background
5. The photo style: lighting, camera angle, mood

Return ONLY valid JSON. Be specific about what you actually see in this image."""

    logger.info(f"Pass 1: Image Analysis with {model}...")
    data = {}
    
    try:
        response1 = ollama.chat(
            model=model, 
            messages=[{'role': 'user', 'content': observation_prompt, 'images': [str(temp_path)]}], 
            options={"temperature": 0.2, "num_predict": 2048, "num_ctx": 8192}
        )
        raw = response1['message']['content']
        logger.info(f"Pass 1 raw (first 300 chars): {raw[:300]}")
        parsed = extract_json_from_text(raw)
        
        if parsed:
            # Convert whatever format the model returns to our standard format
            data["subject"] = {}
            data["pose"] = {}
            data["clothing"] = {}
            data["environment"] = {}
            data["style"] = {}
            
            # Extract person info - check various possible keys
            person = parsed.get("person") or parsed.get("subject") or parsed.get("1") or {}
            if isinstance(person, str):
                data["subject"]["description"] = person
            elif isinstance(person, dict):
                data["subject"]["description"] = person.get("description", "")
                data["subject"]["age"] = person.get("age", "")
                data["subject"]["ethnicity"] = person.get("ethnicity", person.get("gender", ""))
                data["subject"]["body_type"] = person.get("body_type", person.get("body", ""))
                
                # Face
                data["subject"]["face"] = {}
                data["subject"]["face"]["eyes"] = person.get("eye_color", person.get("eyes", ""))
                data["subject"]["face"]["skin"] = person.get("skin_tone", person.get("skin", ""))
                data["subject"]["face"]["expression"] = person.get("expression", "")
                
                # Hair
                data["subject"]["hair"] = {}
                hair = person.get("hair", "")
                if isinstance(hair, dict):
                    data["subject"]["hair"]["color"] = hair.get("color", "")
                    data["subject"]["hair"]["style"] = hair.get("style", "")
                elif isinstance(hair, str):
                    data["subject"]["hair"]["style"] = hair
                    # Try to extract color
                    hair_color = person.get("hair_color", "")
                    if hair_color:
                        data["subject"]["hair"]["color"] = hair_color
            
            # Extract pose info
            pose = parsed.get("pose") or parsed.get("2") or parsed.get("position") or {}
            if isinstance(pose, str):
                data["pose"]["type"] = pose
            elif isinstance(pose, dict):
                data["pose"] = pose
            
            # Extract clothing info
            clothing = parsed.get("clothing") or parsed.get("3") or parsed.get("outfit") or {}
            if isinstance(clothing, str):
                data["clothing"]["outfit"] = clothing
            elif isinstance(clothing, dict):
                data["clothing"] = clothing
            
            # Extract environment info
            env = parsed.get("environment") or parsed.get("4") or parsed.get("background") or parsed.get("location") or {}
            if isinstance(env, str):
                data["environment"]["location"] = env
            elif isinstance(env, dict):
                data["environment"] = env
            
            # Extract style info
            style = parsed.get("style") or parsed.get("5") or parsed.get("photo") or {}
            if isinstance(style, str):
                data["style"]["aesthetic"] = style
            elif isinstance(style, dict):
                data["style"] = style
            
            logger.info(f"Pass 1 parsed successfully")
    
    except Exception as e: 
        logger.error(f"Pass 1 error: {e}")
        import traceback
        logger.error(traceback.format_exc())

    # Fallback if parsing failed
    if not data or not data.get("subject"):
        logger.warning("⚠️ Pass 1 failed. Trying simple fallback...")
        try:
            simple_prompt = "Describe this image in JSON format with keys: subject, pose, clothing, environment, style"
            response_fallback = ollama.chat(
                model=model, 
                messages=[{'role': 'user', 'content': simple_prompt, 'images': [str(temp_path)]}], 
                options={"temperature": 0.3, "num_ctx": 8192}
            )
            data = extract_json_from_text(response_fallback['message']['content'])
        except: 
            data = {}

    # Ensure structure exists
    if not isinstance(data, dict): data = {}
    data.setdefault("subject", {})
    data.setdefault("pose", {})
    data.setdefault("clothing", {})
    data.setdefault("environment", {})
    data.setdefault("style", {})
    if not isinstance(data["subject"].get("face"), dict): data["subject"]["face"] = {}
    if not isinstance(data["subject"].get("hair"), dict): data["subject"]["hair"] = {}

    # PASS 2: MICRO-DETAILS (only if we got valid subject data)
    if data.get("subject") and (data["subject"].get("description") or data["subject"].get("age")):
        detail_prompt = """Look at this image for fine details. Return JSON:
{
    "skin": "skin texture and tone",
    "body": "body proportions and build",
    "pose_detail": "specific posture details",
    "fabric": "clothing material and fit",
    "lighting": "light direction and quality",
    "accessories": "any jewelry or accessories"
}"""
        
        try:
            response2 = ollama.chat(
                model=model, 
                messages=[{'role': 'user', 'content': detail_prompt, 'images': [str(temp_path)]}], 
                options={"temperature": 0.2, "num_ctx": 8192}
            )
            details = extract_json_from_text(response2['message']['content'])
            
            if details:
                if details.get("skin"):
                    data["subject"]["face"]["skin"] = details["skin"]
                if details.get("body"):
                    data["subject"]["body_proportions"] = details["body"] if isinstance(details["body"], dict) else {"build": details["body"]}
                if details.get("pose_detail"):
                    data["pose"]["details"] = details["pose_detail"]
                if details.get("fabric"):
                    data["clothing"]["fit"] = details["fabric"]
                if details.get("lighting"):
                    data["environment"]["lighting"] = details["lighting"]
                if details.get("accessories"):
                    acc_val = details["accessories"]
                    # Clean up "None visible" and similar placeholders
                    if isinstance(acc_val, str):
                        acc_val = re.sub(r',?\s*(None\s*(visible)?|N/A|none)\s*,?', '', acc_val, flags=re.IGNORECASE).strip(", ")
                    if acc_val:
                        data["clothing"]["accessories"] = acc_val
                    
        except Exception as e: 
            logger.error(f"Pass 2 failed: {e}")
    
    return data

# --- SHARED PERSONA LOGIC (Works with OLD format: subject, pose, clothing, environment) ---
def apply_persona_to_data(data: dict, persona_id: str, use_ref_mode: bool = False):
    """
    Injects persona data into the analysis JSON.
    Works with OLD format (subject.face, subject.hair, etc.)
    """
    if persona_id == "none": 
        return data
    
    all_personas = load_json_file(PERSONA_FILE)
    if persona_id not in all_personas: 
        return data
    
    persona = all_personas[persona_id]
    
    # Get persona data - prefer profile, fall back to subject
    p_data = persona.get("profile") or persona.get("subject") or {}
    
    # Ensure target structure exists
    if not isinstance(data.get("subject"), dict):
        data["subject"] = {}
    if not isinstance(data["subject"].get("face"), dict):
        data["subject"]["face"] = {}
    if not isinstance(data["subject"].get("hair"), dict):
        data["subject"]["hair"] = {}
    
    # 1. Name
    data["subject"]["name"] = persona.get("name", "Character")
    
    # 2. Age and Ethnicity
    if p_data.get("age"):
        data["subject"]["age"] = p_data["age"]
    if p_data.get("ethnicity"):
        data["subject"]["ethnicity"] = p_data["ethnicity"]
    
    # 3. Facial features
    face = data["subject"]["face"]
    
    # Check if persona has nested facial_features (new profile format)
    p_facial = p_data.get("facial_features", {})
    if p_facial:
        for key in ["eyes", "eye_color", "nose", "lips", "skin_tone", "expression", "face_structure"]:
            if p_facial.get(key):
                face[key] = p_facial[key]
    else:
        # Old format - fields at root level
        for key in ["eyes", "eye_color", "nose", "lips", "skin", "expression", "face_structure"]:
            if p_data.get(key):
                face[key] = p_data[key]
    
    # 4. Hair
    hair = data["subject"]["hair"]
    if p_facial.get("hair_color"):
        hair["color"] = p_facial["hair_color"]
    elif p_data.get("hair_color"):
        hair["color"] = p_data["hair_color"]
    elif isinstance(p_data.get("hair"), dict) and p_data["hair"].get("color"):
        hair["color"] = p_data["hair"]["color"]
    
    if p_facial.get("hair_style"):
        hair["style"] = p_facial["hair_style"]
    elif p_data.get("hair_style"):
        hair["style"] = p_data["hair_style"]
    elif isinstance(p_data.get("hair"), dict) and p_data["hair"].get("style"):
        hair["style"] = p_data["hair"]["style"]
    
    # 5. Body type and proportions
    if p_data.get("body_type"):
        data["subject"]["body_type"] = p_data["body_type"]
    if p_data.get("body_proportions"):
        data["subject"]["body_proportions"] = p_data["body_proportions"]
    
    # 6. Skin details
    if p_data.get("skin_details"):
        face["skin"] = p_data["skin_details"]
    elif p_data.get("skin"):
        face["skin"] = p_data["skin"]
    
    # 7. Makeup
    if p_data.get("makeup_style"):
        face["makeup"] = p_data["makeup_style"]
    elif p_data.get("makeup"):
        face["makeup"] = p_data["makeup"]
    
    # 8. Eyewear
    if p_data.get("eyewear") and str(p_data["eyewear"]).lower() not in ["none", "no", "n/a", ""]:
        if not isinstance(data.get("clothing"), dict):
            data["clothing"] = {}
        current_acc = data["clothing"].get("accessories", "")
        data["clothing"]["accessories"] = f"{current_acc}, {p_data['eyewear']}".strip(", ")
    
    # 9. Tattoos
    if p_data.get("tattoos") and str(p_data["tattoos"]).lower() not in ["none", "none visible", ""]:
        data["subject"]["tattoos"] = p_data["tattoos"]
    
    # 10. Reference mode
    if use_ref_mode:
        data["reference_image_instruction"] = f"Maintain consistent appearance of {persona.get('name', 'Character')} from reference image."
    elif "reference_image_instruction" in data:
        del data["reference_image_instruction"]
    
    return data

# --- ENDPOINTS ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    if (BASE_DIR / "index.html").exists():
        return templates.TemplateResponse("index.html", {"request": request})
    return {"error": "index.html not found"}

@app.get("/favicon.ico")
async def favicon():
    """Serve favicon if exists, otherwise return empty response"""
    favicon_path = BASE_DIR / "favicon.ico"
    if favicon_path.exists():
        return FileResponse(favicon_path)
    # Return a minimal 1x1 transparent ICO to prevent 404 spam
    return JSONResponse(content={}, status_code=204)

@app.get("/health")
async def health_check(): return {"status": "healthy"}

@app.get("/models")
async def list_models():
    """List all installed Ollama models and identify vision-capable ones"""
    VISION_KEYWORDS = ["vl", "vision", "llava", "moondream", "bakllava", "minicpm-v"]
    
    try:
        installed_names = get_installed_models()
        
        models = []
        vision_models = []
        text_models = []
        
        for name in installed_names:
            is_vision = any(kw in name.lower() for kw in VISION_KEYWORDS)
            models.append({
                "name": name,
                "is_vision": is_vision,
                "size": 0
            })
            if is_vision:
                vision_models.append(name)
            else:
                text_models.append(name)
        
        return {
            "status": "success",
            "models": models,
            # Flat lists for easy dropdown population
            "all_model_names": installed_names,
            "vision_models": vision_models,
            "text_models": text_models,
            "has_vision_model": len(vision_models) > 0,
            "recommended_vision": vision_models[0] if vision_models else None,
            "recommended_text": text_models[0] if text_models else (installed_names[0] if installed_names else None)
        }
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"status": "error", "message": str(e), "models": [], "vision_models": [], "text_models": [], "all_model_names": []}

@app.get("/models/validate/{model_name:path}")
async def validate_model(model_name: str):
    """Check if a specific model is valid and vision-capable"""
    is_valid, result = validate_vision_model(model_name)
    if is_valid:
        return {"status": "valid", "model": result, "message": f"✓ {result} is a valid vision model"}
    else:
        return {"status": "invalid", "error": result}

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
    unloaded = []
    try:
        # Get all installed models and unload them
        installed_names = get_installed_models()
        
        for model_name in installed_names:
            try:
                requests.post("http://localhost:11434/api/generate", 
                            json={"model": model_name, "keep_alive": 0}, timeout=2)
                unloaded.append(model_name)
            except:
                pass
        
        gc.collect()
        logger.info(f"🧹 Unloaded models: {unloaded}")
        return {"status": "success", "message": f"Unloaded {len(unloaded)} models", "unloaded": unloaded}
    except Exception as e: 
        return {"status": "error", "message": str(e)}

@app.post("/analyze")
async def analyze_image(
    file: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None),
    text_prompt: Optional[str] = Form(None), 
    model: str = Form("qwen3-vl:8b"),
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
            response = ollama.chat(model="llama3.2", messages=[{'role': 'system', 'content': 'Convert text to structured JSON with keys: prompt_type, subject_details, pose_and_action, background_environment, lighting_and_atmosphere, technical_specs. Use detailed, evocative descriptions.'}, {'role': 'user', 'content': text_prompt}], format="json", options={"num_ctx": 8192})
            data = extract_json_from_text(response['message']['content'])
        else:
            temp_path = process_uploaded_image(file, image_url)
            filename = file.filename if file else "url_image.jpg"
            data = await enhanced_qwen_analysis(temp_path, model)
            
            # Check if analysis returned an error (model validation failed)
            if data.get("error"):
                return JSONResponse(
                    status_code=400, 
                    content={
                        "error": data["error"],
                        "message": "Vision model not available. Please check Ollama is running and has a vision model installed.",
                        "hint": "Run 'ollama pull qwen2-vl' or 'ollama pull llava' to install a vision model."
                    }
                )

        if not isinstance(data, dict): data = {}
        
        # Ensure OLD structure exists (this is what the model returns)
        data.setdefault("subject", {})
        data.setdefault("pose", {})
        data.setdefault("clothing", {})
        data.setdefault("environment", {})
        data.setdefault("style", {})
        if not isinstance(data["subject"].get("face"), dict): data["subject"]["face"] = {}
        if not isinstance(data["subject"].get("hair"), dict): data["subject"]["hair"] = {}

        # Apply persona data
        data = apply_persona_to_data(data, persona_id, reference_mode)

        # Apply overrides to OLD structure
        if hair_style_override != "auto": 
            data["subject"]["hair"]["style"] = hair_style_override
        if hair_color_override != "auto": 
            data["subject"]["hair"]["color"] = hair_color_override
        if makeup_override != "auto": 
            data["subject"]["face"]["makeup"] = makeup_override
        
        if glasses_override != "auto":
            current_acc = str(data["clothing"].get("accessories", ""))
            if glasses_override == "none":
                clean_acc = re.sub(r"(?i)(,\s*)?(no\s+)?(reading\s+|sun)?glasses(,\s*)?", "", current_acc).strip(", ")
                data["clothing"]["accessories"] = clean_acc if clean_acc else "none"
            else:
                data["clothing"]["accessories"] = f"{current_acc}, {glasses_override}".strip(", ") if current_acc else glasses_override

        if time_override != "auto": 
            data["environment"]["time_indicator"] = time_override
        if expr_override != "auto": 
            data["subject"]["face"]["expression"] = expr_override
        if ratio_override != "auto": 
            data["aspect_ratio"] = ratio_override
        if style_override != "auto": 
            data["style"]["aesthetic"] = style_override

        if quality_override != "auto":
            if not isinstance(data.get("meta_tokens"), list): data["meta_tokens"] = []
            if quality_override == "Best": data["meta_tokens"].extend(["8k", "best quality", "highly detailed"])
            elif quality_override == "Raw": data["meta_tokens"].extend(["raw photo", "film grain", "natural lighting"])
            elif quality_override == "Phone": data["meta_tokens"].extend(["iphone photo", "candid", "natural"])

        # Final cleanup: remove "None visible" and similar from accessories
        if data.get("clothing", {}).get("accessories"):
            acc = data["clothing"]["accessories"]
            if isinstance(acc, str):
                acc = re.sub(r',?\s*(None\s*(visible)?|N/A|none)\s*,?', '', acc, flags=re.IGNORECASE).strip(", ")
                data["clothing"]["accessories"] = acc if acc else None

        data["negative_prompt"] = ["lowres", "bad anatomy", "text", "error", "missing fingers", "cropped", "worst quality", "jpeg artifacts", "blurry", "watermark"]
        save_history({"filename": filename, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "model": model, "persona": persona_id, "json": data})
        return data
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        # AGGRESSIVE CLEANUP after every analysis
        safe_file_cleanup(temp_path)
        gc.collect()
        # Unload vision model from VRAM immediately
        try:
            requests.post("http://localhost:11434/api/generate", 
                         json={"model": model, "keep_alive": 0}, timeout=2)
            logger.info(f"🧹 Unloaded {model} from VRAM")
        except Exception as e:
            logger.debug(f"Model unload request: {e}")

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
    model: str = Form("qwen3-vl:8b"),
    mode: str = Form("create"),
    pid: Optional[str] = Form(None),
    rescan: bool = Form(False)  # New param: whether to re-analyze the image
):
    """
    Create or edit a persona.
    
    Modes:
    - create: Requires file, creates new persona
    - edit: File optional. If file provided OR rescan=True, re-analyzes image.
            Otherwise just updates name/metadata without changing analysis.
    """
    temp_path = None
    try:
        safe_id = pid if (mode == "edit" and pid) else re.sub(r'[^a-zA-Z0-9]', '_', name.lower())
        
        # Determine if we need to run AI analysis
        run_analysis = False
        existing_persona = None
        
        if mode == "edit":
            # Load existing persona data
            all_p = load_json_file(PERSONA_FILE, {})
            existing_persona = all_p.get(safe_id, {})
            
            if file and file.filename:
                # New image uploaded - save it and run analysis
                temp_path = TEMP_DIR / f"scan_{file.filename}"
                with open(temp_path, "wb") as buffer: 
                    shutil.copyfileobj(file.file, buffer)
                shutil.copy(temp_path, IMG_DIR / f"{safe_id}.jpg")
                run_analysis = True
                logger.info(f"Edit mode: New image uploaded for {safe_id}")
            elif rescan:
                # Rescan existing image
                existing_img = IMG_DIR / f"{safe_id}.jpg"
                if existing_img.exists():
                    temp_path = existing_img
                    run_analysis = True
                    logger.info(f"Edit mode: Rescanning existing image for {safe_id}")
                else:
                    raise HTTPException(status_code=400, detail="No existing image to rescan")
            else:
                # Just update name, keep existing data
                if existing_persona:
                    existing_persona["name"] = name
                    all_p[safe_id] = existing_persona
                    save_json_file(PERSONA_FILE, all_p)
                    logger.info(f"Edit mode: Updated name only for {safe_id}")
                    return {"status": "success", "id": safe_id, "data": existing_persona}
                else:
                    raise HTTPException(status_code=404, detail="Persona not found")
        else:
            # Create mode - file required
            if not file or not file.filename:
                raise HTTPException(status_code=400, detail="File required for new persona")
            temp_path = TEMP_DIR / f"scan_{file.filename}"
            with open(temp_path, "wb") as buffer: 
                shutil.copyfileobj(file.file, buffer)
            shutil.copy(temp_path, IMG_DIR / f"{safe_id}.jpg")
            run_analysis = True
        
        if not run_analysis:
            raise HTTPException(status_code=400, detail="No action to perform")
        
        # COMPREHENSIVE PERSONA ANALYSIS
        system_prompt = """You are an expert Character Designer creating a PERSISTENT CHARACTER PROFILE.
Analyze this person with EXTREME detail for AI image generation consistency.

IMPORTANT: Look carefully at the image and extract SPECIFIC details. Do NOT use generic descriptions.

Return strictly valid JSON in this EXACT format:
{
  "age": "Specific age estimate (e.g. '25', 'early 30s', 'mid 20s')",
  "ethnicity": "Specific ethnicity/heritage (e.g. 'Mediterranean', 'East Asian', 'Caucasian', 'Latina')",
  "facial_features": {
    "face_structure": "Bone structure, jaw shape, cheekbones",
    "eyes": "SPECIFIC eye color (brown/blue/green/hazel/grey) with shape and details",
    "eye_color": "Just the color (e.g. 'dark brown', 'hazel', 'blue-green')",
    "nose": "Shape and size",
    "lips": "Shape, fullness, color",
    "skin_tone": "Specific skin tone description",
    "hair_color": "Specific hair color (e.g. 'dark brown', 'black', 'blonde', 'auburn')",
    "hair_style": "Length and style (e.g. 'long wavy', 'shoulder-length straight', 'short curly')"
  },
  "eyewear": "Describe any glasses/sunglasses OR 'None' if not wearing any",
  "body_type": "Single word (Slim/Athletic/Curvy/Petite/Average/Muscular/Voluptuous)",
  "body_proportions": {
    "build": "Overall physique description",
    "chest": "Bust/chest description",
    "shoulders": "Width and shape",
    "waist_to_chest_ratio": "Body shape (hourglass/pear/athletic/rectangle)"
  },
  "skin_details": "Complexion, texture, any freckles/marks",
  "tattoos": "Describe visible tattoos with placement, OR 'None visible'",
  "makeup_style": "Makeup description if visible, OR 'Natural/minimal'"
}

CRITICAL INSTRUCTIONS:
1. For EYES: Look at the actual eye color in the image. Common colors: brown, dark brown, light brown, hazel, green, blue, grey, blue-green
2. For HAIR COLOR: Be specific - not just "brown" but "dark brown", "chestnut brown", "light brown with highlights"
3. For HAIR STYLE: Include length (short/medium/long) and texture (straight/wavy/curly)
4. For EYEWEAR: If wearing glasses, describe the style (e.g. "black-framed glasses", "gold aviator sunglasses")
5. For AGE: Give a specific number or narrow range, not just "young adult"
6. For ETHNICITY: Be specific based on visible features

Be EXTREMELY detailed. This profile maintains character consistency across generated images."""
        
        response = ollama.chat(
            model=model, 
            messages=[{'role': 'user', 'content': system_prompt, 'images': [str(temp_path)]}], 
            options={"temperature": 0.15, "num_ctx": 8192, "num_predict": 2048}
        )
        ai_data = extract_json_from_text(response['message']['content'])
        logger.info(f"Persona analysis returned: {list(ai_data.keys()) if ai_data else 'None'}")
        
        if not isinstance(ai_data, dict): ai_data = {}
        
        # Build legacy subject format for backward compatibility
        legacy_subject = {}
        
        # Age and Ethnicity (new separate fields)
        if ai_data.get("age"):
            legacy_subject["age"] = ai_data["age"]
        if ai_data.get("ethnicity"):
            legacy_subject["ethnicity"] = ai_data["ethnicity"]
        
        # Facial features
        if ai_data.get("facial_features"):
            ff = ai_data["facial_features"]
            
            # Eyes - get specific eye color
            if ff.get("eyes"):
                legacy_subject["eyes"] = ff["eyes"]
            if ff.get("eye_color"):
                legacy_subject["eye_color"] = ff["eye_color"]
            
            # Other facial features
            for key in ["nose", "lips", "face_structure", "skin_tone"]:
                if ff.get(key):
                    legacy_subject[key] = ff[key]
            
            # Hair - separate color and style
            if ff.get("hair_color"):
                legacy_subject["hair_color"] = ff["hair_color"]
            if ff.get("hair_style"):
                legacy_subject["hair_style"] = ff["hair_style"]
            
            # Combined hair field
            hair_parts = []
            if ff.get("hair_color"): hair_parts.append(ff["hair_color"])
            if ff.get("hair_style"): hair_parts.append(ff["hair_style"])
            if hair_parts:
                legacy_subject["hair"] = {"color": ff.get("hair_color", ""), "style": ff.get("hair_style", "")}
        
        # Eyewear (new field)
        if ai_data.get("eyewear"):
            eyewear_val = ai_data["eyewear"]
            if str(eyewear_val).lower() not in ["none", "no", "n/a", ""]:
                legacy_subject["eyewear"] = eyewear_val
        
        # Body type (single descriptor)
        if ai_data.get("body_type"):
            legacy_subject["body_type"] = ai_data["body_type"]
        
        # Body proportions (detailed)
        if ai_data.get("body_proportions"):
            legacy_subject["body_proportions"] = ai_data["body_proportions"]
        
        # Skin details
        if ai_data.get("skin_details"):
            legacy_subject["skin"] = ai_data["skin_details"]
        
        # Tattoos
        if ai_data.get("tattoos"):
            tattoo_val = ai_data["tattoos"]
            if str(tattoo_val).lower() not in ["none", "none visible", "no visible tattoos", ""]:
                legacy_subject["tattoos"] = tattoo_val
        
        # Makeup
        if ai_data.get("makeup_style"):
            legacy_subject["makeup"] = ai_data["makeup_style"]
        
        all_p = load_json_file(PERSONA_FILE, {})
        all_p[safe_id] = { 
            "name": name, 
            "subject": legacy_subject,  # For backward compatibility
            "profile": ai_data  # New structured format
        }
        save_json_file(PERSONA_FILE, all_p)
        logger.info(f"Persona saved with keys: {list(legacy_subject.keys())}")
        return {"status": "success", "id": safe_id, "data": all_p[safe_id]}
    finally:
        # Only cleanup temp files, not permanent persona images
        if temp_path and temp_path.parent == TEMP_DIR:
            safe_file_cleanup(temp_path)

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
            options={"num_ctx": 8192}
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
        
        # Check if persona data is present
        has_persona = json_data.get("persona_name") or json_data.get("subject_details", {}).get("tattoos")
        
        sys_prompt = f"""You are an expert AI Image Prompt Writer for Flux/Midjourney/Stable Diffusion.
INPUT: JSON object describing an image scene.
TASK: Transform JSON into a vivid, cinematic text prompt.

STYLE RULES: {style_instruction}

QUALITY GUIDELINES:
- Use specific, evocative vocabulary (e.g. "crimson silk" not "red fabric")
- Include sensory details: textures, lighting quality, atmosphere
- Flow naturally: Subject → Appearance → Body → Outfit → Pose → Environment → Mood
- Be SPECIFIC about colors, materials, and spatial relationships
- If persona_name is present, mention the character by name
- Include body_type and body_proportions details naturally (e.g. "her athletic, hourglass figure")
- Include tattoos if present with their description and placement
- Include all facial features: eyes, lips, skin texture, hair

OUTPUT: Only the prompt text. No explanations, no intro/outro."""

        response = ollama.chat(
            model=model, 
            messages=[{'role': 'system', 'content': sys_prompt}, {'role': 'user', 'content': json.dumps(json_data)}],
            options={"num_ctx": 8192, "temperature": 0.7, "num_predict": 1024}
        )
        return {"status": "success", "prompt": response['message']['content']}
    except Exception as e: return {"status": "error", "error": str(e)}

@app.post("/generate-tags")
async def generate_tags(file: Optional[UploadFile] = File(None), image_url: Optional[str] = Form(None), model: str = Form("qwen3-vl:8b"), persona_id: str = Form("none"), reference_mode: bool = Form(False)):
    temp_path = None
    try:
        temp_path = process_uploaded_image(file, image_url)
        
        # Build tag prompt
        tag_prompt = """Generate booru-style tags for this image. 
        FORMAT: Comma-separated tags only. NO sentences.
        ORDER: Quality tags, character count (1girl/1boy), physical features, clothing, pose, environment
        STYLE: Use underscores for multi-word tags (e.g. long_hair, blue_eyes, white_dress)
        Include: hair color/style, eye color, body type, clothing details, pose, background, lighting mood
        Start with: masterpiece, best quality"""
        
        resp = ollama.chat(
            model=model, 
            messages=[{'role': 'user', 'content': tag_prompt, 'images': [str(temp_path)]}],
            options={"num_ctx": 8192, "temperature": 0.3}
        )
        
        raw_tags = resp['message']['content']
        
        # Inject persona tags if selected
        if persona_id != "none":
            all_personas = load_json_file(PERSONA_FILE)
            if persona_id in all_personas:
                persona = all_personas[persona_id]
                p_subject = persona.get("subject", {})
                p_profile = persona.get("profile", {})
                persona_tags = []
                
                # Add name tag
                if persona.get("name"):
                    safe_name = re.sub(r'[^a-z0-9_]', '_', persona["name"].lower())
                    persona_tags.append(safe_name)
                
                # Use new profile format if available, fallback to subject
                facial_features = p_profile.get("facial_features", {})
                
                # Add hair tags (check both formats)
                hair_data = facial_features.get("hair") or p_subject.get("hair")
                if hair_data:
                    if isinstance(hair_data, dict):
                        if hair_data.get("color"):
                            color = hair_data["color"].split(',')[0].strip().lower().replace(' ', '_')
                            persona_tags.append(f"{color}_hair")
                        if hair_data.get("style"):
                            style = hair_data["style"].split(',')[0].strip().lower().replace(' ', '_')
                            if len(style.split('_')) <= 3:
                                persona_tags.append(style)
                    else:
                        # String format - extract first few words
                        hair_str = str(hair_data).lower()
                        if 'blonde' in hair_str: persona_tags.append('blonde_hair')
                        elif 'brown' in hair_str: persona_tags.append('brown_hair')
                        elif 'black' in hair_str: persona_tags.append('black_hair')
                        elif 'red' in hair_str: persona_tags.append('red_hair')
                        elif 'auburn' in hair_str: persona_tags.append('auburn_hair')
                        if 'long' in hair_str: persona_tags.append('long_hair')
                        elif 'short' in hair_str: persona_tags.append('short_hair')
                        if 'wavy' in hair_str: persona_tags.append('wavy_hair')
                        elif 'curly' in hair_str: persona_tags.append('curly_hair')
                        elif 'straight' in hair_str: persona_tags.append('straight_hair')
                
                # Add eye color (check both formats)
                eyes_data = facial_features.get("eyes") or p_subject.get("eyes")
                if eyes_data:
                    eyes_str = str(eyes_data).lower()
                    if 'blue' in eyes_str: persona_tags.append('blue_eyes')
                    elif 'green' in eyes_str: persona_tags.append('green_eyes')
                    elif 'brown' in eyes_str: persona_tags.append('brown_eyes')
                    elif 'hazel' in eyes_str: persona_tags.append('hazel_eyes')
                    elif 'grey' in eyes_str or 'gray' in eyes_str: persona_tags.append('grey_eyes')
                
                # Add body type tag
                body_type = p_profile.get("body_type") or p_subject.get("body_type")
                if body_type:
                    bt = str(body_type).lower().replace(' ', '_')
                    if bt in ['curvy', 'athletic', 'slim', 'petite', 'voluptuous', 'muscular', 'slender']:
                        persona_tags.append(bt)
                
                # Add body proportion tags
                body_props = p_profile.get("body_proportions") or p_subject.get("body_proportions", {})
                if isinstance(body_props, dict):
                    if body_props.get("waist_to_chest_ratio"):
                        ratio = str(body_props["waist_to_chest_ratio"]).lower()
                        if 'hourglass' in ratio: persona_tags.append('hourglass_figure')
                        elif 'pear' in ratio: persona_tags.append('pear_shaped')
                    if body_props.get("chest"):
                        chest = str(body_props["chest"]).lower()
                        if 'large' in chest or 'big' in chest: persona_tags.append('large_breasts')
                        elif 'medium' in chest: persona_tags.append('medium_breasts')
                        elif 'small' in chest: persona_tags.append('small_breasts')
                
                # Add tattoo tags
                tattoos = p_profile.get("tattoos") or p_subject.get("tattoos")
                if tattoos and str(tattoos).lower() not in ["none", "none visible", "no visible tattoos", ""]:
                    persona_tags.append('tattoo')
                    tattoo_str = str(tattoos).lower()
                    if 'neck' in tattoo_str: persona_tags.append('neck_tattoo')
                    if 'arm' in tattoo_str: persona_tags.append('arm_tattoo')
                    if 'back' in tattoo_str: persona_tags.append('back_tattoo')
                    if 'floral' in tattoo_str or 'flower' in tattoo_str: persona_tags.append('floral_tattoo')
                
                # Reference mode tags
                if reference_mode:
                    persona_tags.append("character_reference")
                    persona_tags.append("consistent_character")
                
                # Deduplicate tags
                persona_tags = list(dict.fromkeys(persona_tags))
                
                # Prepend persona tags after quality tags
                if persona_tags:
                    raw_tags = raw_tags.strip()
                    # Insert after first few quality tags
                    parts = raw_tags.split(',', 3)
                    if len(parts) >= 3:
                        raw_tags = f"{parts[0]}, {parts[1]}, {', '.join(persona_tags)}, {','.join(parts[2:])}"
                    else:
                        raw_tags = f"{', '.join(persona_tags)}, {raw_tags}"
        
        negative = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, jpeg artifacts, signature, watermark, blurry"
        
        return {"positive_tags": raw_tags, "negative_prompt": negative}
    finally: safe_file_cleanup(temp_path)

@app.post("/refine")
async def refine_json(request: Request):
    try:
        data = await request.json()
        resp = ollama.chat(model=data.get("model", "llama3.2"), messages=[{'role': 'system', 'content': f"Update JSON: {data.get('instruction')}"}, {'role': 'user', 'content': json.dumps(data.get("current_json"))}], format="json", options={"num_ctx": 8192})
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
            if isinstance(v, dict): clean.append({ "id": k, "name": v.get("name", "Unknown"), "subject": v.get("subject", {}) })
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
    except: return {"models": DEFAULT_MODELS}

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