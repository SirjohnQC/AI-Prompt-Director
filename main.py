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
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse, StreamingResponse
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

TEMP_DIR.mkdir(exist_ok=True)
HISTORY_DIR.mkdir(exist_ok=True)
IMG_DIR.mkdir(exist_ok=True)

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

def load_json_file(path: Path, default={}):
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
        if image_url:
            response = requests.get(image_url, stream=True, timeout=15)
            response.raise_for_status()
            filename = f"url_img_{int(time.time())}.jpg"
            temp_path = TEMP_DIR / filename
            with open(temp_path, 'wb') as out_file: shutil.copyfileobj(response.raw, out_file)
            return temp_path
        elif file:
            temp_path = TEMP_DIR / file.filename
            with open(temp_path, "wb") as buffer: shutil.copyfileobj(file.file, buffer)
            return temp_path
    except Exception as e:
        logger.error(f"Image processing failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

# --- NEW: ROBUST TWO-PASS ANALYSIS ---
async def enhanced_qwen_analysis(temp_path: Path, model: str) -> dict:
    """Two-pass analysis with Safety Fallback"""
    
    # SYSTEM PROMPT 1: The "Dream" (Complex & Detailed)
    complex_prompt = """Analyze this image in EXTREME detail.
    Return valid JSON with this structure:
    {
      "subject": { "description": "...", "physique": "...", "face": { "expression": "...", "makeup": "...", "eyes": "..." }, "hair": { "color": "...", "style": "..." } },
      "clothing": { "top": "...", "bottom": "...", "shoes": "...", "accessories": "...", "fabric_details": "..." },
      "environment": { "location": "...", "background_details": ["..."], "lighting": { "source": "...", "quality": "..." } },
      "camera": { "type": "...", "angle": "..." },
      "style": { "aesthetic": "...", "vibe": "..." },
      "meta_tokens": ["tag1", "tag2"]
    }
    """

    # SYSTEM PROMPT 2: The "Safety Net" (Simple & Reliable)
    simple_prompt = """Describe this image in JSON format.
    Keys: subject, clothing, environment, style.
    Keep it simple and direct."""

    logger.info(f"Pass 1: Trying Complex Analysis with {model}...")
    
    try:
        # Attempt 1: Complex
        response1 = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': complex_prompt, 'images': [str(temp_path)]}],
            options={"temperature": 0.1, "num_predict": 3072}
        )
        data = extract_json_from_text(response1['message']['content'])
    except Exception as e:
        logger.error(f"Pass 1 Error: {e}")
        data = {}

    # FALLBACK: If Complex failed (empty data), run Simple
    if not data or not data.get("subject"):
        logger.warning("⚠️ Complex analysis failed/empty. Switching to Simple Mode.")
        try:
            response_fallback = ollama.chat(
                model=model,
                messages=[{'role': 'user', 'content': simple_prompt, 'images': [str(temp_path)]}],
                options={"temperature": 0.2, "num_predict": 1024}
            )
            data = extract_json_from_text(response_fallback['message']['content'])
        except Exception as e:
            logger.error(f"Fallback failed: {e}")
            data = {}

    # Ensure basics exist so Pass 2 doesn't crash
    data.setdefault("subject", {})
    data.setdefault("clothing", {})
    data.setdefault("environment", {})

    # PASS 2: Detail Enhancement (Only run if we have a valid subject)
    if data.get("subject"):
        detail_prompt = """Provide fine details for:
        1. Fabric textures (silk, denim)
        2. Small accessories (rings, buttons)
        3. Lighting nuances (rim light, shadows)
        4. Makeup details
        
        Return JSON: { "fabric_details": [...], "small_accessories": [...], "lighting_nuances": [...], "makeup_techniques": [...] }"""

        logger.info(f"Pass 2: Detail enhancement")
        try:
            response2 = ollama.chat(
                model=model,
                messages=[{'role': 'user', 'content': detail_prompt, 'images': [str(temp_path)]}],
                options={"temperature": 0.1, "num_predict": 1024}
            )
            details = extract_json_from_text(response2['message']['content'])
            
            # Safe Merging
            if details.get("fabric_details"):
                if "fabric_details" not in data["clothing"]: data["clothing"]["fabric_details"] = []
                # Handle cases where AI returns string instead of list
                if isinstance(details["fabric_details"], list):
                    if isinstance(data["clothing"]["fabric_details"], list):
                         data["clothing"]["fabric_details"].extend(details["fabric_details"])
                elif isinstance(details["fabric_details"], str):
                    data["clothing"]["fabric_details"] = details["fabric_details"]

            if details.get("small_accessories"):
                current = str(data["clothing"].get("accessories", ""))
                new_items = details["small_accessories"]
                if isinstance(new_items, list): new_items = ", ".join(new_items)
                data["clothing"]["accessories"] = f"{current}, {new_items}".strip(", ")

            if details.get("lighting_nuances"):
                if isinstance(data["environment"].get("lighting"), dict):
                    data["environment"]["lighting"]["nuances"] = details["lighting_nuances"]

        except Exception as e:
            logger.error(f"Pass 2 Error: {e}")
            # Do not fail the whole request if Pass 2 fails
            pass
    
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

# --- MODEL MGMT ---
@app.get("/models")
async def get_models():
    try:
        info = ollama.list()
        models = [m['model'] for m in info.get('models', [])] or DEFAULT_MODELS
        return {"models": models, "favorite": ""}
    except: return {"models": DEFAULT_MODELS, "favorite": ""}

@app.delete("/models/{model_name}")
async def delete_model(model_name: str):
    try:
        ollama.delete(model_name)
        return {"status": "deleted"}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

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
    glasses_override: str = Form("auto")
):
    temp_path = None
    try:
        data = {}
        
        # --- PATH A: TEXT PROMPT PARSING ---
        if text_prompt and text_prompt.strip():
            logger.info("Parsing Text Prompt...")
            filename = "text_import.json"
            parser_model = "llama3.2" 
            parser_system = """You are a Prompt Engineer. Convert text to JSON.
            Map to: subject (face, hair), clothing, environment, camera, style, technical_quality, meta_tokens.
            JSON FORMAT: { "subject": {...}, "clothing": {...}, "environment": {...}, "camera": {...}, "style": {...}, "technical_quality": {...}, "meta_tokens": [...] }"""
            response = ollama.chat(
                model=parser_model, 
                messages=[
                    {'role': 'system', 'content': parser_system}, 
                    {'role': 'user', 'content': text_prompt}
                ], 
                options={"temperature": 0.1}, 
                format="json"
            )
            data = extract_json_from_text(response['message']['content'])
            if data is None: 
                data = {}

        # --- PATH B: IMAGE ANALYSIS ---
        else:
            temp_path = process_uploaded_image(file, image_url)
            filename = file.filename if file else "url_image.jpg"
            data = await enhanced_qwen_analysis(temp_path, model)

        # --- CRITICAL: NORMALIZE DATA STRUCTURE FIRST ---
        # Ensure all required keys exist before persona injection
        if "subject" not in data or not isinstance(data["subject"], dict):
            data["subject"] = {}
        if "style" not in data or data["style"] is None: 
            data["style"] = {}
        if "face" not in data["subject"] or not isinstance(data["subject"].get("face"), dict):
            data["subject"]["face"] = {}
        if "hair" not in data["subject"]:
            data["subject"]["hair"] = {}
        if "clothing" not in data:
            data["clothing"] = {}
        if "environment" not in data:
            data["environment"] = {}
        if "meta_tokens" not in data:
            data["meta_tokens"] = []

        # --- PERSONA INJECTION (NOW WORKS FOR BOTH TEXT & IMAGE) ---
        if persona_id != "none":
            all_personas = load_json_file(PERSONA_FILE)
            if persona_id in all_personas:
                persona = all_personas[persona_id]
                
                if persona and isinstance(persona, dict):
                    p_subject = persona.get("subject", {})
                    
                    # Inject name
                    data["subject"]["name"] = persona.get("name", "Character")
                    
                    # Basic attributes
                    if p_subject.get("age"): 
                        data["subject"]["age"] = p_subject["age"]
                    if p_subject.get("ethnicity"): 
                        data["subject"]["ethnicity"] = p_subject["ethnicity"]
                    if p_subject.get("body_type"): 
                        data["subject"]["body_type"] = p_subject["body_type"]
                        data["subject"]["physique"] = p_subject["body_type"]

                    # Hair injection (preserve condition if detected)
                    if "hair" in p_subject:
                        detected_condition = None
                        if isinstance(data["subject"].get("hair"), dict):
                            detected_condition = data["subject"]["hair"].get("condition")
                        
                        if isinstance(p_subject["hair"], dict):
                            data["subject"]["hair"] = p_subject["hair"].copy()
                            if detected_condition and "condition" not in data["subject"]["hair"]:
                                data["subject"]["hair"]["condition"] = detected_condition
                        else:
                            # If persona hair is a string, convert to dict
                            data["subject"]["hair"] = {
                                "style": str(p_subject["hair"]), 
                                "color": "Unknown",
                                "condition": detected_condition or "healthy"
                            }

                    # Face attributes
                    for k in ["face_structure", "eyes", "nose", "lips", "skin", "makeup"]:
                        val = p_subject.get(k)
                        if val and str(val).lower() not in ["none", "detected", ""]:
                            data["subject"]["face"][k] = val
                    
                    # Tattoos
                    if p_subject.get("tattoos"): 
                        data["subject"]["tattoos"] = p_subject.get("tattoos")
                    
                    # Regenerate description
                    p_desc_parts = [data["subject"]["name"]]
                    if data["subject"].get("age"): 
                        p_desc_parts.append(f"{data['subject']['age']} years old")
                    if data["subject"].get("ethnicity"): 
                        p_desc_parts.append(data['subject']['ethnicity'])
                    if data["subject"].get("body_type"): 
                        p_desc_parts.append(data['subject']['body_type'])
                    p_desc_parts.append("woman")
                    data["subject"]["description"] = ", ".join(p_desc_parts)
                    
                    # Meta tokens cleanup & injection
                    banned_words = ["hair", "blonde", "blond", "brunette", "redhead", "ginger", "cut", "style"]
                    data["meta_tokens"] = [
                        t for t in data["meta_tokens"] 
                        if not any(b in t.lower() for b in banned_words)
                    ]
                    
                    # Insert persona-specific tokens
                    p_hair = data["subject"].get("hair", {})
                    if isinstance(p_hair, dict):
                        if p_hair.get("style"): 
                            data["meta_tokens"].insert(0, p_hair.get("style"))
                        if p_hair.get("color"): 
                            data["meta_tokens"].insert(0, f"{p_hair.get('color')} hair")
                    
                    if data["subject"]["name"] not in str(data["meta_tokens"]):
                        data["meta_tokens"].insert(0, data["subject"]["name"])

        # --- OVERRIDES (Applied after persona) ---
        if hair_style_override != "auto":
            if isinstance(data["subject"]["hair"], dict):
                data["subject"]["hair"]["style"] = hair_style_override
                
        if hair_color_override != "auto":
            if isinstance(data["subject"]["hair"], dict):
                data["subject"]["hair"]["color"] = hair_color_override
                
        if makeup_override != "auto": 
            data["subject"]["face"]["makeup"] = makeup_override
        
        # Glasses logic (fixed)
        if glasses_override != "auto":
            outfit_key = "clothing" if "clothing" in data else "outfit"
            if outfit_key not in data: 
                data[outfit_key] = {}
            
            acc = data[outfit_key].get("accessories", "")
            eyes = data["subject"]["face"].get("eyes", "")

            remove_pattern = r"(?i)(,\s*)?(no\s+)?(reading\s+|sun)?glasses(,\s*)?"
            clean_acc = re.sub(remove_pattern, "", str(acc)).strip(", ")
            clean_eyes = re.sub(remove_pattern, "", str(eyes)).strip(", ")

            if glasses_override == "none":
                data[outfit_key]["accessories"] = (clean_acc + ", no glasses").strip(", ")
                data["subject"]["face"]["eyes"] = clean_eyes 
            else:
                data[outfit_key]["accessories"] = f"{clean_acc}, {glasses_override}".strip(", ")
                data["subject"]["face"]["eyes"] = f"{clean_eyes}, wearing {glasses_override}".strip(", ")

        # Environment/Style overrides
        if time_override != "auto": 
            data["environment"]["time_indicator"] = time_override
            
        if expr_override != "auto": 
            data["subject"]["face"]["expression"] = expr_override
            
        if ratio_override != "auto": 
            data["aspect_ratio"] = ratio_override
            
        if style_override != "auto": 
            data["style"]["aesthetic"] = style_override

        # Quality presets
        if quality_override != "auto":
            if quality_override == "Best": 
                data["meta_tokens"].extend(["8k", "best quality", "masterpiece"])
            elif quality_override == "Raw": 
                data["meta_tokens"].extend(["raw photo", "film grain", "analog style"])
            elif quality_override == "Phone": 
                data["meta_tokens"].extend(["iphone photo", "candid", "flash photography"])

        # Negative prompt
        data["negative_prompt"] = [
            "lowres", "bad anatomy", "bad hands", "text", "error", 
            "missing fingers", "extra digit", "fewer digits", "cropped", 
            "worst quality", "low quality", "normal quality", "jpeg artifacts", 
            "signature", "watermark", "username", "blurry"
        ]

        # Final cleanup
        if "clothing" in data and "outfit" in data: 
            del data["outfit"]
        elif "outfit" in data: 
            data["clothing"] = data.pop("outfit")
            
        # Remove empty dictionaries
        keys_to_remove = [k for k, v in data.items() if isinstance(v, dict) and not v]
        for k in keys_to_remove: 
            del data[k]

        # Save to history
        save_history({
            "filename": filename,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": "text-parser" if text_prompt else model,
            "persona": persona_id,
            "json": data
        })
        
        return data

    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        safe_file_cleanup(temp_path)
        
# --- QUICK PERSONA SWAP ENDPOINT ---
@app.post("/inject-persona")
async def inject_persona_endpoint(request: Request):
    """
    Updates an existing JSON object with a new Persona's details
    without re-analyzing the image.
    """
    try:
        req_data = await request.json()
        data = req_data.get("json", {})
        persona_id = req_data.get("persona_id")
        
        if not data or not persona_id:
            return {"status": "error", "message": "Missing JSON or Persona ID"}

        # Normalize data if needed (safety check)
        if "subject" not in data: data["subject"] = {}
        if "meta_tokens" not in data: data["meta_tokens"] = []

        if persona_id != "none":
            all_personas = load_json_file(PERSONA_FILE)
            if persona_id in all_personas:
                persona = all_personas[persona_id]
                if persona and isinstance(persona, dict):
                    p_subject = persona.get("subject", {})
                    
                    # 1. Inject Identity
                    data["subject"]["name"] = persona.get("name", "Character")
                    
                    # 2. Overwrite physical traits
                    if p_subject.get("age"): data["subject"]["age"] = p_subject["age"]
                    if p_subject.get("ethnicity"): data["subject"]["ethnicity"] = p_subject["ethnicity"]
                    if p_subject.get("body_type"): 
                        data["subject"]["body_type"] = p_subject["body_type"]
                        data["subject"]["physique"] = p_subject["body_type"]

                    # 3. Smart Hair Merge
                    if "hair" in p_subject:
                        if "hair" not in data["subject"]: data["subject"]["hair"] = {}
                        
                        if isinstance(p_subject["hair"], dict):
                            # Keep detected condition if it exists
                            current_cond = data["subject"]["hair"].get("condition")
                            # Overwrite with persona details
                            data["subject"]["hair"].update(p_subject["hair"])
                            # Restore condition if persona didn't specify it
                            if current_cond and "condition" not in p_subject["hair"]:
                                data["subject"]["hair"]["condition"] = current_cond
                        else:
                            # Handle string format
                            current_cond = data["subject"]["hair"].get("condition", "healthy")
                            data["subject"]["hair"]["style"] = str(p_subject["hair"])
                            data["subject"]["hair"]["condition"] = current_cond

                    # 4. Face attributes
                    if "face" not in data["subject"]: data["subject"]["face"] = {}
                    for k in ["face_structure", "eyes", "nose", "lips", "skin", "makeup"]:
                        val = p_subject.get(k)
                        if val and str(val).lower() not in ["none", "detected", ""]:
                            data["subject"]["face"][k] = val
                    
                    if p_subject.get("tattoos"): data["subject"]["tattoos"] = p_subject.get("tattoos")

                    # 5. Regenerate Description
                    p_desc_parts = [data["subject"]["name"]]
                    if data["subject"].get("age"): p_desc_parts.append(f"{data['subject']['age']} years old")
                    if data["subject"].get("ethnicity"): p_desc_parts.append(data['subject']['ethnicity'])
                    if data["subject"].get("body_type"): p_desc_parts.append(data['subject']['body_type'])
                    p_desc_parts.append("woman")
                    data["subject"]["description"] = ", ".join(p_desc_parts)

                    # 6. Meta Tokens Scrub & Inject
                    # Remove conflicting tags
                    banned_words = ["hair", "blonde", "blond", "brunette", "redhead", "ginger", "cut", "style"]
                    data["meta_tokens"] = [t for t in data["meta_tokens"] if not any(b in t.lower() for b in banned_words)]
                    
                    # Inject new tags
                    if data["subject"]["name"] not in str(data["meta_tokens"]):
                        data["meta_tokens"].insert(0, data["subject"]["name"])
                        
                    p_hair = data["subject"].get("hair", {})
                    if isinstance(p_hair, dict):
                        if p_hair.get("style"): data["meta_tokens"].insert(1, p_hair.get("style"))
                        if p_hair.get("color"): data["meta_tokens"].insert(1, f"{p_hair.get('color')} hair")

        return {"status": "success", "json": data}

    except Exception as e:
        logger.error(f"Inject Persona Error: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/generate-prompt")
async def generate_natural_prompt(request: Request):
    try:
        from prompt_generator import generate_narrative_prompt
        data = await request.json()
        json_context = data.get("json")
        user_model = data.get("model", "llama3.2")
        prompt_text = generate_narrative_prompt(json_context, preferred_model=user_model)
        return {"status": "success", "prompt": prompt_text}
    except Exception as e: return {"status": "error", "error": str(e)}

@app.post("/generate-tags")
async def generate_tags(
    file: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None),
    model: str = Form("qwen3-vl"), 
    persona_id: str = Form("none"),
    reference_mode: bool = Form(False)
):
    temp_path = None
    try:
        try:
            from enhanced_tag_system import generate_tags_enhanced 
        except ImportError:
            generate_tags_enhanced = None

        temp_path = process_uploaded_image(file, image_url)
        system_prompt = "Describe this image using a list of booru-style tags."
        response = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': system_prompt, 'images': [str(temp_path)]}]
        )
        raw_text = response['message']['content']
        persona_data = None
        if persona_id != "none":
            all_personas = load_json_file(PERSONA_FILE)
            persona_data = all_personas.get(persona_id)

        if generate_tags_enhanced:
            result = generate_tags_enhanced(raw_text, persona_data, reference_mode)
            result["model_used"] = model
            return result
        else:
            return {"positive_tags": raw_text, "negative_prompt": "lowres, error"}
    except Exception as e: return JSONResponse(status_code=500, content={"error": str(e)})
    finally: safe_file_cleanup(temp_path)

@app.post("/refine")
async def refine_json(request: Request):
    try:
        data = await request.json()
        current_json = data.get("current_json")
        instruction = data.get("instruction")
        model = data.get("model", "llama3.2")
        sys_prompt = f"You are a JSON Editor. Update JSON based on: {instruction}. Return ONLY JSON."
        resp = ollama.chat(
            model=model,
            messages=[{'role': 'system', 'content': sys_prompt}, {'role': 'user', 'content': json.dumps(current_json)}],
            options={"temperature": 0.3},
            format="json"
        )
        new_json = extract_json_from_text(resp['message']['content'])
        return {"status": "success", "json": new_json}
    except Exception as e: return {"status": "error", "error": str(e)}

@app.get("/personas")
async def list_personas():
    data = load_json_file(PERSONA_FILE, {})
    return [{"id": k, "name": v["name"], "subject": v.get("subject")} for k, v in data.items()]

@app.get("/persona-image/{pid}")
async def get_persona_image(pid: str):
    img_path = IMG_DIR / f"{pid}.jpg"
    return FileResponse(img_path) if img_path.exists() else JSONResponse(status_code=404, content={})

# --- UPGRADED PERSONA CREATION (Dedicated Profile Scanner) ---
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
        
        # Determine ID
        safe_id = pid if (mode == "edit" and pid) else re.sub(r'[^a-zA-Z0-9]', '_', name.lower())
        
        # Save image
        temp_path = TEMP_DIR / f"scan_{file.filename}"
        with open(temp_path, "wb") as buffer: shutil.copyfileobj(file.file, buffer)
        shutil.copy(temp_path, IMG_DIR / f"{safe_id}.jpg")
        
        # --- DEDICATED PERSONA PROMPT ---
        # We use a specific prompt that maps 1:1 to your UI fields
        system_prompt = f"""Analyze this character image for a database profile. 
        Extract these SPECIFIC details. If unsure, estimate.
        
        Return JSON with these EXACT keys:
        {{
            "age": "Estimated age (e.g. '25')",
            "ethnicity": "Specific ethnicity",
            "body_type": "Body build description (e.g. 'Slim', 'Curvy', 'Muscular')",
            "face_structure": "Face shape (e.g. 'Oval', 'Square', 'Heart')",
            "skin": "Skin tone and texture description",
            "eyes": "Eye color and shape",
            "nose": "Nose shape and size",
            "lips": "Lip shape and color",
            "hair": {{ "color": "Precise color", "style": "Hairstyle description" }},
            "makeup": "Makeup details or 'None'",
            "tattoos": "Visible tattoos or 'None'"
        }}
        """
        
        logger.info(f"Scanning Persona '{name}' with profile scanner...")
        response = ollama.chat(
            model=model, 
            messages=[{'role': 'user', 'content': system_prompt, 'images': [str(temp_path)]}],
            options={"temperature": 0.1}
        )
        
        # Extract and Clean Data
        ai_data = extract_json_from_text(response['message']['content'])
        
        # Robustness: Ensure 'hair' is an object
        if "hair" in ai_data and isinstance(ai_data["hair"], str):
             ai_data["hair"] = {"style": ai_data["hair"], "color": "Unknown"}
        
        # Save to DB
        all_p = load_json_file(PERSONA_FILE, {})
        all_p[safe_id] = { 
            "name": name, 
            "subject": ai_data 
        }
        save_json_file(PERSONA_FILE, all_p)
        
        return {"status": "success", "id": safe_id, "data": all_p[safe_id]}
        
    except Exception as e:
        logger.error(f"Persona creation failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        safe_file_cleanup(temp_path)

@app.put("/personas/{pid}")
async def update_persona(pid: str, request: Request):
    data = await request.json()
    all_p = load_json_file(PERSONA_FILE)
    if pid in all_p:
        all_p[pid]["name"] = data.get("name", all_p[pid]["name"])
        all_p[pid]["subject"] = data.get("subject", all_p[pid]["subject"])
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

@app.get("/history")
async def get_history():
    return load_json_file(HISTORY_DIR / "history.json", [])

@app.delete("/history")
async def clear_history():
    save_json_file(HISTORY_DIR / "history.json", [])
    return {"status": "cleared"}

@app.post("/shutdown")
async def shutdown():
    import threading
    def stop_server():
        time.sleep(1)
        os.kill(os.getpid(), signal.SIGTERM)
    threading.Thread(target=stop_server, daemon=True).start()
    return {"message": "Shutting down"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")