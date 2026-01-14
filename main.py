import os
import shutil
import json
import logging
import time
import re
import requests
import signal
import base64
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import ollama
import uvicorn

try:
    from enhanced_tag_system import generate_tags_enhanced 
    from prompt_generator import generate_narrative_prompt
except ImportError:
    logging.warning("⚠️ Enhanced modules not found. Using fallbacks.")
    generate_tags_enhanced = None
    generate_narrative_prompt = None

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
MAX_HISTORY_ITEMS = 50
DEFAULT_MODELS = ["minicpm-v", "llava:v1.6", "qwen3-vl", "llama3.2"]

# --- CLOUD CONFIGURATION ---
GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
FAL_KEY = os.getenv("FAL_KEY")
XAI_KEY = os.getenv("XAI_API_KEY")
# UPDATE THIS TO YOUR REPO:
GITHUB_VERSION_URL = "https://raw.githubusercontent.com/SirjohnQC/AI-Prompt-Director/main/version.txt"

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
        json_str = re.sub(r'(?<=[}\]"0-9e])\s+(?=")', ', ', json_str)
        json_str = re.sub(r'(?<=true)\s+(?=")', ', ', json_str)
        json_str = re.sub(r'(?<=false)\s+(?=")', ', ', json_str)
        json_str = re.sub(r'(?<=null)\s+(?=")', ', ', json_str)
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        json_str = json_str.replace('True', 'true').replace('False', 'false').replace('None', 'null')
        return json.loads(json_str)
    except Exception as e:
        logger.error(f"JSON extraction failed: {e}")
        return {"error": str(e), "subject": {"note": "JSON Repair Failed"}, "outfit": {}, "pose": {}, "environment": {}, "style_and_realism": {}}

def save_history(entry):
    history = load_json_file(HISTORY_FILE, [])
    if not isinstance(history, list): history = []
    history.insert(0, entry)
    save_json_file(HISTORY_FILE, history[:MAX_HISTORY_ITEMS])

def process_uploaded_image(file: Optional[UploadFile] = None, image_url: Optional[str] = None) -> Path:
    if not file and not image_url: raise HTTPException(status_code=400, detail="No image provided.")
    try:
        if image_url:
            if not image_url.startswith(("http", "https")): raise HTTPException(status_code=400, detail="Invalid URL.")
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

# --- ENDPOINTS ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    if (BASE_DIR / "index.html").exists():
        return templates.TemplateResponse("index.html", {"request": request})
    return {"error": "index.html not found"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# --- VERSION CHECK ---
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

# --- CLOUD IMAGE GENERATION ---
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

# --- MODEL MANAGEMENT ---
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

# --- ANALYZE ENDPOINT (FIXED GLASSES) ---
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
            Map to: subject (age, hair, makeup), outfit, pose, environment, style_and_realism, colors_and_tone.
            JSON FORMAT: { "subject": {...}, "outfit": {...}, "pose": {...}, "environment": {...}, "style_and_realism": {...}, "colors_and_tone": {...} }"""
            response = ollama.chat(model=parser_model, messages=[{'role': 'system', 'content': parser_system}, {'role': 'user', 'content': text_prompt}], options={"temperature": 0.1}, format="json")
            data = extract_json_from_text(response['message']['content'])

        # --- PATH B: IMAGE ANALYSIS ---
        else:
            temp_path = process_uploaded_image(file, image_url)
            filename = file.filename if file else "url_image.jpg"
            system_prompt = """Analyze image in EXTREME DETAIL. Return JSON.
            INSTRUCTIONS: Describe textures, lighting, body position, makeup, and colors.
            JSON STRUCTURE:
            {
              "subject": { "name": "Unknown", "age": "...", "ethnicity": "...", "hair": {...}, "skin": "...", "makeup": "..." },
              "outfit": { "top": "...", "bottom": "...", "accessories": "..." },
              "pose": { "posture": "...", "expression": "...", "hands": "..." },
              "environment": { "setting": "...", "lighting": "...", "atmosphere": "..." },
              "camera": { "shot_type": "...", "angle": "...", "device": "..." },
              "style_and_realism": { "visual_style": "...", "realism": "..." },
              "colors_and_tone": { "dominant_palette": "...", "contrast": "...", "grading": "..." },
              "quality_and_technical_details": { "resolution": "...", "sharpness": "...", "defects": "..." }
            }"""
            logger.info(f"Analyzing with {model}")
            response = ollama.chat(model=model, messages=[{'role': 'user', 'content': system_prompt, 'images': [str(temp_path)]}], options={"temperature": 0.2, "num_predict": 2048})
            data = extract_json_from_text(response['message']['content'])

        # --- DATA CLEANUP ---
        target_keys = ["subject", "outfit", "pose", "environment", "camera", "style_and_realism", "colors_and_tone", "quality_and_technical_details"]
        for key in target_keys: 
            if key not in data: data[key] = {}
        if "makeup" not in data.get("subject", {}): 
            if "subject" not in data: data["subject"] = {}
            data["subject"]["makeup"] = ""

        # --- PERSONA INJECTION ---
        if persona_id != "none":
            all_personas = load_json_file(PERSONA_FILE)
            if persona_id in all_personas:
                persona = all_personas[persona_id]
                p_subject = persona.get("subject", {})
                data["subject"]["name"] = persona.get("name", "Character")
                if p_subject.get("age"): data["subject"]["age"] = p_subject["age"]
                if p_subject.get("ethnicity"): data["subject"]["ethnicity"] = p_subject["ethnicity"]
                if p_subject.get("body_type"): data["subject"]["body_type"] = p_subject["body_type"]
                for k in ["face_structure", "eyes", "nose", "lips", "skin", "makeup"]:
                    if p_subject.get(k) and str(p_subject[k]).lower() not in ["none", "detected", ""]:
                        data["subject"][k] = p_subject[k]
                if p_subject.get("tattoos") and "none" not in str(p_subject["tattoos"]).lower(): 
                    data["subject"]["tattoos"] = p_subject.get("tattoos")
                if "hair" in p_subject: data["subject"]["hair"] = p_subject["hair"]

        # --- OVERRIDES ---
        if hair_style_override != "auto":
            if "hair" not in data["subject"]: data["subject"]["hair"] = {}
            data["subject"]["hair"]["style"] = hair_style_override
        if hair_color_override != "auto":
            if "hair" not in data["subject"]: data["subject"]["hair"] = {}
            data["subject"]["hair"]["color"] = hair_color_override
        if makeup_override != "auto": data["subject"]["makeup"] = makeup_override
        
        # --- FIXED GLASSES LOGIC (Smart Cleanup) ---
        if glasses_override != "auto":
            if "outfit" not in data: data["outfit"] = {}
            if "subject" not in data: data["subject"] = {}
            
            acc = data["outfit"].get("accessories", "")
            eyes = data["subject"].get("eyes", "")

            # Regex to remove existing glasses mentions (e.g. "no glasses", "sunglasses", "round glasses")
            # We remove them first to avoid "no glasses, wearing glasses" conflicts
            remove_pattern = r"(?i)(,\s*)?(no\s+)?(reading\s+|sun)?glasses(,\s*)?"
            
            clean_acc = re.sub(remove_pattern, "", acc).strip(", ")
            clean_eyes = re.sub(remove_pattern, "", eyes).strip(", ")

            if glasses_override == "none":
                # User specifically wants NO glasses
                data["outfit"]["accessories"] = (clean_acc + ", no glasses").strip(", ")
                data["subject"]["eyes"] = clean_eyes # Just remove glasses mention from eyes
            else:
                # User wants specific glasses
                # Add to outfit
                data["outfit"]["accessories"] = f"{clean_acc}, {glasses_override}".strip(", ")
                # Add to eyes (crucial for prompt generation)
                data["subject"]["eyes"] = f"{clean_eyes}, wearing {glasses_override}".strip(", ")

        if time_override != "auto": data["environment"]["time_of_day"] = time_override
        if expr_override != "auto": data["pose"]["expression"] = expr_override
        if ratio_override != "auto": data["aspect_ratio_and_output"] = {"aspect_ratio": ratio_override}
        if style_override != "auto": data["style_and_realism"]["visual_style"] = style_override

        if quality_override == "Best":
            data["quality_and_technical_details"] = {"resolution": "8k uhd", "detail": "maximum", "sharpness": "high"}
        elif quality_override == "Raw":
            data["quality_and_technical_details"] = {"style": "raw photo", "noise": "slight film grain"}
        elif quality_override == "Phone":
            data["quality_and_technical_details"] = {"style": "smartphone photography", "lighting": "flash"}

        data["negative_prompt"] = ["lowres", "bad anatomy", "bad hands", "text", "error", "missing fingers", "extra digit", "fewer digits", "cropped", "worst quality", "low quality", "normal quality", "jpeg artifacts", "signature", "watermark", "username", "blurry"]

        ordered = {}
        for k in target_keys: 
            if k in data: ordered[k] = data[k]
        for k, v in data.items(): 
            if k not in ordered and k not in ["negative_prompt", "reference_image_instruction"]: ordered[k] = v
        
        ordered["negative_prompt"] = data["negative_prompt"]
        if "reference_image_instruction" in data:
            ordered["reference_image_instruction"] = data["reference_image_instruction"]

        save_history({
            "filename": filename,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": "text-parser" if text_prompt else model,
            "persona": persona_id,
            "json": ordered
        })
        
        return ordered

    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        safe_file_cleanup(temp_path)

@app.post("/generate-prompt")
async def generate_natural_prompt(request: Request):
    try:
        data = await request.json()
        json_context = data.get("json")
        user_model = data.get("model", "llama3.2")
        if generate_narrative_prompt:
            prompt_text = generate_narrative_prompt(json_context, preferred_model=user_model)
        else:
            return {"status": "error", "error": "Prompt Generator module missing"}
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

# --- PERSONA MGMT ---
@app.get("/personas")
async def list_personas():
    data = load_json_file(PERSONA_FILE, {})
    return [{"id": k, "name": v["name"], "subject": v.get("subject")} for k, v in data.items()]

@app.get("/persona-image/{pid}")
async def get_persona_image(pid: str):
    img_path = IMG_DIR / f"{pid}.jpg"
    return FileResponse(img_path) if img_path.exists() else JSONResponse(status_code=404, content={})

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
        
        scan_prompt = f"Analyze image for character profile. Name: {name}. Return JSON with age, ethnicity, body_type, face_structure, eyes, nose, lips, skin, tattoos, hair, makeup."
        resp = ollama.chat(model=model, messages=[{'role': 'user', 'content': scan_prompt, 'images': [str(temp_path)]}])
        ai_data = extract_json_from_text(resp['message']['content'])
        
        all_p = load_json_file(PERSONA_FILE, {})
        all_p[safe_id] = { "name": name, "subject": ai_data.get("subject", ai_data) } # Robust fallback
        save_json_file(PERSONA_FILE, all_p)
        return {"status": "success", "id": safe_id, "data": all_p[safe_id]}
    finally: safe_file_cleanup(temp_path)

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