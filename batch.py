"""
Batch Image Analyzer Module
Processes multiple images for LoRA dataset preparation
"""

import os
import io
import re
import json
import time
import zipfile
import asyncio
import logging
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime

from fastapi import APIRouter, File, UploadFile, Form, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image, ImageOps

import ollama

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/batch", tags=["batch"])

# Store batch job status
batch_jobs: Dict[str, dict] = {}

# File paths (same as main.py)
BASE_DIR = Path(__file__).parent
PERSONA_FILE = BASE_DIR / "personas.json"
STYLES_FILE = BASE_DIR / "styles.json"

def load_json_file(filepath, default=None):
    """Load JSON file safely"""
    if default is None:
        default = {}
    try:
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
    except:
        pass
    return default

# --- HELPERS ---

def extract_json_from_text(text: str) -> dict:
    """Extract JSON from model response"""
    if not text:
        return {}
    try:
        # Try direct parse
        return json.loads(text)
    except:
        pass
    # Try to find JSON block
    patterns = [
        r'```json\s*([\s\S]*?)\s*```',
        r'```\s*([\s\S]*?)\s*```',
        r'\{[\s\S]*\}'
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                json_str = match.group(1) if '```' in pattern else match.group(0)
                return json.loads(json_str)
            except:
                continue
    return {}


def get_persona_name(persona_id: str) -> Optional[str]:
    """Get just the persona name/trigger word"""
    if persona_id == "none" or not persona_id:
        return None
    all_personas = load_json_file(PERSONA_FILE)
    if persona_id not in all_personas:
        return None
    return all_personas[persona_id].get("name")


def integrate_trigger_word(caption: str, trigger_word: str) -> str:
    """
    Naturally integrate the trigger word into the caption.
    Examples:
    - "A young woman with..." -> "A young woman Chl03 with..."
    - "Young woman taking..." -> "Young woman Chl03 taking..."
    - "A woman in her 20s..." -> "A woman Chl03 in her 20s..."
    """
    if not caption or not trigger_word:
        return caption
    
    # Already contains trigger word
    if trigger_word.lower() in caption.lower():
        return caption
    
    # Remove surrounding quotes if present
    caption = caption.strip('"\'')
    
    # Patterns to match and insert after
    # Format: (pattern_to_find, insert_position_description)
    insert_patterns = [
        # "A young woman" -> "A young woman NAME"
        (r'^(A young woman)', r'\1 ' + trigger_word),
        (r'^(A woman)', r'\1 ' + trigger_word),
        (r'^(A young man)', r'\1 ' + trigger_word),
        (r'^(A man)', r'\1 ' + trigger_word),
        (r'^(Young woman)', r'Young woman ' + trigger_word),
        (r'^(Young man)', r'Young man ' + trigger_word),
        # "A [adjective] woman" -> "A [adjective] woman NAME"
        (r'^(A \w+ woman)', r'\1 ' + trigger_word),
        (r'^(A \w+ man)', r'\1 ' + trigger_word),
        (r'^(A \w+ \w+ woman)', r'\1 ' + trigger_word),
        (r'^(A \w+ \w+ man)', r'\1 ' + trigger_word),
        # "The woman" -> "The woman NAME"
        (r'^(The woman)', r'\1 ' + trigger_word),
        (r'^(The man)', r'\1 ' + trigger_word),
    ]
    
    for pattern, replacement in insert_patterns:
        if re.match(pattern, caption, re.IGNORECASE):
            return re.sub(pattern, replacement, caption, count=1, flags=re.IGNORECASE)
    
    # Fallback: prepend with "Photo of NAME, "
    return f"Photo of {trigger_word}, {caption[0].lower() + caption[1:] if caption else caption}"


def apply_persona_to_batch_data(data: dict, persona_id: str) -> dict:
    """Inject persona data into the analysis JSON for batch processing"""
    if persona_id == "none" or not persona_id:
        return data
    
    all_personas = load_json_file(PERSONA_FILE)
    if persona_id not in all_personas:
        return data
    
    persona = all_personas[persona_id]
    p_data = persona.get("profile") or persona.get("subject") or {}
    
    if not isinstance(data.get("subject"), dict):
        data["subject"] = {}
    
    data["subject"]["name"] = persona.get("name", "Character")
    
    # Inject basic info
    if p_data.get("age"): data["subject"]["age"] = p_data["age"]
    if p_data.get("ethnicity"): data["subject"]["ethnicity"] = p_data["ethnicity"]
    if p_data.get("body_type"): data["subject"]["body_type"] = p_data["body_type"]
    
    # Inject face
    if "face" not in data["subject"] or not isinstance(data["subject"]["face"], dict):
        data["subject"]["face"] = {}
    
    p_face = p_data.get("facial_features", p_data)
    for key in ["eyes", "skin", "expression", "face_structure", "nose", "lips"]:
        val = p_face.get(key)
        if val: data["subject"]["face"][key] = val
    
    # Inject hair
    if "hair" not in data["subject"] or not isinstance(data["subject"]["hair"], dict):
        data["subject"]["hair"] = {}
    
    hair_data = p_data.get("hair") or p_face.get("hair")
    if hair_data:
        if isinstance(hair_data, dict):
            if hair_data.get("color"): data["subject"]["hair"]["color"] = hair_data["color"]
            if hair_data.get("style"): data["subject"]["hair"]["style"] = hair_data["style"]
        else:
            data["subject"]["hair"]["style"] = str(hair_data)
    
    # Inject body proportions
    if p_data.get("body_proportions"):
        data["subject"]["body_proportions"] = p_data["body_proportions"]
    
    return data


def generate_natural_prompt_sync(json_data: dict, model: str, style_instruction: str) -> str:
    """Generate natural language prompt from JSON data (synchronous for batch)"""
    sys_prompt = f"""You are an AI image caption writer for LoRA training datasets.

INPUT: JSON describing a photograph
OUTPUT: A natural image caption/description

CRITICAL RULES:
1. {style_instruction}
2. Start DIRECTLY with describing what's in the image (e.g. "A woman with..." or "Young woman standing...")
3. NEVER start with "Write", "Create", "Generate", "A photograph of" or any instruction words
4. NEVER use phrases like "picture-perfect", "capturing", "showcasing"
5. Be descriptive but concise - aim for 2-4 sentences max
6. Include: subject appearance, clothing, pose, setting, lighting
7. Use natural descriptive language, not flowery prose
8. NO meta-commentary about the image quality or composition

GOOD EXAMPLE: "A young woman with short brown hair and hazel eyes, wearing a light blue crop top. She holds a smartphone, taking a mirror selfie in a bright bedroom. Natural window light illuminates her athletic figure. Gold hoop earrings and a pendant necklace accessorize the casual look."

BAD EXAMPLE: "Write a detailed photograph capturing a stunning young woman showcasing her picture-perfect smile..."
"""

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {'role': 'system', 'content': sys_prompt},
                {'role': 'user', 'content': f"Write a caption for this image:\n{json.dumps(json_data, indent=2)}"}
            ],
            options={"temperature": 0.5, "num_predict": 512}
        )
        result = response['message']['content'].strip()
        
        # Clean up common AI mistakes
        bad_starts = ["write ", "create ", "generate ", "a photograph of ", "a photo of ", "this image shows ", "the image depicts "]
        result_lower = result.lower()
        for bad in bad_starts:
            if result_lower.startswith(bad):
                result = result[len(bad):]
                result = result[0].upper() + result[1:] if result else result
                break
        
        return result
    except Exception as e:
        logger.error(f"Prompt generation error: {e}")
        return generate_caption_from_data({"data": json_data}, "detailed")


def process_image_for_analysis(file_bytes: bytes, max_size: int = 1536) -> bytes:
    """Optimize image for vision model analysis"""
    img = Image.open(io.BytesIO(file_bytes))
    img = ImageOps.exif_transpose(img)
    
    if img.width > max_size or img.height > max_size:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    if img.mode in ('RGBA', 'P'):
        img = img.convert('RGB')
    
    output = io.BytesIO()
    img.save(output, "JPEG", quality=85, optimize=True)
    img.close()
    return output.getvalue()


def detect_face_region(img: Image.Image) -> Optional[tuple]:
    """
    Try to detect face region using various methods.
    Returns (x, y, width, height) of face region or None.
    """
    try:
        # Try OpenCV face detection if available
        import cv2
        import numpy as np
        
        # Convert PIL to OpenCV
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        
        # Try multiple cascade classifiers
        cascades = [
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
            cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml',
            cv2.data.haarcascades + 'haarcascade_profileface.xml',
        ]
        
        for cascade_path in cascades:
            face_cascade = cv2.CascadeClassifier(cascade_path)
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            
            if len(faces) > 0:
                # Return the largest face
                largest = max(faces, key=lambda f: f[2] * f[3])
                return tuple(largest)
        
        return None
        
    except ImportError:
        logger.warning("OpenCV not available for face detection, using heuristic")
        return None
    except Exception as e:
        logger.warning(f"Face detection failed: {e}")
        return None


def smart_resize_image(file_bytes: bytes, target_size: int, crop_mode: str = "face") -> bytes:
    """
    Smart resize image with face-focused cropping for LoRA training.
    
    Args:
        file_bytes: Original image bytes
        target_size: Target dimension (e.g., 512, 768, 1024)
        crop_mode: "face" (smart), "center", "top", or "fit"
    
    Returns:
        Resized image bytes
    """
    img = Image.open(io.BytesIO(file_bytes))
    img = ImageOps.exif_transpose(img)
    
    if img.mode in ('RGBA', 'P'):
        img = img.convert('RGB')
    
    orig_width, orig_height = img.size
    
    if crop_mode == "fit":
        # Fit without cropping, add padding if needed
        img.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
        
        # Create square canvas and paste centered
        canvas = Image.new('RGB', (target_size, target_size), (0, 0, 0))
        paste_x = (target_size - img.width) // 2
        paste_y = (target_size - img.height) // 2
        canvas.paste(img, (paste_x, paste_y))
        img = canvas
        
    else:
        # Calculate crop region
        aspect_ratio = orig_width / orig_height
        
        if aspect_ratio > 1:  # Wider than tall
            new_height = orig_height
            new_width = orig_height  # Make square
        else:  # Taller than wide
            new_width = orig_width
            new_height = orig_width  # Make square
        
        # Determine crop center based on mode
        if crop_mode == "face":
            face_region = detect_face_region(img)
            
            if face_region:
                fx, fy, fw, fh = face_region
                # Center on face, but give some headroom
                center_x = fx + fw // 2
                center_y = fy + fh // 2 - fh // 4  # Slightly above face center for headroom
            else:
                # No face detected, use upper-center (head usually at top)
                center_x = orig_width // 2
                center_y = orig_height // 3  # Upper third
                
        elif crop_mode == "top":
            center_x = orig_width // 2
            center_y = min(new_height // 2, orig_height // 3)
            
        else:  # center
            center_x = orig_width // 2
            center_y = orig_height // 2
        
        # Calculate crop box
        left = max(0, center_x - new_width // 2)
        top = max(0, center_y - new_height // 2)
        
        # Adjust if crop goes beyond image bounds
        if left + new_width > orig_width:
            left = orig_width - new_width
        if top + new_height > orig_height:
            top = orig_height - new_height
        
        # Ensure non-negative
        left = max(0, left)
        top = max(0, top)
        
        right = min(left + new_width, orig_width)
        bottom = min(top + new_height, orig_height)
        
        # Crop to square
        img = img.crop((left, top, right, bottom))
        
        # Resize to target
        img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    
    # Save as JPEG
    output = io.BytesIO()
    img.save(output, "JPEG", quality=92, optimize=True)
    img.close()
    
    return output.getvalue()


def analyze_single_image(image_path: str, model: str, caption_style: str = "detailed") -> dict:
    """Analyze a single image and return structured data"""
    
    if caption_style == "tags":
        prompt = """Analyze this image and return comma-separated tags describing:
- Subject (age, gender, ethnicity, hair, body type)
- Clothing and accessories
- Pose and expression
- Environment and lighting
- Style and mood

Return ONLY tags, no sentences. Example: young woman, blonde hair, blue eyes, casual dress, sitting, smiling, indoor, natural lighting, candid"""
    
    elif caption_style == "simple":
        prompt = """Describe this image in one detailed sentence covering the subject, their appearance, clothing, pose, and setting."""
    
    else:  # detailed
        prompt = """Analyze this image. Return JSON:
{
    "subject": {
        "description": "Brief action/context",
        "age": "estimate",
        "gender": "observed",
        "ethnicity": "estimate", 
        "hair": {"color": "...", "style": "..."},
        "body_type": "short tag",
        "expression": "emotion"
    },
    "clothing": {
        "outfit": "description",
        "accessories": "items"
    },
    "pose": {
        "position": "stance",
        "orientation": "camera angle"
    },
    "environment": {
        "location": "setting",
        "lighting": "light description"
    },
    "style": "overall aesthetic/mood"
}"""

    try:
        response = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt, 'images': [image_path]}],
            options={"temperature": 0.1, "num_predict": 2048, "num_ctx": 8192}
        )
        
        raw_content = response['message']['content']
        
        if caption_style == "tags" or caption_style == "simple":
            return {"caption": raw_content.strip(), "raw": raw_content}
        else:
            data = extract_json_from_text(raw_content)
            if not data:
                data = {"caption": raw_content.strip()}
            return {"data": data, "raw": raw_content}
            
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        return {"error": str(e)}


def generate_caption_from_data(data: dict, style: str = "detailed") -> str:
    """Convert structured JSON data to a text caption"""
    if "caption" in data:
        return data["caption"]
    
    if "error" in data:
        return f"Error: {data['error']}"
    
    analysis = data.get("data", data)
    
    parts = []
    
    # Subject
    subject = analysis.get("subject", {})
    if isinstance(subject, dict):
        subj_parts = []
        if subject.get("age"): subj_parts.append(subject["age"])
        if subject.get("gender"): subj_parts.append(subject["gender"])
        if subject.get("ethnicity"): subj_parts.append(subject["ethnicity"])
        
        hair = subject.get("hair", {})
        if isinstance(hair, dict):
            hair_desc = f"{hair.get('color', '')} {hair.get('style', '')}".strip()
            if hair_desc: subj_parts.append(f"{hair_desc} hair")
        elif hair:
            subj_parts.append(f"{hair} hair")
        
        if subject.get("body_type"): subj_parts.append(subject["body_type"])
        if subject.get("expression"): subj_parts.append(subject["expression"])
        
        if subj_parts:
            parts.append(", ".join(subj_parts))
    
    # Clothing
    clothing = analysis.get("clothing", {})
    if isinstance(clothing, dict):
        if clothing.get("outfit"): parts.append(clothing["outfit"])
        if clothing.get("accessories"): parts.append(clothing["accessories"])
    elif clothing:
        parts.append(str(clothing))
    
    # Pose
    pose = analysis.get("pose", {})
    if isinstance(pose, dict):
        pose_parts = []
        if pose.get("position"): pose_parts.append(pose["position"])
        if pose.get("orientation"): pose_parts.append(pose["orientation"])
        if pose_parts: parts.append(", ".join(pose_parts))
    elif pose:
        parts.append(str(pose))
    
    # Environment
    env = analysis.get("environment", {})
    if isinstance(env, dict):
        if env.get("location"): parts.append(env["location"])
        if env.get("lighting"): parts.append(env["lighting"])
    elif env:
        parts.append(str(env))
    
    # Style
    style_info = analysis.get("style", "")
    if isinstance(style_info, dict):
        style_info = style_info.get("aesthetic", "")
    if style_info:
        parts.append(str(style_info))
    
    return ", ".join(parts) if parts else "No description available"


# --- ENDPOINTS ---

@router.get("/status/{job_id}")
async def get_batch_status(job_id: str):
    """Get status of a batch job"""
    if job_id not in batch_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = batch_jobs[job_id]
    
    # Return a sanitized version without binary image data
    safe_results = []
    for r in job.get("results", []):
        safe_results.append({
            "filename": r.get("filename"),
            "base_name": r.get("base_name"),
            "caption": r.get("caption", "")[:200] + "..." if len(r.get("caption", "")) > 200 else r.get("caption", ""),
            "success": r.get("success", False),
            "has_image": r.get("image_data") is not None
        })
    
    return {
        "status": job.get("status"),
        "total": job.get("total"),
        "completed": job.get("completed"),
        "current_file": job.get("current_file"),
        "results": safe_results,
        "errors": job.get("errors", []),
        "settings": {k: v for k, v in job.get("settings", {}).items() if k != "style_instruction"},
        "started_at": job.get("started_at"),
        "completed_at": job.get("completed_at")
    }


@router.post("/analyze")
async def batch_analyze(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    model: str = Form("qwen3-vl:8b"),
    caption_style: str = Form("detailed"),  # detailed, simple, tags, prompt
    persona_id: str = Form("none"),
    output_format: str = Form("txt"),  # txt, json, both
    narrative_style: str = Form("default"),
    include_images: bool = Form(False),
    rename_files: bool = Form(True),
    resize_size: str = Form("1024"),  # none, 512, 768, 1024, 1280
    crop_mode: str = Form("face")  # face, center, top, fit
):
    """
    Start batch analysis of multiple images
    Returns job_id to track progress
    """
    # Load narrative style instruction
    style_instruction = "Write a natural, detailed description."
    if narrative_style != "default":
        styles = load_json_file(STYLES_FILE)
        if narrative_style in styles:
            style_instruction = styles[narrative_style]
    
    # Parse resize size
    resize_px = None if resize_size == "none" else int(resize_size)
    
    job_id = f"batch_{int(time.time())}_{len(files)}"
    
    # Initialize job status
    batch_jobs[job_id] = {
        "status": "processing",
        "total": len(files),
        "completed": 0,
        "current_file": "",
        "results": [],
        "errors": [],
        "settings": {
            "model": model,
            "caption_style": caption_style,
            "output_format": output_format,
            "persona_id": persona_id,
            "narrative_style": narrative_style,
            "style_instruction": style_instruction,
            "include_images": include_images,
            "rename_files": rename_files,
            "resize_size": resize_px,
            "crop_mode": crop_mode
        },
        "started_at": datetime.now().isoformat()
    }
    
    # Read all file contents before background task
    file_contents = []
    for i, f in enumerate(files):
        content = await f.read()
        original_ext = Path(f.filename).suffix.lower()
        file_contents.append({
            "filename": f.filename,
            "content": content,
            "original_ext": original_ext,
            "index": i
        })
    
    # Start background processing
    background_tasks.add_task(
        process_batch_job,
        job_id,
        file_contents,
        model,
        caption_style,
        output_format
    )
    
    return {"job_id": job_id, "total": len(files), "status": "started"}


async def process_batch_job(
    job_id: str,
    files: List[dict],
    model: str,
    caption_style: str,
    output_format: str
):
    """Background task to process batch of images"""
    import tempfile
    
    job = batch_jobs[job_id]
    settings = job["settings"]
    persona_id = settings.get("persona_id", "none")
    style_instruction = settings.get("style_instruction", "Write a natural, detailed description.")
    
    # Get a text model for prompt generation (use llama if available)
    text_model = "llama3.2"
    
    for i, file_data in enumerate(files):
        filename = file_data["filename"]
        content = file_data["content"]
        
        job["current_file"] = filename
        job["completed"] = i
        
        try:
            # Process and save temp image
            processed = process_image_for_analysis(content)
            
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp.write(processed)
                tmp_path = tmp.name
            
            # Analyze image
            if caption_style == "prompt":
                # For prompt style, first get detailed JSON then convert to prompt
                result = analyze_single_image(tmp_path, model, "detailed")
                analysis_data = result.get("data", {})
                
                # Apply persona if selected
                if persona_id != "none":
                    analysis_data = apply_persona_to_batch_data(analysis_data, persona_id)
                
                # Generate natural prompt using text model
                caption = generate_natural_prompt_sync(analysis_data, text_model, style_instruction)
                result["data"] = analysis_data
            else:
                # For other styles, just analyze
                result = analyze_single_image(tmp_path, model, caption_style)
                
                # Apply persona to detailed results
                if caption_style == "detailed" and persona_id != "none":
                    if "data" in result:
                        result["data"] = apply_persona_to_batch_data(result["data"], persona_id)
                
                # Generate caption
                caption = generate_caption_from_data(result, caption_style)
            
            # INTEGRATE PERSONA NAME (trigger word) naturally into caption for LoRA training
            if persona_id != "none":
                persona_name = get_persona_name(persona_id)
                if persona_name:
                    caption = integrate_trigger_word(caption, persona_name)
            
            # Determine file naming
            rename_files = settings.get("rename_files", True)
            include_images = settings.get("include_images", False)
            resize_size = settings.get("resize_size")
            crop_mode = settings.get("crop_mode", "face")
            original_ext = file_data.get("original_ext", ".jpg")
            file_index = file_data.get("index", i)
            
            if rename_files:
                # Sequential naming: 001, 002, 003...
                new_base_name = f"{file_index + 1:03d}"
            else:
                new_base_name = Path(filename).stem
            
            # Process image for output (resize if requested)
            output_image_data = None
            output_ext = original_ext
            if include_images:
                if resize_size:
                    # Apply smart resize
                    output_image_data = smart_resize_image(content, resize_size, crop_mode)
                    output_ext = ".jpg"  # Always output as JPEG after resize
                    logger.info(f"Resized {filename} to {resize_size}x{resize_size} ({crop_mode} mode)")
                else:
                    output_image_data = content
            
            # Store result
            job["results"].append({
                "filename": filename,
                "base_name": new_base_name,
                "original_ext": output_ext,
                "caption": caption,
                "data": result.get("data", {}),
                "image_data": output_image_data,
                "success": "error" not in result
            })
            
            # Cleanup temp file
            try:
                os.unlink(tmp_path)
            except:
                pass
                
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            rename_files = settings.get("rename_files", True)
            file_index = file_data.get("index", i)
            original_ext = file_data.get("original_ext", ".jpg")
            
            job["errors"].append({"filename": filename, "error": str(e)})
            job["results"].append({
                "filename": filename,
                "base_name": f"{file_index + 1:03d}" if rename_files else Path(filename).stem,
                "original_ext": original_ext,
                "caption": f"Error: {str(e)}",
                "data": {},
                "image_data": None,
                "success": False
            })
        
        # Small delay to prevent overloading
        await asyncio.sleep(0.1)
    
    job["completed"] = len(files)
    job["current_file"] = ""
    job["status"] = "completed"
    job["completed_at"] = datetime.now().isoformat()
    
    logger.info(f"Batch job {job_id} completed: {len(job['results'])} images processed")


@router.get("/download/{job_id}")
async def download_batch_results(job_id: str, format: str = "txt"):
    """Download batch results as ZIP"""
    if job_id not in batch_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = batch_jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    settings = job.get("settings", {})
    include_images = settings.get("include_images", False)
    
    # Create ZIP in memory
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for result in job["results"]:
            base_name = result["base_name"]
            original_ext = result.get("original_ext", ".jpg")
            
            if format in ["txt", "both"]:
                # Caption text file
                txt_content = result["caption"]
                zf.writestr(f"{base_name}.txt", txt_content)
            
            if format in ["json", "both"]:
                # JSON data file
                json_content = json.dumps(result.get("data", {"caption": result["caption"]}), indent=2)
                zf.writestr(f"{base_name}.json", json_content)
            
            # Include original image if requested
            if include_images and result.get("image_data"):
                zf.writestr(f"{base_name}{original_ext}", result["image_data"])
    
    zip_buffer.seek(0)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    persona_name = get_persona_name(settings.get("persona_id", "none"))
    if persona_name:
        filename = f"{persona_name}_dataset_{timestamp}.zip"
    else:
        filename = f"batch_captions_{timestamp}.zip"
    
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@router.delete("/job/{job_id}")
async def delete_batch_job(job_id: str):
    """Delete a batch job from memory"""
    if job_id in batch_jobs:
        del batch_jobs[job_id]
    return {"status": "deleted"}


@router.get("/jobs")
async def list_batch_jobs():
    """List all batch jobs"""
    return {
        "jobs": [
            {
                "job_id": jid,
                "status": job["status"],
                "total": job["total"],
                "completed": job["completed"],
                "started_at": job.get("started_at")
            }
            for jid, job in batch_jobs.items()
        ]
    }