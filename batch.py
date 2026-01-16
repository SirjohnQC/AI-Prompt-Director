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
    return batch_jobs[job_id]


@router.post("/analyze")
async def batch_analyze(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    model: str = Form("qwen3-vl:8b"),
    caption_style: str = Form("detailed"),  # detailed, simple, tags
    persona_id: str = Form("none"),
    output_format: str = Form("txt")  # txt, json, both
):
    """
    Start batch analysis of multiple images
    Returns job_id to track progress
    """
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
            "persona_id": persona_id
        },
        "started_at": datetime.now().isoformat()
    }
    
    # Read all file contents before background task
    file_contents = []
    for f in files:
        content = await f.read()
        file_contents.append({
            "filename": f.filename,
            "content": content
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
            
            # Analyze
            result = analyze_single_image(tmp_path, model, caption_style)
            
            # Generate caption
            caption = generate_caption_from_data(result, caption_style)
            
            # Store result
            base_name = Path(filename).stem
            job["results"].append({
                "filename": filename,
                "base_name": base_name,
                "caption": caption,
                "data": result.get("data", {}),
                "success": "error" not in result
            })
            
            # Cleanup temp file
            try:
                os.unlink(tmp_path)
            except:
                pass
                
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            job["errors"].append({"filename": filename, "error": str(e)})
            job["results"].append({
                "filename": filename,
                "base_name": Path(filename).stem,
                "caption": f"Error: {str(e)}",
                "data": {},
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
    
    # Create ZIP in memory
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for result in job["results"]:
            base_name = result["base_name"]
            
            if format in ["txt", "both"]:
                # Caption text file
                txt_content = result["caption"]
                zf.writestr(f"{base_name}.txt", txt_content)
            
            if format in ["json", "both"]:
                # JSON data file
                json_content = json.dumps(result.get("data", {"caption": result["caption"]}), indent=2)
                zf.writestr(f"{base_name}.json", json_content)
    
    zip_buffer.seek(0)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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