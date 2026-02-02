"""
ComfyUI Integration Module for AI Prompt Director
Simplified version - auto-detects workflow from ComfyUI
"""

import json
import time
import uuid
import logging
import requests
import base64
import random
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/comfy", tags=["comfyui"])


class ComfyConfig:
    """ComfyUI connection settings"""
    def __init__(self, host: str = "127.0.0.1", port: int = 8188):
        self.host = host
        self.port = port
    
    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


# Global config
_config = ComfyConfig()
_client_id = str(uuid.uuid4())


class GenerateRequest(BaseModel):
    positive_prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 25
    cfg: float = 7.0
    seed: int = -1
    sampler: str = "euler"
    scheduler: str = "normal"
    checkpoint: Optional[str] = None
    workflow_type: str = "sdxl"  # sdxl, sd15, flux (hint only)


class ConfigRequest(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8188


# ============== WORKFLOW TEMPLATES ==============

def get_basic_workflow(
    positive: str,
    negative: str,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    seed: int,
    sampler: str,
    scheduler: str,
    checkpoint: Optional[str] = None
) -> dict:
    """
    Basic txt2img workflow that works with most setups.
    ComfyUI will use whatever checkpoint is currently loaded if none specified.
    """
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)
    
    workflow = {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "cfg": cfg,
                "denoise": 1,
                "latent_image": ["5", 0],
                "model": ["4", 0],
                "negative": ["7", 0],
                "positive": ["6", 0],
                "sampler_name": sampler,
                "scheduler": scheduler,
                "seed": seed,
                "steps": steps
            }
        },
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": checkpoint or "sd_xl_base_1.0.safetensors"
            }
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "batch_size": 1,
                "height": height,
                "width": width
            }
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["4", 1],
                "text": positive
            }
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["4", 1],
                "text": negative
            }
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["3", 0],
                "vae": ["4", 2]
            }
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {
                "filename_prefix": "PromptDirector",
                "images": ["8", 0]
            }
        }
    }
    
    return workflow


# ============== API FUNCTIONS ==============

def check_connection() -> dict:
    """Check if ComfyUI is running"""
    try:
        r = requests.get(f"{_config.base_url}/system_stats", timeout=3)
        if r.status_code == 200:
            return {"connected": True, "status": "ok", "stats": r.json()}
    except requests.exceptions.ConnectionError:
        pass
    except requests.exceptions.Timeout:
        pass
    except Exception as e:
        logger.error(f"Connection check error: {e}")
    
    return {
        "connected": False,
        "status": "error",
        "message": f"ComfyUI not running at {_config.base_url}. Start it with: python main.py --listen"
    }


def get_checkpoints() -> list:
    """Get list of available checkpoint models"""
    try:
        r = requests.get(f"{_config.base_url}/object_info/CheckpointLoaderSimple", timeout=5)
        if r.status_code == 200:
            data = r.json()
            checkpoints = data.get("CheckpointLoaderSimple", {}).get("input", {}).get("required", {}).get("ckpt_name", [[]])[0]
            return checkpoints if isinstance(checkpoints, list) else []
    except Exception as e:
        logger.error(f"Failed to get checkpoints: {e}")
    return []


def queue_prompt(workflow: dict) -> Optional[str]:
    """Queue a workflow for generation"""
    try:
        payload = {
            "prompt": workflow,
            "client_id": _client_id
        }
        r = requests.post(f"{_config.base_url}/prompt", json=payload, timeout=10)
        if r.status_code == 200:
            return r.json().get("prompt_id")
        else:
            logger.error(f"Queue failed: {r.status_code} - {r.text}")
    except Exception as e:
        logger.error(f"Queue error: {e}")
    return None


def get_history(prompt_id: str) -> Optional[dict]:
    """Get generation history"""
    try:
        r = requests.get(f"{_config.base_url}/history/{prompt_id}", timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        logger.error(f"History error: {e}")
    return None


def get_image(filename: str, subfolder: str = "", folder_type: str = "output") -> Optional[bytes]:
    """Download generated image"""
    try:
        params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        r = requests.get(f"{_config.base_url}/view", params=params, timeout=30)
        if r.status_code == 200:
            return r.content
    except Exception as e:
        logger.error(f"Get image error: {e}")
    return None


def wait_for_result(prompt_id: str, timeout: int = 300) -> dict:
    """Wait for generation to complete"""
    start = time.time()
    
    while time.time() - start < timeout:
        history = get_history(prompt_id)
        
        if history and prompt_id in history:
            data = history[prompt_id]
            
            # Check for errors
            status = data.get("status", {})
            if status.get("status_str") == "error":
                messages = status.get("messages", [])
                error_msg = messages[0][1] if messages else "Unknown error"
                return {"status": "error", "message": error_msg}
            
            # Check for outputs
            outputs = data.get("outputs", {})
            if outputs:
                images = []
                for node_id, node_output in outputs.items():
                    if "images" in node_output:
                        for img_info in node_output["images"]:
                            img_data = get_image(
                                img_info["filename"],
                                img_info.get("subfolder", ""),
                                img_info.get("type", "output")
                            )
                            if img_data:
                                images.append({
                                    "filename": img_info["filename"],
                                    "data": base64.b64encode(img_data).decode("utf-8")
                                })
                
                if images:
                    return {"status": "success", "images": images, "prompt_id": prompt_id}
        
        time.sleep(0.5)
    
    return {"status": "error", "message": f"Generation timed out after {timeout}s"}


# ============== API ROUTES ==============

@router.get("/status")
async def comfy_status():
    """Check ComfyUI connection"""
    return check_connection()


@router.post("/configure")
async def configure(config: ConfigRequest):
    """Update connection settings"""
    global _config
    _config = ComfyConfig(host=config.host, port=config.port)
    return {"status": "configured", "url": _config.base_url}


@router.get("/checkpoints")
async def list_checkpoints():
    """List available models"""
    conn = check_connection()
    if not conn.get("connected"):
        raise HTTPException(status_code=503, detail=conn.get("message"))
    return {"checkpoints": get_checkpoints()}


@router.post("/generate")
async def generate(req: GenerateRequest):
    """Generate an image"""
    # Check connection
    conn = check_connection()
    if not conn.get("connected"):
        raise HTTPException(status_code=503, detail=conn.get("message"))
    
    # Get available checkpoints if none specified
    checkpoint = req.checkpoint
    if not checkpoint:
        checkpoints = get_checkpoints()
        if checkpoints:
            checkpoint = checkpoints[0]  # Use first available
            logger.info(f"Using checkpoint: {checkpoint}")
    
    # Build workflow
    workflow = get_basic_workflow(
        positive=req.positive_prompt,
        negative=req.negative_prompt,
        width=req.width,
        height=req.height,
        steps=req.steps,
        cfg=req.cfg,
        seed=req.seed,
        sampler=req.sampler,
        scheduler=req.scheduler,
        checkpoint=checkpoint
    )
    
    # Queue the prompt
    prompt_id = queue_prompt(workflow)
    if not prompt_id:
        raise HTTPException(status_code=500, detail="Failed to queue prompt")
    
    logger.info(f"Queued prompt: {prompt_id}")
    
    # Wait for result
    result = wait_for_result(prompt_id, timeout=300)
    
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("message"))
    
    return result


@router.post("/cancel")
async def cancel():
    """Cancel current generation"""
    try:
        r = requests.post(f"{_config.base_url}/interrupt", timeout=5)
        return {"status": "cancelled" if r.status_code == 200 else "failed"}
    except:
        return {"status": "failed"}
