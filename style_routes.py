from fastapi import APIRouter, Request
import ollama
import json
from utils import STYLES_FILE, load_json_file, save_json_file

router = APIRouter()

@router.get("/styles")
async def list_styles():
    return load_json_file(STYLES_FILE, {})

@router.post("/styles/analyze")
async def analyze_style_structure(request: Request):
    """
    Takes a raw prompt (e.g. from Civitai) and reverse-engineers the structure.
    """
    try:
        data = await request.json()
        raw_prompt = data.get("prompt")
        model = data.get("model", "llama3.2")
        
        system_prompt = """You are a Prompt Engineer. Analyze the USER'S PROMPT. 
        Determine the structural rules used to create it.
        
        Look for:
        1. Format (Sentences vs Comma-separated tags)
        2. Ordering (Subject first? Camera details last?)
        3. Specific vocabulary (e.g. "masterpiece", "best quality")
        4. Punctuation style.

        Return a SHORT, precise instruction set that I can feed to an AI to replicate this style for ANY image. 
        Example Output: "Use comma-separated tags. Start with subject description. End with camera model and '8k, best quality'. Do not use verbs."
        """
        
        response = ollama.chat(model=model, messages=[
            {'role': 'system', 'content': system_prompt}, 
            {'role': 'user', 'content': f"Analyze this prompt style:\n\n{raw_prompt}"}
        ])
        
        return {"status": "success", "instruction": response['message']['content']}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.post("/styles/save")
async def save_style(request: Request):
    try:
        data = await request.json()
        name = data.get("name")
        instruction = data.get("instruction")
        
        if not name or not instruction: return {"status": "error"}
        
        styles = load_json_file(STYLES_FILE, {})
        styles[name] = instruction
        save_json_file(STYLES_FILE, styles)
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.delete("/styles/{name}")
async def delete_style(name: str):
    styles = load_json_file(STYLES_FILE)
    if name in styles:
        del styles[name]
        save_json_file(STYLES_FILE, styles)
    return {"status": "deleted"}