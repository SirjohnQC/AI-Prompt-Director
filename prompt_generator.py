"""
Natural Language Prompt Generation Module
Optimized for Flux, Midjourney, and Grok (Narrative Style).
"""

import json
import logging
import ollama

logger = logging.getLogger(__name__)

def generate_narrative_prompt(json_data: dict, preferred_model: str = None) -> str:
    """
    Generates a descriptive, narrative-style prompt from a JSON object.
    """
    
    # 1. Use User Selection if provided
    selected_model = preferred_model
    
    # 2. If no selection (or 'auto'), try to find a good text model
    if not selected_model or selected_model == "auto":
        try:
            available_models = [m['model'] for m in ollama.list()['models']]
            text_candidates = ["llama3.2", "mistral", "qwen2.5", "gemma2", "neural-chat"]
            for candidate in text_candidates:
                if any(candidate in m for m in available_models):
                    selected_model = next(m for m in available_models if candidate in m)
                    break
        except:
            pass
            
    if not selected_model:
        selected_model = "llama3.2" # Hard fallback
        
    logger.info(f"Generating narrative prompt using {selected_model}")

    # 2. Construct the "Novel Writer" Prompt
    # This prompt forces the AI to break out of the "JSON List" mindset
    system_msg = """You are a Visual Novelist and expert Image Prompter (Flux/Midjourney style).
    Your goal is to write a Vivid, Natural Language Description of the character and scene.
    
    RULES:
    1. DO NOT output a comma-separated list (e.g. "Oval face, green eyes, ...").
    2. Write in FULL SENTENCES (e.g. "She has an oval face with striking green eyes...").
    3. CONNECT the details naturally. Flow from Subject -> Outfit -> Pose -> Environment -> Mood.
    4. Focus on lighting, texture, and atmosphere.
    5. Output ONE single, fluid paragraph.
    """
    
    # We strip the reference instruction for the prompt generation to keep the story clean
    # (The frontend appends it purely for the technical instruction if needed, 
    # but for the visual description, we want pure imagery).
    clean_data = json_data.copy()
    if "reference_image_instruction" in clean_data:
        del clean_data["reference_image_instruction"]

    user_msg = f"""Convert this character data into a rich, descriptive prompt for Flux/Midjourney.

    DATA:
    {json.dumps(clean_data, indent=2)}

    Desired Output Format Example:
    "A stunning medium shot of [Name], a [Age] year old [Ethnicity] woman with [Feature]. She is wearing [Outfit], crouching on a wooden bench. The setting is [Environment] with [Lighting]. The image has a [Style] aesthetic with [Quality] details."

    WRITE THE PROMPT:"""

    logger.info(f"Generating narrative prompt using {selected_model}")

    try:
        response = ollama.chat(
            model=selected_model,
            messages=[
                {'role': 'system', 'content': system_msg},
                {'role': 'user', 'content': user_msg}
            ],
            options={
                "temperature": 0.7, # Creativity allowed
                "num_predict": 1024
            }
        )
        
        prompt_text = response['message']['content'].strip()
        
        # Cleanup quotes
        if prompt_text.startswith('"') and prompt_text.endswith('"'):
            prompt_text = prompt_text[1:-1]
        if prompt_text.startswith("Here is the prompt:"):
            prompt_text = prompt_text.replace("Here is the prompt:", "").strip()
            
        return prompt_text

    except Exception as e:
        logger.error(f"Prompt generation failed: {e}")
        return f"Error generating prompt: {str(e)}"