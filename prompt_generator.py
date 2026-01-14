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
    Generates a descriptive, narrative-style prompt from JSON.
    Now with adaptive style based on detected aesthetic.
    """
    
    selected_model = preferred_model
    
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
        selected_model = "llama3.2"
        
    # Extract metadata
    clean_data = json_data.copy()
    
    if "reference_image_instruction" in clean_data:
        del clean_data["reference_image_instruction"]
    
    # Extract aesthetic for adaptive prompting
    aesthetic = clean_data.get("style", {}).get("aesthetic", "")
    vibe = clean_data.get("style", {}).get("vibe", "")
    
    meta_tags = []
    if "meta_tokens" in clean_data:
        meta_raw = clean_data.pop("meta_tokens")
        if isinstance(meta_raw, list):
            meta_tags = meta_raw
        elif isinstance(meta_raw, str):
            meta_tags = [meta_raw]

    # Adaptive system message based on detected style
    style_guidance = ""
    if aesthetic:
        style_guidance = f"\nThe aesthetic is '{aesthetic}' with a '{vibe}' vibe. Incorporate this mood into your writing style."

    system_msg = f"""You are a Visual Novelist and expert Image Prompter (Flux/Midjourney/Grok style).
Your goal is to write a Vivid, Natural Language Description based on JSON data.

RULES:
1. Write ONE fluid, cinematic paragraph (NO bullet points)
2. Use FULL SENTENCES with natural flow
3. Structure: Subject → Appearance → Outfit → Pose → Environment → Lighting → Mood
4. Be SPECIFIC: "crimson silk blouse" not "red shirt", "shoulder-length chestnut waves" not "brown hair"
5. Include sensory details: textures, lighting quality, atmosphere
6. Camera angles and technical details should feel natural ("captured in a medium shot" not "camera: medium")
7. Let the aesthetic guide your tone:{style_guidance}
8. Do NOT use technical quality terms ('8k', 'masterpiece') - they're added separately

QUALITY MARKERS:
- Specific color names and fabric types
- Precise physical descriptions
- Atmospheric/mood language
- Spatial relationships
- Lighting characteristics"""
    
    # Build context summary for better prompting
    subject_name = clean_data.get("subject", {}).get("name", "the subject")
    
    user_msg = f"""Convert this data into a rich, descriptive image prompt for {subject_name}.

DATA:
{json.dumps(clean_data, indent=2)}

EXAMPLE FORMAT (adapt to your data):
"A medium cinematic shot captures [Name], a [age] [ethnicity] woman with [specific hair description], in a [vibe] [aesthetic] setting. She wears a [specific outfit with fabrics and colors], [pose description with body language]. The [environment with specific details] is bathed in [lighting description with direction and quality], creating [atmosphere]. Her [expression] conveys [emotional state], while [notable details like accessories or makeup]."

Write a flowing, vivid paragraph that brings this scene to life:"""

    logger.info(f"Generating narrative prompt using {selected_model}")

    try:
        response = ollama.chat(
            model=selected_model,
            messages=[
                {'role': 'system', 'content': system_msg},
                {'role': 'user', 'content': user_msg}
            ],
            options={
                "temperature": 0.75,  # Slightly higher for more creative language
                "num_predict": 1536,   # More space for detailed prose
                "top_p": 0.92,
                "repeat_penalty": 1.15  # Prevent repetitive phrasing
            }
        )
        
        prompt_text = response['message']['content'].strip()
        
        # Cleanup
        if prompt_text.startswith('"') and prompt_text.endswith('"'):
            prompt_text = prompt_text[1:-1]
        
        # Remove any lingering markdown
        prompt_text = prompt_text.replace('**', '')

        # Append meta tokens
        if meta_tags:
            # Deduplicate tags
            unique_tags = list(dict.fromkeys(meta_tags))
            meta_string = ", ".join(unique_tags)
            prompt_text = f"{prompt_text}, {meta_string}"
            
        return prompt_text

    except Exception as e:
        logger.error(f"Prompt generation failed: {e}")
        return f"Error generating prompt: {str(e)}"