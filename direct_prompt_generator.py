"""
Direct Image-to-Prompt Generator
Generates prompts DIRECTLY from images - no JSON intermediate step.
The JSON analysis becomes a separate, optional feature.
"""

import json
import logging
import io
import re
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# File paths (same as main.py)
BASE_DIR = Path(__file__).parent
PERSONA_FILE = BASE_DIR / "personas.json"
WARDROBE_FILE = BASE_DIR / "wardrobe.json"
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


@dataclass
class PromptContext:
    """Context for prompt generation"""
    # Persona info (if selected)
    persona_name: Optional[str] = None
    persona_age: Optional[str] = None
    persona_ethnicity: Optional[str] = None
    persona_hair_color: Optional[str] = None
    persona_hair_style: Optional[str] = None
    persona_eyes: Optional[str] = None
    persona_body_type: Optional[str] = None
    persona_skin: Optional[str] = None
    persona_makeup: Optional[str] = None
    persona_eyewear: Optional[str] = None
    
    # NSFW persona details (when nsfw_mode enabled)
    persona_body_proportions: Optional[dict] = None
    persona_intimate_details: Optional[dict] = None
    
    # Wardrobe info (if selected)
    wardrobe_description: Optional[str] = None
    
    # Style overrides
    time_of_day: Optional[str] = None
    aesthetic: Optional[str] = None
    expression: Optional[str] = None
    makeup_override: Optional[str] = None
    
    # Generation settings
    prompt_style: str = "narrative"  # narrative, cinematic, booru, minimal
    style_instruction: Optional[str] = None
    include_quality_tags: bool = True
    reference_mode: bool = False
    nsfw_mode: bool = False  # Enable explicit body descriptions


def build_persona_context(persona_id: str) -> dict:
    """Extract persona details for prompt context"""
    if not persona_id or persona_id == "none":
        return {}
    
    all_personas = load_json_file(PERSONA_FILE)
    if persona_id not in all_personas:
        return {}
    
    persona = all_personas[persona_id]
    p_data = persona.get("profile") or persona.get("subject") or {}
    
    context = {
        "name": persona.get("name"),
        "age": p_data.get("age"),
        "ethnicity": p_data.get("ethnicity"),
        "body_type": p_data.get("body_type"),
        "skin": p_data.get("skin"),
        "eyes": p_data.get("eyes"),
        "makeup": p_data.get("makeup"),
        "eyewear": p_data.get("eyewear"),
    }
    
    # Hair
    hair = p_data.get("hair", {})
    if isinstance(hair, dict):
        context["hair_color"] = hair.get("color")
        context["hair_style"] = hair.get("style")
        # NSFW: pubic hair if available
        if hair.get("pubic"):
            context["pubic_hair"] = hair["pubic"]
    elif isinstance(hair, str):
        context["hair_style"] = hair
    
    # Face features
    face = p_data.get("facial_features", p_data)
    if face.get("eyes"):
        context["eyes"] = face["eyes"]
    if face.get("makeup"):
        context["makeup"] = face["makeup"]
    if face.get("eyewear"):
        context["eyewear"] = face["eyewear"]
    
    # NSFW: Body proportions (detailed)
    body_props = p_data.get("body_proportions", {})
    if body_props:
        context["body_proportions"] = body_props
    
    # NSFW: Intimate details
    intimate = p_data.get("intimate_details", {})
    if intimate:
        context["intimate_details"] = intimate
    
    # NSFW: Piercings
    if p_data.get("piercings"):
        context["piercings"] = p_data["piercings"]
    
    # Tattoos
    if p_data.get("tattoos"):
        context["tattoos"] = p_data["tattoos"]
    
    return {k: v for k, v in context.items() if v}  # Remove None values


def build_wardrobe_context(wardrobe_id: str) -> dict:
    """Extract wardrobe details for prompt context"""
    if not wardrobe_id or wardrobe_id == "none":
        return {}
    
    wardrobe = load_json_file(WARDROBE_FILE)
    if wardrobe_id not in wardrobe:
        return {}
    
    item = wardrobe[wardrobe_id]
    return {
        "outfit_name": item.get("name"),
        "outfit_description": item.get("description"),
    }


def generate_system_prompt(ctx: PromptContext) -> str:
    """Build the system prompt based on style and context"""
    
    # Base instruction varies by style
    if ctx.prompt_style == "booru":
        base = """You are an expert at writing booru-style tags for AI image generation (Stable Diffusion, Pony, Illustrious).

OUTPUT FORMAT: Comma-separated tags only. No sentences.
TAG ORDER: quality tags, subject count, character traits (hair, eyes, body), makeup (if visible), clothing, pose, environment, lighting, style
EXAMPLE: masterpiece, best quality, 1girl, long brown hair, blue eyes, red lipstick, smoky eyeshadow, white sundress, standing, garden, golden hour, soft lighting, realistic"""
        
        if ctx.nsfw_mode:
            base += """

NSFW TAG ORDER: quality tags, rating (explicit/questionable), subject count, character traits, body details (breasts, nipples, pussy, ass if visible), clothing state (nude/topless/bottomless), pose, sexual acts if any, environment"""

    elif ctx.prompt_style == "cinematic":
        base = """You are a film director describing a shot for AI image generation.

STYLE: Use cinematic language - camera movements, lighting setups, emotional beats.
EXAMPLE: "A medium close-up tracks slowly as she turns toward the rain-streaked window, her silhouette rimmed by the cold blue light of the city below. The shallow depth of field isolates her contemplative expression against the bokeh of distant lights."
"""

    elif ctx.prompt_style == "minimal":
        base = """You are writing ultra-concise image prompts.

STYLE: Maximum impact, minimum words. 1-2 punchy phrases.
EXAMPLE: "Fierce redhead in leather jacket, neon-lit alley, rain"
"""

    else:  # narrative (default)
        base = """You are an expert image prompt writer for AI art generation (Flux, Midjourney, DALL-E).

STYLE: Vivid, flowing prose. Paint a picture with words. 2-4 rich sentences.
EXAMPLE: "A young woman with windswept auburn hair stands at the edge of a cliff overlooking a turbulent sea. Storm clouds gather behind her as golden light breaks through, illuminating her determined expression. Her vintage leather jacket whips in the wind, contrasting with the delicate silver pendant at her throat."
"""

    # Add custom style instruction if provided
    if ctx.style_instruction:
        base += f"\n\nADDITIONAL STYLE RULES:\n{ctx.style_instruction}"
    
    # Add persona constraints if provided
    persona_rules = []
    if ctx.persona_name:
        persona_rules.append(f"- Character name/trigger: {ctx.persona_name} (include naturally in prompt)")
    if ctx.persona_age:
        persona_rules.append(f"- Age: {ctx.persona_age}")
    if ctx.persona_ethnicity:
        persona_rules.append(f"- Ethnicity: {ctx.persona_ethnicity}")
    if ctx.persona_hair_color or ctx.persona_hair_style:
        hair_desc = " ".join(filter(None, [ctx.persona_hair_color, ctx.persona_hair_style]))
        persona_rules.append(f"- Hair: {hair_desc}")
    if ctx.persona_eyes:
        persona_rules.append(f"- Eyes: {ctx.persona_eyes}")
    if ctx.persona_body_type:
        persona_rules.append(f"- Body type: {ctx.persona_body_type}")
    if ctx.persona_makeup:
        persona_rules.append(f"- Makeup: {ctx.persona_makeup}")
    if ctx.persona_eyewear and ctx.persona_eyewear.lower() not in ["none", "no"]:
        persona_rules.append(f"- Eyewear: {ctx.persona_eyewear}")
    
    if persona_rules:
        base += "\n\nCHARACTER CONSTRAINTS (must include these traits):\n" + "\n".join(persona_rules)
    
    # NSFW mode - add detailed body descriptions
    if ctx.nsfw_mode:
        nsfw_rules = []
        if ctx.persona_body_proportions:
            bp = ctx.persona_body_proportions
            if bp.get("chest"):
                nsfw_rules.append(f"- Bust/chest: {bp['chest']}")
            if bp.get("hips"):
                nsfw_rules.append(f"- Hips: {bp['hips']}")
            if bp.get("butt"):
                nsfw_rules.append(f"- Buttocks: {bp['butt']}")
            if bp.get("thighs"):
                nsfw_rules.append(f"- Thighs: {bp['thighs']}")
            if bp.get("waist"):
                nsfw_rules.append(f"- Waist: {bp['waist']}")
        
        if ctx.persona_intimate_details:
            intimate = ctx.persona_intimate_details
            if intimate.get("nipples"):
                nsfw_rules.append(f"- Nipples: {intimate['nipples']}")
            if intimate.get("areolas"):
                nsfw_rules.append(f"- Areolas: {intimate['areolas']}")
        
        if nsfw_rules:
            base += "\n\nBODY DETAILS (NSFW - include when visible):\n" + "\n".join(nsfw_rules)
        
        base += "\n\nNSFW MODE: Include explicit anatomical details when visible. Describe body parts accurately and sensually."
    
    # Add wardrobe if provided
    if ctx.wardrobe_description:
        base += f"\n\nOUTFIT TO USE: {ctx.wardrobe_description}"
    
    # Add overrides
    overrides = []
    if ctx.time_of_day and ctx.time_of_day != "auto":
        overrides.append(f"Time of day: {ctx.time_of_day}")
    if ctx.aesthetic and ctx.aesthetic != "auto":
        overrides.append(f"Visual style: {ctx.aesthetic}")
    if ctx.expression and ctx.expression != "auto":
        overrides.append(f"Expression: {ctx.expression}")
    if ctx.makeup_override and ctx.makeup_override != "auto":
        overrides.append(f"Makeup: {ctx.makeup_override}")
    
    if overrides:
        base += "\n\nSCENE REQUIREMENTS:\n- " + "\n- ".join(overrides)
    
    # Reference mode
    if ctx.reference_mode and ctx.persona_name:
        base += f"\n\nREFERENCE MODE: This is for character consistency. Emphasize {ctx.persona_name}'s recognizable features."
    
    # Final rules
    base += """

CRITICAL RULES:
1. Start DIRECTLY with the description - never "Here is..." or "A photo of..."
2. Be specific: "crimson silk blouse" not "red shirt", "honey-blonde waves" not "blonde hair"
3. Include: subject, appearance, makeup (if visible), clothing, pose/action, environment, lighting, mood
4. Output ONLY the prompt - no explanations, no quotes around it
5. Do NOT mention image quality terms like "8k" or "masterpiece" unless doing booru tags"""

    return base


def generate_prompt_from_image(
    image_path: str,
    model: str = "qwen2.5-vl",
    persona_id: str = "none",
    wardrobe_id: str = "none",
    prompt_style: str = "narrative",
    style_instruction: str = None,
    time_override: str = "auto",
    aesthetic_override: str = "auto",
    expression_override: str = "auto",
    makeup_override: str = "auto",
    reference_mode: bool = False,
    nsfw_mode: bool = False,
    quality_tags: List[str] = None
) -> Dict[str, Any]:
    """
    Generate a prompt directly from an image.
    
    This is the main entry point - bypasses JSON entirely.
    """
    import ollama
    
    # Build context from persona/wardrobe
    persona_ctx = build_persona_context(persona_id)
    wardrobe_ctx = build_wardrobe_context(wardrobe_id)
    
    # Create prompt context
    ctx = PromptContext(
        persona_name=persona_ctx.get("name"),
        persona_age=persona_ctx.get("age"),
        persona_ethnicity=persona_ctx.get("ethnicity"),
        persona_hair_color=persona_ctx.get("hair_color"),
        persona_hair_style=persona_ctx.get("hair_style"),
        persona_eyes=persona_ctx.get("eyes"),
        persona_body_type=persona_ctx.get("body_type"),
        persona_skin=persona_ctx.get("skin"),
        persona_makeup=persona_ctx.get("makeup"),
        persona_eyewear=persona_ctx.get("eyewear"),
        persona_body_proportions=persona_ctx.get("body_proportions"),
        persona_intimate_details=persona_ctx.get("intimate_details"),
        wardrobe_description=wardrobe_ctx.get("outfit_description"),
        time_of_day=time_override,
        aesthetic=aesthetic_override,
        expression=expression_override,
        makeup_override=makeup_override,
        prompt_style=prompt_style,
        style_instruction=style_instruction,
        reference_mode=reference_mode,
        nsfw_mode=nsfw_mode
    )
    
    # Build system prompt
    system_prompt = generate_system_prompt(ctx)
    
    # User prompt
    if prompt_style == "booru":
        user_prompt = "Analyze this image and output booru-style tags:"
    else:
        user_prompt = "Look at this image and write a compelling prompt to recreate it:"
    
    logger.info(f"Generating {prompt_style} prompt directly from image using {model}")
    logger.info(f"System prompt length: {len(system_prompt)}")
    logger.info(f"User prompt: {user_prompt}")
    
    try:
        response = ollama.chat(
            model=model,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt, 'images': [image_path]}
            ],
            options={
                "temperature": 0.7 if prompt_style in ["narrative", "cinematic"] else 0.3,
                "num_predict": 1024,
                "num_ctx": 8192
            }
        )
        
        raw_response = response['message']['content']
        logger.info(f"=== RAW MODEL RESPONSE ({len(raw_response)} chars) ===")
        logger.info(raw_response[:1000] if raw_response else "COMPLETELY EMPTY")
        logger.info("=== END RAW RESPONSE ===")
        
        prompt = raw_response.strip()
        
        # Handle qwen3-vl thinking tags
        # The model often puts EVERYTHING in <think> tags - we need to extract, not discard!
        import re
        
        # Check if response contains think tags
        think_match = re.search(r'<think>(.*?)</think>', prompt, flags=re.DOTALL | re.IGNORECASE)
        non_think_content = re.sub(r'<think>.*?</think>', '', prompt, flags=re.DOTALL | re.IGNORECASE).strip()
        
        logger.info(f"Has think tags: {think_match is not None}")
        logger.info(f"Non-think content length: {len(non_think_content)}")
        
        if think_match:
            think_content = think_match.group(1).strip()
            logger.info(f"Think content: {len(think_content)} chars")
            logger.info(f"Think preview: {think_content[:300] if think_content else 'EMPTY'}")
            
            # If there's good content outside think tags, use that
            if non_think_content and len(non_think_content) > 30:
                prompt = non_think_content
                logger.info(f"Using non-think content: {prompt[:200]}")
            # Otherwise, the actual prompt might be INSIDE the think tags
            elif think_content and len(think_content) > 30:
                # Try to find a prompt-like section in think content
                # Often it's after "prompt:" or the last sentence
                prompt_patterns = [
                    r'(?:prompt|description|output):\s*["\']?(.+?)["\']?\s*(?:\n\n|$)',
                    r'["\']([^"\']{50,})["\']',  # Long quoted text
                    r'\n\n(.{50,})$',  # Last paragraph
                ]
                
                extracted = False
                for pattern in prompt_patterns:
                    match = re.search(pattern, think_content, re.IGNORECASE | re.DOTALL)
                    if match:
                        prompt = match.group(1).strip()
                        logger.info(f"Extracted from think via pattern: {prompt[:200]}")
                        extracted = True
                        break
                
                if not extracted:
                    # Use the think content itself, but clean it up
                    # Remove reasoning markers
                    cleaned = re.sub(r'\*\*.*?\*\*', '', think_content)  # Remove bold markers
                    cleaned = re.sub(r'^[-*]\s+', '', cleaned, flags=re.MULTILINE)  # Remove list markers
                    
                    # Find the longest paragraph that looks like a description
                    paragraphs = [p.strip() for p in cleaned.split('\n\n') if len(p.strip()) > 50]
                    if paragraphs:
                        # Use the last substantial paragraph (usually the final answer)
                        prompt = paragraphs[-1]
                        logger.info(f"Using last paragraph from think: {prompt[:200]}")
                    else:
                        # Last resort: use cleaned think content
                        prompt = cleaned.strip()
                        logger.info(f"Using cleaned think content: {prompt[:200]}")
        
        # CRITICAL: If prompt is still empty, generate a basic description
        if not prompt or len(prompt) < 20:
            logger.warning("Prompt is empty after processing! Generating fallback...")
            # Try a simpler prompt
            try:
                fallback_response = ollama.chat(
                    model=model,
                    messages=[
                        {'role': 'user', 'content': 'Describe this image in 2-3 sentences. Be specific about the subject, their appearance, clothing, and setting.', 'images': [image_path]}
                    ],
                    options={"temperature": 0.5, "num_predict": 512}
                )
                prompt = fallback_response['message']['content'].strip()
                # Clean think tags from fallback too
                prompt = re.sub(r'<think>.*?</think>', '', prompt, flags=re.DOTALL | re.IGNORECASE).strip()
                logger.info(f"Fallback prompt: {prompt[:200]}")
            except Exception as fb_e:
                logger.error(f"Fallback also failed: {fb_e}")
                prompt = "A person in a photograph."
        
        # Clean up common issues
        prompt = prompt.strip('"\'')
        
        # Remove intro phrases
        bad_starts = [
            "here is", "here's", "sure,", "certainly,", 
            "a photo of", "a photograph of", "an image of",
            "this image shows", "the image depicts",
            "the prompt:", "prompt:", "description:",
            "i would describe", "looking at this image"
        ]
        prompt_lower = prompt.lower()
        for bad in bad_starts:
            if prompt_lower.startswith(bad):
                prompt = prompt[len(bad):].strip()
                prompt = prompt.lstrip(",:. ")
                if prompt:
                    prompt = prompt[0].upper() + prompt[1:]
                break
        
        logger.info(f"Final prompt ({len(prompt)} chars): {prompt[:300]}")
        
        # Add quality tags for booru style
        if prompt_style == "booru" and quality_tags:
            existing_tags = prompt.lower()
            new_tags = [t for t in quality_tags if t.lower() not in existing_tags]
            if new_tags:
                prompt = ", ".join(new_tags) + ", " + prompt
        
        # Build negative prompt
        if prompt_style == "booru":
            negative = (
                "lowres, bad anatomy, bad hands, text, error, missing fingers, "
                "extra digit, fewer digits, cropped, worst quality, low quality, "
                "normal quality, jpeg artifacts, signature, watermark, username, "
                "blurry, deformed, disfigured, mutation, ugly"
            )
        else:
            negative = "blurry, distorted, low quality, text, watermark"
        
        # Build reference instruction if needed
        reference_instruction = None
        if reference_mode and ctx.persona_name:
            reference_instruction = f"Use reference image img0.png to maintain consistent appearance of {ctx.persona_name}."
        
        return {
            "status": "success",
            "prompt": prompt,
            "negative": negative,
            "style": prompt_style,
            "reference_instruction": reference_instruction,
            "persona_used": ctx.persona_name
        }
        
    except Exception as e:
        logger.error(f"Direct prompt generation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "status": "error",
            "error": str(e)
        }


def generate_prompt_from_text(
    text_description: str,
    model: str = "llama3.2",
    persona_id: str = "none",
    prompt_style: str = "narrative",
    style_instruction: str = None
) -> Dict[str, Any]:
    """
    Generate/enhance a prompt from text input (no image).
    Useful for refining or transforming existing prompts.
    """
    import ollama
    
    # Build context from persona
    persona_ctx = build_persona_context(persona_id)
    
    ctx = PromptContext(
        persona_name=persona_ctx.get("name"),
        persona_age=persona_ctx.get("age"),
        persona_ethnicity=persona_ctx.get("ethnicity"),
        persona_hair_color=persona_ctx.get("hair_color"),
        persona_hair_style=persona_ctx.get("hair_style"),
        persona_eyes=persona_ctx.get("eyes"),
        persona_makeup=persona_ctx.get("makeup"),
        persona_eyewear=persona_ctx.get("eyewear"),
        prompt_style=prompt_style,
        style_instruction=style_instruction
    )
    
    system_prompt = generate_system_prompt(ctx)
    system_prompt += "\n\nYou are enhancing/rewriting an existing prompt. Keep the core idea but improve the writing."
    
    try:
        response = ollama.chat(
            model=model,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': f"Rewrite this prompt in {prompt_style} style:\n\n{text_description}"}
            ],
            options={
                "temperature": 0.7,
                "num_predict": 512
            }
        )
        
        prompt = response['message']['content'].strip().strip('"\'')
        
        return {
            "status": "success",
            "prompt": prompt,
            "negative": "blurry, distorted, low quality, text, watermark",
            "style": prompt_style
        }
        
    except Exception as e:
        logger.error(f"Text prompt generation failed: {e}")
        return {"status": "error", "error": str(e)}


# ============================================================
# BATCH PROCESSING SUPPORT
# ============================================================

def generate_caption_for_lora(
    image_path: str,
    model: str = "qwen2.5-vl",
    persona_id: str = "none",
    style_instruction: str = None,
    caption_style: str = "detailed"  # detailed, simple, tags
) -> str:
    """
    Generate a caption for LoRA training.
    Optimized for dataset preparation - includes trigger word naturally.
    """
    import ollama
    
    # Get persona trigger word
    trigger_word = None
    persona_traits = ""
    if persona_id and persona_id != "none":
        persona_ctx = build_persona_context(persona_id)
        trigger_word = persona_ctx.get("name")
        
        # Build traits string for context
        traits = []
        if persona_ctx.get("age"):
            traits.append(persona_ctx["age"])
        if persona_ctx.get("ethnicity"):
            traits.append(persona_ctx["ethnicity"])
        if persona_ctx.get("hair_color") or persona_ctx.get("hair_style"):
            hair = " ".join(filter(None, [persona_ctx.get("hair_color"), persona_ctx.get("hair_style")]))
            traits.append(f"{hair} hair")
        if persona_ctx.get("eyes"):
            traits.append(f"{persona_ctx['eyes']} eyes")
        
        if traits:
            persona_traits = f"The subject is: {', '.join(traits)}."
    
    # Build system prompt based on caption style
    if caption_style == "tags":
        system_prompt = f"""Write comma-separated tags for this image.
Include: subject traits, clothing, pose, environment, lighting.
{f'Include the trigger word "{trigger_word}" early in the tags.' if trigger_word else ''}
{persona_traits}
Output ONLY tags, no sentences."""
    
    elif caption_style == "simple":
        system_prompt = f"""Write ONE detailed sentence describing this image.
{f'Naturally include the name "{trigger_word}" when describing the subject.' if trigger_word else ''}
{persona_traits}
Be specific about appearance, clothing, pose, and setting."""
    
    else:  # detailed
        system_prompt = f"""Write a natural image caption for LoRA training (2-4 sentences).

{f'IMPORTANT: Include "{trigger_word}" naturally when first describing the subject.' if trigger_word else ''}
{persona_traits}

RULES:
1. Start directly with the subject description
2. Be specific: exact colors, materials, styles
3. Include: appearance, clothing, pose, setting, lighting
4. Natural language, not a list
5. No meta-commentary about the image

{f'EXAMPLE: "A young woman {trigger_word} with long brown hair sits on a park bench, wearing a cream-colored sweater..."' if trigger_word else ''}"""

    if style_instruction:
        system_prompt += f"\n\nADDITIONAL STYLE: {style_instruction}"
    
    try:
        response = ollama.chat(
            model=model,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': 'Write the caption for this image:', 'images': [image_path]}
            ],
            options={
                "temperature": 0.4,
                "num_predict": 512,
                "num_ctx": 4096
            }
        )
        
        caption = response['message']['content'].strip()
        
        # Clean up
        caption = caption.strip('"\'')
        bad_starts = ["here", "sure", "this image", "the image", "caption:"]
        for bad in bad_starts:
            if caption.lower().startswith(bad):
                caption = caption.split(":", 1)[-1].strip() if ":" in caption[:20] else caption[len(bad):].strip()
                caption = caption.lstrip(",:. ")
                if caption:
                    caption = caption[0].upper() + caption[1:]
                break
        
        return caption
        
    except Exception as e:
        logger.error(f"Caption generation failed: {e}")
        return f"Error: {str(e)}"