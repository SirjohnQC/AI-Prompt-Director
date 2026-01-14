"""
Enhanced Tag Generation Module for AI Prompt Director
Optimized for SDXL/Pony/Illustrious/SD1.5 booru-style tags
"""

import re
import logging
from typing import Dict, List, Set, Tuple, Any

logger = logging.getLogger(__name__)

class TagGenerator:
    """Comprehensive tag generation system for image-to-prompt conversion."""
    
    QUALITY_TAGS = [
        "masterpiece", "best quality", "high resolution", 
        "8k uhd", "extremely detailed", "ultra detailed"
    ]
    
    ETHNICITY_MAP = {
        'caucasian': ['caucasian', 'white', 'pale_skin'],
        'asian': ['asian', 'east_asian'],
        'african': ['dark_skin', 'african', 'black'],
        'latina': ['latina', 'hispanic', 'tan_skin'],
        'middle_eastern': ['middle_eastern', 'arabic'],
        'indian': ['indian', 'south_asian', 'brown_skin']
    }
    
    HAIR_COLORS = [
        'black_hair', 'brown_hair', 'blonde_hair', 'red_hair',
        'auburn_hair', 'platinum_blonde_hair', 'grey_hair', 'white_hair',
        'blue_hair', 'pink_hair', 'purple_hair', 'green_hair',
        'silver_hair', 'orange_hair', 'multicolored_hair'
    ]
    
    HAIR_STYLES = [
        'long_hair', 'short_hair', 'medium_hair', 'very_long_hair',
        'ponytail', 'twintails', 'braid', 'twin_braids', 'side_braid',
        'bun', 'double_bun', 'hair_bun', 'messy_hair',
        'straight_hair', 'wavy_hair', 'curly_hair',
        'bob_cut', 'pixie_cut', 'undercut', 'bangs', 'side_swept_bangs',
        'ahoge', 'hair_ribbon', 'hair_ornament'
    ]
    
    CLOTHING_ITEMS = [
        'shirt', 'dress', 'blouse', 'sweater', 'jacket', 'hoodie',
        'tank_top', 'crop_top', 'jeans', 'pants', 'skirt', 'shorts',
        'leggings', 'stockings', 'thighhighs', 'pantyhose'
    ]
    
    COMMON_NEGATIVE = (
        "lowres, bad anatomy, bad hands, text, error, missing fingers, "
        "extra digit, fewer digits, cropped, worst quality, low quality, "
        "normal quality, jpeg artifacts, signature, watermark, username, blurry, "
        "artist name, deformed, disfigured, mutation, mutated, extra limbs, "
        "missing limbs, floating limbs, disconnected limbs, malformed hands, "
        "long neck, ugly, poorly drawn, extra legs, fused fingers, too many fingers, "
        "long body, bad proportions, gross proportions, missing arms, missing legs, "
        "extra arms, extra legs, mutated hands, poorly drawn hands, poorly drawn face"
    )
    
    def __init__(self):
        self.seen_tags: Set[str] = set()
        
    def clean_raw_response(self, raw_text: str) -> str:
        """Clean AI model response for tag extraction."""
        # Remove markdown code blocks
        text = re.sub(r'```(?:tags|markdown|text)?\s*', '', raw_text, flags=re.IGNORECASE)
        text = re.sub(r'```\s*', '', text)
        
        # Remove common chatty prefixes
        prefixes = [
            r'^(here|sure|okay|certainly|of course|analysis|tags|output|result).*?[:]\s*',
            r'^(here\'s|here are|i\'ll|let me).*?[:]\s*',
            r'^based on.*?[:]\s*',
        ]
        for prefix_pattern in prefixes:
            text = re.sub(prefix_pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
        
        # Remove numbered/bulleted lists
        text = re.sub(r'^\d+[\.\)]\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'^[-*•]\s*', '', text, flags=re.MULTILINE)
        
        # Remove explanatory text in parentheses
        text = re.sub(r'\([^)]*\)', '', text)
        
        # Convert various separators to commas
        text = text.replace('\n', ', ')
        text = text.replace(';', ',')
        text = text.replace('|', ',')
        
        return text.strip()
    
    def extract_tags_from_text(self, text: str) -> List[str]:
        """Extract individual tags from cleaned text."""
        tags = []
        
        # Split by comma
        raw_tags = text.split(',')
        
        for tag in raw_tags:
            # Clean whitespace
            tag = tag.strip()
            
            # Remove trailing periods (FIX: Don't discard the whole tag)
            tag = tag.rstrip('.')
            
            # Skip empty or very short tags
            if len(tag) < 2:
                continue
                
            # Skip if it's clearly a full sentence intro (very long with few underscores)
            if len(tag) > 50 and '_' not in tag and ' ' in tag:
                continue
            
            # Convert to lowercase and replace spaces with underscores
            tag = tag.lower().replace(' ', '_')
            
            # Remove trailing punctuation again just in case
            tag = tag.rstrip('.,;:!?')
            
            # Remove "wearing_" prefix if present
            tag = re.sub(r'^wearing_', '', tag)
            
            # Skip if contains invalid characters
            if re.search(r'[^\w\-_(),]', tag):
                continue
            
            tags.append(tag)
        
        return tags
    
    def standardize_tag(self, tag: str) -> str:
        """Standardize tag format to match booru conventions."""
        # Handle color descriptors
        color_map = {
            'blond': 'blonde', 'blondie': 'blonde',
            'brunette': 'brown', 'ginger': 'red',
            'raven': 'black', 'silver': 'grey'
        }
        
        for old, new in color_map.items():
            if old in tag:
                tag = tag.replace(old, new)
        
        # Standardize clothing color format (e.g., "red shirt" -> "red_shirt")
        for item in self.CLOTHING_ITEMS:
            if item in tag and '_' not in tag:
                tag = tag.replace(' ', '_')
        
        # Fix common misspellings
        fixes = {
            'glasess': 'glasses',
            'accesory': 'accessory',
            'accesories': 'accessories',
            'jewlery': 'jewelry',
            'make_up': 'makeup',
            'eye_glass': 'glasses'
        }
        
        for wrong, right in fixes.items():
            if wrong in tag:
                tag = tag.replace(wrong, right)
        
        return tag
    
    def categorize_tags(self, tags: List[str]) -> Dict[str, List[str]]:
        """Organize tags into categories for better structure."""
        categories = {
            'quality': [],
            'subject': [],
            'appearance': [],
            'clothing': [],
            'pose': [],
            'environment': [],
            'technical': []
        }
        
        subject_keywords = ['girl', 'boy', 'woman', 'man', 'person', 'character', 'solo']
        appearance_keywords = ['hair', 'eye', 'skin', 'face', 'body']
        clothing_keywords = ['shirt', 'dress', 'pants', 'skirt', 'jacket', 'shoes', 'hat']
        pose_keywords = ['standing', 'sitting', 'lying', 'looking', 'smile', 'pose']
        env_keywords = ['indoor', 'outdoor', 'background', 'sky', 'room', 'street']
        tech_keywords = ['depth', 'lighting', 'angle', 'shot', 'focus', 'blur']
        
        for tag in tags:
            tag_lower = tag.lower()
            
            if any(q in tag_lower for q in ['quality', 'masterpiece', 'detailed', 'resolution']):
                categories['quality'].append(tag)
            elif any(s in tag_lower for s in subject_keywords):
                categories['subject'].append(tag)
            elif any(a in tag_lower for a in appearance_keywords):
                categories['appearance'].append(tag)
            elif any(c in tag_lower for c in clothing_keywords):
                categories['clothing'].append(tag)
            elif any(p in tag_lower for p in pose_keywords):
                categories['pose'].append(tag)
            elif any(e in tag_lower for e in env_keywords):
                categories['environment'].append(tag)
            elif any(t in tag_lower for t in tech_keywords):
                categories['technical'].append(tag)
            else:
                categories['appearance'].append(tag)
        
        return categories
    
    def deduplicate_and_prioritize(self, tags: List[str]) -> List[str]:
        """Remove duplicates and prioritize important tags."""
        seen = set()
        prioritized = []
        
        # Priority order for tag placement
        priority_patterns = [
            lambda t: any(q in t for q in ['masterpiece', 'best_quality', 'high_resolution']),
            lambda t: re.match(r'^\d+(girl|boy|woman|man)', t) or t == 'solo',
            lambda t: any(h in t for h in ['_hair', 'hair_']),
            lambda t: any(c in t for c in ['dress', 'shirt', 'jacket', 'uniform', 'costume']),
        ]
        
        # Sort by priority
        for priority_func in priority_patterns:
            for tag in tags:
                if tag not in seen and priority_func(tag):
                    seen.add(tag)
                    prioritized.append(tag)
        
        # Add remaining tags
        for tag in tags:
            if tag not in seen:
                seen.add(tag)
                prioritized.append(tag)
        
        return prioritized
    
    def enhance_with_persona(self, tags: List[str], persona_data: Dict[str, Any]) -> List[str]:
        """Enhance tags with persona information."""
        persona_tags = []
        
        if not persona_data:
            return tags
        
        # Add character name tag
        if persona_data.get('name'):
            safe_name = re.sub(r'[^a-z0-9_]', '_', persona_data['name'].lower())
            persona_tags.append(safe_name)
        
        persona_tags.append('consistent_character')
        
        subject = persona_data.get('subject', {})
        
        # Remove conflicting hair tags and add persona hair
        if 'hair' in subject and isinstance(subject['hair'], dict):
            tags = [t for t in tags if not any(h in t for h in self.HAIR_COLORS + self.HAIR_STYLES)]
            
            if subject['hair'].get('color'):
                color = subject['hair']['color'].lower().replace(' ', '_')
                persona_tags.append(f"{color}_hair")
            
            if subject['hair'].get('style'):
                style = subject['hair']['style'].lower().replace(' ', '_')
                persona_tags.append(style)
        
        # Add ethnicity if specified
        if subject.get('ethnicity'):
            ethnicity = subject['ethnicity'].lower()
            for key, aliases in self.ETHNICITY_MAP.items():
                if key in ethnicity or ethnicity in key:
                    persona_tags.append(aliases[0])
                    break
        
        # Add other features if they are short (1-2 words)
        for field in ['eyes', 'body_type']:
            val = subject.get(field, "")
            if val and len(val.split()) <= 2:
                persona_tags.append(val.lower().replace(' ', '_'))

        # Combine persona tags with existing (insert after quality tags)
        quality_count = sum(1 for t in tags if any(q in t for q in self.QUALITY_TAGS))
        insert_pos = max(quality_count, 3)
        
        for ptag in reversed(persona_tags):
            if ptag not in tags:
                tags.insert(insert_pos, ptag)
        
        return tags
    
    def generate_from_response(
        self, 
        raw_response: str, 
        persona_data: Dict[str, Any] = None,
        reference_mode: bool = False
    ) -> Tuple[str, str]:
        """
        Main method to generate tags from AI response.
        """
        self.seen_tags.clear()
        
        # Clean and extract tags
        cleaned = self.clean_raw_response(raw_response)
        raw_tags = self.extract_tags_from_text(cleaned)
        
        # Standardize tags
        tags = [self.standardize_tag(tag) for tag in raw_tags]
        
        # Remove empty tags
        tags = [t for t in tags if t]
        
        # Add quality tags if not present
        for qtag in reversed(self.QUALITY_TAGS[:3]):
            if not any(qtag in t for t in tags):
                tags.insert(0, qtag)
        
        # Add reference mode tag if enabled
        if reference_mode:
            if 'character_reference' not in tags:
                tags.insert(3, 'character_reference')
        
        # Enhance with persona data
        if persona_data:
            tags = self.enhance_with_persona(tags, persona_data)
        
        # Deduplicate and prioritize
        final_tags = self.deduplicate_and_prioritize(tags)
        
        # Join tags
        positive_tags = ", ".join(final_tags)
        
        return positive_tags, self.COMMON_NEGATIVE


# Integration function for main.py
def generate_tags_enhanced(
    raw_ai_response: str,
    persona_data: Dict[str, Any] = None,
    reference_mode: bool = False
) -> Dict[str, Any]:
    """Wrapper function to integrate with existing main.py"""
    generator = TagGenerator()
    
    try:
        positive_tags, negative_prompt = generator.generate_from_response(
            raw_ai_response,
            persona_data,
            reference_mode
        )
        
        tag_list = [t.strip() for t in positive_tags.split(',')]
        
        return {
            "positive_tags": positive_tags,
            "negative_prompt": negative_prompt,
            "tag_count": len(tag_list),
            "status": "success"
        }
    
    except Exception as e:
        logger.error(f"Tag generation failed: {e}")
        return {
            "positive_tags": "masterpiece, best quality, 1girl",
            "negative_prompt": TagGenerator.COMMON_NEGATIVE,
            "tag_count": 3,
            "status": "error",
            "error": str(e)
        }