"""
FLUX.2 Klein Text Conditioning Enhancer - Corrected Math
"""

import torch
import math
import gc

try:
    import comfy.model_management as mm
    HAS_COMFY = True
except ImportError:
    HAS_COMFY = False


class Flux2KleinTextEnhancer:
    """
    Modify text conditioning tokens.
    
    Operations (in order):
    1. Normalize: Equalize token magnitudes
    2. Contrast: Amplify/reduce differences between tokens
    3. Magnitude: Global scaling
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "magnitude": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.05,
                    "tooltip": "Scale text embeddings. <1=weaker prompt, >1=stronger"
                }),
            },
            "optional": {
                "contrast": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Token differentiation. >0=sharper, <0=blended"
                }),
                "normalize_strength": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Equalize token magnitudes"
                }),
                "skip_bos": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Skip token 0 (BOS token with huge norm)"
                }),
                "debug": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "enhance"
    CATEGORY = "conditioning/flux2klein"

    def enhance(self, conditioning, magnitude=1.0, contrast=0.0, 
                normalize_strength=0.0, skip_bos=True, debug=False):
        
        if not conditioning:
            return (conditioning,)
        
        # No-op check
        if magnitude == 1.0 and contrast == 0.0 and normalize_strength == 0.0:
            return (conditioning,)
        
        output = []
        
        for idx, (cond_tensor, meta) in enumerate(conditioning):
            cond = cond_tensor.float().clone()
            batch, seq_len, embed_dim = cond.shape
            
            # Get active region
            attn_mask = meta.get("attention_mask", None)
            if attn_mask is not None and attn_mask.dim() == 2:
                nonzero = attn_mask[0].nonzero()
                active_end = int(nonzero[-1].item()) + 1 if len(nonzero) > 0 else 77
            else:
                active_end = min(77, seq_len)
            
            start_idx = 1 if skip_bos else 0
            active = cond[:, start_idx:active_end, :]
            
            if debug:
                print(f"\n[TextEnhancer] Active tokens: [{start_idx}:{active_end}]")
                print(f"  Initial mean norm: {active.norm(dim=-1).mean():.4f}")
            
            # 1. Normalize
            if normalize_strength > 0.0:
                norms = active.norm(dim=-1, keepdim=True)
                mean_norm = norms.mean()
                normalized = active / (norms + 1e-8) * mean_norm
                active = active * (1 - normalize_strength) + normalized * normalize_strength
            
            # 2. Contrast (safe negative handling)
            if contrast != 0.0:
                seq_mean = active.mean(dim=1, keepdim=True)
                deviation = active - seq_mean
                
                if contrast >= 0:
                    scale = 1.0 + contrast
                else:
                    scale = math.exp(contrast)  # Never inverts
                
                active = seq_mean + deviation * scale
                
                if debug:
                    print(f"  Contrast scale: {scale:.4f}")
            
            # 3. Magnitude
            if magnitude != 1.0:
                active = active * magnitude
            
            if debug:
                print(f"  Final mean norm: {active.norm(dim=-1).mean():.4f}")
            
            # Write back
            cond[:, start_idx:active_end, :] = active
            output.append((cond.to(cond_tensor.dtype), meta))
        
        return (output,)


NODE_CLASS_MAPPINGS = {
    "Flux2KleinTextEnhancer": Flux2KleinTextEnhancer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux2KleinTextEnhancer": "FLUX.2 Klein Text Enhancer",
}
