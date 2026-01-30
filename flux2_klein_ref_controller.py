"""
FLUX.2 Klein Reference Latent Controller

Directly manipulates the reference latent in conditioning metadata.
"""

import torch
import gc

try:
    import comfy.model_management as mm
    HAS_COMFY = True
except ImportError:
    HAS_COMFY = False


class Flux2KleinRefLatentController:
    """
    Control the reference latent strength and characteristics.
    
    The reference latent is stored in metadata as [1, 128, H, W].
    This node modifies it directly to control structure preservation.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 5.0,
                    "step": 0.05,
                    "tooltip": "Scale reference latent. 0=ignore reference, 1=normal, >1=stronger structure"
                }),
            },
            "optional": {
                "blend_with_noise": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Blend reference with noise. 0=pure reference, 1=pure noise"
                }),
                "channel_mask_start": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 127,
                    "tooltip": "Start channel for selective modification"
                }),
                "channel_mask_end": ("INT", {
                    "default": 128,
                    "min": 1,
                    "max": 128,
                    "tooltip": "End channel for selective modification (0=all)"
                }),
                "spatial_fade": (["none", "center_out", "edges_out", "top_down", "left_right"], {
                    "default": "none",
                    "tooltip": "Apply spatial gradient to reference strength"
                }),
                "spatial_fade_strength": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1,
                }),
                "debug": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "control"
    CATEGORY = "conditioning/flux2klein"

    def _create_spatial_mask(self, h, w, mode, strength):
        """Create spatial gradient mask."""
        if mode == "none":
            return torch.ones(h, w)
        
        y = torch.linspace(0, 1, h).unsqueeze(1).expand(h, w)
        x = torch.linspace(0, 1, w).unsqueeze(0).expand(h, w)
        
        if mode == "center_out":
            # Center = 1, edges = (1-strength)
            cy, cx = 0.5, 0.5
            dist = torch.sqrt((y - cy)**2 + (x - cx)**2)
            dist = dist / dist.max()
            mask = 1.0 - dist * strength
        elif mode == "edges_out":
            # Edges = 1, center = (1-strength)
            cy, cx = 0.5, 0.5
            dist = torch.sqrt((y - cy)**2 + (x - cx)**2)
            dist = dist / dist.max()
            mask = (1.0 - strength) + dist * strength
        elif mode == "top_down":
            mask = 1.0 - y * strength
        elif mode == "left_right":
            mask = 1.0 - x * strength
        else:
            mask = torch.ones(h, w)
        
        return mask.clamp(0, 1)

    def control(self, conditioning, strength=1.0, blend_with_noise=0.0,
                channel_mask_start=0, channel_mask_end=128,
                spatial_fade="none", spatial_fade_strength=0.5, debug=False):
        
        if not conditioning:
            return (conditioning,)
        
        # Check if anything needs to be done
        if strength == 1.0 and blend_with_noise == 0.0 and spatial_fade == "none":
            if debug:
                print("[RefLatentController] All parameters neutral, passing through")
            return (conditioning,)
        
        output = []
        
        for idx, (cond_tensor, meta) in enumerate(conditioning):
            new_meta = meta.copy()
            
            # Check for reference latents
            ref_latents = meta.get("reference_latents", None)
            
            if ref_latents is None or len(ref_latents) == 0:
                if debug:
                    print(f"[RefLatentController] Item {idx}: No reference latents found")
                output.append((cond_tensor, new_meta))
                continue
            
            # Get the reference latent tensor
            ref = ref_latents[0]  # [1, 128, H, W]
            original_dtype = ref.dtype
            ref = ref.float().clone()
            
            if debug:
                print(f"\n[RefLatentController] Item {idx}")
                print(f"  Reference shape: {ref.shape}")
                print(f"  Original stats: mean={ref.mean():.4f}, std={ref.std():.4f}")
            
            _, c, h, w = ref.shape
            
            # Create channel mask
            ch_start = min(channel_mask_start, c - 1)
            ch_end = min(channel_mask_end, c)
            
            # Create spatial mask
            spatial_mask = self._create_spatial_mask(h, w, spatial_fade, spatial_fade_strength)
            spatial_mask = spatial_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            spatial_mask = spatial_mask.to(ref.device)
            
            # Apply modifications to selected channels
            modified = ref.clone()
            
            # 1. Blend with noise
            if blend_with_noise > 0.0:
                noise = torch.randn_like(ref[:, ch_start:ch_end, :, :])
                noise = noise * ref[:, ch_start:ch_end, :, :].std()  # Match scale
                modified[:, ch_start:ch_end, :, :] = (
                    ref[:, ch_start:ch_end, :, :] * (1 - blend_with_noise) +
                    noise * blend_with_noise
                )
                if debug:
                    print(f"  Noise blend: {blend_with_noise:.2f} on channels [{ch_start}:{ch_end}]")
            
            # 2. Apply strength scaling with spatial mask
            full_mask = torch.ones_like(ref)
            full_mask[:, ch_start:ch_end, :, :] = spatial_mask.expand(-1, ch_end - ch_start, -1, -1)
            
            # Strength applied through mask: 1.0 = no change to mask, other values scale
            effective_strength = full_mask * strength + (1 - full_mask) * 1.0
            modified = modified * effective_strength
            
            if debug:
                print(f"  Strength: {strength:.2f}")
                print(f"  Spatial fade: {spatial_fade}")
                print(f"  Modified stats: mean={modified.mean():.4f}, std={modified.std():.4f}")
            
            # Store modified reference latent
            new_meta["reference_latents"] = [modified.to(original_dtype)]
            output.append((cond_tensor, new_meta))
        
        gc.collect()
        return (output,)


class Flux2KleinTextRefBalance:
    """
    Simple node to balance text conditioning vs reference latent.
    
    Uses a single slider: 0 = reference only, 0.5 = balanced, 1 = text only
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "balance": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "0=reference structure, 0.5=balanced, 1=follow text prompt"
                }),
            },
            "optional": {
                "debug": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "balance_streams"
    CATEGORY = "conditioning/flux2klein"

    def balance_streams(self, conditioning, balance=0.5, debug=False):
        if not conditioning:
            return (conditioning,)
        
        # Convert balance to individual scalings
        # balance=0: text=0, ref=1
        # balance=0.5: text=1, ref=1
        # balance=1: text=1, ref=0
        
        if balance <= 0.5:
            # 0 to 0.5: text scales 0 to 1, ref stays at 1
            text_scale = balance * 2
            ref_scale = 1.0
        else:
            # 0.5 to 1: text stays at 1, ref scales 1 to 0
            text_scale = 1.0
            ref_scale = (1.0 - balance) * 2
        
        if debug:
            print(f"[TextRefBalance] balance={balance:.2f} -> text={text_scale:.2f}, ref={ref_scale:.2f}")
        
        output = []
        
        for idx, (cond_tensor, meta) in enumerate(conditioning):
            # Scale text conditioning
            modified_cond = cond_tensor.clone()
            
            # Get active region from attention mask
            attn_mask = meta.get("attention_mask", None)
            if attn_mask is not None and attn_mask.dim() == 2:
                nonzero = attn_mask[0].nonzero()
                active_end = int(nonzero[-1].item()) + 1 if len(nonzero) > 0 else 77
            else:
                active_end = 77
            
            # Scale active text tokens (skip token 0 which is BOS with huge norm)
            if text_scale != 1.0:
                modified_cond[:, 1:active_end, :] = modified_cond[:, 1:active_end, :] * text_scale
            
            # Scale reference latent
            new_meta = meta.copy()
            ref_latents = meta.get("reference_latents", None)
            
            if ref_latents is not None and len(ref_latents) > 0 and ref_scale != 1.0:
                ref = ref_latents[0].clone()
                ref = ref * ref_scale
                new_meta["reference_latents"] = [ref]
            
            output.append((modified_cond, new_meta))
        
        return (output,)


NODE_CLASS_MAPPINGS = {
    "Flux2KleinRefLatentController": Flux2KleinRefLatentController,
    "Flux2KleinTextRefBalance": Flux2KleinTextRefBalance,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux2KleinRefLatentController": "FLUX.2 Klein Ref Latent Controller",
    "Flux2KleinTextRefBalance": "FLUX.2 Klein Text/Ref Balance",
}
