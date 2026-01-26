"""
FLUX.2 Klein Conditioning Enhancer - Fixed Version

Architecture:
- Shape: [batch, 512, 12288]
- Active region: Dynamic (from attention_mask, typically ~77 tokens)
- Padding: Remaining positions (low variance ~2.3)
- Metadata: pooled_output, attention_mask, reference_latents (edit mode)

FIXED: Previous version had mean-recentering that undid the scaling.
This version uses direct operations with measurable effects.
"""

from .flux2_klein_enhancer import (
    Flux2KleinEnhancer,
    Flux2KleinDetailController,
)

from .flux2_sectioned_encoder import (
    Flux2KleinSectionedEncoder,
)

NODE_CLASS_MAPPINGS = {
    "Flux2KleinEnhancer": Flux2KleinEnhancer,
    "Flux2KleinDetailController": Flux2KleinDetailController,
    "Flux2KleinSectionedEncoder": Flux2KleinSectionedEncoder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux2KleinEnhancer": "FLUX.2 Klein Enhancer",
    "Flux2KleinDetailController": "FLUX.2 Klein Detail Controller",
    "Flux2KleinSectionedEncoder": "FLUX.2 Klein Sectioned Encoder",
}

__version__ = "2.3.0"
