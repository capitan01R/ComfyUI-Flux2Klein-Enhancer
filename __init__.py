from .flux2_klein_ref_controller import NODE_CLASS_MAPPINGS as REF_NODES, NODE_DISPLAY_NAME_MAPPINGS as REF_NAMES
from .flux2_klein_text_enhancer import NODE_CLASS_MAPPINGS as TEXT_NODES, NODE_DISPLAY_NAME_MAPPINGS as TEXT_NAMES
from .flux2_klein_enhancer import Flux2KleinEnhancer, Flux2KleinDetailController
from .flux2_sectioned_encoder import Flux2KleinSectionedEncoder
from .flux2_klein_mask_ref_controller import Flux2KleinMaskRefController
from .flux2_klein_color_anchor import Flux2KleinColorAnchor
from .identity_guidance import IdentityGuidance
from .identity_feature_transfer import IdentityFeatureTransfer, IdentityFeatureTransferAdvanced, IdentityFeatureTransferV3
from .multi_reference_latent import NODE_CLASS_MAPPINGS as MULTI_REF_NODES, NODE_DISPLAY_NAME_MAPPINGS as MULTI_REF_NAMES
from .Flux2klein_Ksampler_exp import NODE_CLASS_MAPPINGS as EXP_NODES, NODE_DISPLAY_NAME_MAPPINGS as EXP_NAMES

NODE_CLASS_MAPPINGS = {
    **REF_NODES,
    **TEXT_NODES,
    **MULTI_REF_NODES,
    **EXP_NODES,
    "Flux2KleinEnhancer": Flux2KleinEnhancer,
    "Flux2KleinDetailController": Flux2KleinDetailController,
    "Flux2KleinSectionedEncoder": Flux2KleinSectionedEncoder,
    "Flux2KleinMaskRefController": Flux2KleinMaskRefController,
    "Flux2KleinColorAnchor": Flux2KleinColorAnchor,
    "IdentityGuidance": IdentityGuidance,
    "IdentityFeatureTransfer": IdentityFeatureTransfer,
    "IdentityFeatureTransferAdvanced": IdentityFeatureTransferAdvanced,
    "IdentityFeatureTransferV3": IdentityFeatureTransferV3,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **REF_NAMES,
    **TEXT_NAMES,
    **MULTI_REF_NAMES,
    **EXP_NAMES,
    "Flux2KleinEnhancer": "FLUX.2 Klein Enhancer",
    "Flux2KleinDetailController": "FLUX.2 Klein Detail Controller",
    "Flux2KleinSectionedEncoder": "FLUX.2 Klein Sectioned Encoder",
    "Flux2KleinMaskRefController": "FLUX.2 Klein Mask Ref Controller",
    "Flux2KleinColorAnchor": "FLUX.2 Klein Color Anchor",
    "IdentityGuidance": "FLUX.2 Klein Identity Guidance",
    "IdentityFeatureTransfer": "FLUX.2 Klein Identity Feature Transfer",
    "IdentityFeatureTransferAdvanced": "FLUX.2 Klein Identity Feature Transfer Advanced",
    "IdentityFeatureTransferV3": "FLUX.2 Klein Identity Feature Transfer V3",
}

__version__ = "3.3.1"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
