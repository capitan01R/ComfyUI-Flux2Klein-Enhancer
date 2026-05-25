import torch


def _samples(latent):
    if latent is None:
        return None
    if isinstance(latent, dict):
        latent = latent.get("samples")
    if torch.is_tensor(latent) and latent.ndim == 4:
        return latent
    return None


class MultiReferenceLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "latent_1": ("LATENT",),
            },
            "optional": {
                "latent_2": ("LATENT",),
                "latent_3": ("LATENT",),
                "latent_4": ("LATENT",),
                "latent_5": ("LATENT",),
                "latent_6": ("LATENT",),
                "latent_7": ("LATENT",),
                "latent_8": ("LATENT",),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "apply"
    CATEGORY = "conditioning/flux2klein"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def apply(self, conditioning, latent_1, **optional):
        refs = []
        for _, latent in [("latent_1", latent_1)] + sorted(optional.items()):
            z = _samples(latent)
            if z is None:
                continue
            for b in range(z.shape[0]):
                refs.append(z[b:b + 1].detach())

        if not refs:
            return (conditioning,)

        out = []
        for cond, meta in conditioning:
            meta = meta.copy()
            meta["reference_latents"] = list(refs)
            meta["reference_latents_method"] = "index"
            out.append([cond, meta])
        return (out,)


NODE_CLASS_MAPPINGS = {
    "Flux2KleinMultiReferenceLatent": MultiReferenceLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux2KleinMultiReferenceLatent": "Multi ReferenceLatent",
}
