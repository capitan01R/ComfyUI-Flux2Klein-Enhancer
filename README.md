# ComfyUI-Flux2Klein-Enhancer

[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-Support-yellow.svg)](https://buymeacoffee.com/capitan01r)
[![License: PolyForm NC 1.0.0](https://img.shields.io/badge/License-PolyForm%20NC%201.0.0-blue.svg)](LICENSE)

Conditioning, reference-latent, identity-transfer, color-control, and sampling tools for FLUX.2 Klein in ComfyUI.

The primary target is FLUX.2 Klein 9B. The conditioning enhancer also recognizes the three-slice `7680`-wide conditioning used by the smaller Klein variant, but model-hook schedules are designed around the 9B block layout unless stated otherwise.

## Current Workflow

For multi-reference identity-preserving image editing:

1. Encode every reference image with the FLUX.2 VAE.
2. Send the encoded latents to **Multi ReferenceLatent**.
3. Send its conditioning output to the sampler.
4. Patch the model with **Identity Feature Transfer Final**.
5. Optionally connect one mask per reference to `subject_mask_1` through `subject_mask_8`.

```text
positive conditioning -> Multi ReferenceLatent -> sampler positive
reference images -> VAE Encode -> latent_1 ... latent_8

diffusion model -> Identity Feature Transfer Final -> sampler model
```

Reference order is shared between both nodes: `latent_1` corresponds to `subject_mask_1`, `latent_2` to `subject_mask_2`, and so on.

The output canvas is still controlled by the latent supplied to the sampler. Reference dimensions do not automatically determine the generated image dimensions.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/capitan01R/ComfyUI-Flux2Klein-Enhancer.git
```

Restart ComfyUI after installing or updating.

No additional Python packages are required beyond the dependencies already provided by ComfyUI.

## Included Nodes

| Node | Purpose |
|---|---|
| **Identity Feature Transfer Final** | Current multi-reference feature-transfer node with schedules, presets, per-reference masks, and optional sigma-aware strength scaling. |
| **Multi ReferenceLatent** | Places up to eight encoded reference latents into one conditioning object using Klein's indexed reference method. |
| **FLUX.2 Klein Color Anchor** | Corrects denoised latent channel means toward a selected reference over the sampling schedule. |
| **FLUX.2 Klein Enhancer** | Applies explicit scaling, whitening, norm equalization, and per-Qwen-layer scaling to text conditioning. |
| **FLUX.2 Klein Text Enhancer** | Simpler active-token magnitude, contrast, and norm control. |
| **FLUX.2 Klein Sectioned Encoder** | Encodes FRONT/MID/END prompt sections and records their real token ranges. |
| **FLUX.2 Klein Detail Controller** | Scales the token ranges produced by Sectioned Encoder. |
| **FLUX.2 Klein Ref Latent Controller** | Scales one reference's attention keys and values, optionally with a spatial fade. |
| **FLUX.2 Klein Ref Latent Weight** | Lightweight model-only per-reference key/value multiplier. |
| **FLUX.2 Klein Text/Ref Balance** | Attenuates text or reference keys/values around a neutral midpoint. |
| **FLUX.2 Klein Mask Ref Controller** | Directly attenuates black regions of one encoded reference latent. |
| **FLUX.2 Klein Identity Guidance** | Sampling-output correction toward an identity latent. |
| **Identity Feature Transfer / Advanced / V3** | Earlier identity-transfer implementations retained for workflow compatibility and experimentation. |
| **Flux2Klein KSampler Experimental** | Standalone experimental Euler sampler with a resolution-aware shifted schedule. |

## Identity Feature Transfer Final

This is the current feature-transfer implementation. It operates on Klein's attention output, separates generated and reference image tokens using the runtime `reference_image_num_tokens` metadata, and builds a masked reference bank from the selected references.

The transfer performs:

1. Per-image centering of generated and reference features.
2. Normalized similarity matching.
3. Similarity-floor filtering.
4. Temperature-controlled reference pooling.
5. Confidence-gated transfer at the scheduled double and single blocks.

### Main Controls

| Parameter | Default | Meaning |
|---|---:|---|
| `preset` | `HARD_LOCK` | `HARD_LOCK`, `MID_LOCK`, and `SOFT_LOCK` replace the manual similarity, temperature, mask threshold, and block schedules. Use `custom` to edit them directly. |
| `enabled` | `true` | Returns an unmodified model clone when disabled. |
| `reference_indices` | `all` | Zero-based references used by the transfer. Accepts `all`, comma-separated indices such as `0,2,3`, or ranges such as `0-3`. |
| `reference_index` | `0` | Fallback reference when `reference_indices` resolves to no valid entries. |
| `similarity_floor` | `0.040` | Minimum centered similarity allowed to contribute. Higher values reject more matches. |
| `softmax_temperature` | `0.0250` | Match sharpness. Lower values concentrate on fewer reference tokens; higher values blend more candidates. |
| `mask_threshold` | `1.00` | Minimum pooled mask value required for a reference token. White is included; black is excluded. |
| `double_blocks` | `0-7:mid_img=0.55` | Per-double-block transfer strengths. |
| `single_blocks` | tested sparse schedule | Per-single-block transfer strengths. Empty text disables single-block transfer. |
| `sigmas` | optional | Rescales each block strength by `delta_sigma_0 / delta_sigma_step` using the connected sampler schedule. |
| `debug` | `false` | Prints active settings and sigma scaling. |

### Schedule Syntax

```text
0-7:mid_img=0.55
0:mid_img=0.22; 1:mid_img=0.24; 3:mid_img=0.28
```

Double blocks use indices `0-7`. Single blocks use indices `0-23`. Unlisted blocks are inactive.

### Multiple References

All selected references are combined into the reference bank. The mask inputs follow reference order:

```text
latent_1 <-> subject_mask_1
latent_2 <-> subject_mask_2
...
latent_8 <-> subject_mask_8
```

An unwired mask leaves that reference unrestricted.

### Mask Behavior

`mask_behavior` has two modes:

- **`focus_only`**: original behavior. The mask limits which tokens enter the feature-transfer bank, while Klein's native attention still sees the complete reference image.
- **`zero_unmasked_tokens`**: the same transfer-bank filtering, plus an attention-source gate in every block. Unmasked tokens from each wired reference are blocked as attention sources. The implementation does not zero block residual outputs, which avoids discontinuities and static artifacts. References without a wired mask remain complete.

Example: leave `subject_mask_1` unwired so the first portrait supplies full context, then connect a t-shirt/outfit mask to `subject_mask_2`. In `zero_unmasked_tokens`, reference 2 can supply only its white t-shirt/outfit region while reference 1 remains fully available.

## Multi ReferenceLatent

Accepts one required latent and up to seven optional latents. Every batch item is split into an individual reference and stored in conditioning as:

```python
meta["reference_latents"] = refs
meta["reference_latents_method"] = "index"
```

This node replaces the conditioning object's existing reference list with the supplied list. Inputs must be encoded `LATENT` values, not raw images. It returns one `CONDITIONING` output and exposes no weighting or append mode.

## Color Anchor

**FLUX.2 Klein Color Anchor** reads one reference latent from conditioning and applies a sampler post-CFG correction to the denoised latent's per-channel spatial mean. Spatial deviations are left unchanged; this is color-statistic anchoring, not identity transfer.

| Parameter | Default | Meaning |
|---|---:|---|
| `strength` | `0.5` | Maximum mean correction. `0` disables the node; `1` applies the full scheduled correction. |
| `ramp_curve` | `1.5` | Uses `progress^(1/curve)`. `1` is linear, values above `1` engage faster, and values below `1` delay the correction. |
| `ref_index` | `0` | Reference latent used as the color source. |
| `channel_weights` | `uniform` | `by_variance` trusts low-spatial-variance reference channels more strongly. |

## Text Conditioning Tools

### FLUX.2 Klein Enhancer

Applies explicit operations to the active conditioning region:

- `active_scale`: global active-token multiplier.
- `per_token_whiten`: expands or compresses deviation from the sequence mean.
- `norm_equalize`: blends token norms toward the sequence mean norm.
- `early_layer_scale`, `mid_layer_scale`, `late_layer_scale`: independently scale the three stacked Qwen hidden-layer slices.
- `preserve_original`: blends the modified active region back toward its original value.
- `active_end_override`: manual active-token boundary; `0` uses the attention mask and otherwise falls back to the full sequence.

Neutral values make this node an exact pass-through.

### FLUX.2 Klein Text Enhancer

A simpler conditioning transform with `magnitude`, `contrast`, `normalize_strength`, and `skip_bos`. It modifies active text embeddings directly. It does not parse prompt meaning or assign semantic roles to words.

### Sectioned Encoder and Detail Controller

Use these together when different prompt sections need different weights.

```text
CLIP -> Sectioned Encoder -> Detail Controller -> sampler
```

Sectioned Encoder accepts separate FRONT/MID/END text boxes or a combined prompt:

```text
[FRONT] subject and primary action
[MID] clothing and scene details
[END] lighting and rendering style
```

It encodes one final prompt and stores tokenizer-derived section ranges in `meta["klein_sections"]`. Detail Controller scales those exact ranges with `front_mult`, `mid_mult`, and `end_mult`.

Without Sectioned Encoder metadata, Detail Controller falls back to arbitrary 25%/50%/25% sequence slices for backward compatibility. That fallback does not imply those positions have fixed semantic roles.

## Reference Controls

### Ref Latent Controller

Scales one reference's attention keys and values in every block. `spatial_fade` supports `center_out`, `edges_out`, `top_down`, and `left_right`. It returns both the patched model and unchanged conditioning.

### Ref Latent Weight

Model-only version of per-reference key/value scaling. It has no conditioning input and no spatial fade.

### Text/Ref Balance

`balance=0.5` is neutral: text and references both remain at scale `1`.

- From `0` to `0.5`, reference scale stays at `1` while text rises from `0` to `1`.
- From `0.5` to `1`, text stays at `1` while reference scale falls from `1` to `0`.

This is attenuation around a neutral midpoint, not an independent gain control for both streams.

### Mask Ref Controller

Directly modifies one encoded reference latent in conditioning. White regions remain unchanged; black regions are multiplied by `1 - strength`. `invert_mask` flips the interpretation and `feather` blurs the boundary in latent space.

This is different from Identity Feature Transfer Final masks: Mask Ref Controller mutates the latent before Klein consumes it, while Final masks filter or isolate reference tokens inside the model.

## Earlier Identity Nodes

The following nodes remain registered so existing workflows continue to load:

- **Identity Feature Transfer**: basic `cosine_pull`, `topk_replace`, or `mean_transfer` over a block range.
- **Identity Feature Transfer Advanced**: separate double/single ranges and strengths, block curves, similarity floor, and one optional mask.
- **Identity Feature Transfer V3**: commit-based matching with `MIDUM_LOCK`, `HARD_LOCK`, `SOFT_LOCK`, and custom schedules.
- **Identity Guidance**: sampler post-CFG latent correction with `adaptive`, `direct`, and `channel_match` modes.

These are alternatives, not required companions for Identity Feature Transfer Final.

## Experimental Sampler

**Flux2Klein KSampler Experimental** directly calls the diffusion model with a shifted Euler schedule. It supports:

- Resolution-dependent `base_shift` and `max_shift`.
- Full denoise or latent-to-latent denoise.
- Optional negative conditioning and CFG when `cfg_scale > 1`.
- Optional embedded guidance only when the loaded model exposes a guidance embedding layer.
- Reference latents found in positive conditioning metadata.

This sampler is experimental. It is not a drop-in replacement for every ComfyUI sampler workflow and does not expose every standard sampler feature.

## Architecture Notes

- Klein reference latents are stored separately from text conditioning as `[batch, 128, H, W]` tensors.
- FLUX.2 patchifies generated and reference latents independently, appends the reference token sequences to the generated image sequence, and exposes each reference's exact runtime token count through `reference_image_num_tokens`.
- Token counts depend on latent resolution and are not fixed globally.
- For the 9B architecture targeted by the identity schedules, there are 8 double blocks and 24 single blocks.
- Text and image are separate residual streams in double blocks, but their Q/K/V tensors participate in joint attention. Single blocks operate on the concatenated sequence.
- The 9B conditioning width is `12288`, formed from three `4096`-wide Qwen hidden-state slices. The model projects it to its internal hidden width; `12288` is not the joint-attention head dimension.

## Example Workflows

The `example_workflow` directory currently includes:

- `Iden_feat_final_fixed.json`
- `Iden_feat_final_fixed_sigma.json`
- `iden_transfer_v3.json`
- `Sample_color_anchor.json`
- `ref__latent.json`
- `Flux2Klein_Ksampler_exp.json`
- `adv_wf.json`
- `iden_wf (1).json`

## License

This project is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE).

- Free for noncommercial use, including personal projects, research, education, and nonprofit work.
- Modification and redistribution are allowed for noncommercial purposes under the license terms.
- Commercial use requires a separate license.

For commercial licensing, open an issue on the [GitHub repository](https://github.com/capitan01R/ComfyUI-Flux2Klein-Enhancer/issues) or contact the author through the GitHub profile.
