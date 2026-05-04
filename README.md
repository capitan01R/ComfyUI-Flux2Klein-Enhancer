# ComfyUI-Flux2Klein-Enhancer
[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-Support-yellow.svg)](https://buymeacoffee.com/capitan01r)
[![License: PolyForm NC 1.0.0](https://img.shields.io/badge/License-PolyForm%20NC%201.0.0-blue.svg)](LICENSE)

Conditioning enhancement and reference latent control for FLUX.2 Klein 9B in ComfyUI. Built from empirical analysis and forward-pass tracing of the model's dual-stream architecture.

## What This Does (( NEW DISCOVERY!! ))

FLUX.2 Klein uses a Qwen3 8B text encoder that outputs conditioning tensors of shape `[batch, 512, 12288]`. Through diagnostic analysis and model hook tracing, I verified:

- **Text Conditioning**: `[1, 512, 12288]` tensor with ~67 active tokens (auto-detected from attention mask)
- **Reference Latent**: Stored separately in metadata as `[1, 128, H, W]` - NOT merged into text conditioning
- **Dual-Stream Architecture**: Text and image streams are processed separately through double_blocks, then concatenated for single_blocks

### Verified Architecture (from forward-pass hooks)

```
┌─────────────────────────────────────────────────────────────────┐
│                    FLUX.2 Klein Forward Pass                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Reference Latent [1,128,55,74]  →  patchify  →  [1, 4070, 128] │
│  Noisy Latent [1,128,55,74]      →  patchify  →  [1, 4070, 128] │
│                                         │                       │
│                                    CONCATENATE                  │
│                                         ↓                       │
│                                  [1, 8140, 128]                 │
│                                         │                       │
│                                      img_in                     │
│                                         ↓                       │
│                                  [1, 8140, 4096]                │
│                                                                 │
│  Text Conditioning [1,512,12288] → txt_in → [1, 512, 4096]      │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  DOUBLE BLOCKS (×8): Separate streams                           │
│    img_stream: [1, 8140, 4096]                                  │
│    txt_stream: [1, 512, 4096]                                   │
├─────────────────────────────────────────────────────────────────┤
│  SINGLE BLOCKS (×24): Concatenated                              │
│    combined: [1, 8652, 4096]  (8140 + 512)                      │
└─────────────────────────────────────────────────────────────────┘
```

**Without reference latent**: img_in receives `[1, 4070, 128]`, single_blocks process `[1, 4582, 4096]`

This means:
- **Text enhancement** modifies the txt_stream input
- **Reference control** modifies half of the img_stream input (the reference portion)
- These are **independent controls** that can be combined

## Installation

1. Navigate to your ComfyUI custom nodes folder:
   ```
   cd ComfyUI/custom_nodes/
   ```

2. Clone this repository:
   ```
   git clone https://github.com/capitan01R/ComfyUI-Flux2Klein-Enhancer.git
   ```

3. Restart ComfyUI

## Nodes

### FLUX.2 Klein Enhancer

General-purpose text conditioning enhancement for both text-to-image and image editing workflows.

#### Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `magnitude` | 1.0 | 0.0 to 3.0 | Direct scaling of active region embeddings. Values above 1.0 increase prompt influence, below 1.0 decrease it. |
| `contrast` | 0.0 | -1.0 to 2.0 | Amplifies differences between tokens. Positive values sharpen concept separation, negative values blend them. |
| `normalize_strength` | 0.0 | 0.0 to 1.0 | Equalizes token magnitudes. Higher values balance emphasis across all tokens in the prompt. |
| `edit_text_weight` | 1.0 | 0.0 to 3.0 | Image edit mode only. Values below 1.0 preserve more of the original image, above 1.0 follows the prompt more strongly. |
| `active_end_override` | 0 | 0 to 512 | Manual override for active region end. 0 = auto-detect from attention mask. |
| `low_vram` | False | True/False | Use float16 computation on CUDA devices. |
| `device` | auto | auto/cpu/cuda:N | Compute device selection. |
| `debug` | False | True/False | Prints tensor statistics and modification details to console. |

### FLUX.2 Klein Detail Controller

Regional control over prompt conditioning. Divides active tokens into front/mid/end sections.

#### Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `front_mult` | 1.0 | 0.0 to 3.0 | Multiplier for first 25% of active tokens (typically subject/main concept). |
| `mid_mult` | 1.0 | 0.0 to 3.0 | Multiplier for middle 50% of active tokens (typically details/modifiers). |
| `end_mult` | 1.0 | 0.0 to 3.0 | Multiplier for last 25% of active tokens (typically style/quality terms). |
| `emphasis_start` | 0 | 0 to 200 | Start position of custom emphasis region. |
| `emphasis_end` | 0 | 0 to 200 | End position of custom emphasis region. 0 = disabled. |
| `emphasis_mult` | 1.0 | 0.0 to 3.0 | Multiplier for the custom emphasis region. |
| `low_vram` | False | True/False | Use float16 computation on CUDA devices. |
| `device` | auto | auto/cpu/cuda:N | Compute device selection. |
| `debug` | False | True/False | Prints debug information to console. |

---

### FLUX.2 Klein Ref Latent Controller

Controls how strongly a specific reference image influences the generation. Requires a `MODEL` input and returns an updated `MODEL`. Chain multiple nodes to control each reference independently.

#### Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `strength` | 1.0 | 0.0 to 1000.0 | Reference attention strength. 0 = reference ignored, 1 = normal, >1 = stronger structure. |
| `reference_index` | 0 | 0 to 7 | Which reference image to control (0 = first). |
| `spatial_fade` | none | none/center_out/edges_out/top_down/left_right | Per-token spatial gradient applied to the strength. |
| `spatial_fade_strength` | 0.5 | 0.0 to 1.0 | Intensity of the spatial fade. |
| `debug` | False | True/False | Prints block index, token range, and strength to console. |

### FLUX.2 Klein Text/Ref Balance

Single slider to balance text conditioning vs. all reference images. Requires a `MODEL` input and returns an updated `MODEL`.

#### Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `balance` | 0.005 | 0.000 to 1.000 | 0 = reference only, 0.5 = balanced, 1 = text only. |
| `debug` | False | True/False | Prints text and ref scale factors per block to console. |


---

## Preserve Original - Solving FLUX Klein's Preservation Problem

FLUX Klein has a consistency problem. Sometimes it nails the preservation of subjects and objects. Sometimes it completely ignores what you're trying to keep and generates something else entirely. There was no native way to control this.

This node exposes preservation control that FLUX Klein doesn't provide. You can now control exactly how much original structure is maintained versus how much the prompt can modify the generation.

### The Modes

#### dampen (Recommended)
Reduces modification strength before applying changes. This is the most reliable mode for precise preservation.

**For consistent identity/object preservation: 1.20 to 1.30**

#### linear
Applies full modifications, then blends the result back with the original.

#### hybrid
Dampens parameters first, then blends the result.

#### blend_after
Same as linear, just a different name.

### Usage

- **1.20-1.30 (dampen)**: Recommended starting point for solid preservation
- **1.40-1.50**: Tighter control when needed, very prompt-dependent
- **0.0-1.0**: Standard range from full enhancement to balanced preservation


---

## How It Works

### Text Conditioning Enhancement

#### Magnitude
Direct scaling of all embedding vectors in the active region:
```python
active = active * magnitude
```

#### Contrast (Safe Implementation)
Computes the mean embedding across the sequence, then amplifies deviations:
```python
seq_mean = active.mean(dim=1, keepdim=True)
deviation = active - seq_mean
if contrast >= 0:
    scale = 1.0 + contrast
else:
    scale = math.exp(contrast)  # Never inverts, asymptotes to 0
active = seq_mean + deviation * scale
```

Note: Negative contrast uses exponential scaling to prevent semantic inversion that occurs with linear scaling below -1.

#### Normalize Strength
Equalizes token magnitudes toward a uniform value:
```python
token_norms = active.norm(dim=-1, keepdim=True)
mean_norm = token_norms.mean()
normalized = active / token_norms * mean_norm
active = active * (1.0 - normalize_strength) + normalized * normalize_strength
```

### Reference Latent Control

Strength is applied to `k` and `v` inside every attention block via `attn1_patch`, after all normalisation has completed:

```python
# Token layout: [ txt_tokens | main_img_tokens | ref_0_tokens | ... | ref_N_tokens ]
k[:, :, seq_start:seq_end, :] *= strength
v[:, :, seq_start:seq_end, :] *= strength
```

`reference_image_num_tokens` from `extra_options` gives the exact token count per reference, allowing precise per-reference indexing without affecting other tokens.

### Active Region Detection
Auto-detected from attention mask:
```python
attn_mask = meta.get("attention_mask", None)
nonzero = attn_mask[0].nonzero()
active_end = int(nonzero[-1].item()) + 1
```

---

## Presets

### Text-to-Image

```
              BASE   GENTLE   MOD   STRONG   AGG     MAX    CRAZY
              ----    ----    ----    ----    ----    ----    ----
magnitude:    1.20    1.15    1.25    1.35    1.50    1.75    2.50
contrast:     0.00    0.10    0.20    0.30    0.40    0.60    1.20
normalize:    0.00    0.00    0.00    0.15    0.25    0.35    0.60
edit_weight:  1.00    1.00    1.00    1.00    1.00    1.00    1.00
```

### Image Edit

```
              PRESERVE   SUBTLE   BALANCED   FOLLOW   FORCE
              --------   ------   --------   ------   -----
magnitude:       0.85     1.00       1.10     1.20    1.35
contrast:        0.00     0.05       0.10     0.15    0.25
normalize:       0.00     0.00       0.10     0.10    0.15
edit_weight:     0.70     0.85       1.00     1.25    1.50
ref_strength:    1.50     1.20       1.00     0.70    0.30
```

---

## BETA: Mask-Guided Reference Latent Controller

> Experimental — results are promising but behavior may vary depending on prompt and image complexity.

### FLUX.2 Klein Mask Ref Controller

Spatially controls the reference latent using a mask. Masked area gets targeted by the prompt while staying true to its original structure. Unmasked area gets fully freed up for the prompt to take over.

Not inpainting — works entirely at the conditioning level through the reference latent stream.

#### Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `mask` | — | MASK | Defines targeted vs free regions. Connect from any ComfyUI mask node. |
| `strength` | 1.0 | 0.0 to 1.0 | How free the unmasked area is. 1.0 = reference fully removed there. 0.5 = half reference kept. Lower values also bleed influence into surrounding areas. |
| `invert_mask` | False | True/False | Flip targeted and free regions. |
| `feather` | 0 | 0 to 64 | Gaussian blur on mask edges in latent space. Reduces hard seams at boundaries. |
| `channel_mode` | all | all/low/high | Which latent channels the mask affects. `low` = structure/layout (ch 0-63), `high` = texture/detail (ch 64-127). |
| `debug` | False | True/False | Print spatial stats and attenuation to console. |

#### Notes
- Requires image edit mode — reference latents must be present in conditioning metadata
- `strength` also controls boundary bleed — 1.0 is a tight boundary, lower values spread influence into neighboring regions
- Useful for targeting a specific subject within a scene while leaving the rest open to the prompt

## Identity Preservation Nodes

Two nodes that approach identity preservation from outside the model's conditioning pipeline. They bypass text streams, attention masks, and token injection entirely.

### FLUX.2 Klein Identity Guidance

Operates in the sampling loop. After the model predicts the denoised image at each step, compares it to the reference latent and pulls it back. The model runs freely, then gets corrected.

Takes a VAE-encoded reference image directly. ReferenceLatent should still be connected on the conditioning path so the model has reference context.

**Wiring:**
```
[Checkpoint] → MODEL → [Identity Guidance] → MODEL → [KSampler]
                              ↑
                    [VAE Encode of reference]
```

#### Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `strength` | 0.3 | 0.0 to 1.0 | How hard to pull toward the reference each step. 0.3 = move 30% of the distance. |
| `start_percent` | 0.0 | 0.0 to 1.0 | When to start correcting. 0.0 = beginning of denoising. |
| `end_percent` | 0.8 | 0.0 to 1.0 | When to stop correcting. 0.8 = last 20% runs freely for texture refinement. |
| `mode` | adaptive | adaptive/direct/channel_match | How correction is applied (see below). |

#### Modes

- **adaptive**: Pulls only where the prediction already resembles the reference. Preserves prompt-driven changes like new backgrounds or poses.
- **direct**: Pulls everywhere equally. Strongest identity lock, least prompt freedom.
- **channel_match**: Forces the generation's color and feature statistics to match the reference without copying spatial content.

---

### FLUX.2 Klein Identity Feature Transfer

Operates inside the model's attention layers. After each attention block computes, finds where the generation's features are similar to the reference's features and pushes them closer.

**Requires** ReferenceLatent connected. The reference must already be in the image stream.

**Wiring:**
```
[Checkpoint] → MODEL → [Identity Feature Transfer] → MODEL → [KSampler]
                                                        ↑
                        [ReferenceLatent] → CONDITIONING → [KSampler]
```

#### Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `strength` | 0.15 | 0.0 to 1.0 | Per-block blend factor. Fires at every active block (cumulative). Start at 0.10 to 0.20. |
| `start_block` | 0 | 0 to 23 | First block index to apply. Index is shared across double and single blocks (resets when single blocks begin). |
| `end_block` | 23 | 0 to 23 | Last block index to apply. Covers 8 double blocks (0-7) then 24 single blocks (index resets 0-23). |
| `mode` | cosine_pull | cosine_pull/topk_replace/mean_transfer | How features are transferred (see below). |
| `top_k_percent` | 0.25 | 0.01 to 1.0 | topk_replace mode only. Fraction of tokens to affect. |

#### Modes

- **cosine_pull**: Each generation token finds its most similar reference token and gets pulled toward it. Strength scales with similarity.
- **topk_replace**: Only the top K% most similar tokens are affected. Everything else stays untouched.
- **mean_transfer**: Shifts the overall feature distribution toward the reference without spatial matching.

---

### FLUX.2 Klein Identity Feature Transfer Advanced

Same approach as the standard transfer node, with finer control: per-band strength on double vs single blocks, a similarity floor that gates how tight or loose the pull is, a block schedule, and an optional subject mask that restricts the pull to the masked area of the reference.

**Requires** ReferenceLatent connected. The reference must already be in the image stream.

**Wiring:**
```
[Checkpoint] → MODEL → [Identity Feature Transfer Advanced] → MODEL → [KSampler]
                                              ↑                          ↑
                                       [MASK (optional)]    [ReferenceLatent] → CONDITIONING
```

#### Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `reference_index` | 0 | 0 to 15 | Which reference image to draw features from when multiple are connected (0 = first). |
| `mode` | cosine_pull | cosine_pull/topk_replace/mean_transfer | Same modes as the standard node. cosine_pull is the only mode that uses the subject mask. |
| `top_k_percent` | 0.25 | 0.01 to 1.0 | `topk_replace` mode only. Fraction of generation tokens to affect. |
| `double_enable` | True | True/False | Apply transfer on double blocks (0-7). These shape pose, color, and identity early. |
| `double_strength` | 0.15 | 0.0 to 1.0 | Per-block blend factor for double blocks. Cumulative. Raise (0.4-0.6) for stronger guidance, especially with multi-subject references. |
| `double_start` / `double_end` | 0 / 7 | 0 to 7 | Range of double blocks to apply on. |
| `single_enable` | True | True/False | Apply transfer on single blocks (0-23). These refine style and texture. |
| `single_strength` | 0.15 | 0.0 to 1.0 | Per-block blend factor for single blocks. Cumulative. |
| `single_start` / `single_end` | 0 / 23 | 0 to 23 | Range of single blocks to apply on. |
| `block_schedule` | flat | flat / ramp_down / ramp_up / peak_mid | Strength curve across the active block range. |
| `sim_floor` | 0.20 | 0.0 to 0.95 | Cosine threshold gating which matches contribute. Low (~0.05) = wide pull, tight identity lock, suited to subtle edits. High = sparse pull, more freedom for broader edits. |
| `mask_threshold` | 0.5 | 0.0 to 1.0 | Used only when `subject_mask` is connected. Reference tokens whose pooled mask value falls below this are excluded from the pull. 0.5 keeps boundary tokens; raise toward 1.0 to shrink the effective mask inward. |
| `subject_mask` (optional) | — | MASK | When connected, the cosine pull samples only from masked-in reference tokens. The conditioning latent is not modified. Mask aspect must match the encoded reference aspect. |

#### Tuning notes

- **Subtle edit (outfit swap, same character):** lower `sim_floor` to ~0.05, keep strengths around 0.15-0.20. The pull is wide and locks identity tightly so only the prompted change lands.
- **Broader edit (new scene, same person):** raise `sim_floor` toward 0.4-0.6. The pull becomes sparse and gives the model freedom to drift.
- **Reference contains multiple subjects:** paint the desired one as `subject_mask` and raise both block strengths to 0.4-0.6. The default 0.15 is too soft to compete with the model's natural attention to the unmasked subject.
- **Subject mask source:** ComfyUI's built-in MaskEditor on the reference image works fine. Any MASK output is accepted.

---

### FLUX.2 Klein Identity Feature Transfer V3

![Identity Feature Transfer V3 process diagram](example_workflow/identity_feature_transfer_v3_diagram.svg)

Shipment-ready identity preservation node for FLUX.2 Klein. This is the easier version to use.

V3 uses a commit system instead of constantly averaging the generation toward the whole reference. Each generation token looks for the best matching reference token. If the same match stays stable, it locks to that match and then keeps a lighter anchor instead of continuing to pull hard every block.

This gives cleaner preservation and less feature mush.

**Requires** ReferenceLatent connected. The reference must already be in the image stream.

**Wiring:**
```
[Checkpoint] -> MODEL -> [Identity Feature Transfer V3] -> MODEL -> [KSampler]
                                             ↑                         ↑
                                      [MASK optional]    [ReferenceLatent] -> CONDITIONING
```

#### Presets

| Preset | Use |
|--------|-----|
| `MIDUM_LOCK` | Recommended starting point. Uses the tested middle preset. |
| `HARD_LOCK` | Strongest preset. Use when the reference keeps drifting. |
| `SOFT_LOCK` | Softer preset. Use when the lock is too aggressive. |
| `custom` | Uses your manual schedules and settings. |

Any preset except `custom` overrides the manual settings below. This is intentional so the node is easier to use.

#### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `preset` | MIDUM_LOCK | Main control. Pick how strongly the reference should hold. |
| `reference_index` | 0 | Which reference image to use when multiple references are connected. |
| `double_schedule` | 0-3:mid=0.25; 4:mid=0.35; 5:mid=0.65; 6-7:mid=0.45 | Custom preset only. Controls double blocks. |
| `single_schedule` | 0:mid=0.35; 1:mid=0.25; 2-10:mid=0.30; 11-19:mid=0.25; 20:mid=0.08; 21:mid=0.10; 22:mid=0.15; 23:mid=0.20 | Custom preset only. Controls single blocks. |
| `double_sim` | 0.020 | Custom preset only. Higher means fewer double-block matches are allowed. |
| `single_sim` | 0.020 | Custom preset only. Higher means fewer single-block matches are allowed. |
| `commit_margin` | 0.035 | Custom preset only. Higher means the match has to be more obvious before it locks. |
| `commit_confirm` | 2 | Custom preset only. How many times the same match must repeat before it locks. |
| `commit_anchor` | 0.35 | Custom preset only. How much pull remains after a token has locked. |
| `mask_threshold` | 0.25 | Used only when `subject_mask` is connected. Lower keeps more edge tokens. |
| `subject_mask` | optional | Use this when the reference has more than one subject. |

#### Schedule format

```
4-7:0.20
2-10:0.10; 14-20:0.07
```

Left side is the block range. Right side is the strength.

Double blocks are `0-7`.
Single blocks are `0-23`.

#### Starting points

Use `MIDUM_LOCK` first.

If identity is weak, switch to `HARD_LOCK`.

If the reference is copying too much or distorting the result, switch to `SOFT_LOCK`.

Use `custom` only after you know what you want to change.

---

### Combining Both Nodes

Both nodes can be stacked. Identity Guidance handles macro-level correction in latent space. Feature Transfer handles micro-level feature alignment inside attention. They operate at different stages and don't interfere.

```
[Checkpoint] → MODEL → [Identity Feature Transfer] → MODEL → [Identity Guidance] → MODEL → [KSampler]
```
---


## Technical Details

- **Model**: FLUX.2 Klein 9B
- **Text Encoder**: Qwen3 8B (4096 hidden dim, 36 layers)
- **Conditioning Shape**: [batch, 512, 12288]
- **Reference Latent Shape**: [batch, 128, H, W] (stored in metadata)
- **Joint Attention Dim**: 12288
- **Architecture**: 8 double_blocks (separate streams) + 24 single_blocks (concatenated)
- **img_in projection**: [128] → [4096]
- **txt_in projection**: [12288] → [4096]
- **Guidance Embeds**: False (step-distilled model, no CFG)

## Methodology

All findings were verified through:
1. **Conditioning Diagnostic**: Tensor structure analysis
2. **Forward Hook Tracing**: Attached hooks to img_in, txt_in, double_blocks, single_blocks
3. **Comparative Analysis**: With vs without reference latent runs
4. **Model Introspection**: forward() signature analysis

The reference latent concatenation was confirmed by observing:
- With reference: `img_in` receives `[1, 8140, 128]` (4070 × 2 patches)
- Without reference: `img_in` receives `[1, 4070, 128]`

## Acknowledgments

Built through empirical analysis and forward-pass hook tracing of FLUX.2 Klein's conditioning structure and dual-stream architecture.

## License

This project is licensed under the **[PolyForm Noncommercial License 1.0.0](LICENSE)**.

- ✅ **Free for noncommercial use** — personal projects, research, hobby work, education, charitable/public-interest organizations, and other noncommercial purposes.
- ✅ **You may modify, distribute, and build on it** for any noncommercial purpose.
- ❌ **Commercial use requires a separate license.** If you (or your company) want to use this in a commercial product, paid service, or any revenue-generating context, please contact me first to arrange commercial licensing.

For commercial licensing inquiries, open an issue on the [GitHub repository](https://github.com/capitan01R/ComfyUI-Flux2Klein-Enhancer/issues) or reach out via the contact info on my GitHub profile.
