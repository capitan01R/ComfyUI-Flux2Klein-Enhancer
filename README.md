# ComfyUI-Flux2Klein-Enhancer
[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-Support-yellow.svg)](https://buymeacoffee.com/capitan01r)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

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
| `balance` | 0.500 | 0.000 to 1.000 | Adjusts the relative strength of text and reference conditioning. Values below `0.5` reduce text strength while keeping reference strength constant. Values above `0.5` reduce reference strength while keeping text strength constant. |
| `debug` | False | True/False | Prints text and ref scale factors per block to console. |

### FLUX.2 Klein Ref Latent Weight

Minimal per-reference k/v scaler. Takes and returns `MODEL` only. Chain one node per reference for independent per-reference control.

#### Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `reference_index` | 0 | 0 to 7 | Which reference image to weight (0 = first). |
| `weight` | 1.0 | 0.0 to 5.0 | 1.0 = unchanged, 0.0 = invisible, >1.0 = stronger influence. |



---

## Preserve Original - Solving FLUX Klein's Preservation Problem

FLUX Klein has a consistency problem. Sometimes it nails the preservation of subjects and objects. Sometimes it completely ignores what you're trying to keep and generates something else entirely. There was no native way to control this.

This node exposes preservation control that FLUX Klein doesn't provide. You can now control exactly how much original structure is maintained versus how much the prompt can modify the generation.

### The Modes
  - `dampen`: Reduces modification strength before applying changes. This is the most reliable mode for precise preservation. (Recommended)
  - `linear`: Applies full modifications, then blends the result back with the original.
  - `hybrid`: Dampens parameters first, then blends the result.
  - `blend_after`: Same as `linear` (legacy naming).

### Usage

- **1.20-1.30 (dampen)**: Recommended starting point for consistent identity/object preservation.
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
