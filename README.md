# ComfyUI-Flux2Klein-Enhancer

Conditioning enhancement node for FLUX.2 Klein 9B in ComfyUI. Built from empirical analysis of the model's conditioning structure.

## What This Does

FLUX.2 Klein uses a Qwen3 8B text encoder that outputs conditioning tensors of shape `[batch, 512, 12288]`. Through diagnostic analysis, I found:

- **Positions 0-77**: Active text embeddings (std ~40.7)
- **Positions 77-511**: Padding/inactive tokens (std ~2.3)
- **Image edit mode**: Adds `reference_latents` to metadata

This node modifies only the active text region to affect prompt adherence and edit behavior. The padding region is left untouched. Active region end is auto-detected from attention mask.

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
    New Nodes Update #2.1.0
<a href="examples/node_fixed.png">
  <img src="examples/node_fixed.png" alt="FLUX.2 Klein Enhancer" width="900">
</a>

  
<a href="examples/updated_sample.png">
  <img src="examples/updated_sample.png" alt="FLUX.2 Klein Enhancer" width="900">
</a>
   
   

   
### FLUX.2 Klein Enhancer

General-purpose conditioning enhancement for both text-to-image and image editing workflows.

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


# Example of purely seperating the: 

front(subject)/mid(details)/end(style) [results in these examples only meant to showcase the capability of pure seperation]

<a href="examples/front.png">
  <img src="examples/front.png" alt="FLUX.2 Detail Controller" width="500">
</a>
<a href="examples/mid.png">
  <img src="examples/mid.png" alt="FLUX.2 Detail Controller" width="500">
</a>
<a href="examples/end.png">
  <img src="examples/end.png" alt="FLUX.2 Detail Controller" width="500">
</a>

**Prompt:** turn only the ground into a mirror surface reflecting the sky, keep the full dog and its body unchanged and add the dog's reflection below

# Added new support node to the FLUX.2 Detail Controller

- Target: A user friendly node to help users comfortably section their prompts to add target mult to them using the FLUX.2 Detail Controller.

- The node is sectioned to three parts front/mid/end just like the Detail Controller node + a combined box "Basically for Vanilla use".

  <a href="examples/new_node.png">
  <img src="examples/new_node.png" alt="FLUX.2 Detail Controller" width="500">
</a>
<a href="examples/eg1.png">
  <img src="examples/eg1.png" alt="FLUX.2 Detail Controller" width="500">
</a>
<a href="examples/eg2.png">
  <img src="examples/eg2.png" alt="FLUX.2 Detail Controller" width="500">
</a>
</a>
<a href="examples/eg3.png">
  <img src="examples/eg3.png" alt="FLUX.2 Detail Controller" width="500">
</a>

## Preserve Original - Solving FLUX Klein's Preservation Problem

FLUX Klein has a consistency problem. Sometimes it nails the preservation of subjects and objects. Sometimes it completely ignores what you're trying to keep and generates something else entirely. There was no native way to control this.

This node exposes preservation control that FLUX Klein doesn't provide. You can now control exactly how much original structure is maintained versus how much the prompt can modify the generation.

### The Modes

#### dampen (Recommended)
Reduces modification strength before applying changes. This is the most reliable mode for precise preservation.

**For consistent identity/object preservation: 1.20 to 1.30**

This range provides solid preservation without killing your prompt's ability to make changes.

#### linear
Applies full modifications, then blends the result back with the original. Less predictable than dampen but useful when you want aggressive changes with some safety rails.

#### hybrid
Dampens parameters first, then blends the result. Combines both approaches. Can be overkill for most cases.

#### blend_after
Same as linear, just a different name.

---

### Usage

The optimal value changes depending on your prompt and what you're preserving.

- **1.20-1.30 (dampen)**: Recommended starting point for solid preservation
- **1.40-1.50**: Tighter control when needed, very prompt-dependent
- **0.0-1.0**: Standard range from full enhancement to balanced preservation
- **Negative values**: Experimental, not recommended for production

You might need 1.25 for one generation and 1.45 for another. That's normal. The node gives you the precision to dial in what each specific prompt needs.

---

### Why It Matters

FLUX Klein doesn't expose preservation controls. This node creates that capability from scratch. Instead of rolling the dice on whether your subject's identity or object consistency survives the generation, you can lock it down while still letting the prompt do its work.


# Addidtion:
<a href="examples/updated_01_26.png">
  <img src="examples/updated_01_26.png" alt="Preservation_01" width="1280">
</a>
<a href="examples/added_preservation.png">
  <img src="examples/added_preservation.png" alt="Preservation_02" width="1280">
</a>



# Examples:
<a href="examples/Figure_01.png">
  <img src="examples/Figure_01.png" alt="Preservation" width="1280">
</a>



## How It Works

### Magnitude

Direct scaling of all embedding vectors in the active region:

```
active = active * magnitude
```

A value of 1.25 results in 25% stronger conditioning signal to the diffusion model. This directly affects cross-attention key magnitudes.

### Contrast

Computes the mean embedding across the sequence, then amplifies deviations from that mean:

```
seq_mean = active.mean(dim=1, keepdim=True)
deviation = active - seq_mean
active = seq_mean + deviation * (1.0 + contrast)
```

This increases the distinctiveness of individual token embeddings, helping separate concepts in complex prompts.

### Normalize Strength

Equalizes token magnitudes toward a uniform value:

```
token_norms = active.norm(dim=-1, keepdim=True)
mean_norm = token_norms.mean()
normalized = active / token_norms * mean_norm
active = active * (1.0 - normalize_strength) + normalized * normalize_strength
```

This prevents any single token from dominating the conditioning signal.

### Image Edit Mode Detection

The node automatically detects image edit mode by checking for `reference_latents` in the conditioning metadata. When detected, `edit_text_weight` provides additional scaling.

### Active Region Detection

The active region end is auto-detected from the attention mask in metadata:

```
attn_mask = meta.get("attention_mask", None)
nonzero = attn_mask[0].nonzero()
active_end = int(nonzero[-1].item()) + 1
```

This ensures only meaningful tokens are modified, regardless of prompt length.

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
```

## Usage Examples

### Text-to-Image: Stronger Prompt Adherence

```
magnitude: 1.25
contrast: 0.20
normalize_strength: 0.00
edit_text_weight: 1.00
```

### Text-to-Image: Complex Prompt with Multiple Concepts

```
magnitude: 1.35
contrast: 0.30
normalize_strength: 0.15
edit_text_weight: 1.00
```

### Image Edit: Follow Prompt More

```
magnitude: 1.20
contrast: 0.15
normalize_strength: 0.10
edit_text_weight: 1.25
```

### Image Edit: Preserve Original More

```
magnitude: 0.85
contrast: 0.00
normalize_strength: 0.00
edit_text_weight: 0.70
```

### Image Edit: Force Prompt Adherence

```
magnitude: 1.35
contrast: 0.25
normalize_strength: 0.15
edit_text_weight: 1.50
```

## Debugging

Enable `debug: True` to see console output:

```
==================================================
Flux2KleinEnhancer Item 0
==================================================
Shape: torch.Size([1, 512, 12288])
Active region: 0 to 71
Edit mode: True
Active std: 42.4100
Padding std: 2.3094

Before modifications:
  Active region mean norm: 893.77

Contrast (+0.20): deviation scaled by 1.20

Magnitude (1.25): all active tokens scaled

Edit text weight (1.15): applied for image edit mode

Final state:
  Active region mean norm: 893.77 -> 1284.56
  Output change: mean=42.5341, max=1506.23
```

The `Output change` line confirms the conditioning tensor was modified. If it shows `0.000000`, no changes were applied.

## Technical Details

- **Model**: FLUX.2 Klein 9B
- **Text Encoder**: Qwen3 8B (4096 hidden dim, 36 layers)
- **Conditioning Shape**: [batch, 512, 12288]
- **Joint Attention Dim**: 12288
- **Active Region**: Dynamic, detected from attention mask (typically 0-77)
- **Guidance Embeds**: False (step-distilled model, no CFG)

## Visual Results: Vanilla vs. With Flux2Klein-Enhancer

Exact same workflow, seed and prompt - only difference is using the node or not.  
Click images to view full size.

### Source Photo
[![Source](examples/source_02.jpg)](examples/source_02.jpg)


### Comparison 1
**Prompt:** turn only the ground into a mirror surface reflecting the sky, keep the full dog and its body unchanged and add the dog's reflection below

Vanilla Flux.2 Klein          |  With Enhancer Node
:-----------------------------:|:-----------------------------:
[![Vanilla](examples/vanilla_01.png)](examples/vanilla_01.png) | [![With Node](examples/with_node_01.png)](examples/with_node_01.png)

### Comparison 2
**Prompt:** replace the grass with shallow ocean water and add realistic water reflections of the dog, keep the sunny lighting

Vanilla                       |  With Enhancer Node
:-----------------------------:|:-----------------------------:
[![Vanilla](examples/vanilla_02.png)](examples/vanilla_02.png) | [![With Node](examples/with_node_02.png)](examples/with_node_02.png)


## Acknowledgments

Built through empirical analysis of FLUX.2 Klein's conditioning structure using diagnostic tools to inspect actual tensor shapes and statistics during inference.
