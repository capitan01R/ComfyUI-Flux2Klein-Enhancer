import re
from typing import Dict, List, Optional, Sequence, Tuple
import torch
import torch.nn.functional as F


class IdentityFeatureTransfer:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "Requires ReferenceLatent connected. The reference must be in the image stream.",
                }),
                "strength": ("FLOAT", {
                    "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Per-block blend factor. Fires at every active block so the effect is cumulative. Start at 0.10 to 0.20.",
                }),
                "start_block": ("INT", {
                    "default": 0, "min": 0, "max": 23,
                    "tooltip": "First block index to apply. 0 = earliest. Index is shared across double and single blocks (resets when single blocks begin).",
                }),
                "end_block": ("INT", {
                    "default": 23, "min": 0, "max": 23,
                    "tooltip": "Last block index to apply. Covers 8 double blocks (0-7) then 24 single blocks (index resets 0-23). Higher values extend coverage into later single blocks.",
                }),
                "mode": (["cosine_pull", "topk_replace", "mean_transfer"], {
                    "default": "cosine_pull",
                    "tooltip": "cosine_pull: pulls each gen token toward its best-matching ref token. topk_replace: only affects the top K%% most similar tokens. mean_transfer: shifts overall feature distribution toward the reference.",
                }),
                "top_k_percent": ("FLOAT", {
                    "default": 0.25, "min": 0.01, "max": 1.0, "step": 0.01,
                    "tooltip": "topk_replace mode only. Fraction of generation tokens to affect. 0.25 = top 25%% most similar.",
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "conditioning/flux2klein"

    def apply(self, model, strength=0.15, start_block=0, end_block=23,
              mode="cosine_pull", top_k_percent=0.25):
        m = model.clone()

        _strength = strength
        _start = start_block
        _end = end_block
        _mode = mode
        _topk_pct = top_k_percent

        def output_patch(attn, extra_options):
            ref_tokens_list = extra_options.get("reference_image_num_tokens", [])
            if not ref_tokens_list:
                return attn

            block_idx = extra_options.get("block_index", 0)
            if block_idx < _start or block_idx > _end:
                return attn

            img_slice = extra_options.get("img_slice", None)
            if img_slice is None:
                return attn

            txt_end = img_slice[0]
            total_seq = img_slice[1]

            total_ref = sum(ref_tokens_list)
            if total_ref <= 0:
                return attn

            gen_start = txt_end
            gen_end = total_seq - total_ref
            ref_start = total_seq - total_ref
            ref_end = total_seq

            if gen_end <= gen_start or ref_end <= ref_start:
                return attn

            gen_features = attn[:, gen_start:gen_end]
            ref_features = attn[:, ref_start:ref_end]

            if _mode == "cosine_pull":
                gen_norm = F.normalize(gen_features.float(), dim=-1)
                ref_norm = F.normalize(ref_features.float(), dim=-1)

                sim = torch.bmm(gen_norm, ref_norm.transpose(1, 2))
                max_sim, max_idx = sim.max(dim=-1)

                weight = max_sim.clamp(0.0, 1.0) * _strength
                weight = weight.unsqueeze(-1).to(attn.dtype)

                max_idx_expanded = max_idx.unsqueeze(-1).expand(-1, -1, ref_features.shape[-1])
                best_ref = torch.gather(ref_features, 1, max_idx_expanded)

                new_gen = gen_features + (best_ref - gen_features) * weight

                attn = attn.clone()
                attn[:, gen_start:gen_end] = new_gen

            elif _mode == "topk_replace":
                gen_norm = F.normalize(gen_features.float(), dim=-1)
                ref_norm = F.normalize(ref_features.float(), dim=-1)
                sim = torch.bmm(gen_norm, ref_norm.transpose(1, 2))
                max_sim, max_idx = sim.max(dim=-1)

                k = max(1, int(gen_features.shape[1] * _topk_pct))
                topk_vals, topk_indices = max_sim.topk(k, dim=-1)

                max_idx_expanded = max_idx.unsqueeze(-1).expand(-1, -1, ref_features.shape[-1])
                best_ref = torch.gather(ref_features, 1, max_idx_expanded)

                attn = attn.clone()
                for b in range(attn.shape[0]):
                    for i in range(k):
                        idx = topk_indices[b, i].item()
                        sim_val = topk_vals[b, i].item()
                        if sim_val > 0:
                            w = min(sim_val * _strength, 1.0)
                            pos = gen_start + idx
                            attn[b, pos] = (1.0 - w) * attn[b, pos] + w * best_ref[b, idx]

            elif _mode == "mean_transfer":
                gen_mean = gen_features.mean(dim=1, keepdim=True)
                ref_mean = ref_features.mean(dim=1, keepdim=True)

                delta = (ref_mean - gen_mean) * _strength
                new_gen = gen_features + delta

                attn = attn.clone()
                attn[:, gen_start:gen_end] = new_gen

            return attn

        m.set_model_attn1_output_patch(output_patch)
        return (m,)




class IdentityFeatureTransferAdvanced:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "Requires ReferenceLatent connected. The reference must be in the image stream.",
                }),
                "reference_index": ("INT", {
                    "default": 0, "min": 0, "max": 15, "step": 1,
                    "tooltip": "Which reference image to draw features from when multiple are connected (0 = first).",
                }),
                "mode": (["cosine_pull", "topk_replace", "mean_transfer"], {
                    "default": "cosine_pull",
                    "tooltip": "cosine_pull: each generation token is pulled toward similar reference tokens. topk_replace: only the top K%% most similar tokens are affected. mean_transfer: shifts the overall feature distribution toward the reference.",
                }),
                "top_k_percent": ("FLOAT", {
                    "default": 0.25, "min": 0.01, "max": 1.0, "step": 0.01,
                    "tooltip": "topk_replace mode only. Fraction of generation tokens to affect.",
                }),
                "double_enable": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply transfer on double blocks (0-7). These shape pose, color, and identity early in the network.",
                }),
                "double_strength": ("FLOAT", {
                    "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Per-block blend factor for double blocks. Cumulative across blocks. Raise for stronger identity guidance, especially when the reference contains multiple subjects.",
                }),
                "double_start": ("INT", {
                    "default": 0, "min": 0, "max": 7,
                    "tooltip": "First double block to apply on (0-7).",
                }),
                "double_end": ("INT", {
                    "default": 7, "min": 0, "max": 7,
                    "tooltip": "Last double block to apply on (0-7).",
                }),
                "single_enable": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply transfer on single blocks (0-23). These refine style and texture later in the network.",
                }),
                "single_strength": ("FLOAT", {
                    "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Per-block blend factor for single blocks. Cumulative across blocks.",
                }),
                "single_start": ("INT", {
                    "default": 0, "min": 0, "max": 23,
                    "tooltip": "First single block to apply on (0-23).",
                }),
                "single_end": ("INT", {
                    "default": 23, "min": 0, "max": 23,
                    "tooltip": "Last single block to apply on (0-23).",
                }),
                "block_schedule": (["flat", "ramp_down", "ramp_up", "peak_mid"], {
                    "default": "flat",
                    "tooltip": "Strength curve across the active block range. flat = constant. ramp_down = stronger on early blocks. ramp_up = stronger on later blocks. peak_mid = strongest in the middle.",
                }),
                "sim_floor": ("FLOAT", {
                    "default": 0.20, "min": 0.0, "max": 0.95, "step": 0.01,
                    "tooltip": "Cosine similarity threshold gating which reference-to-generation matches contribute. Low (~0.05) = wide pull, tight identity lock, suited to subtle edits like outfit swaps. High = sparse pull, more freedom for broader edits.",
                }),
                "mask_threshold": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Used only when subject_mask is connected. Reference tokens whose pooled mask value falls below this are excluded from the pull. 0.5 keeps boundary tokens; raise toward 1.0 to shrink the effective mask inward.",
                }),
            },
            "optional": {
                "subject_mask": ("MASK", {
                    "tooltip": "Optional subject mask for the reference image. When connected, the cosine pull samples only from masked-in reference tokens, leaving everything else out of the transfer. The conditioning latent is not modified, so the model still sees the full reference. Mask aspect must match the encoded reference aspect.",
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "conditioning/flux2klein"

    def apply(self, model, reference_index=0, mode="cosine_pull", top_k_percent=0.25,
              double_enable=True, double_strength=0.15, double_start=0, double_end=7,
              single_enable=True, single_strength=0.15, single_start=0, single_end=23,
              block_schedule="flat", sim_floor=0.20, mask_threshold=0.5,
              subject_mask=None):
        m = model.clone()

        _ref_idx = reference_index
        _mode = mode
        _topk_pct = top_k_percent
        _d_enable = double_enable
        _d_strength = double_strength
        _d_start = double_start
        _d_end = double_end
        _s_enable = single_enable
        _s_strength = single_strength
        _s_start = single_start
        _s_end = single_end
        _schedule = block_schedule
        _sim_floor = float(sim_floor)
        _mask_thresh = float(mask_threshold)

        if subject_mask is not None:
            mk = subject_mask
            if mk.dim() == 3:
                mk = mk[0]
            elif mk.dim() == 4:
                mk = mk[0, 0]
            _src_mask = mk.detach().float().cpu()
        else:
            _src_mask = None

        _idx_cache = {}

        TEMPERATURE = 0.07

        def _subject_indices(Nr, device):
            if _src_mask is None:
                return None
            if Nr in _idx_cache:
                idx = _idx_cache[Nr]
                return idx.to(device) if idx is not None else None

            mh, mw = _src_mask.shape[-2], _src_mask.shape[-1]
            target = mh / max(mw, 1)
            best = (1, Nr)
            best_err = float("inf")
            limit = int(Nr ** 0.5) + 2
            for h in range(1, limit):
                if Nr % h == 0:
                    w = Nr // h
                    for hh, ww in ((h, w), (w, h)):
                        err = abs(hh / max(ww, 1) - target)
                        if err < best_err:
                            best_err = err
                            best = (hh, ww)
            h_ref, w_ref = best

            mask_2d = _src_mask.unsqueeze(0).unsqueeze(0)
            pooled = F.adaptive_avg_pool2d(mask_2d, (h_ref, w_ref))
            flat = pooled.view(-1)
            keep = flat >= _mask_thresh
            if keep.sum().item() == 0:
                _idx_cache[Nr] = None
                return None
            idx = torch.nonzero(keep, as_tuple=False).squeeze(-1)
            _idx_cache[Nr] = idx
            return idx.to(device)

        def _schedule_multiplier(block_idx, start, end):
            if end <= start:
                return 1.0
            t = (block_idx - start) / max(end - start, 1)
            if _schedule == "flat":
                return 1.0
            elif _schedule == "ramp_down":
                return 1.0 - t
            elif _schedule == "ramp_up":
                return t
            elif _schedule == "peak_mid":
                return 1.0 - abs(2.0 * t - 1.0)
            return 1.0

        def _apply_steering(gen_features, ref_features, strength, attn, gen_start, gen_end):
            if _mode == "cosine_pull":
                gen_f = gen_features.float()
                ref_f = ref_features.float()

                full_Nr = ref_f.shape[1]
                subj_idx = _subject_indices(full_Nr, ref_f.device)
                if subj_idx is not None:
                    ref_f = ref_f.index_select(1, subj_idx)

                gen_c = gen_f - gen_f.mean(dim=1, keepdim=True)
                ref_c = ref_f - ref_f.mean(dim=1, keepdim=True)

                gen_norm = F.normalize(gen_c, dim=-1)
                ref_norm = F.normalize(ref_c, dim=-1)
                sim = torch.bmm(gen_norm, ref_norm.transpose(1, 2))

                B, Ng, Nr = sim.shape
                neg_inf = torch.finfo(sim.dtype).min

                if subj_idx is None and Ng == Nr:
                    diag = torch.arange(Ng, device=sim.device)
                    sim[:, diag, diag] = neg_inf

                sim = torch.where(sim >= _sim_floor, sim,
                                  torch.full_like(sim, neg_inf))

                attn_w = torch.softmax(sim / TEMPERATURE, dim=-1)
                attn_w = torch.nan_to_num(attn_w, nan=0.0)

                pooled_ref = torch.bmm(attn_w, ref_f)

                max_sim = sim.max(dim=-1).values
                max_sim = torch.where(torch.isinf(max_sim),
                                      torch.zeros_like(max_sim), max_sim)
                denom = max(1.0 - _sim_floor, 1e-6)
                conf = ((max_sim - _sim_floor) / denom).clamp(0.0, 1.0)
                weight = (conf * strength).unsqueeze(-1).to(attn.dtype)

                new_gen = gen_features + (pooled_ref.to(attn.dtype) - gen_features) * weight
                attn = attn.clone()
                attn[:, gen_start:gen_end] = new_gen

            elif _mode == "topk_replace":
                gen_norm = F.normalize(gen_features.float(), dim=-1)
                ref_norm = F.normalize(ref_features.float(), dim=-1)
                sim = torch.bmm(gen_norm, ref_norm.transpose(1, 2))
                max_sim, max_idx = sim.max(dim=-1)

                k = max(1, int(gen_features.shape[1] * _topk_pct))
                topk_vals, topk_indices = max_sim.topk(k, dim=-1)

                max_idx_expanded = max_idx.unsqueeze(-1).expand(-1, -1, ref_features.shape[-1])
                best_ref = torch.gather(ref_features, 1, max_idx_expanded)

                attn = attn.clone()
                for b in range(attn.shape[0]):
                    for i in range(k):
                        idx = topk_indices[b, i].item()
                        sim_val = topk_vals[b, i].item()
                        if sim_val > 0:
                            w = min(sim_val * strength, 1.0)
                            pos = gen_start + idx
                            attn[b, pos] = (1.0 - w) * attn[b, pos] + w * best_ref[b, idx]

            elif _mode == "mean_transfer":
                gen_mean = gen_features.mean(dim=1, keepdim=True)
                ref_mean = ref_features.mean(dim=1, keepdim=True)
                delta = (ref_mean - gen_mean) * strength
                new_gen = gen_features + delta
                attn = attn.clone()
                attn[:, gen_start:gen_end] = new_gen

            return attn

        def output_patch(attn, extra_options):
            ref_tokens_list = extra_options.get("reference_image_num_tokens", [])
            if not ref_tokens_list:
                return attn

            block_type = extra_options.get("block_type", "double")
            block_idx = extra_options.get("block_index", 0)

            if block_type == "double":
                if not _d_enable:
                    return attn
                if block_idx < _d_start or block_idx > _d_end:
                    return attn
                base_strength = _d_strength
                sched_mult = _schedule_multiplier(block_idx, _d_start, _d_end)
            elif block_type == "single":
                if not _s_enable:
                    return attn
                if block_idx < _s_start or block_idx > _s_end:
                    return attn
                base_strength = _s_strength
                sched_mult = _schedule_multiplier(block_idx, _s_start, _s_end)
            else:
                return attn

            strength = base_strength * sched_mult
            if strength <= 0.0:
                return attn

            img_slice = extra_options.get("img_slice", None)
            if img_slice is None:
                return attn

            txt_end = img_slice[0]
            total_seq = img_slice[1]
            total_ref = sum(ref_tokens_list)
            if total_ref <= 0:
                return attn

            gen_start = txt_end
            gen_end = total_seq - total_ref

            if gen_end <= gen_start:
                return attn

            clamped_idx = min(_ref_idx, len(ref_tokens_list) - 1)
            ref_offset = sum(ref_tokens_list[:clamped_idx])
            ref_token_count = ref_tokens_list[clamped_idx]
            ref_start = (total_seq - total_ref) + ref_offset
            ref_end = ref_start + ref_token_count

            if ref_end > total_seq or ref_token_count <= 0:
                return attn

            gen_features = attn[:, gen_start:gen_end]
            ref_features = attn[:, ref_start:ref_end]

            attn = _apply_steering(gen_features, ref_features, strength,
                                   attn, gen_start, gen_end)
            return attn

        m.set_model_attn1_output_patch(output_patch)
        return (m,)


class IdentityFeatureTransferV3:

    PRESETS = {
        "HARD_LOCK": {
            "double_schedule": "0-3:mid=0.25; 4:mid=0.35; 5:mid=0.65; 6-7:mid=0.45",
            "single_schedule": "0:mid=0.35; 1:mid=0.25; 2-10:mid=0.30; 11-19:mid=0.25; 20:mid=0.08; 21:mid=0.10; 22:mid=0.15; 23:mid=0.20",
            "double_sim": 0.001,
            "single_sim": 0.001,
            "commit_margin": 0.001,
            "commit_confirm": 1,
            "commit_anchor": 1.0,
        },
        "MIDUM_LOCK": {
            "double_schedule": "0-3:mid=0.25; 4:mid=0.35; 5:mid=0.65; 6-7:mid=0.45",
            "single_schedule": "0:mid=0.35; 1:mid=0.25; 2-10:mid=0.30; 11-19:mid=0.25; 20:mid=0.08; 21:mid=0.10; 22:mid=0.15; 23:mid=0.20",
            "double_sim": 0.020,
            "single_sim": 0.020,
            "commit_margin": 0.035,
            "commit_confirm": 2,
            "commit_anchor": 0.50,
        },
        "SOFT_LOCK": {
            "double_schedule": "0-3:mid=0.25; 4:mid=0.35; 5:mid=0.65; 6-7:mid=0.45",
            "single_schedule": "0:mid=0.35; 1:mid=0.25; 2-10:mid=0.30; 11-19:mid=0.25; 20:mid=0.08; 21:mid=0.10; 22:mid=0.15; 23:mid=0.20",
            "double_sim": 0.020,
            "single_sim": 0.020,
            "commit_margin": 0.050,
            "commit_confirm": 2,
            "commit_anchor": 0.55,
        },
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "Connect this after your checkpoint model. ReferenceLatent must also be connected in the conditioning path.",
                }),
                "preset": (["MIDUM_LOCK", "HARD_LOCK", "SOFT_LOCK", "custom"], {
                    "default": "MIDUM_LOCK",
                    "tooltip": "Pick how strongly the reference should hold. Any preset except custom ignores the manual settings below.",
                }),
                "reference_index": ("INT", {
                    "default": 0, "min": 0, "max": 15, "step": 1,
                    "tooltip": "Which reference image to use. 0 means the first reference.",
                }),
                "double_schedule": ("STRING", {
                    "default": "0-3:mid=0.25; 4:mid=0.35; 5:mid=0.65; 6-7:mid=0.45",
                    "multiline": False,
                    "tooltip": "Custom preset only. Double block schedule.",
                }),
                "single_schedule": ("STRING", {
                    "default": "0:mid=0.35; 1:mid=0.25; 2-10:mid=0.30; 11-19:mid=0.25; 20:mid=0.08; 21:mid=0.10; 22:mid=0.15; 23:mid=0.20",
                    "multiline": False,
                    "tooltip": "Custom preset only. Single block schedule.",
                }),
                "double_sim": ("FLOAT", {
                    "default": 0.020, "min": 0.0, "max": 0.95, "step": 0.001,
                    "tooltip": "Custom preset only. Higher means fewer double-block matches are allowed.",
                }),
                "single_sim": ("FLOAT", {
                    "default": 0.020, "min": 0.0, "max": 0.95, "step": 0.001,
                    "tooltip": "Custom preset only. Higher means fewer single-block matches are allowed.",
                }),
                "commit_margin": ("FLOAT", {
                    "default": 0.035, "min": 0.0, "max": 0.5, "step": 0.005,
                    "tooltip": "Custom preset only. Higher means the match has to be more obvious before it locks.",
                }),
                "commit_confirm": ("INT", {
                    "default": 2, "min": 1, "max": 16, "step": 1,
                    "tooltip": "Custom preset only. How many times the same match must repeat before it locks.",
                }),
                "commit_anchor": ("FLOAT", {
                    "default": 0.50, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Custom preset only. How much pull remains after a token has locked.",
                }),
                "mask_threshold": ("FLOAT", {
                    "default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Used only when a mask is connected. Lower keeps more edge tokens. Higher keeps only the strongest mask area.",
                }),
                "debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print the active preset and schedules to the console.",
                }),
            },
            "optional": {
                "subject_mask": ("MASK", {
                    "tooltip": "Optional mask for the reference image. Use it when the reference has more than one subject.",
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "conditioning/flux2klein"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return False

    @staticmethod
    def _parse_schedule(text, max_block):
        out = {}
        if not text:
            return out
        for part in str(text).split(";"):
            part = part.strip()
            if not part or ":" not in part:
                continue
            block_part, value_part = part.split(":", 1)
            value_part = value_part.strip()
            if "=" in value_part:
                value_part = value_part.split("=", 1)[1].strip()
            try:
                value = float(value_part)
            except ValueError:
                continue
            try:
                if "-" in block_part:
                    start, end = block_part.split("-", 1)
                    start, end = int(start.strip()), int(end.strip())
                else:
                    start = end = int(block_part.strip())
            except ValueError:
                continue
            if start > end:
                start, end = end, start
            start = max(0, start)
            end = min(max_block, end)
            for idx in range(start, end + 1):
                out[idx] = value
        return out

    def apply(self, model, preset="MIDUM_LOCK", reference_index=0,
              double_schedule="0-3:mid=0.25; 4:mid=0.35; 5:mid=0.65; 6-7:mid=0.45",
              single_schedule="0:mid=0.35; 1:mid=0.25; 2-10:mid=0.30; 11-19:mid=0.25; 20:mid=0.08; 21:mid=0.10; 22:mid=0.15; 23:mid=0.20",
              double_sim=0.020, single_sim=0.020,
              commit_margin=0.035, commit_confirm=2, commit_anchor=0.50,
              mask_threshold=0.25, debug=False, subject_mask=None):
        m = model.clone()

        if preset in self.PRESETS:
            cfg = self.PRESETS[preset].copy()
        else:
            cfg = {
                "double_schedule": double_schedule,
                "single_schedule": single_schedule,
                "double_sim": float(double_sim),
                "single_sim": float(single_sim),
                "commit_margin": float(commit_margin),
                "commit_confirm": int(commit_confirm),
                "commit_anchor": float(commit_anchor),
            }

        ref_idx = int(reference_index)
        double_map = self._parse_schedule(cfg["double_schedule"], 7)
        single_map = self._parse_schedule(cfg["single_schedule"], 23)
        mask_threshold = float(mask_threshold)

        if subject_mask is not None:
            mk = subject_mask
            if mk.dim() == 4:
                mk = mk[0, 0]
            elif mk.dim() == 3:
                mk = mk[0]
            src_mask = mk.detach().float().cpu()
        else:
            src_mask = None

        idx_cache = {}
        commit_assign = {}
        commit_hits = {}

        def subject_indices(count, device):
            if src_mask is None:
                return None
            if count in idx_cache:
                cached = idx_cache[count]
                return cached.to(device) if cached is not None else None

            mh, mw = src_mask.shape[-2:]
            target = mh / max(mw, 1)
            best = (1, count)
            best_err = float("inf")
            limit = int(count ** 0.5) + 3
            for h in range(1, limit):
                if count % h == 0:
                    w = count // h
                    for hh, ww in ((h, w), (w, h)):
                        err = abs(hh / max(ww, 1) - target)
                        if err < best_err:
                            best_err = err
                            best = (hh, ww)

            pooled = F.adaptive_avg_pool2d(src_mask[None, None], best).view(-1)
            keep = pooled >= mask_threshold
            if keep.sum().item() == 0:
                idx_cache[count] = None
                return None
            idx = torch.nonzero(keep, as_tuple=False).squeeze(-1)
            idx_cache[count] = idx
            return idx.to(device)

        def ref_slice(ref_tokens, base_offset):
            if not ref_tokens:
                return None
            idx = min(ref_idx, len(ref_tokens) - 1)
            offset = sum(ref_tokens[:idx])
            count = ref_tokens[idx]
            if count <= 0:
                return None
            return base_offset + offset, base_offset + offset + count

        def commit_delta(gen_features, ref_features, strength, sim_floor, cache_key):
            if strength <= 0.0:
                return None

            gen_f = gen_features.float()
            ref_f = ref_features.float()
            gen_count = gen_f.shape[1]
            ref_count = ref_f.shape[1]
            if ref_count <= 0:
                return None

            gen_norm = F.normalize(gen_f - gen_f.mean(dim=1, keepdim=True), dim=-1)
            ref_norm = F.normalize(ref_f - ref_f.mean(dim=1, keepdim=True), dim=-1)
            sim = torch.bmm(gen_norm, ref_norm.transpose(1, 2))

            subj_idx = subject_indices(ref_count, gen_f.device)
            if subj_idx is not None:
                mask = torch.zeros((1, 1, ref_count), device=sim.device, dtype=sim.dtype)
                mask[0, 0, subj_idx] = 1.0
                sim = torch.where(mask > 0.5, sim, torch.full_like(sim, torch.finfo(sim.dtype).min))
            elif gen_count == ref_count:
                diag = torch.arange(gen_count, device=sim.device)
                sim[:, diag, diag] = torch.finfo(sim.dtype).min

            k = 2 if ref_count > 1 else 1
            top_vals, top_idx = torch.topk(sim, k=k, dim=-1)
            best_sim = top_vals[..., 0]
            best_idx = top_idx[..., 0]
            second_sim = top_vals[..., 1] if k == 2 else torch.zeros_like(best_sim)
            margin = best_sim - second_sim
            valid = torch.isfinite(best_sim) & (best_sim >= sim_floor)

            prev = commit_assign.get(cache_key)
            prev_hits = commit_hits.get(cache_key)
            if prev is None or prev.shape != best_idx.shape:
                hits = torch.ones_like(best_idx, dtype=torch.int16)
            else:
                if prev_hits is None or prev_hits.shape != best_idx.shape:
                    prev_hits = torch.zeros_like(best_idx, dtype=torch.int16)
                hits = torch.where(
                    (prev == best_idx) & valid,
                    (prev_hits + 1).clamp(max=32767),
                    torch.ones_like(prev_hits),
                )
            hits = torch.where(valid, hits, torch.zeros_like(hits))
            commit_assign[cache_key] = best_idx.detach()
            commit_hits[cache_key] = hits.detach()

            committed = valid & (hits >= cfg["commit_confirm"]) & (margin >= cfg["commit_margin"])
            gather_idx = best_idx.clamp(min=0).unsqueeze(-1).expand(-1, -1, ref_f.shape[-1])
            best_ref = torch.gather(ref_f, 1, gather_idx)

            denom = max(1.0 - sim_floor, 1e-6)
            confidence = ((best_sim - sim_floor) / denom).clamp(0.0, 1.0)
            if cfg["commit_margin"] > 0.0:
                margin_weight = (margin / cfg["commit_margin"]).clamp(0.0, 1.0)
            else:
                margin_weight = torch.ones_like(confidence)
            anchor = torch.where(
                committed,
                torch.full_like(confidence, cfg["commit_anchor"]),
                torch.ones_like(confidence),
            )
            weight = (confidence * margin_weight * anchor * strength).unsqueeze(-1)
            weight = torch.where(valid.unsqueeze(-1), weight, torch.zeros_like(weight))
            return (best_ref.to(gen_features.dtype) - gen_features) * weight.to(gen_features.dtype)

        def output_patch(attn, extra_options):
            ref_tokens = extra_options.get("reference_image_num_tokens", []) or []
            img_slice = extra_options.get("img_slice")
            if not ref_tokens or img_slice is None:
                return attn

            block_type = extra_options.get("block_type", "double")
            block_idx = int(extra_options.get("block_index", 0))
            if block_type == "double":
                strength = double_map.get(block_idx, 0.0)
                sim_floor = float(cfg["double_sim"])
            elif block_type == "single":
                strength = single_map.get(block_idx, 0.0)
                sim_floor = float(cfg["single_sim"])
            else:
                return attn
            if strength <= 0.0:
                return attn

            txt_end, total_seq = int(img_slice[0]), int(img_slice[1])
            total_ref = sum(ref_tokens)
            gen_start = txt_end
            gen_end = total_seq - total_ref
            rs = ref_slice(ref_tokens, total_seq - total_ref)
            if gen_end <= gen_start or rs is None or rs[1] > total_seq:
                return attn

            gen_features = attn[:, gen_start:gen_end]
            ref_features = attn[:, rs[0]:rs[1]]
            delta = commit_delta(gen_features, ref_features, strength, sim_floor, (block_type, block_idx, "mid_img"))
            if delta is None:
                return attn

            out = attn.clone()
            out[:, gen_start:gen_end] = gen_features + delta
            return out

        if debug:
            print(
                "[IdentityFeatureTransferV3] "
                f"preset={preset} "
                f"double_sim={cfg['double_sim']} single_sim={cfg['single_sim']} "
                f"commit_margin={cfg['commit_margin']} "
                f"commit_confirm={cfg['commit_confirm']} "
                f"commit_anchor={cfg['commit_anchor']} "
                f"doubles={double_map} singles={single_map}"
            )

        m.set_model_attn1_output_patch(output_patch)
        return (m,)


HARD_DOUBLE = "0-7:mid_img=0.55"
HARD_SINGLE = (
    "0:mid_img=0.22; "
    "1:mid_img=0.24; "
    "3:mid_img=0.28; "
    "4:mid_img=0.22; "
    "6:mid_img=0.26; "
    "7:mid_img=0.27; "
    "8:mid_img=0.25; "
    "10:mid_img=0.27; "
    "13:mid_img=0.27"
)


class IdentityFeatureTransferFinal:
    PRESETS = {
        "HARD_LOCK": {
            "double_blocks": HARD_DOUBLE,
            "single_blocks": HARD_SINGLE,
            "similarity_floor": 0.040,
            "softmax_temperature": 0.0250,
            "mask_threshold": 1.0,
        },
        "MID_LOCK": {
            "double_blocks": HARD_DOUBLE,
            "single_blocks": HARD_SINGLE,
            "similarity_floor": 0.200,
            "softmax_temperature": 0.0700,
            "mask_threshold": 1.0,
        },
        "SOFT_LOCK": {
            "double_blocks": HARD_DOUBLE,
            "single_blocks": HARD_SINGLE,
            "similarity_floor": 0.500,
            "softmax_temperature": 0.0700,
            "mask_threshold": 1.0,
        },
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "preset": (["HARD_LOCK", "MID_LOCK", "SOFT_LOCK", "custom"], {"default": "HARD_LOCK"}),
                "enabled": ("BOOLEAN", {"default": True}),
                "reference_index": ("INT", {"default": 0, "min": 0, "max": 15, "step": 1}),
                "reference_indices": ("STRING", {"default": "all", "multiline": False}),
                "similarity_floor": ("FLOAT", {"default": 0.040, "min": 0.0, "max": 0.95, "step": 0.001}),
                "softmax_temperature": ("FLOAT", {"default": 0.0250, "min": 0.0001, "max": 0.25, "step": 0.0001}),
                "mask_threshold": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "double_blocks": ("STRING", {"default": HARD_DOUBLE, "multiline": False}),
                "single_blocks": ("STRING", {"default": HARD_SINGLE, "multiline": False}),
                "debug": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "sigmas": ("SIGMAS", {
                    "forceInput": True,
                    "tooltip": (
                        "Optional sampler sigma schedule. When connected, the "
                        "existing block strengths decay per sampling step by "
                        "delta_sigma_0 / delta_sigma_step."
                    ),
                }),
                "subject_mask_1": ("MASK",),
                "subject_mask_2": ("MASK",),
                "subject_mask_3": ("MASK",),
                "subject_mask_4": ("MASK",),
                "subject_mask_5": ("MASK",),
                "subject_mask_6": ("MASK",),
                "subject_mask_7": ("MASK",),
                "subject_mask_8": ("MASK",),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "conditioning/flux2klein"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return False

    @staticmethod
    def _parse_ref_indices(text: str, count: int) -> List[int]:
        if count <= 0:
            return []
        value = str(text or "all").strip().lower()
        if value in ("", "all", "*"):
            return list(range(count))
        out = set()
        for part in re.split(r"[;, ]+", value):
            if not part:
                continue
            try:
                if "-" in part:
                    a, b = part.split("-", 1)
                    lo, hi = int(a), int(b)
                    if lo > hi:
                        lo, hi = hi, lo
                    for i in range(lo, hi + 1):
                        if 0 <= i < count:
                            out.add(i)
                else:
                    i = int(part)
                    if 0 <= i < count:
                        out.add(i)
            except ValueError:
                continue
        return sorted(out)

    @staticmethod
    def _parse_schedule(text: str, max_block: int) -> Dict[int, float]:
        out: Dict[int, float] = {}
        for row in str(text or "").split(";"):
            row = row.strip()
            if not row or ":" not in row:
                continue
            block_part, value_part = row.split(":", 1)
            value_part = value_part.strip()
            if "=" in value_part:
                key, value_part = value_part.split("=", 1)
                if key.strip().lower() not in ("mid", "mid_img"):
                    continue
            try:
                strength = float(value_part.strip())
            except ValueError:
                continue
            try:
                if "-" in block_part:
                    lo_s, hi_s = block_part.split("-", 1)
                    lo, hi = int(lo_s.strip()), int(hi_s.strip())
                else:
                    lo = hi = int(block_part.strip())
            except ValueError:
                continue
            if lo > hi:
                lo, hi = hi, lo
            lo = max(0, lo)
            hi = min(max_block, hi)
            for idx in range(lo, hi + 1):
                out[idx] = strength
        return out

    @staticmethod
    def _prep_mask(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if mask is None or not torch.is_tensor(mask):
            return None
        x = mask.detach().float().cpu()
        if x.dim() == 4:
            if x.shape[-1] in (1, 3, 4):
                x = x[0].mean(dim=-1)
            else:
                x = x[0, 0]
        elif x.dim() == 3:
            if x.shape[-1] in (1, 3, 4) and x.shape[0] != 1:
                x = x.mean(dim=-1)
            else:
                x = x[0]
        elif x.dim() != 2:
            return None
        return x.contiguous()

    @staticmethod
    def _grid_for_tokens(count: int, mask: torch.Tensor) -> Tuple[int, int]:
        count = max(1, int(count))
        mh, mw = mask.shape[-2:]
        target = mh / max(mw, 1)
        best = (1, count)
        best_err = float("inf")
        for h in range(1, int(count ** 0.5) + 3):
            if count % h != 0:
                continue
            w = count // h
            for hh, ww in ((h, w), (w, h)):
                err = abs((hh / max(ww, 1)) - target)
                if err < best_err:
                    best = (hh, ww)
                    best_err = err
        return best

    @staticmethod
    def _scalar_float(value) -> Optional[float]:
        if value is None:
            return None
        try:
            if torch.is_tensor(value):
                if value.numel() == 0:
                    return None
                return float(value.detach().flatten()[0].double().cpu().item())
            return float(value)
        except (TypeError, ValueError, RuntimeError):
            return None

    @staticmethod
    def _sigma_equal_energy_schedule(sigmas):
        if sigmas is None:
            return None
        if not torch.is_tensor(sigmas):
            raise ValueError("Identity Feature Transfer Final: sigmas must be a SIGMAS tensor.")

        values = sigmas.detach().flatten().double().cpu()
        if values.numel() < 2:
            raise ValueError("Identity Feature Transfer Final: at least two sigma values are required.")
        if not torch.isfinite(values).all().item():
            raise ValueError("Identity Feature Transfer Final: sigma schedule contains a non-finite value.")

        deltas = (values[:-1] - values[1:]).abs()
        if deltas[0].item() <= 0.0:
            raise ValueError("Identity Feature Transfer Final: Step 0 sigma interval must be greater than zero.")
        zero_steps = torch.nonzero(deltas <= 0.0, as_tuple=False).flatten()
        if zero_steps.numel() > 0:
            step = int(zero_steps[0].item())
            raise ValueError(
                f"Identity Feature Transfer Final: sigma interval for step {step} is zero."
            )

        ratios = deltas[0] / deltas
        return values, ratios

    @staticmethod
    def _sigma_step_index(values: torch.Tensor, current_sigma: float) -> int:
        step_values = values[:-1]
        differences = torch.abs(step_values - float(current_sigma))
        tolerance = max(1e-7, abs(float(current_sigma)) * 1e-6)
        exact = torch.nonzero(differences <= tolerance, as_tuple=False).flatten()
        if exact.numel() > 0:
            return int(exact[0].item())

        for step_idx in range(values.numel() - 1):
            start = float(values[step_idx].item())
            end = float(values[step_idx + 1].item())
            lo, hi = min(start, end), max(start, end)
            if lo <= float(current_sigma) <= hi:
                return step_idx

        return int(torch.argmin(differences).item())

    def apply(
        self,
        model,
        preset="HARD_LOCK",
        enabled=True,
        reference_index=0,
        reference_indices="all",
        similarity_floor=0.040,
        softmax_temperature=0.0250,
        mask_threshold=1.0,
        double_blocks=HARD_DOUBLE,
        single_blocks=HARD_SINGLE,
        debug=False,
        subject_mask_1=None,
        subject_mask_2=None,
        subject_mask_3=None,
        subject_mask_4=None,
        subject_mask_5=None,
        subject_mask_6=None,
        subject_mask_7=None,
        subject_mask_8=None,
        sigmas=None,
    ):
        m = model.clone()
        if not bool(enabled):
            return (m,)

        if preset in self.PRESETS:
            cfg = self.PRESETS[preset]
            double_blocks = cfg["double_blocks"]
            single_blocks = cfg["single_blocks"]
            similarity_floor = cfg["similarity_floor"]
            softmax_temperature = cfg["softmax_temperature"]
            mask_threshold = cfg["mask_threshold"]

        double_map = self._parse_schedule(double_blocks, 7)
        single_map = self._parse_schedule(single_blocks, 23)
        sim_floor = float(max(0.0, min(0.95, similarity_floor)))
        temperature = float(max(1e-6, softmax_temperature))
        mask_threshold = float(max(0.0, min(1.0, mask_threshold)))
        ref_idx = int(reference_index)
        ref_indices_text = str(reference_indices)
        sigma_schedule = self._sigma_equal_energy_schedule(sigmas)
        sigma_debug_seen = set()
        sigma_missing_warned = False

        def sigma_strength_multiplier(extra_options):
            nonlocal sigma_missing_warned
            if sigma_schedule is None:
                return 1.0, None, None

            current_sigma = self._scalar_float(extra_options.get("sigmas"))
            if current_sigma is None:
                if debug and not sigma_missing_warned:
                    sigma_missing_warned = True
                    print(
                        "[IdentityFeatureTransferFinal] SIGMAS connected, but "
                        "the current model sigma is unavailable; using unscaled strengths."
                    )
                return 1.0, None, None

            values, ratios = sigma_schedule
            step_idx = self._sigma_step_index(values, current_sigma)
            return float(ratios[step_idx].item()), step_idx, current_sigma

        masks = [
            self._prep_mask(subject_mask_1),
            self._prep_mask(subject_mask_2),
            self._prep_mask(subject_mask_3),
            self._prep_mask(subject_mask_4),
            self._prep_mask(subject_mask_5),
            self._prep_mask(subject_mask_6),
            self._prep_mask(subject_mask_7),
            self._prep_mask(subject_mask_8),
        ]
        mask_cache: Dict[Tuple[int, int, float], Optional[torch.Tensor]] = {}

        def mask_indices(ref_id: int, count: int, device):
            if ref_id < 0 or ref_id >= len(masks):
                return None
            src = masks[ref_id]
            if src is None:
                return None
            key = (ref_id, int(count), mask_threshold)
            if key in mask_cache:
                cached = mask_cache[key]
                return cached.to(device) if cached is not None else None
            grid = self._grid_for_tokens(int(count), src)
            pooled = F.adaptive_avg_pool2d(src[None, None], grid).view(-1)
            keep = pooled >= mask_threshold
            if keep.sum().item() == 0:
                mask_cache[key] = torch.empty((0,), dtype=torch.long)
            else:
                mask_cache[key] = torch.nonzero(keep, as_tuple=False).squeeze(-1).to(torch.long).cpu()
            return mask_cache[key].to(device)

        def selected_ref_slices(ref_tokens: Sequence[int], base: int):
            selected = self._parse_ref_indices(ref_indices_text, len(ref_tokens))
            if not selected:
                selected = [min(max(ref_idx, 0), len(ref_tokens) - 1)]
            selected_set = set(selected)
            slices = []
            offset = 0
            for i, count in enumerate(ref_tokens):
                start = base + offset
                end = start + int(count)
                if i in selected_set and end > start:
                    slices.append((i, start, end))
                offset += int(count)
            return slices

        def reference_bank(tokens: torch.Tensor, slices):
            parts = []
            for ref_id, start, end in slices:
                ref = tokens[:, start:end]
                idx = mask_indices(ref_id, end - start, tokens.device)
                if idx is not None:
                    if idx.numel() == 0:
                        continue
                    ref = ref.index_select(1, idx.to(torch.long))
                if ref.shape[1] > 0:
                    parts.append(ref)
            if not parts:
                return None
            return torch.cat(parts, dim=1)

        def pull_delta(gen: torch.Tensor, ref: torch.Tensor, strength: float):
            if strength <= 0.0 or ref is None or ref.shape[1] <= 0:
                return None
            gen_f = gen.float()
            ref_f = ref.float()
            gen_norm = F.normalize(gen_f - gen_f.mean(dim=1, keepdim=True), dim=-1)
            ref_norm = F.normalize(ref_f - ref_f.mean(dim=1, keepdim=True), dim=-1)
            sim = torch.bmm(gen_norm, ref_norm.transpose(1, 2))
            neg = torch.finfo(sim.dtype).min
            sim = torch.where(sim >= sim_floor, sim, torch.full_like(sim, neg))
            weights = torch.softmax(sim / temperature, dim=-1)
            weights = torch.nan_to_num(weights, nan=0.0)
            pooled = torch.bmm(weights, ref_f)
            best = sim.max(dim=-1).values
            best = torch.where(torch.isfinite(best), best, torch.zeros_like(best))
            confidence = ((best - sim_floor) / max(1.0 - sim_floor, 1e-6)).clamp(0.0, 1.0)
            weight = (confidence * float(strength)).unsqueeze(-1).to(gen.dtype)
            return (pooled.to(gen.dtype) - gen) * weight

        def output_patch(attn, extra_options):
            ref_tokens = extra_options.get("reference_image_num_tokens", []) or []
            img_slice = extra_options.get("img_slice")
            if not ref_tokens or img_slice is None:
                return attn
            block_type = extra_options.get("block_type", "double")
            block_idx = int(extra_options.get("block_index", 0))
            if block_type == "double":
                strength = double_map.get(block_idx, 0.0)
            elif block_type == "single":
                strength = single_map.get(block_idx, 0.0)
            else:
                return attn
            sigma_multiplier, sigma_step, current_sigma = sigma_strength_multiplier(extra_options)
            strength *= sigma_multiplier
            if strength == 0.0:
                return attn
            if debug and sigma_step is not None and sigma_step not in sigma_debug_seen:
                sigma_debug_seen.add(sigma_step)
                print(
                    "[IdentityFeatureTransferFinal sigma] "
                    f"step={sigma_step} sigma={current_sigma:.7f} "
                    f"strength_multiplier={sigma_multiplier:.7f}"
                )
            txt_end, total_seq = int(img_slice[0]), int(img_slice[1])
            total_ref = int(sum(ref_tokens))
            gen_start = txt_end
            gen_end = total_seq - total_ref
            if gen_end <= gen_start:
                return attn
            ref = reference_bank(attn, selected_ref_slices(ref_tokens, total_seq - total_ref))
            if ref is None:
                return attn
            gen = attn[:, gen_start:gen_end]
            delta = pull_delta(gen, ref, strength)
            if delta is None:
                return attn
            out = attn.clone()
            out[:, gen_start:gen_end] = gen + delta
            return out

        if bool(debug):
            print(
                "[IdentityFeatureTransferFinal] "
                f"preset={preset} sim={sim_floor:.4f} temp={temperature:.4f} "
                f"mask={mask_threshold:.2f} "
                f"double={double_map} single={single_map} "
                f"sigma_aware={sigma_schedule is not None}"
            )

        m.set_model_attn1_output_patch(output_patch)
        return (m,)


NODE_CLASS_MAPPINGS = {
    "IdentityFeatureTransfer": IdentityFeatureTransfer,
    "IdentityFeatureTransferAdvanced": IdentityFeatureTransferAdvanced,
    "IdentityFeatureTransferV3": IdentityFeatureTransferV3,
    "IdentityFeatureTransferFinal": IdentityFeatureTransferFinal,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IdentityFeatureTransfer": "FLUX.2 Klein Identity Feature Transfer",
    "IdentityFeatureTransferAdvanced": "FLUX.2 Klein Identity Feature Transfer Advanced",
    "IdentityFeatureTransferV3": "FLUX.2 Klein Identity Feature Transfer V3",
    "IdentityFeatureTransferFinal": "Identity Feature Transfer Final",
}
