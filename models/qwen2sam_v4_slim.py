"""
V4-Slim Bridge Experiment: Slim Projector (4096→512→1024→plug→256)
with Frozen Qwen + SAM Architectural Plug + Distillation.

Purpose: Isolate the Information Bottleneck's impact on domain generalization.
If RWTD mIoU stays stable at ep10 (unlike V4's drift), the slim projector
is the key to generalization — independent of [SEG] token or LoRA.

Architecture:
  Qwen (FROZEN, no LoRA) → override_tex_embeds (pre-computed GT)
    → Slim Projector: Linear(4096→512) + LN + GELU + Linear(512→1024) + GELU
    → Frozen SAM Plug: LN(1024) + Linear(1024→256)
    → Batch Multiplexed SAM3 → masks

The 512-dim bottleneck forces generalization. The frozen SAM plug anchors
output in SAM's native pre-projection space. Cosine distillation provides
directional alignment.
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# Reuse shared components from the V5 model file
from models.qwen2sam_detexture import (
    MAX_TEXTURES, NUM_QUERY_SLOTS,
    MultiTextureMaskHead,
    load_qwen_processor, load_qwen_model,
    load_sam3,
)
from models.orthogonal_lora import apply_orthogonal_lora_to_mha


# ===================================================================== #
#  Slim Projector with SAM Architectural Plug                             #
# ===================================================================== #

class SlimProjectorWithPlug(nn.Module):
    """
    V4-Slim bottleneck projector with frozen SAM text-head transplant.

    Trainable trunk:
        Linear(4096 → 512) + LayerNorm(512) + GELU   [bottleneck]
        Linear(512 → 1024) + GELU                     [expand to plug input dim]

    Frozen transplant (from SAM3 text encoder):
        LayerNorm(1024)           [sam_ln_final]
        Linear(1024 → 256)       [sam_resizer]

    Total trainable: ~2.6M params (vs V4's 10.5M — 4x reduction)
    """

    def __init__(self, llm_dim: int = 4096, bottleneck_dim: int = 512):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(llm_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, 1024),
            nn.GELU(),
        )
        # Frozen transplant — overwritten by _transplant_sam_text_head
        self.sam_ln_final = nn.LayerNorm(1024)
        self.sam_resizer = nn.Linear(1024, 256)

    def forward(self, x):
        h = self.trunk(x)
        h = self.sam_ln_final(h)
        h = self.sam_resizer(h)
        return h

    def trainable_parameters(self):
        return self.trunk.parameters()

    def frozen_transplant_modules(self):
        return (self.sam_ln_final, self.sam_resizer)


# ===================================================================== #
#  V4-Slim Model                                                          #
# ===================================================================== #

# Keep TEX tokens for tokenizer compatibility with pre-computed embeddings
TEX_TOKENS = [f"<TEX_{i}>" for i in range(1, MAX_TEXTURES + 1)]


def add_tex_tokens(processor, model):
    tokenizer = processor.tokenizer
    num_added = tokenizer.add_tokens(TEX_TOKENS, special_tokens=True)
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
    return {t: tokenizer.convert_tokens_to_ids(t) for t in TEX_TOKENS}


class Qwen2SAMDeTextureV4Slim(nn.Module):
    """
    V4-Slim: Frozen Qwen + Slim Projector + SAM Plug + Distillation.
    Bridge experiment to isolate the bottleneck's impact.
    """

    def __init__(self, cfg: dict, device: str = "cuda"):
        super().__init__()
        self.device = torch.device(device)
        self.cfg = cfg
        model_cfg = cfg["model"]

        # ---- Qwen (FROZEN, no LoRA) ------------------------------------ #
        qwen_dtype = getattr(torch, model_cfg.get("qwen_dtype", "bfloat16"))
        self.processor = load_qwen_processor(model_cfg["qwen_model"])
        self.qwen = load_qwen_model(model_cfg["qwen_model"], dtype=qwen_dtype)

        self.tex_token_ids = add_tex_tokens(self.processor, self.qwen)
        self.tex_id_list = [self.tex_token_ids[t] for t in TEX_TOKENS]

        for p in self.qwen.parameters():
            p.requires_grad = False
        self.qwen.eval()
        self.qwen.to(self.device)

        # ---- SAM3 ------------------------------------------------------ #
        self.sam3 = load_sam3(model_cfg, self.device)
        self._freeze_sam3()
        self.sam3_lora_modules = self._apply_sam3_orthogonal_lora(model_cfg)
        self.sam3.to(device=self.device)

        # ---- LLM dim --------------------------------------------------- #
        qwen_cfg = getattr(self.qwen.config, "text_config", self.qwen.config)
        self.llm_dim = qwen_cfg.hidden_size

        # ---- Slim Projector with Architectural Plug --------------------- #
        bottleneck = model_cfg.get("projector_bottleneck_dim", 512)
        self.projector = SlimProjectorWithPlug(
            llm_dim=self.llm_dim, bottleneck_dim=bottleneck,
        ).to(self.device)
        self._transplant_sam_text_head()

        # ---- Dustbin + Mask head ---------------------------------------- #
        self.dustbin_embed = nn.Parameter(
            torch.randn(1, 1, self.llm_dim, device=self.device) * 0.02
        )
        self.mask_head = MultiTextureMaskHead(
            embed_dim=256, mask_dim=256,
        ).to(self.device)

    # ------------------------------------------------------------------ #
    #  SAM3 setup                                                          #
    # ------------------------------------------------------------------ #

    def _freeze_sam3(self):
        for p in self.sam3.parameters():
            p.requires_grad = False

    def _apply_sam3_orthogonal_lora(self, model_cfg):
        r = model_cfg.get("sam3_lora_r", 8)
        alpha = model_cfg.get("sam3_lora_alpha", 16.0)
        n_sing = model_cfg.get("sam3_lora_n_singular", 32)
        targets = tuple(model_cfg.get("sam3_lora_targets", ["q", "v"]))
        all_lora = []
        for layer in self.sam3.transformer.encoder.layers:
            if hasattr(layer, "cross_attn_image"):
                d = apply_orthogonal_lora_to_mha(
                    layer.cross_attn_image, r=r, alpha=alpha,
                    n_singular=n_sing, target_projections=targets,
                )
                all_lora.extend(d.values())
        seg_head = self.sam3.segmentation_head
        if seg_head and hasattr(seg_head, "cross_attend_prompt"):
            if seg_head.cross_attend_prompt is not None:
                d = apply_orthogonal_lora_to_mha(
                    seg_head.cross_attend_prompt, r=r, alpha=alpha,
                    n_singular=n_sing, target_projections=targets,
                )
                all_lora.extend(d.values())
        n_p = sum(p.numel() for m in all_lora for p in [m.lora_A, m.lora_B])
        print(f"  SAM3 Orth LoRA: {len(all_lora)} modules, {n_p:,} params")
        return all_lora

    # ------------------------------------------------------------------ #
    #  Architectural Plug transplant                                       #
    # ------------------------------------------------------------------ #

    def _transplant_sam_text_head(self):
        text_enc = self.sam3.backbone.language_backbone
        src_ln = text_enc.encoder.ln_final
        src_resizer = text_enc.resizer
        dst_ln = self.projector.sam_ln_final
        dst_resizer = self.projector.sam_resizer

        with torch.no_grad():
            dst_ln.weight.copy_(src_ln.weight.detach())
            dst_ln.bias.copy_(src_ln.bias.detach())
            dst_resizer.weight.copy_(src_resizer.weight.detach())
            dst_resizer.bias.copy_(src_resizer.bias.detach())

        for p in dst_ln.parameters():
            p.requires_grad = False
        for p in dst_resizer.parameters():
            p.requires_grad = False
        dst_ln.to(self.device)
        dst_resizer.to(self.device)

        n = sum(p.numel() for p in dst_ln.parameters()) + \
            sum(p.numel() for p in dst_resizer.parameters())
        print(f"  Architectural Plug: {n:,} frozen params transplanted")

    # ------------------------------------------------------------------ #
    #  SAM text encoder (for distillation targets)                         #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def encode_text_sam(self, descriptions):
        if not descriptions:
            return torch.zeros(0, 256, device=self.device)
        text_enc = self.sam3.backbone.language_backbone
        mask, mem, _ = text_enc(descriptions, device=self.device)
        feats = mem.transpose(0, 1).float()
        valid = (~mask).float().unsqueeze(-1)
        return (feats * valid).sum(1) / valid.sum(1).clamp(min=1)

    @torch.no_grad()
    def encode_descriptions_batch(self, descriptions_batch):
        B = len(descriptions_batch)
        out = torch.zeros(B, MAX_TEXTURES, 256, device=self.device)
        flat, offsets = [], []
        for b, descs in enumerate(descriptions_batch):
            for i, d in enumerate(descs[:MAX_TEXTURES]):
                flat.append(d)
                offsets.append((b, i))
        if not flat:
            return out
        targets = self.encode_text_sam(flat)
        for (b, i), v in zip(offsets, targets):
            out[b, i] = v
        return out

    # ------------------------------------------------------------------ #
    #  Query slot assembly                                                 #
    # ------------------------------------------------------------------ #

    def build_query_slots(self, tex_embeds, k_preds):
        B = tex_embeds.shape[0]
        device = tex_embeds.device
        dustbin = self.dustbin_embed.expand(B, 1, self.llm_dim)
        query_embeds = torch.cat([dustbin, tex_embeds], dim=1)
        pad_mask = torch.zeros(B, NUM_QUERY_SLOTS, dtype=torch.bool, device=device)
        for b in range(B):
            kp = int(k_preds[b].item())
            pad_mask[b, kp + 1:] = True
        return query_embeds, pad_mask

    # ------------------------------------------------------------------ #
    #  Batch Multiplexed SAM3 pass                                         #
    # ------------------------------------------------------------------ #

    def _get_img_feats(self, backbone_out, img_ids):
        n = self.sam3.num_feature_levels
        vf = backbone_out["backbone_fpn"][-n:]
        vp = backbone_out["vision_pos_enc"][-n:]
        sizes = [x.shape[-2:] for x in vp]
        feats = [x[img_ids].flatten(2).permute(2, 0, 1) for x in vf]
        pos = [x[img_ids].flatten(2).permute(2, 0, 1) for x in vp]
        return feats, pos, sizes

    def run_sam3_semantic(self, backbone_out, query_256, pad_mask):
        B, N, D = query_256.shape
        device = query_256.device
        image_ids = torch.arange(B, device=device).repeat_interleave(N)
        img_feats, img_pos, sizes = self._get_img_feats(backbone_out, image_ids)

        qf = query_256.reshape(B * N, 1, D)
        prompt = qf.transpose(0, 1)
        prompt_pos = torch.zeros_like(prompt)

        mem = self.sam3.transformer.encoder(
            src=[f.clone() for f in img_feats],
            src_key_padding_mask=None,
            src_pos=[p.clone() for p in img_pos],
            prompt=prompt, prompt_pos=prompt_pos,
            prompt_key_padding_mask=None,
            feat_sizes=sizes,
        )
        enc_hs = mem["memory"]

        sh = self.sam3.segmentation_head
        if sh.cross_attend_prompt is not None:
            t2 = sh.cross_attn_norm(enc_hs)
            t2 = sh.cross_attend_prompt(
                query=t2, key=prompt, value=prompt,
                key_padding_mask=None,
            )[0]
            enc_hs = t2 + enc_hs

        pixel_embed = sh._embed_pixels(
            backbone_feats=backbone_out["backbone_fpn"],
            image_ids=image_ids,
            encoder_hidden_states=enc_hs,
        )
        logits = self.mask_head(qf, pixel_embed)
        return logits.reshape(B, N, *logits.shape[-2:])

    # ------------------------------------------------------------------ #
    #  Forward (V4-style: override_tex_embeds)                             #
    # ------------------------------------------------------------------ #

    def forward(self, qwen_inputs, sam_images, seg_grad_to_lm=False,
                override_tex_embeds=None):
        if override_tex_embeds is not None:
            tex_embeds = override_tex_embeds
            B = tex_embeds.shape[0]
            k_preds = torch.zeros(B, dtype=torch.long, device=tex_embeds.device)
            for b in range(B):
                for i in range(MAX_TEXTURES):
                    if tex_embeds[b, i].norm() > 0.01:
                        k_preds[b] = i + 1
            lm_loss = torch.tensor(0.0, device=self.device)
        else:
            qwen_out = self.qwen(**qwen_inputs, output_hidden_states=True)
            hidden = qwen_out.hidden_states[-1]
            lm_loss = qwen_out.loss if qwen_out.loss is not None else \
                torch.tensor(0.0, device=self.device)
            tex_embeds, k_preds = self._extract_tex_hidden(hidden, qwen_inputs["input_ids"])

        query_embeds, pad_mask = self.build_query_slots(tex_embeds, k_preds)
        query_256 = self.projector(query_embeds)

        with torch.no_grad():
            backbone_out = self.sam3.backbone.forward_image(sam_images)
            backbone_out["img_batch_all_stages"] = sam_images

        mask_logits = self.run_sam3_semantic(backbone_out, query_256, pad_mask)

        return {
            "mask_logits": mask_logits,
            "tex_embeds": tex_embeds,
            "query_256": query_256,
            "k_preds": k_preds,
            "pad_mask": pad_mask,
            "lm_loss": lm_loss,
        }

    def _extract_tex_hidden(self, hidden_states, input_ids):
        B = hidden_states.shape[0]
        embeds = torch.zeros(B, MAX_TEXTURES, self.llm_dim,
                             device=hidden_states.device, dtype=hidden_states.dtype)
        k_preds = torch.zeros(B, dtype=torch.long, device=hidden_states.device)
        for b in range(B):
            ids = input_ids[b]
            count = 0
            for i, tid in enumerate(self.tex_id_list):
                pos = (ids == tid).nonzero(as_tuple=True)[0]
                if len(pos) > 0:
                    embeds[b, i] = hidden_states[b, pos[0].item()]
                    count = i + 1
            k_preds[b] = count
        return embeds, k_preds

    # ------------------------------------------------------------------ #
    #  Inference forward (V4-style regex fallback, max_new_tokens=300)      #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def inference_forward(self, qwen_inputs, sam_images, max_new_tokens=300):
        gen_out = self.qwen.generate(
            **qwen_inputs, max_new_tokens=max_new_tokens,
            output_hidden_states=True, return_dict_in_generate=True,
            do_sample=False,
        )
        generated_ids = gen_out.sequences
        prompt_len = qwen_inputs["input_ids"].shape[1]
        B = generated_ids.shape[0]

        gen_hidden_list = []
        for sh in gen_out.hidden_states:
            ll = sh[-1]
            if ll.shape[1] > 1:
                ll = ll[:, -1:, :]
            gen_hidden_list.append(ll)
        gen_hidden = torch.cat(gen_hidden_list, dim=1)

        full_hidden = torch.zeros(
            B, generated_ids.shape[1], self.llm_dim,
            device=generated_ids.device, dtype=gen_hidden.dtype,
        )
        full_hidden[:, prompt_len:prompt_len + gen_hidden.shape[1]] = gen_hidden

        tex_embeds, k_preds = self._extract_tex_hidden(full_hidden, generated_ids)

        generated_text = []
        for b in range(B):
            text = self.processor.tokenizer.decode(
                generated_ids[b, prompt_len:], skip_special_tokens=False,
            )
            generated_text.append(text)

            if k_preds[b].item() == 0 and "TEXTURE_" in text:
                was_truncated = "<|im_end|>" not in text
                lines = text.strip().split("\n")
                tc = 0
                for li, line in enumerate(lines):
                    m = re.match(r"TEXTURE_(\d+):", line.strip())
                    if not m or tc >= MAX_TEXTURES:
                        continue
                    if was_truncated and li == len(lines) - 1:
                        continue
                    desc = line.strip().split(":", 1)[-1].strip()
                    if len(desc.split()) < 3:
                        continue
                    text_up_to = "\n".join(lines[:li + 1])
                    tokens_up_to = self.processor.tokenizer.encode(
                        text_up_to, add_special_tokens=False,
                    )
                    pos = prompt_len + min(len(tokens_up_to) - 1,
                                           gen_hidden.shape[1] - 1)
                    if 0 <= pos < full_hidden.shape[1]:
                        tex_embeds[b, tc] = full_hidden[b, pos]
                        tc += 1
                k_preds[b] = tc

        query_embeds, pad_mask = self.build_query_slots(tex_embeds, k_preds)
        query_256 = self.projector(query_embeds)

        backbone_out = self.sam3.backbone.forward_image(sam_images)
        backbone_out["img_batch_all_stages"] = sam_images
        mask_logits = self.run_sam3_semantic(backbone_out, query_256, pad_mask)

        return {
            "mask_logits": mask_logits, "tex_embeds": tex_embeds,
            "k_preds": k_preds, "pad_mask": pad_mask,
            "lm_loss": torch.tensor(0.0, device=self.device),
            "generated_text": generated_text,
        }

    # ------------------------------------------------------------------ #
    #  Parameter groups                                                    #
    # ------------------------------------------------------------------ #

    def get_parameter_groups(self, base_lr):
        sam3_lr_scale = self.cfg["model"].get("sam3_lr_scale", 0.1)
        proj_params = [p for p in self.projector.parameters() if p.requires_grad]
        mh_params = list(self.mask_head.parameters())
        dustbin_params = [self.dustbin_embed]
        sam3_lora = []
        for m in self.sam3_lora_modules:
            sam3_lora.extend([m.lora_A, m.lora_B])

        groups = [
            {"params": proj_params, "lr": base_lr, "name": "projector_trunk"},
            {"params": mh_params, "lr": base_lr, "name": "mask_head"},
            {"params": dustbin_params, "lr": base_lr, "name": "dustbin"},
        ]
        if sam3_lora:
            groups.append({"params": sam3_lora, "lr": base_lr * sam3_lr_scale,
                           "name": "sam3_orth_lora"})
        return groups

    def num_trainable_params(self):
        trunk_n = sum(p.numel() for p in self.projector.trainable_parameters())
        plug_n = sum(p.numel() for m in self.projector.frozen_transplant_modules()
                     for p in m.parameters())
        mh_n = sum(p.numel() for p in self.mask_head.parameters())
        db_n = self.dustbin_embed.numel()
        sam_n = sum(p.numel() for m in self.sam3_lora_modules
                    for p in [m.lora_A, m.lora_B])
        return {
            "projector_trunk": trunk_n,
            "projector_plug_frozen": plug_n,
            "mask_head": mh_n,
            "dustbin": db_n,
            "sam3_orth_lora": sam_n,
            "total_trainable": trunk_n + mh_n + db_n + sam_n,
        }
