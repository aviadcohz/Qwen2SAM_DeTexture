#!/usr/bin/env python3
"""Sanity-check the V4 Architectural Plug without loading Qwen."""

import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from sam3.model_builder import build_sam3_image_model
from models.projector import DescriptionProjector

print("Building SAM3 (CPU)...")
sam3 = build_sam3_image_model(checkpoint_path=None, device="cpu")

# Build projector standalone (same dims as model)
proj = DescriptionProjector(llm_dim=4096, sam_dim=256)

# Manually perform the transplant (mirrors Qwen2SAMDeTexture._transplant_sam_text_head)
src_ln = sam3.backbone.language_backbone.encoder.ln_final
src_resizer = sam3.backbone.language_backbone.resizer

with torch.no_grad():
    proj.sam_ln_final.weight.copy_(src_ln.weight)
    proj.sam_ln_final.bias.copy_(src_ln.bias)
    proj.sam_resizer.weight.copy_(src_resizer.weight)
    proj.sam_resizer.bias.copy_(src_resizer.bias)

for p in proj.sam_ln_final.parameters():
    p.requires_grad = False
for p in proj.sam_resizer.parameters():
    p.requires_grad = False

# Verify weights match exactly
assert torch.allclose(proj.sam_ln_final.weight, src_ln.weight)
assert torch.allclose(proj.sam_ln_final.bias, src_ln.bias)
assert torch.allclose(proj.sam_resizer.weight, src_resizer.weight)
assert torch.allclose(proj.sam_resizer.bias, src_resizer.bias)
print("OK: transplant weights are bit-identical to SAM source.")

# Forward pass shape check: (B, 7, 4096) -> (B, 7, 256)
B = 2
x = torch.randn(B, 7, 4096)
y = proj(x)
assert y.shape == (B, 7, 256), y.shape
print(f"OK: forward shape {tuple(y.shape)}")

# Grad flow check: only trunk params have grads, transplant does not.
loss = y.sum()
loss.backward()
trunk_has_grad = any(
    p.grad is not None and p.grad.abs().sum() > 0 for p in proj.trunk.parameters()
)
plug_has_grad = any(
    p.grad is not None for m in proj.frozen_transplant_modules() for p in m.parameters()
)
assert trunk_has_grad, "Trunk params did not receive gradients!"
assert not plug_has_grad, "Frozen plug layers should NOT have gradients!"
print("OK: trunk receives gradients, transplanted layers are frozen.")

# Parameter counts
n_trunk = sum(p.numel() for p in proj.trainable_parameters())
n_plug = sum(p.numel() for m in proj.frozen_transplant_modules() for p in m.parameters())
print(f"\nProjector trainable (trunk): {n_trunk:,}")
print(f"Projector frozen (plug):     {n_plug:,}")
print(f"  expected plug: {1024*2 + 256*1024 + 256:,} (2*LN 1024 + Linear 1024->256 + bias)")

# SAM text encoder forward: encode a fake description and check shape
descs = ["rough stone surface in the foreground",
         "smooth water reflecting sky in the center"]
with torch.no_grad():
    mask, mem, emb = sam3.backbone.language_backbone(descs, device="cpu")
print(f"\nSAM text encoder output: text_memory_resized shape {tuple(mem.shape)} "
      f"(expect (seq_len, B, 256))")
print(f"                        attention_mask shape {tuple(mask.shape)}")

# Pooled target shape
feats = mem.transpose(0, 1).float()
valid = (~mask).float().unsqueeze(-1)
pooled = (feats * valid).sum(1) / valid.sum(1).clamp(min=1)
print(f"Pooled per-description: {tuple(pooled.shape)} (expect (B, 256))")
print("\nV4 Architectural Plug smoke test PASSED.")
