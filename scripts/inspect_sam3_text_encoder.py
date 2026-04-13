#!/usr/bin/env python3
"""Inspect the SAM3 text encoder for the Architectural Plug."""

import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from sam3.model_builder import build_sam3_image_model

print("Loading SAM3 (CPU)...")
sam3 = build_sam3_image_model(checkpoint_path=None, device="cpu")
sam3.eval()

# Locate text encoder
text_enc = sam3.backbone.language_backbone
print("\n=== TEXT ENCODER OBJECT ===")
print(type(text_enc).__name__)
print(text_enc)

# Resizer (the candidate for transplant)
print("\n=== RESIZER (final projection layer) ===")
print(f"  type:   {type(text_enc.resizer).__name__}")
print(f"  weight: {tuple(text_enc.resizer.weight.shape)}")
print(f"  bias:   {tuple(text_enc.resizer.bias.shape) if text_enc.resizer.bias is not None else None}")

# Inner TextTransformer width
print("\n=== INNER TextTransformer ===")
print(f"  width (hidden dim): {text_enc.encoder.width}")
print(f"  output_dim:         {text_enc.encoder.output_dim}")
print(f"  layers:             {text_enc.encoder.transformer.layers}")
print(f"  ln_final:           {type(text_enc.encoder.ln_final).__name__}")
if hasattr(text_enc.encoder, "text_projection"):
    tp = text_enc.encoder.text_projection
    if isinstance(tp, torch.nn.Parameter):
        print(f"  text_projection:    Parameter shape {tuple(tp.shape)}")
    else:
        print(f"  text_projection:    {type(tp).__name__} {tuple(tp.weight.shape)}")

# State dict keys
print("\n=== STATE_DICT KEYS (text encoder) ===")
sd = sam3.state_dict()
text_keys = [k for k in sd.keys() if "backbone.language_backbone" in k]
print(f"  Total keys under backbone.language_backbone: {len(text_keys)}")
print("  Resizer keys:")
for k in text_keys:
    if "resizer" in k:
        print(f"    {k:60s} {tuple(sd[k].shape)}")
print("  ln_final / text_projection keys:")
for k in text_keys:
    if "ln_final" in k or "text_projection" in k:
        print(f"    {k:60s} {tuple(sd[k].shape)}")

# Quick sanity check: are the resizer weights non-trivial (i.e. loaded from ckpt)?
w = text_enc.resizer.weight
print(f"\n=== RESIZER WEIGHT STATS ===")
print(f"  shape:   {tuple(w.shape)}")
print(f"  mean:    {w.mean().item():+.6f}")
print(f"  std:     {w.std().item():.6f}")
print(f"  min/max: {w.min().item():+.4f} / {w.max().item():+.4f}")
print(f"  → if std > 0.01, this looks like trained weights (not init)")
