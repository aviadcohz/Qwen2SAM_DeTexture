#!/usr/bin/env python3
"""
Offline Qwen GT Embedding Generation for Self-Distillation.

Runs the frozen pretrained Qwen3-VL-8B (NO LoRA) on each training/val sample
with the FULL image+text input, extracts hidden states at <TEX_i> token
positions, and saves them as a .pt file.

These serve as the "teacher" embeddings for self-distillation training.

Usage:
    cd /home/aviad/Qwen2SAM_DeTexture
    python scripts/prepare_qwen_gt_embeds.py --config configs/detexture.yaml
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from models.qwen2sam_detexture import (
    load_qwen_model, load_qwen_processor, add_tex_tokens,
    MAX_TEXTURES, TEX_TOKENS,
)
from data.dataset import SYSTEM_PROMPT, TRAIN_USER_PROMPT, build_assistant_text


def main():
    parser = argparse.ArgumentParser(description="Pre-compute Qwen GT embeddings")
    parser.add_argument("--config", type=str, default="configs/detexture.yaml")
    parser.add_argument("--output", type=str, default=None,
                        help="Output .pt path (default: from config)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size (1 recommended for image processing)")
    args = parser.parse_args()

    from training.utils import load_config
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_path = args.output or cfg["data"].get(
        "qwen_gt_embeds_path",
        "/home/aviad/datasets/ADE20K_textured_images/qwen_gt_embeds.pt",
    )

    # ---- Load Qwen (base model, NO LoRA = the "teacher") ----
    model_name = cfg["model"]["qwen_model"]
    print(f"Loading Qwen teacher: {model_name}")
    qwen_dtype = getattr(torch, cfg["model"].get("qwen_dtype", "bfloat16"))
    processor = load_qwen_processor(model_name)
    model = load_qwen_model(model_name, dtype=qwen_dtype)

    # Add TEX special tokens
    tex_token_ids = add_tex_tokens(processor, model)
    tex_id_list = [tex_token_ids[t] for t in TEX_TOKENS]

    model.eval().to(device)
    print(f"Model loaded on {device}, dtype={qwen_dtype}")

    # Get hidden size
    qwen_cfg = getattr(model.config, "text_config", model.config)
    llm_dim = qwen_cfg.hidden_size  # 4096

    # ---- Load all metadata ----
    train_path = cfg["data"]["train_metadata"]
    val_path = cfg["data"]["val_metadata"]

    all_samples = []
    for path in [train_path, val_path]:
        with open(path) as f:
            samples = json.load(f)
        all_samples.extend(samples)
        print(f"Loaded {len(samples)} samples from {path}")
    print(f"Total samples: {len(all_samples)}")

    # ---- Process each sample (with resume support) ----
    # Load existing progress if the file already exists
    if Path(output_path).exists():
        embeddings = torch.load(output_path, map_location="cpu")
        print(f"Resuming: loaded {len(embeddings)} existing embeddings from {output_path}")
    else:
        embeddings = {}
    t0 = time.time()

    with torch.no_grad():
        for idx, sample in enumerate(all_samples):
            image_path = sample["image_path"]

            # Skip if already computed
            if image_path in embeddings:
                continue

            textures = sample["textures"][:MAX_TEXTURES]
            descriptions = [t["description"] for t in textures]
            k = len(descriptions)

            # Build the FULL chat with image (matching student's input exactly)
            assistant_text = build_assistant_text(descriptions)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": TRAIN_USER_PROMPT},
                ]},
                {"role": "assistant", "content": assistant_text},
            ]

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )

            # Load image
            img = Image.open(image_path).convert("RGB")

            # Tokenize with image
            inputs = processor(
                text=[text], images=[img], return_tensors="pt", padding=False,
            )
            inputs = {k_: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k_, v in inputs.items()}
            inputs.pop("token_type_ids", None)

            # Forward
            out = model(**inputs, output_hidden_states=True)
            hidden = out.hidden_states[-1]  # (1, seq_len, llm_dim)
            input_ids = inputs["input_ids"][0]

            # Extract TEX token hidden states
            tex_embeds = torch.zeros(k, llm_dim, dtype=qwen_dtype)
            for i in range(k):
                tid = tex_id_list[i]
                positions = (input_ids == tid).nonzero(as_tuple=True)[0]
                if len(positions) > 0:
                    pos = positions[0].item()
                    tex_embeds[i] = hidden[0, pos].cpu()

            # Store with image_path as key, move to CPU bfloat16
            embeddings[image_path] = tex_embeds.to(torch.bfloat16).cpu()

            if (idx + 1) % 100 == 0 or idx == 0:
                elapsed = time.time() - t0
                n_new = len(embeddings) - (len(embeddings) if idx == 0 else 0)
                rate = max(n_new, 1) / max(elapsed, 1)
                remaining = len(all_samples) - len(embeddings)
                eta = remaining / max(rate, 0.01)
                print(f"  [{len(embeddings)}/{len(all_samples)}] "
                      f"{rate:.1f} new/sec, ETA: {eta/60:.1f} min")

            # Checkpoint every 500 samples
            if len(embeddings) % 500 == 0 and len(embeddings) > 0:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(embeddings, output_path)
                print(f"  Checkpoint saved: {len(embeddings)} embeddings")

    # ---- Save ----
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, output_path)

    # Stats
    total_size = sum(v.numel() * 2 for v in embeddings.values())  # bfloat16 = 2 bytes
    print(f"\nSaved {len(embeddings)} embeddings to {output_path}")
    print(f"  File size: ~{total_size / 1e6:.0f} MB")
    print(f"  Sample shapes: {[v.shape for v in list(embeddings.values())[:3]]}")
    print(f"  Total time: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
