#!/usr/bin/env python3
"""
Generate V4 Qwen GT embeddings aligned with the inference-time extraction path.

V4 training was feeding the projector pre-computed hidden states extracted at
<TEX_i> token positions (teacher-forced text containing <TEX_1>..<TEX_K>).
But V4 inference runs a PERMANENTLY FROZEN Qwen that was never trained to
emit <TEX_i>, so inference_forward falls back to a regex parser that extracts
hidden states at the last token of each "TEXTURE_N: ..." line.

That Train-Test mismatch caused the V4 test mIoU to peak at ep5 (0.686) and
drift back to ~0.55-0.60 as the projector overfit to the training embedding
distribution. This script fixes the mismatch at its source.

For each training/val sample it:
  1. Builds the EXACT chat used at inference: system + user + assistant,
     where the assistant content is "TEXTURE_N: <desc>" lines with NO <TEX_i>
     tokens appended.
  2. Runs a single teacher-forced forward pass through frozen Qwen3-VL with
     the real image and output_hidden_states=True.
  3. Uses the EXACT SAME regex fallback from inference_forward() to locate
     the end-of-line token position for each TEXTURE line and extract the
     hidden state there.
  4. Saves {image_path: (K, llm_dim) tensor} to qwen_gt_embeds_v4.pt.

Usage:
    cd /home/aviad/Qwen2SAM_DeTexture
    python scripts/generate_gt_embeds_v4.py \
        --output /home/aviad/datasets/ADE20K_textured_images/qwen_gt_embeds_v4.pt

Point `data.qwen_gt_embeds_path` in configs/detexture.yaml at the new file
and launch a fresh V4 run.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import cv2
import torch
from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from models.qwen2sam_detexture import (
    MAX_TEXTURES,
    add_tex_tokens,
    load_qwen_model,
    load_qwen_processor,
)
from data.dataset import SYSTEM_PROMPT, TRAIN_USER_PROMPT


# ===================================================================== #
#  Helpers                                                                #
# ===================================================================== #

def build_assistant_text_no_tex(descriptions):
    """
    Build the exact assistant text that V4 inference produces:
    'TEXTURE_1: <desc>\\nTEXTURE_2: <desc>\\n...'  (no <TEX_i> tokens).
    """
    return "\n".join(f"TEXTURE_{i+1}: {d}" for i, d in enumerate(descriptions))


def load_image(image_path: str, image_size: int) -> Image.Image:
    """Same loading pipeline as data/dataset.py so Qwen sees the same pixels."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    return Image.fromarray(img)


# ===================================================================== #
#  Per-sample extraction                                                   #
# ===================================================================== #

@torch.no_grad()
def extract_embeds_for_sample(
    model,
    processor,
    image: Image.Image,
    descriptions: list,
    device,
    llm_dim: int,
) -> torch.Tensor:
    """
    Teacher-forced forward through Qwen3-VL and extract one hidden state per
    TEXTURE line using the exact same regex fallback as inference_forward.

    Returns:
        (K, llm_dim) float32 CPU tensor with K = len(descriptions) (capped at
        MAX_TEXTURES). Returns a zero tensor if extraction fails.
    """
    K = min(len(descriptions), MAX_TEXTURES)
    if K == 0:
        return torch.zeros(0, llm_dim)

    descriptions = descriptions[:K]
    assistant_text = build_assistant_text_no_tex(descriptions)

    # ---- Build prompt-only sequence (to measure prompt_len) -------------- #
    messages_prompt_only = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": TRAIN_USER_PROMPT},
        ]},
    ]
    prompt_text = processor.apply_chat_template(
        messages_prompt_only, tokenize=False, add_generation_prompt=True,
    )
    prompt_inputs = processor(
        text=[prompt_text], images=[image], return_tensors="pt", padding=True,
    )
    prompt_inputs.pop("token_type_ids", None)
    prompt_len = prompt_inputs["input_ids"].shape[1]

    # ---- Build full teacher-forced sequence ------------------------------ #
    messages_full = messages_prompt_only + [
        {"role": "assistant", "content": assistant_text},
    ]
    full_text = processor.apply_chat_template(
        messages_full, tokenize=False, add_generation_prompt=False,
    )
    full_inputs = processor(
        text=[full_text], images=[image], return_tensors="pt", padding=True,
    )
    full_inputs.pop("token_type_ids", None)
    full_inputs = {k: v.to(device) for k, v in full_inputs.items()}

    # Guard: prompt_len must be a prefix of the full sequence
    if prompt_len > full_inputs["input_ids"].shape[1]:
        raise RuntimeError(
            f"prompt_len ({prompt_len}) exceeds full seq "
            f"({full_inputs['input_ids'].shape[1]}) — chat template drift"
        )

    # ---- Forward through frozen Qwen ------------------------------------- #
    out = model(**full_inputs, output_hidden_states=True)
    full_hidden = out.hidden_states[-1][0]  # (seq, llm_dim)
    seq_len = full_hidden.shape[0]

    # ---- Decode assistant portion (mirrors inference_forward exactly) ---- #
    assistant_ids = full_inputs["input_ids"][0, prompt_len:]
    decoded = processor.tokenizer.decode(assistant_ids, skip_special_tokens=False)

    # ---- Apply the SAME regex fallback as inference_forward -------------- #
    tex_embeds = torch.zeros(MAX_TEXTURES, llm_dim, dtype=torch.float32)
    lines = decoded.strip().split("\n")
    tex_count = 0
    for line in lines:
        match = re.match(r"TEXTURE_(\d+):", line.strip())
        if match and tex_count < MAX_TEXTURES:
            # Encode the assistant text up to and including this line
            text_up_to = "\n".join(lines[: tex_count + 1])
            tokens_up_to = processor.tokenizer.encode(
                text_up_to, add_special_tokens=False,
            )
            # Position in the full sequence = prompt_len + (len - 1).
            # Clamp to the assistant-content portion of the sequence.
            max_assistant_idx = seq_len - prompt_len - 1
            pos_in_assistant = min(len(tokens_up_to) - 1, max_assistant_idx)
            pos = prompt_len + pos_in_assistant
            if 0 <= pos < seq_len:
                tex_embeds[tex_count] = full_hidden[pos].float().cpu()
                tex_count += 1

    if tex_count == 0:
        raise RuntimeError(
            "Regex fallback found 0 TEXTURE lines in decoded assistant text: "
            f"{decoded[:200]!r}"
        )
    if tex_count < K:
        print(f"  [WARN] expected {K} textures, extracted {tex_count}")

    return tex_embeds[:K]


# ===================================================================== #
#  Main                                                                   #
# ===================================================================== #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-metadata",
        default="/home/aviad/datasets/ADE20K_textured_images/train_metadata.json",
    )
    parser.add_argument(
        "--val-metadata",
        default="/home/aviad/datasets/ADE20K_textured_images/val_metadata.json",
    )
    parser.add_argument(
        "--output",
        default="/home/aviad/datasets/ADE20K_textured_images/qwen_gt_embeds_v4.pt",
    )
    parser.add_argument("--qwen-model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--image-size", type=int, default=1008)
    parser.add_argument("--limit", type=int, default=None,
                        help="Debug: only process the first N samples per split")
    parser.add_argument("--save-every", type=int, default=1000,
                        help="Save intermediate checkpoint every N samples")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    print(f"Loading Qwen: {args.qwen_model}")
    processor = load_qwen_processor(args.qwen_model)
    model = load_qwen_model(args.qwen_model, dtype=dtype)

    # Add TEX tokens to match the training-time tokenizer state (the tokens
    # are never emitted in the text we build, but adding them keeps the
    # tokenizer + embedding table identical to the one used by the live model).
    add_tex_tokens(processor, model)

    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    llm_cfg = getattr(model.config, "text_config", model.config)
    llm_dim = llm_cfg.hidden_size
    print(f"llm_dim = {llm_dim}")

    # ---- Load metadata --------------------------------------------------- #
    splits = []
    for split_name, meta_path in [("train", args.train_metadata),
                                   ("val", args.val_metadata)]:
        with open(meta_path) as f:
            samples = json.load(f)
        if args.limit:
            samples = samples[: args.limit]
        print(f"  {split_name}: {len(samples)} samples")
        splits.append((split_name, samples))

    total = sum(len(s) for _, s in splits)
    print(f"Total samples to process: {total}")

    # ---- Extract embeddings ---------------------------------------------- #
    results = {}
    t0 = time.time()
    n_done = 0
    n_fail = 0

    for split_name, samples in splits:
        print(f"\n=== Processing {split_name} ({len(samples)} samples) ===")
        for i, meta in enumerate(samples):
            image_path = meta["image_path"]
            textures = meta["textures"][:MAX_TEXTURES]
            descriptions = [t["description"] for t in textures]

            try:
                image = load_image(image_path, args.image_size)
                embeds = extract_embeds_for_sample(
                    model, processor, image, descriptions, device, llm_dim,
                )
                results[image_path] = embeds
            except Exception as e:
                n_fail += 1
                print(f"  [FAIL] {i}: {image_path}: {type(e).__name__}: {e}")
                continue

            n_done += 1

            if (n_done % 50 == 0):
                elapsed = time.time() - t0
                rate = n_done / elapsed
                eta = (total - n_done) / max(rate, 1e-6)
                print(
                    f"  [{n_done}/{total}] "
                    f"rate={rate:.2f}/s  eta={eta/60:.1f}min  fails={n_fail}",
                    flush=True,
                )

            if n_done % args.save_every == 0:
                tmp_path = Path(str(args.output) + ".partial")
                tmp_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(results, tmp_path)
                print(f"  [checkpoint] saved partial: {tmp_path} ({len(results)} entries)")

    elapsed = time.time() - t0
    print(f"\nFinished {n_done}/{total} samples in {elapsed/60:.1f}min "
          f"({n_fail} failures)")

    # ---- Save final ------------------------------------------------------ #
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, out_path)
    print(f"Saved: {out_path}")
    size_mb = out_path.stat().st_size / 1024**2
    print(f"Size: {size_mb:.1f} MB")

    # Remove partial file if it exists
    partial = Path(str(args.output) + ".partial")
    if partial.exists():
        partial.unlink()

    # ---- Sanity check ---------------------------------------------------- #
    first_key = next(iter(results.keys()))
    first = results[first_key]
    print(f"\nSanity check — first sample: {first_key}")
    print(f"  shape: {tuple(first.shape)}  dtype: {first.dtype}")
    print(f"  per-slot norms: {[f'{v:.3f}' for v in first.norm(dim=-1).tolist()]}")
    print(f"  mean: {first.mean().item():+.4f}  std: {first.std().item():.4f}")


if __name__ == "__main__":
    main()
