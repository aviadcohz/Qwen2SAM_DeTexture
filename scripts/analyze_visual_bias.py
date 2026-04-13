#!/usr/bin/env python3
"""
Zero-Shot Ablation: Orthogonal Subspace Projection on Qwen Hidden States.

Hypothesis (INSID3-inspired): Qwen's texture-token hidden states suffer
from Global Visual Bias — self-attention pulls them toward the image's
global context, artificially inflating cosine similarity between distinct
textures in the same image (~0.74 measured in V4).

This script proves that the bias lives in a low-rank linear subspace
of the image-patch hidden states, and can be neutralised via
Gram-Schmidt orthogonal projection.

Pipeline:
  1. Baseline:     cos(tex1, tex2) — raw hidden states
  2. Mean Removal: subtract mean(v_img) direction → re-measure
  3. PCA Removal:  subtract PC1, PC2, PC3 of centered v_img → re-measure each

If cosine similarity drops significantly after projection, the bias IS
a linear subspace that a learned projector (or explicit de-biasing layer)
can neutralise.

Usage:
    cd /home/aviad/Qwen2SAM_DeTexture
    python scripts/analyze_visual_bias.py
"""

import sys
import re
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from data.dataset import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


# ===================================================================== #
#  Helpers                                                                #
# ===================================================================== #

def cosine_sim(a, b):
    """Cosine similarity between two 1-D tensors."""
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1).item()


def project_out(v, direction):
    """
    Orthogonal projection: remove the component of v along direction.
    v, direction: (D,) tensors. direction need not be unit-length.
    Returns v_perp = v - (v·d̂)d̂
    """
    d_hat = direction / direction.norm().clamp(min=1e-8)
    return v - (v @ d_hat) * d_hat


def project_out_batch(V, direction):
    """Project each row of V (N, D) away from direction (D,)."""
    d_hat = direction / direction.norm().clamp(min=1e-8)
    dots = V @ d_hat  # (N,)
    return V - dots.unsqueeze(1) * d_hat.unsqueeze(0)


# ===================================================================== #
#  Qwen inference + token-level hidden state extraction                    #
# ===================================================================== #

@torch.no_grad()
def extract_hidden_states(model, processor, image_pil, prompt_text, device):
    """
    Run Qwen forward with a single image and prompt.
    Returns:
        full_hidden: (seq_len, 4096) — last-layer hidden states for all tokens
        input_ids:   (seq_len,)
        generated_text: str
        prompt_len:  int
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt_text},
        ]},
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = processor(
        text=[text], images=[image_pil], return_tensors="pt", padding=True,
    )
    inputs.pop("token_type_ids", None)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    gen_out = model.generate(
        **inputs,
        max_new_tokens=300,
        output_hidden_states=True,
        return_dict_in_generate=True,
        do_sample=False,
    )

    generated_ids = gen_out.sequences[0]  # (total_len,)

    # Collect last-layer hidden states
    gen_hidden_list = []
    for step_hidden in gen_out.hidden_states:
        last_layer = step_hidden[-1][0]  # (seq, dim) for batch 0
        if last_layer.shape[0] > 1:
            last_layer = last_layer[-1:]
        gen_hidden_list.append(last_layer)
    gen_hidden = torch.cat(gen_hidden_list, dim=0)  # (gen_len, dim)

    # We need the prompt hidden states too (for image tokens).
    # Run a separate forward pass on just the prompt to get those.
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        prompt_out = model(**inputs, output_hidden_states=True)
    prompt_hidden = prompt_out.hidden_states[-1][0]  # (prompt_len, dim)

    # Full hidden: prompt + generated
    full_hidden = torch.cat([prompt_hidden, gen_hidden], dim=0).float()

    decoded = processor.tokenizer.decode(
        generated_ids[prompt_len:], skip_special_tokens=False,
    )

    return full_hidden, generated_ids, decoded, prompt_len


def identify_token_regions(input_ids, prompt_len, processor, generated_text):
    """
    Classify tokens into regions:
      - 'image': visual/image patch tokens (in the prompt, before text)
      - 'text_prompt': system + user text tokens
      - 'tex_line': generated TEXTURE_N: line tokens
    Returns dict of region → list of positions.
    """
    tokenizer = processor.tokenizer

    # Image tokens in Qwen3-VL are typically represented as special tokens
    # in the range of visual token IDs. We detect them by checking for
    # tokens that are NOT in the text vocabulary (high IDs or special patterns).
    # Simpler heuristic: image tokens are contiguous block in the prompt
    # between the image placeholder markers.

    # For Qwen3-VL, image tokens have specific IDs. Let's find them by
    # looking for the <|image_pad|> or <|vision_start|>/<|vision_end|> markers.
    regions = {"image": [], "text_prompt": [], "tex_line_1": [], "tex_line_2": []}

    ids = input_ids.tolist()

    # Find vision token range in the prompt
    # Qwen3-VL uses <|vision_start|> and <|vision_end|> markers
    vision_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")

    in_vision = False
    for i in range(prompt_len):
        if ids[i] == vision_start_id:
            in_vision = True
            continue
        if ids[i] == vision_end_id:
            in_vision = False
            continue
        if in_vision:
            regions["image"].append(i)
        else:
            regions["text_prompt"].append(i)

    # Find TEXTURE line positions in generated text
    lines = generated_text.strip().split("\n")
    for line_idx, line in enumerate(lines):
        match = re.match(r"TEXTURE_(\d+):", line.strip())
        if match:
            tex_num = int(match.group(1))
            # Encode text up to end of this line
            text_up_to = "\n".join(lines[:line_idx + 1])
            tokens_up_to = tokenizer.encode(text_up_to, add_special_tokens=False)
            end_pos = prompt_len + min(len(tokens_up_to) - 1,
                                        len(ids) - prompt_len - 1)

            # Encode text up to start of this line
            if line_idx > 0:
                text_before = "\n".join(lines[:line_idx]) + "\n"
                tokens_before = tokenizer.encode(text_before, add_special_tokens=False)
                start_pos = prompt_len + len(tokens_before)
            else:
                start_pos = prompt_len

            key = f"tex_line_{tex_num}" if tex_num <= 2 else None
            if key and key in regions:
                regions[key] = list(range(start_pos, end_pos + 1))

    return regions


# ===================================================================== #
#  Orthogonal Subspace Projection Analysis                                 #
# ===================================================================== #

def analyze_one_sample(model, processor, image_path, device, sample_label):
    """Full analysis pipeline for one image."""
    # Load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (1008, 1008))
    image_pil = Image.fromarray(img)

    prompt = USER_PROMPT_TEMPLATE.format(N="2")

    print(f"\n{'='*70}")
    print(f"  Sample: {sample_label}")
    print(f"  Image:  {image_path}")
    print(f"{'='*70}")

    # Extract hidden states
    full_hidden, input_ids, gen_text, prompt_len = extract_hidden_states(
        model, processor, image_pil, prompt, device,
    )
    D = full_hidden.shape[1]

    # Show generated text
    tex_lines = [l.strip() for l in gen_text.split("\n") if "TEXTURE" in l]
    print(f"\n  Generated ({len(tex_lines)} textures):")
    for l in tex_lines[:4]:
        print(f"    {l[:100]}")

    # Identify token regions
    regions = identify_token_regions(input_ids, prompt_len, processor, gen_text)
    print(f"\n  Token regions:")
    print(f"    image patches: {len(regions['image'])} tokens")
    print(f"    text prompt:   {len(regions['text_prompt'])} tokens")
    print(f"    tex_line_1:    {len(regions['tex_line_1'])} tokens")
    print(f"    tex_line_2:    {len(regions['tex_line_2'])} tokens")

    if len(regions["tex_line_1"]) == 0 or len(regions["tex_line_2"]) == 0:
        print("  SKIP: could not locate both texture line regions")
        return None

    if len(regions["image"]) == 0:
        print("  SKIP: no image tokens found")
        return None

    # Extract vectors
    # v_tex: last token of each texture line (causal LM concentrates info there)
    v_tex1 = full_hidden[regions["tex_line_1"][-1]].float()
    v_tex2 = full_hidden[regions["tex_line_2"][-1]].float()

    # v_img: all image patch hidden states
    img_positions = regions["image"]
    v_img = full_hidden[img_positions].float()  # (N_img, D)

    print(f"\n  v_tex1 norm: {v_tex1.norm():.2f}")
    print(f"  v_tex2 norm: {v_tex2.norm():.2f}")
    print(f"  v_img  shape: {tuple(v_img.shape)}, mean norm: {v_img.norm(dim=1).mean():.2f}")

    # ---- Baseline ---------------------------------------------------- #
    baseline_cos = cosine_sim(v_tex1, v_tex2)

    # ---- Step 1: Mean Removal ---------------------------------------- #
    img_mean = v_img.mean(dim=0)  # (D,)
    t1_mean_rm = project_out(v_tex1, img_mean)
    t2_mean_rm = project_out(v_tex2, img_mean)
    mean_rm_cos = cosine_sim(t1_mean_rm, t2_mean_rm)

    # ---- Step 2: PCA/SVD Removal ------------------------------------- #
    # Center image tokens
    v_img_centered = v_img - img_mean.unsqueeze(0)
    # SVD to find top principal components
    U, S, Vt = torch.linalg.svd(v_img_centered, full_matrices=False)
    # Vt[i] is the i-th principal component direction (D,)
    n_pcs = min(10, Vt.shape[0])

    # Progressively project out PCs from the mean-removed texture vectors
    t1_proj = t1_mean_rm.clone()
    t2_proj = t2_mean_rm.clone()
    pc_results = []

    for pc_idx in range(n_pcs):
        pc_dir = Vt[pc_idx]  # (D,)
        t1_proj = project_out(t1_proj, pc_dir)
        t2_proj = project_out(t2_proj, pc_dir)
        cos_after = cosine_sim(t1_proj, t2_proj)
        variance_explained = (S[pc_idx] ** 2).item()
        pc_results.append({
            "pc": pc_idx + 1,
            "cos": cos_after,
            "var_explained": variance_explained,
            "t1_norm": t1_proj.norm().item(),
            "t2_norm": t2_proj.norm().item(),
        })

    # ---- Print results ------------------------------------------------ #
    print(f"\n  {'Step':<25} {'Cosine Sim':>12} {'Δ from prev':>12} {'Δ from base':>12}")
    print(f"  {'-'*65}")
    print(f"  {'Baseline (raw)':.<25} {baseline_cos:>12.4f} {'—':>12} {'—':>12}")
    print(f"  {'Mean Removal':.<25} {mean_rm_cos:>12.4f} "
          f"{mean_rm_cos - baseline_cos:>+12.4f} "
          f"{mean_rm_cos - baseline_cos:>+12.4f}")

    prev = mean_rm_cos
    for r in pc_results:
        delta_prev = r["cos"] - prev
        delta_base = r["cos"] - baseline_cos
        label = f"+ Remove PC{r['pc']}"
        print(f"  {label:.<25} {r['cos']:>12.4f} {delta_prev:>+12.4f} {delta_base:>+12.4f}")
        prev = r["cos"]

    # Norm retention
    print(f"\n  Norm retention after all {n_pcs} PCs removed:")
    final = pc_results[-1] if pc_results else {}
    print(f"    tex1: {v_tex1.norm():.2f} → {final.get('t1_norm', 0):.2f} "
          f"({final.get('t1_norm', 0) / v_tex1.norm().item() * 100:.1f}%)")
    print(f"    tex2: {v_tex2.norm():.2f} → {final.get('t2_norm', 0):.2f} "
          f"({final.get('t2_norm', 0) / v_tex2.norm().item() * 100:.1f}%)")

    # Singular value spectrum
    print(f"\n  Image token SVD — top 10 singular values:")
    print(f"    {', '.join(f'{s:.1f}' for s in S[:10].tolist())}")
    total_var = (S ** 2).sum().item()
    cum_var = 0
    for i in range(min(10, len(S))):
        cum_var += (S[i] ** 2).item()
        pct = cum_var / total_var * 100
        if i < 5 or i == 9:
            print(f"    Cumulative variance PC1-{i+1}: {pct:.1f}%")

    return {
        "sample": sample_label,
        "baseline_cos": baseline_cos,
        "mean_rm_cos": mean_rm_cos,
        "pc_results": pc_results,
        "n_img_tokens": len(img_positions),
        "tex_lines": tex_lines[:2],
    }


# ===================================================================== #
#  Main                                                                   #
# ===================================================================== #

def main():
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--rwtd-metadata",
                        default="/home/aviad/datasets/RWTD/metadata.json")
    parser.add_argument("--n-samples", type=int, default=5,
                        help="Number of RWTD samples to analyze")
    parser.add_argument("--qwen-model", default="Qwen/Qwen3-VL-8B-Instruct")
    args = parser.parse_args()

    device = torch.device("cuda")

    # Load RWTD metadata for sample images
    with open(args.rwtd_metadata) as f:
        samples = json.load(f)

    # Pick a spread of samples
    indices = list(range(0, len(samples),
                         max(1, len(samples) // args.n_samples)))[:args.n_samples]

    print(f"Loading Qwen: {args.qwen_model}")
    from models.qwen2sam_detexture import load_qwen_processor, load_qwen_model
    processor = load_qwen_processor(args.qwen_model)
    model = load_qwen_model(args.qwen_model, dtype=torch.bfloat16)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    all_results = []

    for idx in indices:
        meta = samples[idx]
        image_path = meta["image_path"]
        result = analyze_one_sample(
            model, processor, image_path, device,
            sample_label=f"RWTD #{idx}",
        )
        if result:
            all_results.append(result)
        torch.cuda.empty_cache()

    # ---- Aggregate summary ------------------------------------------- #
    if all_results:
        print(f"\n{'='*70}")
        print(f"  AGGREGATE SUMMARY ({len(all_results)} samples)")
        print(f"{'='*70}")

        baseline_arr = np.array([r["baseline_cos"] for r in all_results])
        mean_rm_arr = np.array([r["mean_rm_cos"] for r in all_results])

        print(f"\n  {'Metric':<30} {'Mean':>10} {'Std':>10}")
        print(f"  {'-'*50}")
        print(f"  {'Baseline cosine':.<30} {baseline_arr.mean():>10.4f} {baseline_arr.std():>10.4f}")
        print(f"  {'After mean removal':.<30} {mean_rm_arr.mean():>10.4f} {mean_rm_arr.std():>10.4f}")
        print(f"  {'Δ (mean removal)':.<30} {(mean_rm_arr - baseline_arr).mean():>+10.4f}")

        # PC-by-PC aggregate
        max_pcs = min(len(r["pc_results"]) for r in all_results)
        print(f"\n  Progressive PC removal (mean across {len(all_results)} samples):")
        print(f"  {'Step':<20} {'Mean cos':>10} {'Δ from base':>12}")
        print(f"  {'-'*45}")
        print(f"  {'Baseline':.<20} {baseline_arr.mean():>10.4f} {'—':>12}")
        print(f"  {'Mean removal':.<20} {mean_rm_arr.mean():>10.4f} "
              f"{(mean_rm_arr - baseline_arr).mean():>+12.4f}")

        for pc_idx in range(min(max_pcs, 10)):
            cos_arr = np.array([r["pc_results"][pc_idx]["cos"] for r in all_results])
            delta = (cos_arr - baseline_arr).mean()
            print(f"  {'+ PC' + str(pc_idx+1):.<20} {cos_arr.mean():>10.4f} {delta:>+12.4f}")

    # Save
    out_path = Path("ablation") / "visual_bias_analysis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
