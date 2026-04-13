#!/usr/bin/env python3
"""
Analytical test: measure cosine similarity between TEXTURE_1 and TEXTURE_2
embeddings at two levels to prove or disprove Representation Collapse.

Measures:
  1. Pre-Projector  (Qwen 4096-D):  cos(tex_embed[0], tex_embed[1])
  2. Post-Projector  (SAM 256-D):   cos(proj(tex_embed[0]), proj(tex_embed[1]))

If the 4096-D similarity is low (Qwen distinguishes) but 256-D similarity
spikes to ~0.95+ on RWTD ep10 (while staying lower on ADE20K and ep5),
then Representation Collapse through the projector is proven.

Conditions:
  A. RWTD @ epoch_5   (cross-domain, early)
  B. RWTD @ epoch_10  (cross-domain, drifted)
  C. ADE20K_DeTexture @ epoch_10  (in-domain control)

Usage:
    python scripts/analyze_vector_collapse.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from models.qwen2sam_detexture import Qwen2SAMDeTexture, MAX_TEXTURES
from data.dataset import DeTextureDataset, DeTextureCollator
from training.utils import load_config, load_checkpoint


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two 1-D tensors."""
    return F.cosine_similarity(
        a.float().unsqueeze(0), b.float().unsqueeze(0), dim=-1
    ).item()


@torch.no_grad()
def analyze_condition(model, loader, device, label):
    """
    Run inference_forward on each sample. For every sample with k_pred >= 2,
    compute cosine similarity between TEXTURE_1 and TEXTURE_2:
      - Pre-projector:  in Qwen's 4096-D space
      - Post-projector: in SAM's 256-D space (after trunk + frozen plug)

    Also computes per-slot norms to check for magnitude collapse.
    """
    model.eval()
    pre_sims = []      # cosine similarity in 4096-D
    post_sims = []     # cosine similarity in 256-D
    pre_norms = []     # (norm_t1, norm_t2) in 4096-D
    post_norms = []    # (norm_t1, norm_t2) in 256-D
    per_sample = []    # detailed per-sample records

    for idx, batch in enumerate(loader):
        qwen_inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch["qwen_inputs"].items()
        }
        sam_images = batch["sam_images"].to(device)
        k_gt = int(batch["k_gts"][0].item())

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model.inference_forward(
                qwen_inputs=qwen_inputs, sam_images=sam_images,
            )

        k_pred = int(out["k_preds"][0].item())
        tex_embeds = out["tex_embeds"]  # (1, MAX_TEXTURES, 4096)

        if k_pred < 2:
            continue

        # Extract TEXTURE_1 and TEXTURE_2 raw hidden states (4096-D)
        t1_4096 = tex_embeds[0, 0]  # (4096,)
        t2_4096 = tex_embeds[0, 1]  # (4096,)

        # Pre-projector cosine similarity
        pre_cos = cosine_sim(t1_4096, t2_4096)
        pre_sims.append(pre_cos)
        pre_norms.append((t1_4096.float().norm().item(),
                          t2_4096.float().norm().item()))

        # Pass through projector (trunk + frozen SAM plug)
        # Projector is pointwise — can pass (2, 4096) directly
        stacked = torch.stack([t1_4096, t2_4096], dim=0)  # (2, 4096)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            projected = model.projector(stacked.unsqueeze(0))  # (1, 2, 256)
        t1_256 = projected[0, 0]  # (256,)
        t2_256 = projected[0, 1]  # (256,)

        # Post-projector cosine similarity
        post_cos = cosine_sim(t1_256, t2_256)
        post_sims.append(post_cos)
        post_norms.append((t1_256.float().norm().item(),
                           t2_256.float().norm().item()))

        per_sample.append({
            "idx": idx,
            "k_gt": k_gt,
            "k_pred": k_pred,
            "pre_cos": pre_cos,
            "post_cos": post_cos,
            "pre_norm_t1": pre_norms[-1][0],
            "pre_norm_t2": pre_norms[-1][1],
            "post_norm_t1": post_norms[-1][0],
            "post_norm_t2": post_norms[-1][1],
            "text_t1": out.get("generated_text", [""])[0].split("\n")[0][:100],
        })

        if (idx + 1) % 50 == 0:
            print(f"    {idx+1}/{len(loader)}", flush=True)

    # ---- Statistics -------------------------------------------------- #
    pre_arr = np.array(pre_sims)
    post_arr = np.array(post_sims)

    print(f"\n  [{label}]  ({len(pre_sims)} samples with k_pred >= 2)")
    print(f"  {'':>25} {'Pre (4096-D)':>15} {'Post (256-D)':>15} {'Δ':>10}")
    print(f"  {'-'*65}")
    print(f"  {'Mean cosine sim':>25} {pre_arr.mean():>15.4f} {post_arr.mean():>15.4f} {post_arr.mean()-pre_arr.mean():>+10.4f}")
    print(f"  {'Median':>25} {np.median(pre_arr):>15.4f} {np.median(post_arr):>15.4f}")
    print(f"  {'Std':>25} {pre_arr.std():>15.4f} {post_arr.std():>15.4f}")
    print(f"  {'Min':>25} {pre_arr.min():>15.4f} {post_arr.min():>15.4f}")
    print(f"  {'Max':>25} {pre_arr.max():>15.4f} {post_arr.max():>15.4f}")

    # Fraction above thresholds
    for thresh in [0.8, 0.9, 0.95, 0.99]:
        pre_pct = (pre_arr > thresh).mean() * 100
        post_pct = (post_arr > thresh).mean() * 100
        print(f"  {'> ' + str(thresh):>25} {pre_pct:>14.1f}% {post_pct:>14.1f}%")

    # Norm statistics
    pre_n = np.array(pre_norms)
    post_n = np.array(post_norms)
    print(f"\n  {'Mean norm T1':>25} {pre_n[:,0].mean():>15.2f} {post_n[:,0].mean():>15.4f}")
    print(f"  {'Mean norm T2':>25} {pre_n[:,1].mean():>15.2f} {post_n[:,1].mean():>15.4f}")
    print(f"  {'Norm ratio T2/T1':>25} {(pre_n[:,1]/pre_n[:,0].clip(1e-6)).mean():>15.4f} {(post_n[:,1]/post_n[:,0].clip(1e-6)).mean():>15.4f}")

    # Show most collapsed samples (highest post-projector similarity)
    sorted_samples = sorted(per_sample, key=lambda x: -x["post_cos"])
    print(f"\n  Top 5 MOST COLLAPSED (highest post-projector cos sim):")
    for s in sorted_samples[:5]:
        print(f"    #{s['idx']} pre={s['pre_cos']:.4f} post={s['post_cos']:.4f} "
              f"k_pred={s['k_pred']} | {s['text_t1']}")

    # Show most separated samples
    print(f"\n  Top 5 MOST SEPARATED (lowest post-projector cos sim):")
    for s in sorted_samples[-5:]:
        print(f"    #{s['idx']} pre={s['pre_cos']:.4f} post={s['post_cos']:.4f} "
              f"k_pred={s['k_pred']} | {s['text_t1']}")

    return {
        "label": label,
        "n_samples": len(pre_sims),
        "pre_cos_mean": float(pre_arr.mean()),
        "pre_cos_std": float(pre_arr.std()),
        "post_cos_mean": float(post_arr.mean()),
        "post_cos_std": float(post_arr.std()),
        "post_above_0.95": float((post_arr > 0.95).mean()),
        "per_sample": per_sample,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/detexture.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda")

    # Define the 3 conditions
    conditions = [
        ("checkpoints/epoch_5.pt",  "/home/aviad/datasets/RWTD/metadata.json",
         "RWTD_ep5"),
        ("checkpoints/epoch_10.pt", "/home/aviad/datasets/RWTD/metadata.json",
         "RWTD_ep10"),
        ("checkpoints/epoch_10.pt", "/home/aviad/datasets/ADE20k_DeTexture/metadata.json",
         "ADE20K_ep10"),
    ]

    print("Building model...")
    model = Qwen2SAMDeTexture(cfg, device="cuda")

    all_results = {}
    prev_ckpt = None

    for ckpt_path, metadata_path, label in conditions:
        if not Path(ckpt_path).exists():
            print(f"\n  SKIP: {ckpt_path} not found")
            continue

        print(f"\n{'='*70}")
        print(f"  Condition: {label}")
        print(f"  Checkpoint: {ckpt_path}")
        print(f"  Dataset: {metadata_path}")
        print(f"{'='*70}")

        # Only reload checkpoint if different from previous
        if ckpt_path != prev_ckpt:
            load_checkpoint(model, None, ckpt_path, device="cuda")
            prev_ckpt = ckpt_path

        ds = DeTextureDataset(
            metadata_path, image_size=1008, augment=False,
            qwen_gt_embeds_path=cfg["data"].get("qwen_gt_embeds_path"),
        )
        collator = DeTextureCollator(model.processor, inference=True)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collator,
        )

        result = analyze_condition(model, loader, device, label)
        all_results[label] = result

    # ---- Cross-condition comparison ---------------------------------- #
    print(f"\n{'='*70}")
    print(f"  CROSS-CONDITION COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Condition':<20} {'n':>5} {'Pre 4096-D':>12} {'Post 256-D':>12} {'Δ':>8} {'>0.95':>8}")
    print(f"  {'-'*65}")
    for label, r in all_results.items():
        delta = r["post_cos_mean"] - r["pre_cos_mean"]
        pct95 = r["post_above_0.95"] * 100
        print(f"  {label:<20} {r['n_samples']:>5} {r['pre_cos_mean']:>12.4f} "
              f"{r['post_cos_mean']:>12.4f} {delta:>+8.4f} {pct95:>7.1f}%")

    print(f"\n  INTERPRETATION:")
    print(f"  If Post-256D spikes on RWTD_ep10 vs RWTD_ep5 → Projector Collapse")
    print(f"  If Post-256D similar across all → Projector is fine, issue elsewhere")
    print(f"  If Pre-4096D already high → Qwen doesn't distinguish textures")

    # Save
    out_path = Path("ablation") / "vector_collapse_analysis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
