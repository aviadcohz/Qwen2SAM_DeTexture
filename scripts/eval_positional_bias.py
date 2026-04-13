#!/usr/bin/env python3
"""
Positional Bias Test for SAM3 cross-attention.

Tests whether SAM's query position affects mask allocation by running
3 conditions on RWTD (253 samples, 2 textures each):

  A) Baseline:   [Dustbin, T1, T2, PAD, PAD, PAD, PAD]
  B) Swapped:    [Dustbin, T2, T1, PAD, PAD, PAD, PAD]
  C) Isolated:   Run SAM twice: [Dustbin, T1, PAD...] and [Dustbin, T2, PAD...]
                  then combine logits.

If positional bias exists, Condition B will show T2 (now in slot 1)
dominating, and Condition C (no position competition) will have
more balanced channel allocation.

Usage:
    cd /home/aviad/Qwen2SAM_DeTexture
    python scripts/eval_positional_bias.py --checkpoint checkpoints/epoch_10.pt
"""

import argparse
import json
import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from models.qwen2sam_detexture import (
    Qwen2SAMDeTexture, MAX_TEXTURES, NUM_QUERY_SLOTS,
)
from data.dataset import DeTextureDataset, DeTextureCollator
from training.utils import load_config, load_checkpoint
from training.monitor import _compute_ari
from scipy.optimize import linear_sum_assignment


def compute_miou(pred, gt, k_pred, k_gt):
    cost = np.zeros((max(k_pred, 1), max(k_gt, 1)))
    for pi in range(k_pred):
        for gi in range(k_gt):
            inter = ((pred == pi + 1) & (gt == gi + 1)).sum()
            union = ((pred == pi + 1) | (gt == gi + 1)).sum()
            cost[pi, gi] = 1.0 - inter / max(union, 1)
    r, c = linear_sum_assignment(cost)
    ious = [1.0 - cost[ri, ci] for ri, ci in zip(r, c) if ri < k_pred and ci < k_gt]
    return np.mean(ious) if ious else 0.0


def run_sam_with_embeds(model, sam_images, tex_embeds, k):
    """
    Run SAM3 semantic path with given texture embeddings.

    Args:
        model: Qwen2SAMDeTexture model
        sam_images: (1, 3, 1008, 1008)
        tex_embeds: (1, MAX_TEXTURES, 4096) — texture embeddings to use
        k: number of active textures

    Returns:
        mask_logits: (1, NUM_QUERY_SLOTS, H, W)
        pad_mask: (1, NUM_QUERY_SLOTS)
    """
    B = 1
    k_preds = torch.tensor([k], dtype=torch.long, device=tex_embeds.device)

    # Build query slots
    query_embeds, pad_mask = model.build_query_slots(tex_embeds, k_preds)

    # Project to SAM space
    query_256 = model.projector(query_embeds)

    # SAM3 backbone
    with torch.no_grad():
        backbone_out = model.sam3.backbone.forward_image(sam_images)
        backbone_out["img_batch_all_stages"] = sam_images

    # SAM3 semantic path
    mask_logits = model.run_sam3_semantic(backbone_out, query_256, pad_mask)

    return mask_logits, pad_mask


def get_channel_areas(mask_logits, pad_mask):
    """Compute pixel allocation percentage per channel."""
    masked = mask_logits.clone()
    inf_mask = pad_mask.unsqueeze(-1).unsqueeze(-1).expand_as(masked)
    masked[inf_mask] = float("-inf")
    preds = masked[0].argmax(dim=0)  # (H, W)
    total = preds.numel()
    areas = {}
    for c in range(mask_logits.shape[1]):
        areas[c] = (preds == c).sum().item() / total
    return areas, preds.cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/detexture.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/epoch_10.pt")
    parser.add_argument("--test-metadata", default="/home/aviad/datasets/RWTD/metadata.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda")

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = Qwen2SAMDeTexture(cfg, device="cuda")
    load_checkpoint(model, None, args.checkpoint, device="cuda")
    model.eval()

    # Load GT embeddings for RWTD
    gt_embeds_path = cfg["data"].get("qwen_gt_embeds_path")
    gt_embeds_dict = {}
    if gt_embeds_path and Path(gt_embeds_path).exists():
        gt_embeds_dict = torch.load(gt_embeds_path, map_location="cpu")
        print(f"Loaded {len(gt_embeds_dict)} GT embeddings")

    # Load test data
    test_ds = DeTextureDataset(
        args.test_metadata, image_size=1008, augment=False,
        qwen_gt_embeds_path=gt_embeds_path,
    )
    collator = DeTextureCollator(model.processor, inference=True)
    loader = torch.utils.data.DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collator,
    )

    # Load GT descriptions
    with open(args.test_metadata) as f:
        gt_data = json.load(f)

    # Results storage
    results_a, results_b, results_c = [], [], []

    print(f"\nRunning positional bias test on {len(test_ds)} samples...")
    print(f"Each sample tested under 3 conditions: A(baseline), B(swapped), C(isolated)\n")

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            sam_images = batch["sam_images"].to(device)
            index_masks = batch["index_masks"]
            k_gt = int(batch["k_gts"][0].item())
            qwen_gt = batch["qwen_gt_embeds"].to(device)  # (1, MAX, 4096)

            gt = index_masks[0].numpy()
            H_gt, W_gt = gt.shape

            # Only test samples with exactly 2 textures
            if k_gt != 2:
                continue

            T1 = qwen_gt[0, 0:1]  # (1, 4096)
            T2 = qwen_gt[0, 1:2]  # (1, 4096)

            # ---- Condition A: Baseline [Dustbin, T1, T2, PAD...] ----
            embeds_a = torch.zeros(1, MAX_TEXTURES, model.llm_dim, device=device)
            embeds_a[0, 0] = T1
            embeds_a[0, 1] = T2
            logits_a, pad_a = run_sam_with_embeds(model, sam_images, embeds_a, k=2)
            if logits_a.shape[2] != H_gt:
                logits_a = F.interpolate(logits_a.float(), size=(H_gt, W_gt),
                                          mode="bilinear", align_corners=False)
            areas_a, pred_a = get_channel_areas(logits_a, pad_a)
            miou_a = compute_miou(pred_a, gt, 2, 2)

            # ---- Condition B: Swapped [Dustbin, T2, T1, PAD...] ----
            embeds_b = torch.zeros(1, MAX_TEXTURES, model.llm_dim, device=device)
            embeds_b[0, 0] = T2
            embeds_b[0, 1] = T1
            logits_b, pad_b = run_sam_with_embeds(model, sam_images, embeds_b, k=2)
            if logits_b.shape[2] != H_gt:
                logits_b = F.interpolate(logits_b.float(), size=(H_gt, W_gt),
                                          mode="bilinear", align_corners=False)
            areas_b, pred_b = get_channel_areas(logits_b, pad_b)
            miou_b = compute_miou(pred_b, gt, 2, 2)

            # ---- Condition C: Isolated (run SAM twice, combine) ----
            # Run with T1 only
            embeds_c1 = torch.zeros(1, MAX_TEXTURES, model.llm_dim, device=device)
            embeds_c1[0, 0] = T1
            logits_c1, pad_c1 = run_sam_with_embeds(model, sam_images, embeds_c1, k=1)
            if logits_c1.shape[2] != H_gt:
                logits_c1 = F.interpolate(logits_c1.float(), size=(H_gt, W_gt),
                                           mode="bilinear", align_corners=False)

            # Run with T2 only
            embeds_c2 = torch.zeros(1, MAX_TEXTURES, model.llm_dim, device=device)
            embeds_c2[0, 0] = T2
            logits_c2, pad_c2 = run_sam_with_embeds(model, sam_images, embeds_c2, k=1)
            if logits_c2.shape[2] != H_gt:
                logits_c2 = F.interpolate(logits_c2.float(), size=(H_gt, W_gt),
                                           mode="bilinear", align_corners=False)

            # Combine: build a 3-channel logit [dustbin, T1_score, T2_score]
            # Use channel 1 logits from each isolated run
            combined = torch.zeros(1, 3, H_gt, W_gt, device=device)
            combined[0, 0] = torch.min(logits_c1[0, 0], logits_c2[0, 0])  # dustbin = min of both
            combined[0, 1] = logits_c1[0, 1]  # T1's texture channel
            combined[0, 2] = logits_c2[0, 1]  # T2's texture channel
            pred_c = combined[0].argmax(dim=0).cpu().numpy()
            total_px = pred_c.size
            areas_c = {
                0: (pred_c == 0).sum() / total_px,
                1: (pred_c == 1).sum() / total_px,
                2: (pred_c == 2).sum() / total_px,
            }
            miou_c = compute_miou(pred_c, gt, 2, 2)

            results_a.append({"idx": idx, "miou": miou_a, "slot1": areas_a.get(1, 0), "slot2": areas_a.get(2, 0), "dustbin": areas_a.get(0, 0)})
            results_b.append({"idx": idx, "miou": miou_b, "slot1": areas_b.get(1, 0), "slot2": areas_b.get(2, 0), "dustbin": areas_b.get(0, 0)})
            results_c.append({"idx": idx, "miou": miou_c, "slot1": areas_c.get(1, 0), "slot2": areas_c.get(2, 0), "dustbin": areas_c.get(0, 0)})

            if (idx + 1) % 50 == 0:
                print(f"  {idx+1}/{len(test_ds)}", flush=True)

    # ---- Summary ----
    n = len(results_a)
    print(f"\n{'='*70}")
    print(f"  POSITIONAL BIAS TEST RESULTS ({n} samples, all k_gt=2)")
    print(f"{'='*70}")
    print(f"{'Condition':<25} | {'mIoU':>7} | {'Slot1 %':>8} | {'Slot2 %':>8} | {'Dustbin %':>10}")
    print("-" * 70)

    for name, results in [("A: Baseline [T1,T2]", results_a),
                           ("B: Swapped [T2,T1]", results_b),
                           ("C: Isolated (2 runs)", results_c)]:
        avg_miou = np.mean([r["miou"] for r in results])
        avg_s1 = np.mean([r["slot1"] for r in results])
        avg_s2 = np.mean([r["slot2"] for r in results])
        avg_dust = np.mean([r["dustbin"] for r in results])
        print(f"{name:<25} | {avg_miou:>7.4f} | {avg_s1:>7.1%} | {avg_s2:>7.1%} | {avg_dust:>9.1%}")

    print()
    print("Interpretation:")
    avg_a_s1 = np.mean([r["slot1"] for r in results_a])
    avg_a_s2 = np.mean([r["slot2"] for r in results_a])
    avg_b_s1 = np.mean([r["slot1"] for r in results_b])
    avg_b_s2 = np.mean([r["slot2"] for r in results_b])
    avg_c_s1 = np.mean([r["slot1"] for r in results_c])
    avg_c_s2 = np.mean([r["slot2"] for r in results_c])

    bias_ratio_a = avg_a_s1 / max(avg_a_s2, 0.001)
    bias_ratio_b = avg_b_s1 / max(avg_b_s2, 0.001)

    if bias_ratio_a > 2.0 and bias_ratio_b > 2.0:
        print("  POSITIONAL BIAS CONFIRMED: Slot 1 dominates regardless of content.")
        print(f"  Baseline: slot1/slot2 = {bias_ratio_a:.1f}x")
        print(f"  Swapped:  slot1/slot2 = {bias_ratio_b:.1f}x (T2 now dominates)")
    elif bias_ratio_a > 2.0 and bias_ratio_b < 1.5:
        print("  CONTENT BIAS (not positional): T1 content inherently stronger than T2.")
        print(f"  Baseline: slot1/slot2 = {bias_ratio_a:.1f}x (T1 in slot 1)")
        print(f"  Swapped:  slot1/slot2 = {bias_ratio_b:.1f}x (T2 in slot 1, less dominant)")
    else:
        print("  NO CLEAR POSITIONAL BIAS: Channels are reasonably balanced.")
        print(f"  Baseline: slot1/slot2 = {bias_ratio_a:.1f}x")
        print(f"  Swapped:  slot1/slot2 = {bias_ratio_b:.1f}x")

    balance_c = min(avg_c_s1, avg_c_s2) / max(avg_c_s1, avg_c_s2, 0.001)
    print(f"\n  Isolated run balance: {balance_c:.1%} (1.0 = perfectly balanced)")
    miou_gain_c = np.mean([r["miou"] for r in results_c]) - np.mean([r["miou"] for r in results_a])
    print(f"  Isolated mIoU gain over baseline: {miou_gain_c:+.4f}")

    # Save detailed results
    output = {
        "checkpoint": args.checkpoint,
        "n_samples": n,
        "condition_A": {"avg_miou": float(np.mean([r["miou"] for r in results_a])),
                        "avg_slot1": float(avg_a_s1), "avg_slot2": float(avg_a_s2)},
        "condition_B": {"avg_miou": float(np.mean([r["miou"] for r in results_b])),
                        "avg_slot1": float(avg_b_s1), "avg_slot2": float(avg_b_s2)},
        "condition_C": {"avg_miou": float(np.mean([r["miou"] for r in results_c])),
                        "avg_slot1": float(avg_c_s1), "avg_slot2": float(avg_c_s2)},
        "per_sample_A": results_a,
        "per_sample_B": results_b,
        "per_sample_C": results_c,
    }
    out_path = Path("ablation") / "positional_bias_test.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
