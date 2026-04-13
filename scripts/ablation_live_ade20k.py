#!/usr/bin/env python3
"""
Control Group Ablation: Live inference on IN-DOMAIN ADE20K_DeTexture.

Purpose: Determine whether the ep5→ep10 mIoU drop is caused by:
  (A) Domain gap (ADE20K → RWTD)  → in-domain test stays high/climbing
  (B) Engineering bug in inference_forward → in-domain test ALSO crashes

Runs inference_forward (live Qwen generation + regex fallback extraction)
on ADE20K_DeTexture and computes mIoU with Hungarian matching.

Usage:
    python scripts/ablation_live_ade20k.py \
        --checkpoints checkpoints/epoch_5.pt checkpoints/epoch_10.pt checkpoints/best.pt \
        --test-metadata /home/aviad/datasets/ADE20k_DeTexture/metadata.json
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from models.qwen2sam_detexture import Qwen2SAMDeTexture, MAX_TEXTURES
from data.dataset import DeTextureDataset, DeTextureCollator
from training.utils import load_config, load_checkpoint
from scipy.optimize import linear_sum_assignment


def compute_miou_hungarian(pred, gt, k_pred, k_gt):
    """
    mIoU with Hungarian matching — handles k_pred != k_gt.
    pred/gt: (H, W) numpy arrays with class indices.
    """
    if k_pred == 0 or k_gt == 0:
        return 0.0
    cost = np.zeros((k_pred, k_gt))
    for pi in range(k_pred):
        for gi in range(k_gt):
            inter = ((pred == pi + 1) & (gt == gi + 1)).sum()
            union = ((pred == pi + 1) | (gt == gi + 1)).sum()
            cost[pi, gi] = 1.0 - inter / max(union, 1)
    r, c = linear_sum_assignment(cost)
    ious = [1.0 - cost[ri, ci] for ri, ci in zip(r, c) if ri < k_pred and ci < k_gt]
    return np.mean(ious) if ious else 0.0


def evaluate_checkpoint(model, loader, device, ckpt_label):
    """Run inference_forward on all samples and compute metrics."""
    model.eval()
    ious = []
    k_preds_list = []
    n_tex_found = []
    sample_details = []

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            qwen_inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch["qwen_inputs"].items()
            }
            sam_images = batch["sam_images"].to(device)
            index_masks = batch["index_masks"]
            k_gt = int(batch["k_gts"][0].item())

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model.inference_forward(
                    qwen_inputs=qwen_inputs,
                    sam_images=sam_images,
                )

            mask_logits = out["mask_logits"]
            pad_mask = out["pad_mask"]
            k_pred = int(out["k_preds"][0].item())
            gen_text = out.get("generated_text", [""])[0]

            k_preds_list.append(k_pred)
            n_tex_found.append(k_pred)

            # Upsample + argmax
            masked = mask_logits.clone()
            inf_mask = pad_mask.unsqueeze(-1).unsqueeze(-1).expand_as(masked)
            masked[inf_mask] = float("-inf")
            H, W = index_masks.shape[1], index_masks.shape[2]
            if masked.shape[2] != H:
                masked = F.interpolate(
                    masked.float(), size=(H, W),
                    mode="bilinear", align_corners=False,
                )
            pred = masked[0].argmax(dim=0).cpu().numpy()
            gt = index_masks[0].numpy()

            miou = compute_miou_hungarian(pred, gt, k_pred, k_gt)
            ious.append(miou)

            # Channel areas
            total_px = pred.size
            slot_areas = {c: (pred == c).sum() / total_px for c in range(MAX_TEXTURES + 1)}

            if idx < 5 or (idx + 1) % 50 == 0:
                detail = {
                    "idx": idx, "k_gt": k_gt, "k_pred": k_pred, "miou": miou,
                    "slot_areas": {k: f"{v:.1%}" for k, v in slot_areas.items() if v > 0.001},
                    "text_preview": gen_text[:200],
                }
                sample_details.append(detail)

            if (idx + 1) % 50 == 0:
                print(f"    {idx+1}/{len(loader)} avg mIoU={np.mean(ious):.4f}", flush=True)

    mean_iou = np.mean(ious)
    k_dist = Counter(k_preds_list)

    print(f"\n  [{ckpt_label}] RESULTS:")
    print(f"    mIoU:     {mean_iou:.4f}")
    print(f"    n_samples: {len(ious)}")
    print(f"    k_pred distribution: {dict(sorted(k_dist.items()))}")
    print(f"    Samples with mIoU > 0.7: {sum(1 for x in ious if x > 0.7)}/{len(ious)}")
    print(f"    Samples with mIoU < 0.3: {sum(1 for x in ious if x < 0.3)}/{len(ious)}")

    print(f"\n    Sample details:")
    for d in sample_details[:8]:
        print(f"      #{d['idx']} k_gt={d['k_gt']} k_pred={d['k_pred']} "
              f"mIoU={d['miou']:.4f} slots={d['slot_areas']}")
        lines = [l.strip() for l in d["text_preview"].split("\n") if "TEXTURE" in l]
        for l in lines[:3]:
            print(f"        {l[:100]}")

    return {
        "miou": float(mean_iou),
        "n_samples": len(ious),
        "k_pred_distribution": dict(sorted(k_dist.items())),
        "per_sample_ious": [float(x) for x in ious],
        "sample_details": sample_details,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/detexture.yaml")
    parser.add_argument("--test-metadata",
                        default="/home/aviad/datasets/ADE20k_DeTexture/metadata.json")
    parser.add_argument("--checkpoints", nargs="+", default=[
        "checkpoints/epoch_5.pt",
        "checkpoints/epoch_10.pt",
        "checkpoints/best.pt",
    ])
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda")

    # ---- Build model once, swap checkpoints ----------------------------- #
    print("Building model...")
    model = Qwen2SAMDeTexture(cfg, device="cuda")

    # ---- Dataset -------------------------------------------------------- #
    test_ds = DeTextureDataset(
        args.test_metadata,
        image_size=cfg["data"].get("image_size", 1008),
        augment=False,
        qwen_gt_embeds_path=cfg["data"].get("qwen_gt_embeds_path"),
    )
    collator = DeTextureCollator(model.processor, inference=True)
    loader = torch.utils.data.DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collator,
    )
    print(f"Test dataset: {args.test_metadata} ({len(test_ds)} samples)")

    # ---- Evaluate each checkpoint --------------------------------------- #
    all_results = {}
    for ckpt_path in args.checkpoints:
        ckpt_path = str(ckpt_path)
        if not Path(ckpt_path).exists():
            print(f"\n  SKIP: {ckpt_path} not found")
            continue

        ckpt_label = Path(ckpt_path).stem
        print(f"\n{'='*70}")
        print(f"  Loading checkpoint: {ckpt_path}")
        print(f"{'='*70}")
        load_checkpoint(model, None, ckpt_path, device="cuda")
        model.eval()

        results = evaluate_checkpoint(model, loader, device, ckpt_label)
        all_results[ckpt_label] = results

    # ---- Side-by-side summary ------------------------------------------- #
    print(f"\n{'='*70}")
    print(f"  CONTROL GROUP ABLATION: Live ADE20K_DeTexture")
    print(f"{'='*70}")
    print(f"\n  {'Checkpoint':<20} {'mIoU':>8}  {'n':>5}  {'k_pred dist':<35}")
    print(f"  {'-'*70}")
    for label, r in all_results.items():
        kdist = " ".join(f"{k}:{v}" for k, v in sorted(r["k_pred_distribution"].items()))
        print(f"  {label:<20} {r['miou']:>8.4f}  {r['n_samples']:>5}  {kdist}")

    # ---- Compare with RWTD test results from run 2 ---------------------- #
    print(f"\n  COMPARISON: same checkpoints on RWTD (from training log):")
    print(f"  {'epoch_5':<20} RWTD test mIoU=0.6921")
    print(f"  {'epoch_10':<20} RWTD test mIoU=0.6177")
    print()
    print(f"  INTERPRETATION:")
    print(f"  If ADE20K_DeTexture mIoU stays high → Domain Gap (RWTD breaks projector)")
    print(f"  If ADE20K_DeTexture mIoU ALSO crashes → Engineering Bug in inference_forward")

    # ---- Save results --------------------------------------------------- #
    out_path = Path("ablation") / "live_ade20k_control.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
