#!/usr/bin/env python3
"""
V3 Diagnostic: 3 surgical experiments on epoch 15 checkpoint.

Experiment 1: GT Bypass Test (Qwen vs Vision isolation)
Experiment 2: Over-generation Check (k_pred distribution + sample texts)
Experiment 3: Positional Bias Re-test (verify Batch Multiplexing works)

Usage:
    python scripts/v3_diagnosis.py --checkpoint checkpoints/epoch_15.pt
"""

import argparse
import json
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


def get_channel_areas(mask_logits, pad_mask):
    masked = mask_logits.clone()
    inf_mask = pad_mask.unsqueeze(-1).unsqueeze(-1).expand_as(masked)
    masked[inf_mask] = float("-inf")
    preds = masked[0].argmax(dim=0)
    total = preds.numel()
    areas = {}
    for c in range(mask_logits.shape[1]):
        areas[c] = (preds == c).sum().item() / total
    return areas, preds.cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/detexture.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/epoch_15.pt")
    parser.add_argument("--test-metadata", default="/home/aviad/datasets/RWTD/metadata.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda")

    print(f"Loading model from {args.checkpoint}...")
    model = Qwen2SAMDeTexture(cfg, device="cuda")
    load_checkpoint(model, None, args.checkpoint, device="cuda")
    model.eval()

    test_ds = DeTextureDataset(
        args.test_metadata, image_size=1008, augment=False,
        qwen_gt_embeds_path=cfg["data"].get("qwen_gt_embeds_path"),
    )
    collator = DeTextureCollator(model.processor, inference=True)
    loader = torch.utils.data.DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collator,
    )

    # =================================================================== #
    #  Experiment 1: GT Bypass Test                                         #
    # =================================================================== #
    print("\n" + "=" * 70)
    print("  EXPERIMENT 1: GT Bypass Test (Vision vs LLM Isolation)")
    print("=" * 70)
    print("Running RWTD inference with pre-computed GT embeddings (bypassing Qwen)...")

    ious_gt = []
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            sam_images = batch["sam_images"].to(device)
            index_masks = batch["index_masks"]
            k_gt = int(batch["k_gts"][0].item())
            qwen_gt = batch["qwen_gt_embeds"].to(device)

            # Forward with override_tex_embeds (Phase 1 style)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(
                    qwen_inputs=None,
                    sam_images=sam_images,
                    seg_grad_to_lm=False,
                    override_tex_embeds=qwen_gt,
                )

            mask_logits = out["mask_logits"]
            pad_mask = out["pad_mask"]
            k_pred = int(out["k_preds"][0].item())

            masked = mask_logits.clone()
            inf_mask = pad_mask.unsqueeze(-1).unsqueeze(-1).expand_as(masked)
            masked[inf_mask] = float("-inf")
            H, W = index_masks.shape[1], index_masks.shape[2]
            if masked.shape[2] != H:
                masked = F.interpolate(masked.float(), size=(H, W),
                                        mode="bilinear", align_corners=False)
            pred = masked[0].argmax(dim=0).cpu().numpy()
            gt = index_masks[0].numpy()

            ious_gt.append(compute_miou(pred, gt, k_pred, k_gt))

            if (idx + 1) % 50 == 0:
                print(f"  {idx+1}/{len(test_ds)}", flush=True)

    miou_gt = np.mean(ious_gt)
    print(f"\n  RESULT: mIoU with GT embeddings (Qwen bypass) = {miou_gt:.4f}")
    print(f"  Baseline (Phase 1 @ ep 10): 0.6825")
    print(f"  Current Phase 2 @ ep 15:    0.5883")
    if miou_gt > 0.65:
        print(f"  → Vision path WORKS. Qwen is the bottleneck (overfitting).")
    else:
        print(f"  → Vision path BROKEN. V3 or projector regressed.")

    # =================================================================== #
    #  Experiment 2: Over-generation Check                                  #
    # =================================================================== #
    print("\n" + "=" * 70)
    print("  EXPERIMENT 2: Over-generation Check (Live Inference)")
    print("=" * 70)
    print("Running RWTD with LIVE Qwen generation...")

    k_preds_list = []
    sample_texts = []
    ious_live = []

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            qwen_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                           for k, v in batch["qwen_inputs"].items()}
            sam_images = batch["sam_images"].to(device)
            index_masks = batch["index_masks"]
            k_gt = int(batch["k_gts"][0].item())

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model.inference_forward(qwen_inputs=qwen_inputs, sam_images=sam_images)

            k_pred = int(out["k_preds"][0].item())
            k_preds_list.append(k_pred)
            gen_text = out.get("generated_text", [""])[0]

            if idx in [0, 50, 100, 150, 200]:
                sample_texts.append({"idx": idx, "k_gt": k_gt, "k_pred": k_pred, "text": gen_text[:500]})

            mask_logits = out["mask_logits"]
            pad_mask = out["pad_mask"]
            masked = mask_logits.clone()
            inf_mask = pad_mask.unsqueeze(-1).unsqueeze(-1).expand_as(masked)
            masked[inf_mask] = float("-inf")
            H, W = index_masks.shape[1], index_masks.shape[2]
            if masked.shape[2] != H:
                masked = F.interpolate(masked.float(), size=(H, W),
                                        mode="bilinear", align_corners=False)
            pred = masked[0].argmax(dim=0).cpu().numpy()
            gt = index_masks[0].numpy()
            ious_live.append(compute_miou(pred, gt, k_pred, k_gt))

            if (idx + 1) % 50 == 0:
                print(f"  {idx+1}/{len(test_ds)}", flush=True)

    miou_live = np.mean(ious_live)
    k_dist = Counter(k_preds_list)
    print(f"\n  RESULT: mIoU with live Qwen = {miou_live:.4f}")
    print(f"  k_pred distribution (all RWTD samples have k_gt=2):")
    for k in sorted(k_dist.keys()):
        print(f"    k_pred={k}: {k_dist[k]:>3} samples ({k_dist[k]/len(k_preds_list)*100:.0f}%)")

    print(f"\n  Sample generated descriptions:")
    for s in sample_texts:
        print(f"  --- Sample #{s['idx']} (k_gt={s['k_gt']}, k_pred={s['k_pred']}) ---")
        lines = [l.strip() for l in s['text'].split('\n') if 'TEXTURE' in l]
        for l in lines[:4]:
            print(f"    {l[:120]}")

    # =================================================================== #
    #  Experiment 3: Positional Bias Re-test                                #
    # =================================================================== #
    print("\n" + "=" * 70)
    print("  EXPERIMENT 3: Positional Bias Re-test (V3 Batch Multiplexing)")
    print("=" * 70)
    print("Testing channel allocation with GT embeddings...")

    areas_a = []  # Baseline [T1, T2]
    areas_b = []  # Swapped [T2, T1]
    ious_a, ious_b = [], []

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            sam_images = batch["sam_images"].to(device)
            index_masks = batch["index_masks"]
            k_gt = int(batch["k_gts"][0].item())
            qwen_gt = batch["qwen_gt_embeds"].to(device)

            if k_gt != 2:
                continue

            T1 = qwen_gt[0, 0:1]
            T2 = qwen_gt[0, 1:2]

            # Condition A: [T1, T2]
            embeds_a = torch.zeros(1, MAX_TEXTURES, model.llm_dim, device=device)
            embeds_a[0, 0] = T1
            embeds_a[0, 1] = T2
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out_a = model(
                    qwen_inputs=None, sam_images=sam_images,
                    seg_grad_to_lm=False, override_tex_embeds=embeds_a,
                )
            ml_a = out_a["mask_logits"]
            H, W = index_masks.shape[1], index_masks.shape[2]
            if ml_a.shape[2] != H:
                ml_a = F.interpolate(ml_a.float(), size=(H, W), mode="bilinear", align_corners=False)
            area_a, pred_a = get_channel_areas(ml_a, out_a["pad_mask"])
            areas_a.append(area_a)
            ious_a.append(compute_miou(pred_a, index_masks[0].numpy(), 2, 2))

            # Condition B: [T2, T1]
            embeds_b = torch.zeros(1, MAX_TEXTURES, model.llm_dim, device=device)
            embeds_b[0, 0] = T2
            embeds_b[0, 1] = T1
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out_b = model(
                    qwen_inputs=None, sam_images=sam_images,
                    seg_grad_to_lm=False, override_tex_embeds=embeds_b,
                )
            ml_b = out_b["mask_logits"]
            if ml_b.shape[2] != H:
                ml_b = F.interpolate(ml_b.float(), size=(H, W), mode="bilinear", align_corners=False)
            area_b, pred_b = get_channel_areas(ml_b, out_b["pad_mask"])
            areas_b.append(area_b)
            ious_b.append(compute_miou(pred_b, index_masks[0].numpy(), 2, 2))

            if (idx + 1) % 50 == 0:
                print(f"  {idx+1}/{len(test_ds)}", flush=True)

    def avg(key, areas_list):
        return np.mean([a.get(key, 0) for a in areas_list])

    print(f"\n  RESULT ({len(areas_a)} samples with k_gt=2):")
    print(f"  {'Condition':<20} | {'mIoU':>7} | {'Slot0':>7} | {'Slot1':>7} | {'Slot2':>7} | {'Slot3+':>7}")
    print("  " + "-" * 68)

    for name, areas, ious in [("A: [T1, T2]", areas_a, ious_a), ("B: [T2, T1]", areas_b, ious_b)]:
        s0 = avg(0, areas)
        s1 = avg(1, areas)
        s2 = avg(2, areas)
        s_rest = sum(avg(k, areas) for k in range(3, MAX_TEXTURES + 1))
        mIoU = np.mean(ious)
        print(f"  {name:<20} | {mIoU:>7.4f} | {s0:>6.1%} | {s1:>6.1%} | {s2:>6.1%} | {s_rest:>6.1%}")

    s2_avg = avg(2, areas_a)
    print(f"\n  V2 ep10 Slot 2 allocation: 0.0%")
    print(f"  V3 ep15 Slot 2 allocation: {s2_avg:.1%}")
    if s2_avg > 0.2:
        print(f"  → Batch Multiplexing WORKING. SAM uses multiple channels.")
    elif s2_avg > 0.05:
        print(f"  → Batch Multiplexing PARTIALLY working. Some slot 2 usage.")
    else:
        print(f"  → Batch Multiplexing FAILED. Still channel collapse.")

    # =================================================================== #
    #  Summary                                                              #
    # =================================================================== #
    print("\n" + "=" * 70)
    print("  FINAL DIAGNOSIS")
    print("=" * 70)
    print(f"  Exp 1 (GT bypass):     mIoU = {miou_gt:.4f}")
    print(f"  Exp 2 (live Qwen):     mIoU = {miou_live:.4f}")
    print(f"  Exp 3 (slot 2 area):   {s2_avg:.1%}")
    print()
    print(f"  Training val mIoU:     0.7002 (ep 15)")
    print(f"  Training test mIoU:    0.5883 (ep 15)")
    print()
    print(f"  Gap analysis:")
    print(f"    GT bypass vs live:   {miou_gt - miou_live:+.4f} (pure Qwen degradation)")
    print(f"    V1/V2 ZS baseline:   0.7063")

    # Save results
    result = {
        "checkpoint": args.checkpoint,
        "experiment_1_gt_bypass": {"miou": float(miou_gt), "n_samples": len(ious_gt)},
        "experiment_2_over_generation": {
            "miou_live": float(miou_live),
            "k_pred_distribution": dict(sorted(k_dist.items())),
            "sample_texts": sample_texts,
        },
        "experiment_3_positional_bias": {
            "condition_a": {
                "miou": float(np.mean(ious_a)),
                "slot0": float(avg(0, areas_a)),
                "slot1": float(avg(1, areas_a)),
                "slot2": float(avg(2, areas_a)),
            },
            "condition_b": {
                "miou": float(np.mean(ious_b)),
                "slot0": float(avg(0, areas_b)),
                "slot1": float(avg(1, areas_b)),
                "slot2": float(avg(2, areas_b)),
            },
        },
    }
    out_path = Path("ablation") / "v3_diagnosis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
