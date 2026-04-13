#!/usr/bin/env python3
"""
Focused test: "exactly 2" prompt on RWTD, ep5 + ep10.
No baseline — we already know '1 to 6' gives 0.6939 (ep5) / 0.6181 (ep10).
"""

import argparse, json, sys, re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from models.qwen2sam_detexture import Qwen2SAMDeTexture, MAX_TEXTURES
from data.dataset import (
    DeTextureDataset, DeTextureCollator,
    SYSTEM_PROMPT, USER_PROMPT_TEMPLATE,
)
from training.utils import load_config, load_checkpoint
from scipy.optimize import linear_sum_assignment


class ExactKCollator(DeTextureCollator):
    def __init__(self, processor, k=2):
        super().__init__(processor, inference=True)
        self.user_prompt = USER_PROMPT_TEMPLATE.format(N=str(k))

    def __call__(self, samples):
        texts, images = [], []
        for s in samples:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.user_prompt},
                ]},
            ]
            texts.append(self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            ))
            images.append(s["image"])
        qwen_inputs = self.processor(
            text=texts, images=images, return_tensors="pt", padding=True,
        )
        qwen_inputs.pop("token_type_ids", None)
        return {
            "qwen_inputs": qwen_inputs,
            "sam_images": torch.stack([s["sam_image"] for s in samples]),
            "index_masks": torch.stack([s["index_mask"] for s in samples]),
            "k_gts": torch.tensor([s["k_gt"] for s in samples], dtype=torch.long),
            "qwen_gt_embeds": torch.stack([s["qwen_gt_embeds"] for s in samples]),
        }


def compute_miou(pred, gt, k_pred, k_gt):
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


def run(model, loader, device, label):
    model.eval()
    ious, k_preds_list, details = [], [], []
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            qi = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in batch["qwen_inputs"].items()}
            si = batch["sam_images"].to(device)
            im = batch["index_masks"]
            kg = int(batch["k_gts"][0].item())
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model.inference_forward(qwen_inputs=qi, sam_images=si)
            ml, pm = out["mask_logits"], out["pad_mask"]
            kp = int(out["k_preds"][0].item())
            gt_text = out.get("generated_text", [""])[0]
            k_preds_list.append(kp)
            m = ml.clone()
            inf_m = pm.unsqueeze(-1).unsqueeze(-1).expand_as(m)
            m[inf_m] = float("-inf")
            H, W = im.shape[1], im.shape[2]
            if m.shape[2] != H:
                m = F.interpolate(m.float(), size=(H, W), mode="bilinear", align_corners=False)
            pred = m[0].argmax(dim=0).cpu().numpy()
            gt = im[0].numpy()
            miou = compute_miou(pred, gt, kp, kg)
            ious.append(miou)
            tp = pred.size
            slots = {c: f"{(pred==c).sum()/tp:.1%}" for c in range(MAX_TEXTURES+1) if (pred==c).sum() > 0}
            if idx < 5 or (idx+1) % 50 == 0:
                details.append({"idx": idx, "k_gt": kg, "k_pred": kp, "miou": miou,
                                "slots": slots, "text": gt_text[:300]})
            if (idx+1) % 50 == 0:
                print(f"    {idx+1}/{len(loader)} avg={np.mean(ious):.4f}", flush=True)

    kd = Counter(k_preds_list)
    mean = np.mean(ious)
    print(f"\n  [{label}] mIoU={mean:.4f}  k_pred={dict(sorted(kd.items()))}  "
          f">0.7:{sum(1 for x in ious if x>0.7)}/{len(ious)}  "
          f"<0.3:{sum(1 for x in ious if x<0.3)}/{len(ious)}")
    for d in details[:6]:
        print(f"    #{d['idx']} k_gt={d['k_gt']} k_pred={d['k_pred']} "
              f"mIoU={d['miou']:.4f} slots={d['slots']}")
        lines = [l.strip() for l in d["text"].split("\n") if "TEXTURE" in l]
        for l in lines[:3]:
            print(f"      {l[:120]}")
    return mean, kd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/detexture.yaml")
    parser.add_argument("--test-metadata", default="/home/aviad/datasets/RWTD/metadata.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda")

    print("Building model...")
    model = Qwen2SAMDeTexture(cfg, device="cuda")
    test_ds = DeTextureDataset(args.test_metadata, image_size=1008, augment=False,
                                qwen_gt_embeds_path=cfg["data"].get("qwen_gt_embeds_path"))
    collator = ExactKCollator(model.processor, k=2)
    loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False,
                                          num_workers=0, collate_fn=collator)
    print(f"RWTD: {len(test_ds)} samples, prompt='exactly 2'\n")

    results = {}
    for ckpt in ["checkpoints/epoch_5.pt", "checkpoints/epoch_10.pt"]:
        if not Path(ckpt).exists():
            print(f"  SKIP: {ckpt}"); continue
        label = Path(ckpt).stem
        print(f"{'='*60}\n  {ckpt}\n{'='*60}")
        load_checkpoint(model, None, ckpt, device="cuda")
        miou, kd = run(model, loader, device, label)
        results[label] = miou

    print(f"\n{'='*60}")
    print(f"  SUMMARY: 'exactly 2' prompt on RWTD")
    print(f"{'='*60}")
    print(f"  {'checkpoint':<15} {'1-to-6':>10} {'exactly-2':>10} {'delta':>10}")
    baselines = {"epoch_5": 0.6939, "epoch_10": 0.6181}
    for label, miou in results.items():
        bl = baselines.get(label, 0)
        print(f"  {label:<15} {bl:>10.4f} {miou:>10.4f} {miou-bl:>+10.4f}")
    print(f"\n  ZS baseline: 0.7063")


if __name__ == "__main__":
    main()
