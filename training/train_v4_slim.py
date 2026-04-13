"""
V4-Slim Bridge Experiment training loop.

Same as V4 (frozen Qwen, override_tex_embeds, distillation) but with
the slim projector (4096→512→1024→plug→256, ~2.6M trainable params).

Usage:
    cd /home/aviad/Qwen2SAM_DeTexture
    python -m training.train_v4_slim --config configs/detexture_v4_slim.yaml
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from models.qwen2sam_v4_slim import Qwen2SAMDeTextureV4Slim
from data.dataset import DeTextureDataset, DeTextureCollator
from training.utils import (
    load_config, set_seed, AverageMeter, WarmupCosineScheduler,
    get_lr, save_checkpoint, load_checkpoint,
)
from training.monitor import DataSanityChecker, TrainingLogger, PlotGenerator, TestEvaluator

from models.qwen2sam_detexture import NUM_QUERY_SLOTS


# ===================================================================== #
#  Losses (inline — V4-style with distillation)                           #
# ===================================================================== #

def mask_pad_logits(logits, pad_mask):
    masked = logits.clone()
    inf_mask = pad_mask.unsqueeze(-1).unsqueeze(-1).expand_as(masked)
    masked[inf_mask] = float("-inf")
    return masked


def mask_loss(logits, targets, pad_mask, ce_w=1.0, dice_w=3.0):
    masked = mask_pad_logits(logits, pad_mask)
    ce = F.cross_entropy(masked, targets, reduction="mean")

    probs = F.softmax(masked, dim=1)
    gt_oh = F.one_hot(targets.long(), num_classes=NUM_QUERY_SLOTS).permute(0, 3, 1, 2).float()
    B = logits.shape[0]
    active = ~pad_mask
    dice_sum, n = torch.tensor(0.0, device=logits.device), 0
    for b in range(B):
        for c in range(NUM_QUERY_SLOTS):
            if not active[b, c]:
                continue
            p_c = probs[b, c].reshape(-1)
            g_c = gt_oh[b, c].reshape(-1)
            inter = (p_c * g_c).sum()
            union = p_c.sum() + g_c.sum()
            dice_sum = dice_sum + (1.0 - (2 * inter + 1) / (union + 1))
            n += 1
    dc = dice_sum / max(n, 1)
    return {"mask_total": ce_w * ce + dice_w * dc, "mask_ce": ce, "mask_dice": dc}


def distillation_loss(pred_256, sam_target_256, k_gts):
    B = pred_256.shape[0]
    pred_list, tgt_list = [], []
    for b in range(B):
        k = int(k_gts[b].item())
        for i in range(k):
            pred_list.append(pred_256[b, i])
            tgt_list.append(sam_target_256[b, i])
    if not pred_list:
        return torch.tensor(0.0, device=pred_256.device)
    pred_s = torch.stack(pred_list).float()
    tgt_s = torch.stack(tgt_list).float().detach()
    return (1.0 - F.cosine_similarity(pred_s, tgt_s, dim=-1)).mean()


def orthogonal_reg(model):
    p = None
    for m in model.modules():
        if hasattr(m, "orthogonal_penalty") and callable(m.orthogonal_penalty):
            v = m.orthogonal_penalty()
            p = v if p is None else p + v.to(p.device)
    return p if p is not None else torch.tensor(0.0)


# ===================================================================== #
#  Curriculum                                                             #
# ===================================================================== #

def apply_curriculum(model, epoch, cfg):
    ae = cfg.get("curriculum", {}).get("alignment_epochs", 10)
    if epoch < ae:
        for m in model.sam3_lora_modules:
            if hasattr(m, "lora_A"):
                m.lora_A.requires_grad = False
            if hasattr(m, "lora_B"):
                m.lora_B.requires_grad = False
        return 1
    else:
        for m in model.sam3_lora_modules:
            if hasattr(m, "lora_A"):
                m.lora_A.requires_grad = True
            if hasattr(m, "lora_B"):
                m.lora_B.requires_grad = True
        return 2


# ===================================================================== #
#  Training epoch                                                         #
# ===================================================================== #

def train_one_epoch(model, loader, optimizer, scheduler, scaler, epoch,
                    cfg, device, logger=None, phase=1):
    model.train()
    meters = {k: AverageMeter() for k in [
        "total", "mask_ce", "mask_dice", "distillation", "orthogonal_reg"]}

    accum = cfg["training"]["gradient_accumulation_steps"]
    max_gn = cfg["training"]["max_grad_norm"]
    w = cfg.get("loss", {})
    lam_mask = w.get("mask_weight", 1.0)
    lam_dist = w.get("distillation_weight", 1.0)
    lam_orth = w.get("orthogonal_weight", 0.01)
    ce_w = w.get("ce_weight", 1.0)
    dice_w = w.get("dice_weight", 3.0)

    optimizer.zero_grad()
    t0 = time.time()

    for step, batch in enumerate(loader):
        sam_images = batch["sam_images"].to(device)
        index_masks = batch["index_masks"].to(device)
        k_gts = batch["k_gts"].to(device)
        qwen_gt = batch["qwen_gt_embeds"].to(device)
        descriptions = batch["descriptions"]

        # SAM text encoder targets
        sam_targets = model.encode_descriptions_batch(descriptions)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(
                qwen_inputs=None, sam_images=sam_images,
                seg_grad_to_lm=False, override_tex_embeds=qwen_gt,
            )

            ml = out["mask_logits"]
            q256 = out["query_256"]
            pm = out["pad_mask"]

            # Upsample
            H, W = index_masks.shape[1], index_masks.shape[2]
            if ml.shape[2] != H:
                ml_up = F.interpolate(ml.float(), size=(H, W),
                                       mode="bilinear", align_corners=False)
            else:
                ml_up = ml.float()

            m = mask_loss(ml_up, index_masks, pm, ce_w, dice_w)
            pred_256 = q256[:, 1:, :]  # texture slots (drop dustbin)
            d = distillation_loss(pred_256, sam_targets, k_gts)
            o = orthogonal_reg(model)

            total = lam_mask * m["mask_total"] + lam_dist * d + lam_orth * o
            loss = total / accum

        scaler.scale(loss).backward()

        if (step + 1) % accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_gn)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        meters["total"].update(total.item())
        meters["mask_ce"].update(m["mask_ce"].item())
        meters["mask_dice"].update(m["mask_dice"].item())
        meters["distillation"].update(d.item())
        meters["orthogonal_reg"].update(o.item())

        lr = get_lr(optimizer)
        if (step + 1) % 10 == 0 or step == 0:
            el = time.time() - t0
            print(f"  [S{phase}|{step+1}/{len(loader)}] "
                  f"loss={meters['total'].avg:.4f} "
                  f"ce={meters['mask_ce'].avg:.4f} "
                  f"dice={meters['mask_dice'].avg:.4f} "
                  f"distill={meters['distillation'].avg:.4f} "
                  f"orth={meters['orthogonal_reg'].avg:.6f} "
                  f"lr={lr:.2e} ({el:.1f}s)", flush=True)

        if logger and ((step + 1) % 10 == 0 or step == 0):
            logger.log_step(epoch + 1, step + 1, len(loader),
                           {k: m.val for k, m in meters.items()}, lr)

    return {k: m.avg for k, m in meters.items()}


# ===================================================================== #
#  Validation                                                             #
# ===================================================================== #

@torch.no_grad()
def validate(model, loader, cfg, device):
    model.eval()
    ious = []
    for batch in loader:
        sam_images = batch["sam_images"].to(device)
        index_masks = batch["index_masks"].to(device)
        k_gts = batch["k_gts"]
        qwen_gt = batch["qwen_gt_embeds"].to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(qwen_inputs=None, sam_images=sam_images,
                        seg_grad_to_lm=False, override_tex_embeds=qwen_gt)

        ml = out["mask_logits"]
        pm = out["pad_mask"]
        masked = ml.clone()
        inf_m = pm.unsqueeze(-1).unsqueeze(-1).expand_as(masked)
        masked[inf_m] = float("-inf")
        H, W = index_masks.shape[1], index_masks.shape[2]
        if masked.shape[2] != H:
            masked = F.interpolate(masked.float(), size=(H, W),
                                    mode="bilinear", align_corners=False)
        preds = masked.argmax(dim=1)
        B = ml.shape[0]
        for b in range(B):
            k = int(k_gts[b].item())
            si = []
            for c in range(1, k + 1):
                pc = (preds[b] == c)
                gc = (index_masks[b] == c)
                inter = (pc & gc).sum().float()
                union = (pc | gc).sum().float()
                si.append((inter / union.clamp(min=1)).item())
            if si:
                ious.append(sum(si) / len(si))
    return sum(ious) / len(ious) if ious else 0.0


# ===================================================================== #
#  Main                                                                   #
# ===================================================================== #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/detexture_v4_slim.yaml")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda")

    train_cfg = cfg["training"]
    ckpt_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints_v4_slim"))
    monitor_cfg = cfg.get("monitor", {})
    logger = TrainingLogger(monitor_cfg.get("log_dir", str(ckpt_dir / "logs")))

    baselines = {}
    bp = monitor_cfg.get("baseline_results")
    if bp and Path(bp).exists():
        import json
        with open(bp) as f:
            for a, m in json.load(f).items():
                baselines[a] = {"miou": m.get("mean_iou", 0), "mari": m.get("mean_ari", 0)}
    plotter = PlotGenerator(monitor_cfg.get("plot_dir", str(ckpt_dir / "plots")),
                            baselines=baselines)

    test_evaluator = None
    tm = monitor_cfg.get("test_metadata")
    if tm and Path(tm).exists():
        test_evaluator = TestEvaluator(
            test_metadata=tm,
            output_dir=monitor_cfg.get("test_output_dir", str(ckpt_dir / "test_results")),
            image_size=cfg["data"].get("image_size", 1008),
            eval_every=monitor_cfg.get("test_eval_every", 5),
        )

    # ---- Model --------------------------------------------------------- #
    print("Building V4-Slim model...")
    model = Qwen2SAMDeTextureV4Slim(cfg, device="cuda")
    params = model.num_trainable_params()
    print("Trainable parameters:")
    for k, v in params.items():
        print(f"  {k}: {v:,}")

    # ---- Data ---------------------------------------------------------- #
    data_cfg = cfg["data"]
    train_ds = DeTextureDataset(
        data_cfg["train_metadata"], image_size=data_cfg.get("image_size", 1008),
        augment=data_cfg.get("augment", True),
        qwen_gt_embeds_path=data_cfg.get("qwen_gt_embeds_path"),
    )
    val_ds = DeTextureDataset(
        data_cfg["val_metadata"], image_size=data_cfg.get("image_size", 1008),
        augment=False, qwen_gt_embeds_path=data_cfg.get("qwen_gt_embeds_path"),
    )
    collator = DeTextureCollator(model.processor)
    train_loader = DataLoader(train_ds, batch_size=train_cfg["batch_size"],
                               shuffle=True, num_workers=data_cfg.get("num_workers", 0),
                               collate_fn=collator, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                             num_workers=0, collate_fn=collator)

    # ---- Optimizer ----------------------------------------------------- #
    steps_per_epoch = len(train_loader) // train_cfg["gradient_accumulation_steps"]
    param_groups = model.get_parameter_groups(train_cfg["learning_rate"])
    optimizer = torch.optim.AdamW(param_groups,
                                   weight_decay=train_cfg.get("weight_decay", 0.01))
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_epochs=train_cfg.get("warmup_epochs", 3),
        total_epochs=train_cfg["num_epochs"],
        min_lr=train_cfg.get("min_lr", 1e-6),
        steps_per_epoch=max(steps_per_epoch, 1),
    )
    scaler = torch.amp.GradScaler("cuda")

    # ---- Resume -------------------------------------------------------- #
    start_epoch, best_iou = 0, 0.0
    if args.resume and Path(args.resume).exists():
        start_epoch = load_checkpoint(model, optimizer, args.resume, device="cuda") + 1

    # ---- Train --------------------------------------------------------- #
    print(f"\nV4-Slim Bridge: {train_cfg['num_epochs']} epochs, "
          f"{len(train_ds)} train / {len(val_ds)} val")
    print(f"Projector: 4096 → 512 → 1024 → plug(LN+Linear→256)")
    print(f"Qwen FROZEN. Distillation active.\n")

    for epoch in range(start_epoch, train_cfg["num_epochs"]):
        phase = apply_curriculum(model, epoch, cfg)
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{train_cfg['num_epochs']} [Stage {phase}]")
        print(f"{'='*60}")

        tm = train_one_epoch(model, train_loader, optimizer, scheduler,
                              scaler, epoch, cfg, device, logger, phase)
        vi = validate(model, val_loader, cfg, device)
        print(f"\n  Val mIoU: {vi:.4f}")

        lr = get_lr(optimizer)
        is_best = vi > best_iou
        logger.log_epoch(epoch + 1, tm, vi, lr, is_best, extra={"phase": phase})

        if is_best:
            best_iou = vi
            save_checkpoint(model, optimizer, epoch, str(ckpt_dir / "best.pt"),
                           extra={"val_iou": vi, "phase": phase})
            print(f"  ** New best: {best_iou:.4f} **")

        if (epoch + 1) % train_cfg.get("save_every", 5) == 0:
            save_checkpoint(model, optimizer, epoch,
                           str(ckpt_dir / f"epoch_{epoch+1}.pt"),
                           extra={"val_iou": vi, "phase": phase})

        if test_evaluator and test_evaluator.should_evaluate(epoch):
            print(f"\n  Running RWTD test...")
            try:
                test_metrics = test_evaluator.evaluate(
                    model, model.processor, device, epoch)
                logger.log_test(epoch + 1, test_metrics)
            except Exception as e:
                print(f"  WARNING: Test eval failed: {e}")

        plotter.update(logger)

    logger.close()
    print(f"\nDone. Best val mIoU: {best_iou:.4f}")


if __name__ == "__main__":
    main()
