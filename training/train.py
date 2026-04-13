"""
V5 Training loop for Qwen2SAM-DeTexture.

Architecture: [SEG] Token + Bottleneck Projector + Loss Masking.

Stage 1 (epochs 1-N): Projector + Qwen LoRA (SAM LoRA frozen)
  - Qwen generates text with <|seg|> tokens (teacher-forced)
  - LM loss: -100 on ALL text tokens, CE only on <|seg|> positions
  - Mask loss gradients: SAM → Projector → [SEG] hidden states → Qwen LoRA
  - The pixels teach the [SEG] token where to go

Stage 2 (epochs N+1 onwards): + SAM LoRA unfrozen
  - Joint fine-tuning of projector + Qwen LoRA + SAM LoRA
  - Same loss masking

No distillation loss, no InverseProjector, no pre-computed GT embeddings.
Live Qwen forward in EVERY epoch.

Usage:
    cd /home/aviad/Qwen2SAM_DeTexture
    python -m training.train --config configs/detexture.yaml
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.multiprocessing
import torch.nn.functional as F
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from models.qwen2sam_detexture import Qwen2SAMDeTexture
from models.losses import combined_loss
from data.dataset import DeTextureDataset, DeTextureCollator
from training.utils import (
    load_config, set_seed, AverageMeter, WarmupCosineScheduler,
    get_lr, save_checkpoint, load_checkpoint,
)
from training.monitor import DataSanityChecker, TrainingLogger, PlotGenerator, TestEvaluator


# ===================================================================== #
#  Curriculum                                                             #
# ===================================================================== #

def apply_curriculum(model, epoch, cfg):
    """
    V5 Three-Stage Curriculum (Cold Start Protection).

    Stage 1 — Projector Warmup (epoch < projector_warmup_epochs):
      - ONLY Projector + mask head + dustbin train
      - Qwen LoRA FROZEN (protects pretrained weights from garbage gradients)
      - SAM LoRA FROZEN
      - Projector absorbs initial random-weight shock and finds a baseline
        mapping from base-Qwen [SEG] hidden states to SAM's manifold

    Stage 2 — Joint Co-Adaptation (projector_warmup_epochs <= epoch < e2e_epoch):
      - Qwen LoRA UNFROZEN at conservative LR (e.g. 2e-5 vs projector's 1e-4)
      - Projector continues training at full LR
      - SAM LoRA still FROZEN
      - Qwen LoRA carefully refines [SEG] representations without being
        yanked around by projector instability

    Stage 3 — End-to-End Fine-Tuning (epoch >= e2e_epoch):
      - SAM LoRA UNFROZEN
      - All three components train jointly to convergence
    """
    curriculum = cfg.get("curriculum", {})
    warmup_epochs = curriculum.get("projector_warmup_epochs", 2)
    e2e_epoch = curriculum.get("e2e_epoch", 5)

    def _freeze_qwen_lora(model):
        for n, p in model.qwen.named_parameters():
            if "lora" in n.lower():
                p.requires_grad = False

    def _unfreeze_qwen_lora(model):
        for n, p in model.qwen.named_parameters():
            if "lora" in n.lower():
                p.requires_grad = True

    def _freeze_sam_lora(model):
        for m in model.sam3_lora_modules:
            if hasattr(m, "lora_A"):
                m.lora_A.requires_grad = False
            if hasattr(m, "lora_B"):
                m.lora_B.requires_grad = False

    def _unfreeze_sam_lora(model):
        for m in model.sam3_lora_modules:
            if hasattr(m, "lora_A"):
                m.lora_A.requires_grad = True
            if hasattr(m, "lora_B"):
                m.lora_B.requires_grad = True

    if epoch < warmup_epochs:
        # Stage 1: Projector Warmup — only projector trains
        _freeze_qwen_lora(model)
        _freeze_sam_lora(model)
        return 1, {}
    elif epoch < e2e_epoch:
        # Stage 2: Joint Co-Adaptation — Qwen LoRA (slow) + Projector (fast)
        _unfreeze_qwen_lora(model)
        _freeze_sam_lora(model)
        return 2, {}
    else:
        # Stage 3: End-to-End — all components train
        _unfreeze_qwen_lora(model)
        _unfreeze_sam_lora(model)
        return 3, {}


# ===================================================================== #
#  Training epoch                                                         #
# ===================================================================== #

def train_one_epoch(
    model, dataloader, optimizer, scheduler, scaler, epoch, cfg, device,
    logger=None, phase=1, loss_overrides=None,
):
    model.train()
    meters = {
        k: AverageMeter() for k in [
            "total", "mask_ce", "mask_dice", "lm_loss", "orthogonal_reg",
        ]
    }

    accum_steps = cfg["training"]["gradient_accumulation_steps"]
    max_grad_norm = cfg["training"]["max_grad_norm"]

    effective_cfg = {**cfg}
    if loss_overrides:
        effective_cfg["loss"] = {**cfg.get("loss", {}), **loss_overrides}

    optimizer.zero_grad()
    t0 = time.time()

    for step, batch in enumerate(dataloader):
        # Move to device
        qwen_inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch["qwen_inputs"].items()
        }
        sam_images = batch["sam_images"].to(device)
        index_masks = batch["index_masks"].to(device)
        k_gts = batch["k_gts"].to(device)

        # V5: Live Qwen forward with [SEG] token extraction.
        # Labels already have -100 on text, CE only on [SEG] positions.
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(
                qwen_inputs=qwen_inputs,
                sam_images=sam_images,
                seg_grad_to_lm=True,  # mask gradients flow to Qwen LoRA
            )

            mask_logits = out["mask_logits"]
            k_preds = out["k_preds"]
            pad_mask = out["pad_mask"]
            lm_loss = out["lm_loss"]

            losses = combined_loss(
                mask_logits=mask_logits,
                gt_masks=index_masks,
                pad_mask=pad_mask,
                k_gts=k_gts,
                lm_loss=lm_loss,
                model=model,
                cfg=effective_cfg,
            )

            loss = losses["total"] / accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_grad_norm,
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        for k in meters:
            if k in losses:
                meters[k].update(losses[k].item())

        lr = get_lr(optimizer)
        if (step + 1) % 10 == 0 or step == 0:
            elapsed = time.time() - t0
            print(
                f"  [S{phase}|{step+1}/{len(dataloader)}] "
                f"loss={meters['total'].avg:.4f} "
                f"ce={meters['mask_ce'].avg:.4f} "
                f"dice={meters['mask_dice'].avg:.4f} "
                f"lm={meters['lm_loss'].avg:.4f} "
                f"orth={meters['orthogonal_reg'].avg:.6f} "
                f"lr={lr:.2e} "
                f"({elapsed:.1f}s)",
                flush=True,
            )

        if logger and ((step + 1) % 10 == 0 or step == 0):
            logger.log_step(
                epoch=epoch + 1,
                step=step + 1,
                total_steps=len(dataloader),
                losses={k: m.val for k, m in meters.items()},
                lr=lr,
            )

    return {k: m.avg for k, m in meters.items()}


# ===================================================================== #
#  Validation                                                             #
# ===================================================================== #

@torch.no_grad()
def validate(model, dataloader, cfg, device):
    """V5 validation with live Qwen forward and [SEG] extraction."""
    model.eval()
    ious = []

    for batch in dataloader:
        qwen_inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch["qwen_inputs"].items()
        }
        sam_images = batch["sam_images"].to(device)
        index_masks = batch["index_masks"].to(device)
        k_gts = batch["k_gts"]

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(
                qwen_inputs=qwen_inputs,
                sam_images=sam_images,
                seg_grad_to_lm=False,
            )

        mask_logits = out["mask_logits"]
        pad_mask = out["pad_mask"]

        masked_logits = mask_logits.clone()
        inf_mask = pad_mask.unsqueeze(-1).unsqueeze(-1).expand_as(masked_logits)
        masked_logits[inf_mask] = float("-inf")

        H_gt, W_gt = index_masks.shape[1], index_masks.shape[2]
        if masked_logits.shape[2] != H_gt or masked_logits.shape[3] != W_gt:
            masked_logits = F.interpolate(
                masked_logits.float(), size=(H_gt, W_gt),
                mode="bilinear", align_corners=False,
            )
        preds = masked_logits.argmax(dim=1)

        B = mask_logits.shape[0]
        for b in range(B):
            k = int(k_gts[b].item())
            sample_ious = []
            for c in range(1, k + 1):
                pred_c = (preds[b] == c)
                gt_c = (index_masks[b] == c)
                inter = (pred_c & gt_c).sum().float()
                union = (pred_c | gt_c).sum().float()
                iou = (inter / union.clamp(min=1)).item()
                sample_ious.append(iou)
            if sample_ious:
                ious.append(sum(sample_ious) / len(sample_ious))

    mean_iou = sum(ious) / len(ious) if ious else 0.0
    return mean_iou


# ===================================================================== #
#  Main                                                                   #
# ===================================================================== #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/detexture.yaml")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Monitoring ---------------------------------------------------- #
    train_cfg = cfg["training"]
    ckpt_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints"))
    monitor_cfg = cfg.get("monitor", {})

    log_dir = monitor_cfg.get("log_dir", str(ckpt_dir / "logs"))
    plot_dir = monitor_cfg.get("plot_dir", str(ckpt_dir / "plots"))
    logger = TrainingLogger(log_dir)

    baselines = {}
    baseline_path = monitor_cfg.get("baseline_results")
    if baseline_path and Path(baseline_path).exists():
        import json as _json
        with open(baseline_path) as f:
            bl_data = _json.load(f)
        for approach, metrics in bl_data.items():
            baselines[approach] = {
                "miou": metrics.get("mean_iou", 0),
                "mari": metrics.get("mean_ari", 0),
            }
        print(f"Loaded {len(baselines)} baselines from {baseline_path}")
    plotter = PlotGenerator(plot_dir, baselines=baselines)

    test_evaluator = None
    test_metadata = monitor_cfg.get("test_metadata")
    if test_metadata and Path(test_metadata).exists():
        test_eval_every = monitor_cfg.get("test_eval_every", 5)
        test_output_dir = monitor_cfg.get("test_output_dir",
                                           str(ckpt_dir / "test_results"))
        test_evaluator = TestEvaluator(
            test_metadata=test_metadata,
            output_dir=test_output_dir,
            image_size=cfg["data"].get("image_size", 1008),
            eval_every=test_eval_every,
        )

    # ---- Model --------------------------------------------------------- #
    print("Building Qwen2SAM-DeTexture V5 model...")
    model = Qwen2SAMDeTexture(cfg, device=str(device))
    params = model.num_trainable_params()
    print(f"Trainable parameters:")
    for k, v in params.items():
        print(f"  {k}: {v:,}")

    # ---- Data ---------------------------------------------------------- #
    data_cfg = cfg["data"]

    train_ds = DeTextureDataset(
        data_cfg["train_metadata"],
        image_size=data_cfg.get("image_size", 1008),
        augment=data_cfg.get("augment", True),
        qwen_gt_embeds_path=data_cfg.get("qwen_gt_embeds_path"),
    )
    val_ds = DeTextureDataset(
        data_cfg["val_metadata"],
        image_size=data_cfg.get("image_size", 1008),
        augment=False,
        qwen_gt_embeds_path=data_cfg.get("qwen_gt_embeds_path"),
    )

    collator = DeTextureCollator(model.processor)

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 0),
        collate_fn=collator,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=0, collate_fn=collator,
    )

    # ---- Sanity check -------------------------------------------------- #
    sanity_checker = DataSanityChecker(
        output_dir=str(ckpt_dir / "sanity"),
        n_batches=monitor_cfg.get("sanity_check_batches", 5),
    )
    sanity_ok = sanity_checker.check(train_loader, train_ds)
    if not sanity_ok and not monitor_cfg.get("skip_sanity_check", False):
        print("CRITICAL: Data sanity check failed.")
        sys.exit(1)

    # ---- Optimizer + scheduler ----------------------------------------- #
    curriculum = cfg.get("curriculum", {})
    alignment_epochs = curriculum.get("alignment_epochs", 15)

    steps_per_epoch = len(train_loader) // train_cfg["gradient_accumulation_steps"]
    param_groups = model.get_parameter_groups(train_cfg["learning_rate"])
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=train_cfg.get("weight_decay", 0.01),
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=train_cfg.get("warmup_epochs", 5),
        total_epochs=train_cfg["num_epochs"],
        min_lr=train_cfg.get("min_lr", 1e-6),
        steps_per_epoch=max(steps_per_epoch, 1),
    )
    scaler = torch.amp.GradScaler("cuda")

    # ---- Resume -------------------------------------------------------- #
    start_epoch = 0
    best_iou = 0.0

    resume_path = args.resume
    if resume_path == "auto":
        ckpt_files = sorted(ckpt_dir.glob("epoch_*.pt"))
        if ckpt_files:
            resume_path = str(ckpt_files[-1])
        elif (ckpt_dir / "best.pt").exists():
            resume_path = str(ckpt_dir / "best.pt")
        else:
            resume_path = None

    if resume_path and Path(resume_path).exists():
        print(f"\nResuming from: {resume_path}")
        state = torch.load(resume_path, map_location=device)
        start_epoch = load_checkpoint(model, optimizer, resume_path, device=str(device))
        start_epoch += 1
        best_iou = state.get("val_iou", 0.0)
        total_steps_done = start_epoch * max(steps_per_epoch, 1)
        for _ in range(total_steps_done):
            scheduler.step()
        print(f"  Resumed at epoch {start_epoch + 1}, best_iou={best_iou:.4f}")

    # ---- Training loop ------------------------------------------------- #
    stage_initialized = {1: False, 2: False, 3: False}

    warmup_ep = curriculum.get("projector_warmup_epochs", 2)
    e2e_ep = curriculum.get("e2e_epoch", 5)

    print(f"\nStarting V5 training: epochs {start_epoch+1}-{train_cfg['num_epochs']}")
    print(f"  {len(train_ds)} train / {len(val_ds)} val samples")
    print(f"  Effective batch: {train_cfg['batch_size']} × "
          f"{train_cfg['gradient_accumulation_steps']} = "
          f"{train_cfg['batch_size'] * train_cfg['gradient_accumulation_steps']}")
    print(f"  Stage 1: Projector Warmup (epochs 1-{warmup_ep})")
    print(f"  Stage 2: + Qwen LoRA at conservative LR (epochs {warmup_ep+1}-{e2e_ep})")
    print(f"  Stage 3: + SAM LoRA — End-to-End (epochs {e2e_ep+1}+)")
    print(f"  [SEG] token training with loss masking (CE only on <|seg|>)")

    for epoch in range(start_epoch, train_cfg["num_epochs"]):
        phase, loss_overrides = apply_curriculum(model, epoch, cfg)

        if not stage_initialized[phase]:
            stage_initialized[phase] = True
            n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if phase == 1:
                print(f"\n  *** STAGE 1: Projector Warmup (epoch {epoch+1}) ***")
                print(f"  Qwen LoRA FROZEN. SAM LoRA FROZEN. Only projector trains.")
            elif phase == 2:
                print(f"\n  *** STAGE 2: Joint Co-Adaptation (epoch {epoch+1}) ***")
                print(f"  Qwen LoRA UNFROZEN (conservative LR). Projector continues.")
            elif phase == 3:
                print(f"\n  *** STAGE 3: End-to-End (epoch {epoch+1}) ***")
                print(f"  SAM LoRA UNFROZEN. All components train jointly.")
            print(f"  Trainable params: {n_trainable:,}")

        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{train_cfg['num_epochs']} [Stage {phase}]")
        print(f"{'='*60}")

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, epoch, cfg, device,
            logger=logger, phase=phase, loss_overrides=loss_overrides,
        )

        val_iou = validate(model, val_loader, cfg, device)
        print(f"\n  Val mIoU: {val_iou:.4f}")

        lr = get_lr(optimizer)
        is_best = val_iou > best_iou
        logger.log_epoch(
            epoch=epoch + 1,
            train_metrics=train_metrics,
            val_miou=val_iou,
            lr=lr,
            is_best=is_best,
            extra={"phase": phase},
        )

        if is_best:
            best_iou = val_iou
            save_checkpoint(
                model, optimizer, epoch,
                str(ckpt_dir / "best.pt"),
                extra={"val_iou": val_iou, "phase": phase},
            )
            print(f"  ** New best: {best_iou:.4f} **")

        if (epoch + 1) % train_cfg.get("save_every", 5) == 0:
            save_checkpoint(
                model, optimizer, epoch,
                str(ckpt_dir / f"epoch_{epoch+1}.pt"),
                extra={"val_iou": val_iou, "phase": phase},
            )

        if test_evaluator and test_evaluator.should_evaluate(epoch):
            print(f"\n  Running test evaluation (RWTD)...")
            try:
                test_metrics = test_evaluator.evaluate(
                    model, model.processor, device, epoch,
                )
                logger.log_test(epoch=epoch + 1, metrics=test_metrics)
            except Exception as e:
                print(f"  WARNING: Test evaluation failed: {e}")

        plotter.update(logger)

    logger.close()
    print(f"\nTraining complete. Best val mIoU: {best_iou:.4f}")


if __name__ == "__main__":
    main()
