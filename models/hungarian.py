"""
N-way Hungarian (bipartite) matching for variable-K texture segmentation.

Matches K_pred predicted textures to K_gt ground-truth textures using
a combined cost of text similarity and mask overlap. Unmatched predictions
are assigned to the DUSTBIN channel (index 0).
"""

import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

from models.qwen2sam_detexture import MAX_TEXTURES, NUM_QUERY_SLOTS


def compute_mask_dice(pred_logits: torch.Tensor, gt_mask: torch.Tensor,
                      n_classes: int) -> torch.Tensor:
    """
    Compute pairwise Dice coefficients between predicted mask channels
    and GT texture classes.

    Args:
        pred_logits: (6, H, W) raw logits for one sample.
        gt_mask: (H, W) index mask with values 0..K_gt (0=dustbin).
        n_classes: K_gt (number of GT textures, not counting dustbin).

    Returns:
        dice_matrix: (5, K_gt) — Dice score between pred channel i+1
                     and GT class j+1 (skip dustbin in both).
    """
    # Softmax to get probs, skip channel 0 (dustbin)
    probs = torch.softmax(pred_logits, dim=0)  # (C, H_pred, W_pred)

    # Upsample probs to GT resolution (preserve fine boundary detail)
    H_gt, W_gt = gt_mask.shape[0], gt_mask.shape[1]
    if probs.shape[1] != H_gt or probs.shape[2] != W_gt:
        probs = torch.nn.functional.interpolate(
            probs.unsqueeze(0), size=(H_gt, W_gt),
            mode="bilinear", align_corners=False,
        ).squeeze(0)

    dice = torch.zeros(MAX_TEXTURES, n_classes, device=pred_logits.device)
    for i in range(MAX_TEXTURES):  # pred channels 1..MAX_TEXTURES
        pred_soft = probs[i + 1]  # (H, W) probability for texture i+1
        for j in range(n_classes):  # GT classes 1..K_gt
            gt_bin = (gt_mask == (j + 1)).float()  # (H, W)
            inter = (pred_soft * gt_bin).sum()
            union = pred_soft.sum() + gt_bin.sum()
            dice[i, j] = (2.0 * inter / union.clamp(min=1e-6))
    return dice


def compute_text_cosine(pred_embeds: torch.Tensor,
                        gt_embeds: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise cosine similarity between predicted and GT text embeddings.

    Args:
        pred_embeds: (K_pred, D) predicted text embeddings.
        gt_embeds: (K_gt, D) GT text embeddings.

    Returns:
        sim_matrix: (K_pred, K_gt) cosine similarities in [-1, 1].
    """
    pred_norm = torch.nn.functional.normalize(pred_embeds.float(), dim=-1)
    gt_norm = torch.nn.functional.normalize(gt_embeds.float(), dim=-1)
    return pred_norm @ gt_norm.T


def hungarian_match(
    pred_logits: torch.Tensor,
    gt_mask: torch.Tensor,
    k_pred: int,
    k_gt: int,
    pred_text_embeds: torch.Tensor = None,
    gt_text_embeds: torch.Tensor = None,
    text_weight: float = 0.5,
    mask_weight: float = 0.5,
):
    """
    Hungarian matching for a single sample.

    Matches K_pred predicted textures to K_gt GT textures.
    Unmatched predictions → assigned to DUSTBIN (index 0).
    PAD slots → flagged for exclusion from loss.

    Args:
        pred_logits: (6, H, W) raw mask logits.
        gt_mask: (H, W) index mask, values {0=dustbin, 1..K_gt=textures}.
        k_pred: Number of textures Qwen predicted (1..5).
        k_gt: Number of GT textures (1..5).
        pred_text_embeds: (K_pred, D) text embeddings from Qwen, or None.
        gt_text_embeds: (K_gt, D) GT text embeddings (CLIP), or None.
        text_weight: Weight for text cost.
        mask_weight: Weight for mask cost.

    Returns:
        permutation: (6,) int tensor — GT class index for each output channel.
            0 = dustbin, 1..K_gt = texture, -1 = PAD (ignore).
        pad_mask: (6,) bool tensor — True for PAD channels.
        hallucinated_mask: (6,) bool tensor — True for unmatched preds → dustbin.
    """
    device = pred_logits.device
    permutation = torch.full((NUM_QUERY_SLOTS,), -1, dtype=torch.long, device=device)
    pad_mask = torch.zeros(NUM_QUERY_SLOTS, dtype=torch.bool, device=device)
    hallucinated_mask = torch.zeros(NUM_QUERY_SLOTS, dtype=torch.bool, device=device)

    # Channel 0 is always DUSTBIN
    permutation[0] = 0

    # Mark PAD channels (beyond K_pred + 1 for dustbin)
    for i in range(k_pred + 1, NUM_QUERY_SLOTS):
        pad_mask[i] = True

    if k_pred == 0 or k_gt == 0:
        # Edge case: no predictions or no GT — everything is dustbin
        for i in range(1, k_pred + 1):
            permutation[i] = 0
            hallucinated_mask[i] = True
        return permutation, pad_mask, hallucinated_mask

    # Build cost matrix: (K_pred, K_gt) on same device as inputs
    cost = torch.zeros(k_pred, k_gt, device=pred_logits.device)

    # Mask-based cost (1 - Dice)
    if mask_weight > 0:
        dice = compute_mask_dice(pred_logits.detach(), gt_mask, k_gt)
        # dice is (5, K_gt), we need (K_pred, K_gt)
        cost += mask_weight * (1.0 - dice[:k_pred, :k_gt])

    # Text-based cost (1 - cosine_sim) / 2
    if text_weight > 0 and pred_text_embeds is not None and gt_text_embeds is not None:
        sim = compute_text_cosine(
            pred_text_embeds[:k_pred].detach(),
            gt_text_embeds[:k_gt].detach(),
        )
        cost += text_weight * (1.0 - sim) / 2.0  # normalize to [0, 1]

    # Solve assignment
    row_ind, col_ind = linear_sum_assignment(cost.cpu().numpy())

    # Assign matched predictions
    matched_preds = set()
    for r, c in zip(row_ind, col_ind):
        # pred channel r+1 (skip dustbin at 0) maps to GT class c+1
        permutation[r + 1] = c + 1
        matched_preds.add(r)

    # Unmatched predictions → DUSTBIN
    for i in range(k_pred):
        if i not in matched_preds:
            permutation[i + 1] = 0
            hallucinated_mask[i + 1] = True

    return permutation, pad_mask, hallucinated_mask


def batch_hungarian_match(
    pred_logits: torch.Tensor,
    gt_masks: torch.Tensor,
    k_preds: torch.Tensor,
    k_gts: torch.Tensor,
    pred_text_embeds: torch.Tensor = None,
    gt_text_embeds: torch.Tensor = None,
    text_weight: float = 0.5,
    mask_weight: float = 0.5,
):
    """
    Batched Hungarian matching.

    Args:
        pred_logits: (B, 6, H, W) raw mask logits.
        gt_masks: (B, H, W) index masks.
        k_preds: (B,) number of predicted textures per sample.
        k_gts: (B,) number of GT textures per sample.
        pred_text_embeds: (B, 5, D) or None.
        gt_text_embeds: (B, 5, D) or None.

    Returns:
        permutations: (B, 6) GT class index per channel.
        pad_masks: (B, 6) True for PAD channels.
        hallucinated_masks: (B, 6) True for unmatched → dustbin.
    """
    B = pred_logits.shape[0]
    device = pred_logits.device
    permutations = torch.full((B, NUM_QUERY_SLOTS), -1, dtype=torch.long, device=device)
    pad_masks = torch.zeros(B, NUM_QUERY_SLOTS, dtype=torch.bool, device=device)
    hallucinated_masks = torch.zeros(B, NUM_QUERY_SLOTS, dtype=torch.bool, device=device)

    for b in range(B):
        pte = pred_text_embeds[b] if pred_text_embeds is not None else None
        gte = gt_text_embeds[b] if gt_text_embeds is not None else None

        perm, pad, hall = hungarian_match(
            pred_logits[b], gt_masks[b],
            k_pred=int(k_preds[b].item()),
            k_gt=int(k_gts[b].item()),
            pred_text_embeds=pte,
            gt_text_embeds=gte,
            text_weight=text_weight,
            mask_weight=mask_weight,
        )
        permutations[b] = perm
        pad_masks[b] = pad
        hallucinated_masks[b] = hall

    return permutations, pad_masks, hallucinated_masks


def permute_gt_mask(gt_mask: torch.Tensor, permutation: torch.Tensor,
                    k_gt: int) -> torch.Tensor:
    """
    Create a permuted GT mask that aligns with the predicted channel order.

    After Hungarian matching, each output channel i is assigned to GT class
    permutation[i]. We need to remap the GT mask so that pixels belonging
    to GT class permutation[i] have label i.

    Args:
        gt_mask: (H, W) original GT index mask (0=dustbin, 1..K_gt).
        permutation: (6,) mapping from output channel → GT class.
        k_gt: Number of GT texture classes.

    Returns:
        remapped: (H, W) index mask where values are output channel indices (0..5).
    """
    H, W = gt_mask.shape
    remapped = torch.zeros(H, W, dtype=torch.long, device=gt_mask.device)

    for channel_idx in range(NUM_QUERY_SLOTS):
        gt_class = permutation[channel_idx].item()
        if gt_class < 0:
            continue  # PAD — skip
        if gt_class == 0:
            # DUSTBIN: pixels with gt==0 should map to channel 0
            # (but only assign if not already claimed by a texture)
            pass  # handled below
        else:
            # Texture: pixels with gt==gt_class → channel_idx
            remapped[gt_mask == gt_class] = channel_idx

    # Remaining pixels (gt==0 or unclaimed) → channel 0 (DUSTBIN)
    # remapped is initialized to 0, so this is already correct.
    return remapped
