# Qwen2SAM-DeTexture (V5)

**End-to-End VLM-Guided Multi-Texture Segmentation with [SEG] Token Grounding**

An E2E architecture that fuses a Vision-Language Model (**Qwen3-VL-8B**) with a Geometric Segmentation Engine (**SAM 3**) to segment images into 1-6 distinct texture regions. The model uses a dedicated **[SEG] token** to decouple visual grounding from language context, a **block-diagonal attention mask** to prevent cross-texture contamination, and a **bottleneck projector** (~2M params) that forces domain-agnostic generalization.

---

## Table of Contents

- [Key Innovations](#key-innovations)
- [Architecture Overview](#architecture-overview)
- [Detailed Architecture](#detailed-architecture)
  - [Module A: Qwen3-VL with [SEG] Token](#module-a-qwen3-vl-with-seg-token)
  - [Module B: Bottleneck Projector](#module-b-bottleneck-projector)
  - [Module C: SAM 3 with Batch Multiplexing](#module-c-sam-3-with-batch-multiplexing)
  - [Module D: Multi-Texture Mask Head](#module-d-multi-texture-mask-head)
- [The Dustbin Query](#the-dustbin-query)
- [Block-Diagonal Attention Mask](#block-diagonal-attention-mask)
- [Loss Masking (Anti-Count-Collapse)](#loss-masking-anti-count-collapse)
- [Orthogonal LoRA](#orthogonal-lora)
- [Training Process](#training-process)
  - [Three-Stage Curriculum](#three-stage-curriculum)
  - [Forward Pass Walkthrough](#forward-pass-walkthrough)
  - [Loss Functions](#loss-functions)
  - [Differential Learning Rates](#differential-learning-rates)
- [Inference (Two-Pass Strategy)](#inference-two-pass-strategy)
- [Ablation History (V1-V5)](#ablation-history-v1-v5)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Getting Started](#getting-started)

---

## Key Innovations

| Innovation | What it solves |
|---|---|
| **[SEG] Token Grounding** | Dedicated `<\|seg\|>` token after each texture description. Qwen's LoRA learns to pack visually-grounded spatial information into this token, decoupling it from noisy language context. |
| **Block-Diagonal Attention Mask** | Prevents Context Leakage: TEXTURE_2's `<\|seg\|>` hidden state cannot attend to TEXTURE_1's tokens. Proven to reduce inter-texture cosine similarity from 0.74 to 0.16. |
| **Loss Masking (-100 on text)** | LM loss is computed ONLY on `<\|seg\|>` token positions. Prevents V3's "Count Collapse" where Qwen learned to terminate after one texture to minimize text-prediction loss. |
| **Information Bottleneck (2M params)** | Projector reduced from 10.5M to 2.1M params. V4 proved that larger projectors memorize domain-specific manifold directions (Directional Drift). The bottleneck forces generalization. |
| **Batch Multiplexing** | Each query is fed to SAM independently in its own "slot 1" position. Eliminates SAM3's pretrained positional bias where only the first query slot received attention. |
| **Dustbin Query** | A learned embedding that absorbs non-texture pixels (objects, sky, etc.), preventing false texture assignments. |
| **Orthogonal LoRA** | Fine-tunes SAM3's cross-attention without catastrophic forgetting. LoRA updates are regularized to be orthogonal to SAM3's pretrained weight subspace. |
| **Three-Stage Curriculum** | Cold Start Protection: (1) Projector warmup, (2) Qwen LoRA at conservative LR, (3) SAM LoRA end-to-end. Prevents garbage gradients from a random projector corrupting Qwen's pretrained weights. |
| **Two-Pass Inference** | Pass 1: standard causal generation (no repetition). Pass 2: block-diagonal masked forward (decoupled [SEG] extraction). Train-test parity guaranteed. |

---

## Architecture Overview

```
                        +-----------------------+
                        |      Input Image      |
                        +-----------+-----------+
                                    |
                  +-----------------+-----------------+
                  |                                   |
                  v                                   v
    +----------------------------+     +----------------------------+
    |  MODULE A: Qwen3-VL-8B     |     |  SAM3 Backbone (frozen)    |
    |  (LoRA r=8 on q,v)         |     |  Image Encoder             |
    |                            |     |  1008x1008 -> FPN features |
    |  Input: image + prompt     |     +-------------+--------------+
    |  Output: text descriptions |                   |
    |  with <|seg|> markers      |                   |
    +-------------+--------------+                   |
                  |                                  |
                  v                                  |
    +----------------------------+                   |
    |  Block-Diagonal Mask       |                   |
    |  Extract <|seg|> hidden    |                   |
    |  states (decoupled)        |                   |
    |                            |                   |
    |  K vectors of dim 4096     |                   |
    +-------------+--------------+                   |
                  |                                  |
                  v                                  |
    +----------------------------+                   |
    |  Build 7 Query Slots       |                   |
    |                            |                   |
    |  [DUSTBIN, SEG_1, ...,     |                   |
    |   SEG_K, PAD, ..., PAD]    |                   |
    |                            |                   |
    |  7 vectors of dim 4096     |                   |
    +-------------+--------------+                   |
                  |                                  |
                  v                                  |
    +----------------------------+                   |
    |  MODULE B: Bottleneck      |                   |
    |  Projector (~2M params)    |                   |
    |                            |                   |
    |  4096 -> 512 + LN + GELU  |                    |
    |    + Dropout(0.15)         |                   |
    |  512  -> 256               |                   |
    +-------------+--------------+                   |
                  |                                  |
                  +----------------------------------+
                  |                                  |
                  v                                  v
    +-------------------------------------------------------+
    |  MODULE C: SAM3 — Batch Multiplexed                    |
    |  (B, 7, 256) -> flatten -> (B*7, 1, 256)               |
    |  Each query gets its own "slot 1" position             |
    |                                                        |
    |  Fusion Encoder (frozen + Orthogonal LoRA)             |
    |  SegHead cross_attend_prompt (frozen + Orth. LoRA)     |
    |  Pixel Decoder (frozen) -> pixel_embed                 |
    +----------------------------+---------------------------+
                                 |
                                 v
    +-------------------------------------------------------+
    |  MODULE D: Multi-Texture Mask Head (trainable)        |
    |                                                       |
    |  einsum(queries, pixel_embed) -> (B, 7, H, W)         |
    |                                                       |
    |  Channel 0: DUSTBIN     Channels 1-K: Textures        |
    |  Channels K+1 to 6: PAD (masked to -inf)              |
    +-------------------------------------------------------+
```

---

## Detailed Architecture

### Module A: Qwen3-VL with [SEG] Token

**Model**: `Qwen/Qwen3-VL-8B-Instruct`
**Status**: Frozen base weights + LoRA (r=8) on `q_proj` and `v_proj`.
**Training**: LoRA receives gradients ONLY through `<|seg|>` token positions (loss masking).

**Output format** (teacher-forced during training):
```
TEXTURE_1: Texture of rough mossy stone with granular surface in the foreground <|seg|>
TEXTURE_2: Texture of smooth flowing water with reflective sheen in the center <|seg|>
TEXTURE_3: Texture of dry sandy ground with ripple patterns on the right <|seg|>
```

The `<|seg|>` token is a dedicated grounding anchor. Unlike natural language end-of-line tokens (which absorb context from ALL prior tokens via causal attention), `<|seg|>` hidden states are computed under a **block-diagonal attention mask** that isolates each texture block from previous blocks. This produces clean, decoupled visual representations.

---

### Module B: Bottleneck Projector

**Status**: Fully trainable (~2.1M params).

Maps `<|seg|>` hidden states from Qwen's 4096-D space to SAM3's 256-D query space through a deliberately constrained bottleneck:

```
Linear(4096 -> 512) + LayerNorm(512) + GELU + Dropout(0.15)
Linear(512  -> 256)
```

**Why so small?** V4 ablation studies proved that a larger projector (10.5M params) memorizes ADE20K-specific manifold directions that don't generalize to unseen domains (Directional Drift). The V4-Slim bridge experiment confirmed: reducing to 2.6M params eliminated the drift entirely and produced the first trained model to beat the zero-shot baseline (mIoU 0.7316 vs 0.7063).

---

### Module C: SAM 3 with Batch Multiplexing

Each of the 7 query vectors is fed to SAM **independently** by flattening the query dimension into the batch dimension:

```
(B, 7, 256) -> reshape -> (B*7, 1, 256) -> SAM -> (B*7, 1, H, W) -> reshape -> (B, 7, H, W)
```

This eliminates SAM3's pretrained positional bias (V2 ablation: Slot 1 received 90.5% of pixels, Slot 2 received 0.0%). Image features are indexed via `image_ids` (no memory copy).

| Component | Status | Role |
|---|---|---|
| Image Encoder (ViT) | **Frozen** | Multi-scale visual features (FPN) |
| Fusion Encoder | **Frozen + Orthogonal LoRA** | Cross-attends image features with queries |
| `cross_attend_prompt` | **Frozen + Orthogonal LoRA** | Enriches encoder hidden states |
| Pixel Decoder | **Frozen** | Dense pixel embeddings (B, 256, H, W) |

---

### Module D: Multi-Texture Mask Head

**Status**: Fully trainable.

```python
query_proj  = MLP(queries)          # (B, 7, 256) -> (B, 7, 256)
pixel_proj  = Conv1x1(pixel_embed)  # (B, 256, H, W) -> (B, 256, H, W)
mask_logits = einsum("bqc, bchw -> bqhw", query_proj, pixel_proj)  # (B, 7, H, W)
```

---

## The Dustbin Query

Channel 0 is a learned 4096-dim embedding that absorbs all non-texture pixels.

```
Index:  [  0     ,  1   ,  2   , ...,  K  , K+1 , ...,  6  ]
Role:   [DUSTBIN , SEG_1, SEG_2, ..., SEG_K, PAD , ..., PAD ]
```

PAD slots have logits set to `-inf` before softmax/CE.

---

## Block-Diagonal Attention Mask

During both training and inference extraction, a custom 4D attention mask prevents Context Leakage between texture blocks:

```
         prefix  TEX_1  TEX_2  TEX_3
prefix  [causal  ────   ────   ─── ]
TEX_1   [  ok   causal ────   ──── ]
TEX_2   [  ok    BLOCK causal ──── ]   <- TEX_2 CANNOT see TEX_1
TEX_3   [  ok    BLOCK  BLOCK causal]   <- TEX_3 CANNOT see TEX_1 or TEX_2
```

All blocks can attend to the shared prefix (system + image + user prompt). Within each block, standard causal attention applies.

**Why this matters**: Without the mask, Qwen's causal attention causes `<|seg|>_2` to absorb semantic noise from TEXTURE_1's description. Ablation proved this inflates inter-texture cosine similarity from 0.16 (with mask) to 0.74 (without mask), causing Directional Drift in the projector.

---

## Loss Masking (Anti-Count-Collapse)

```python
labels = torch.full_like(input_ids, -100)  # mask everything
labels[input_ids == seg_token_id] = seg_token_id  # unmask only <|seg|>
```

This prevents the V3 failure mode where Qwen's LoRA learned to terminate after TEXTURE_1 (minimizing text-prediction loss at the expense of finding multiple textures). With loss masking, the LoRA ONLY receives LM gradient at `<|seg|>` positions. The primary training signal flows through mask loss: `Mask Loss -> SAM -> Projector -> [SEG] hidden states -> Qwen LoRA`.

---

## Orthogonal LoRA

Applied to SAM3's cross-attention layers. Constrains weight updates to directions orthogonal to SAM3's pretrained dominant singular vectors:

```
L_orth = || U_k^T @ (B @ A) ||_F^2
```

This preserves SAM3's zero-shot segmentation capability while adapting to texture-specific features.

---

## Training Process

### Three-Stage Curriculum (Cold Start Protection)

| Stage | Epochs | Trainable | Frozen | Purpose |
|---|---|---|---|---|
| **1. Projector Warmup** | 1-2 | Projector + mask head + dustbin | Qwen LoRA, SAM LoRA | Projector escapes random init |
| **2. Joint Co-Adaptation** | 3-5 | + Qwen LoRA (conservative LR) | SAM LoRA | LoRA gently refines [SEG] representations |
| **3. End-to-End** | 6+ | + SAM LoRA | — | Full joint convergence |

### Forward Pass Walkthrough

```
Step 1: Qwen Forward (teacher forcing with <|seg|> tokens)
  |  Image + prompt + "TEXTURE_1: desc <|seg|>\nTEXTURE_2: desc <|seg|>"
  |  Block-diagonal attention mask applied
  |  -> hidden_states + lm_loss (only on <|seg|> positions)
  |
Step 2: Extract <|seg|> Hidden States
  |  Clean token lookup (no regex, no position arithmetic)
  |  K vectors of dim 4096, each decoupled from other texture blocks
  |
Step 3: Build 7 Query Slots
  |  [DUSTBIN, SEG_1, ..., SEG_K, PAD, ..., PAD]
  |
Step 4: Bottleneck Projection
  |  (B, 7, 4096) -> (B, 7, 256)
  |
Step 5: SAM3 Batch Multiplexed
  |  (B, 7, 256) -> (B*7, 1, 256) -> Fusion Encoder -> Pixel Decoder
  |  -> (B, 7, H, W) mask logits
  |
Step 6: Loss
  |  Mask: CE + heavy Dice (weight 3.0) on pixel predictions
  |  LM: CE on <|seg|> tokens only (all text masked to -100)
  |  Orth: regularization on SAM LoRA
```

### Loss Functions

```
L_total = mask_weight * (CE + 3.0 * Dice) + lm_weight * LM_seg + orth_weight * L_orth
```

| Loss | Weight | Notes |
|---|---|---|
| **Cross-Entropy** | 1.0 | Pixel-wise, PAD channels = -inf |
| **Dice** | 3.0 | Per-channel, PAD excluded |
| **LM (seg-only)** | 0.1 | Only `<\|seg\|>` tokens — prevents count collapse |
| **Orthogonal Reg** | 0.01 | SAM LoRA stays in null space of pretrained weights |

### Differential Learning Rates

| Component | LR | Notes |
|---|---|---|
| Projector | 1e-4 (base) | Needs to learn fast during warmup |
| Mask Head + Dustbin | 1e-4 (base) | |
| Qwen LoRA | 2e-5 (0.2x base) | Conservative — protects pretrained weights |
| SAM3 Orth LoRA | 1e-5 (0.1x base) | Frozen in Stages 1-2 |

---

## Inference (Two-Pass Strategy)

```
Pass 1 — Generation (standard causal mask):
  Qwen generates the full texture description sequence naturally.
  Standard causal attention ensures no repetition.

Pass 2 — Extraction (block-diagonal mask):
  The FULL generated sequence is re-run through Qwen with the
  block-diagonal custom mask. Each <|seg|> hidden state is computed
  in isolation from other texture blocks — matching training conditions.

  If <|seg|> tokens are not yet emitted (early training), a regex
  fallback extracts from TEXTURE_N: line-end positions in Pass 1.
```

This two-pass approach guarantees train-test parity while preserving generation quality.

---

## Ablation History (V1-V5)

| Version | Key Change | Best RWTD mIoU | Failure Mode |
|---|---|---:|---|
| V1 | Direct 1-to-1, Qwen bypass P1 | 0.678 | Qwen LoRA overfitting |
| V2 | + Crop aug + cycle loss + constrained LoRA | 0.695 | SAM Slot 1 Addiction |
| V3 | + Batch Multiplexing | 0.703 | Phase 2 LLM co-training collapse |
| V4 | Frozen Qwen Oracle + Architectural Plug | 0.692 | Directional Drift (10.5M projector) |
| V4-Slim | + Information Bottleneck (2.6M) | **0.732** | First to beat ZS baseline (0.706) |
| **V5** | + [SEG] token + block-diagonal mask + loss masking | **TBD** | In progress |

Key discoveries:
- **V2**: SAM3 has extreme positional bias (Slot 1 gets 90.5%, Slot 2 gets 0.0%)
- **V3**: LLM co-training is fundamentally toxic — causes count collapse + projector drift + SAM regression
- **V4**: Directional Drift — the projector learns ADE20K-specific directions that don't generalize. Representation Collapse disproved (vectors get MORE separated, not less)
- **V4-Slim**: Information Bottleneck prevents Directional Drift. 4x param reduction forces generalization
- **V5**: Context Leakage from "1 to 6" prompt inflates inter-texture cosine from 0.16 to 0.74. Block-diagonal mask eliminates this

Full ablation data: `ablation/v1/` through `ablation/v4/`

---

## Project Structure

```
Qwen2SAM_DeTexture/
|
+-- configs/
|   +-- detexture.yaml              # V5 config (main)
|   +-- detexture_v4_slim.yaml      # V4-Slim bridge experiment
|
+-- models/
|   +-- qwen2sam_detexture.py       # V5 main model (SEG token, block mask, two-pass)
|   +-- qwen2sam_v4_slim.py         # V4-Slim bridge model
|   +-- projector.py                # V5 bottleneck projector (4096->512->256)
|   +-- orthogonal_lora.py          # Orthogonal LoRA wrapper
|   +-- losses.py                   # V5 losses (mask + LM_seg + orth)
|
+-- data/
|   +-- dataset.py                  # DeTextureDataset + V5 collator (<|seg|> + loss masking)
|
+-- training/
|   +-- train.py                    # V5 training loop (3-stage curriculum)
|   +-- train_v4_slim.py            # V4-Slim training loop
|   +-- monitor.py                  # Sanity checker, logger, plotter, test evaluator
|   +-- utils.py                    # Config, seed, scheduler, checkpointing
|
+-- scripts/
|   +-- analyze_vector_collapse.py  # Pre/post projector cosine similarity analysis
|   +-- analyze_visual_bias.py      # Orthogonal subspace projection (INSID3-inspired)
|   +-- ablation_live_ade20k.py     # Control group: live inference on in-domain data
|   +-- ablation_k2_only.py         # Prompt count ablation (1-to-6 vs exactly-2)
|   +-- generate_gt_embeds_v4.py    # GT embedding regeneration (V4 train/test alignment)
|   +-- inspect_sam3_text_encoder.py # SAM3 text encoder architecture probe
|
+-- ablation/
|   +-- v1/ v2/ v3/ v4/             # Per-version ablation studies + analyses
|   +-- vector_collapse_analysis.json
|   +-- visual_bias_analysis.json
|   +-- live_ade20k_control.json
```

---

## Configuration

```yaml
model:
  qwen_model: "Qwen/Qwen3-VL-8B-Instruct"
  lora_r: 8                       # Qwen LoRA rank
  lora_alpha: 16
  qwen_lr_scale: 0.2              # Qwen LR = base * 0.2 = 2e-5
  projector_hidden_dim: 512       # Bottleneck dimension
  sam3_lora_r: 32                 # SAM3 Orthogonal LoRA rank
  sam3_lr_scale: 0.1

curriculum:
  projector_warmup_epochs: 2      # Stage 1: only projector
  e2e_epoch: 5                    # Stage 3: SAM LoRA unfreezes

loss:
  mask_weight: 1.0
  ce_weight: 1.0
  dice_weight: 3.0
  lm_weight: 0.1                  # Only on <|seg|> tokens
  orthogonal_weight: 0.01

training:
  batch_size: 2
  gradient_accumulation_steps: 4  # Effective batch = 8
  num_epochs: 60
  learning_rate: 1.0e-4
```

---

## Getting Started

### Prerequisites

- PyTorch >= 2.1
- transformers >= 4.45
- peft >= 0.7
- sam3 (Meta AI)
- scipy, opencv-python, Pillow

### Training

```bash
cd /home/aviad/Qwen2SAM_DeTexture
python -m training.train --config configs/detexture.yaml
```

Resume from checkpoint:
```bash
python -m training.train --config configs/detexture.yaml --resume auto
```

### Data Format

```json
[
  {
    "image_path": "/path/to/image.jpg",
    "textures": [
      {"description": "Texture of rough stone...", "mask_path": "/path/to/mask1.png"},
      {"description": "Texture of smooth water...", "mask_path": "/path/to/mask2.png"}
    ]
  }
]
```

---

## Trainable Parameters

```
+-----------------------------------------------------------+
|                    TRAINABLE COMPONENTS                   |
+-----------------------------------------------------------+
| Qwen3-VL LoRA (r=8, q_proj + v_proj)    ~3.8M params      |
| Bottleneck Projector (4096->512->256)    ~2.1M params     |
| Multi-Texture Mask Head                  ~0.2M params     |
| DUSTBIN embedding (4096-dim)             4,096 params     |
| SAM3 Orthogonal LoRA (cross-attn)       ~0.3M params      |
+-----------------------------------------------------------+
| TOTAL TRAINABLE                          ~6.6M params     |
+-----------------------------------------------------------+

+-----------------------------------------------------------+
|                      FROZEN COMPONENTS                    |
+-----------------------------------------------------------+
| Qwen3-VL base weights                   ~8B params        |
| SAM3 Image Encoder + Fusion + Decoder    ~300M params     |
+-----------------------------------------------------------+
```
