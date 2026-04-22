# Qwen2SAM-Detecture

> *Part of the [Detecture](../) umbrella — this is the **model** component. The **data pipeline** that produces our training data lives at [../DetectureMiner/](../DetectureMiner/); the **benchmark** that scores us against sibling baselines lives at [../Qwen2SAM_Detecture_Benchmark/](../Qwen2SAM_Detecture_Benchmark/).*

**End-to-End VLM-Guided Multi-Texture Segmentation with [SEG] Token Grounding**

An end-to-end architecture that fuses a Vision-Language Model (**Qwen3-VL-8B**) with a Geometric Segmentation Engine (**SAM 3**) to segment images into 1–6 distinct texture regions. The model uses a dedicated **[SEG] token** to decouple visual grounding from language context, a **block-diagonal attention mask** to prevent cross-texture contamination, and the **Bridge** — a trainable projection into SAM 3's *native* 1024-dim text-encoder space that then reuses SAM's own frozen 1024 → 256 resizer, so Qwen's semantic richness is shape-matched into SAM instead of being squashed through a learned bottleneck.

Two additional mechanisms make live free-generation work at inference: a **Shifted-Zero LM cliff** that weights the grounding-token loss one position forward to preserve the emission-learning gradient, and **masked-row trainable SEG rows** (with warm-start from `<|im_end|>`) that calibrate `lm_head[SEG]` so the model actually chooses the token at decode time.

This README documents the current architecture. See [ablation/](ablation/) for the full experimental history and the pathologies each predecessor exposed.

---

## Table of Contents

- [Key Innovations](#key-innovations)
- [Architecture Overview](#architecture-overview)
- [Detailed Architecture](#detailed-architecture)
  - [Module A: Qwen3-VL with [SEG] Token](#module-a-qwen3-vl-with-seg-token)
  - [Module B: Bridge + Frozen SAM Resizer](#module-b-bridge--frozen-sam-resizer)
  - [Module C: SAM 3 with Batch Multiplexing](#module-c-sam-3-with-batch-multiplexing)
  - [Module D: Multi-Texture Mask Head](#module-d-multi-texture-mask-head)
- [The Dustbin Query](#the-dustbin-query)
- [Block-Diagonal Attention Mask](#block-diagonal-attention-mask)
- [Shifted-Zero LM Loss](#shifted-zero-lm-loss)
- [Masked-Row Trainable SEG Rows](#masked-row-trainable-seg-rows)
- [Training Process](#training-process)
  - [Two-Stage Curriculum](#two-stage-curriculum)
  - [Micro-Warmup Restart (Extension Runs)](#micro-warmup-restart-extension-runs)
  - [Forward Pass Walkthrough](#forward-pass-walkthrough)
  - [Loss Functions](#loss-functions)
  - [Differential Learning Rates](#differential-learning-rates)
- [Inference (Two-Pass Strategy)](#inference-two-pass-strategy)
- [Experimental History](#experimental-history)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Getting Started](#getting-started)

---

## Key Innovations

| Innovation | What it solves |
|---|---|
| **[SEG] Token Grounding** | Dedicated `<\|seg\|>` token after each texture description. Qwen's LoRA learns to pack visually-grounded spatial information into this token, decoupling it from noisy language context. |
| **Bridge to SAM's Native Width** | `Linear(4096 → 1024) + LayerNorm + GELU + Dropout(0.4)` maps Qwen [SEG] into SAM 3's native 1024-D text-encoder space; SAM's *frozen* pretrained resizer (`Linear 1024 → 256`) handles the final projection. Replaces the conventional `4096 → 512 → 256` bottleneck that squashed Qwen's semantic richness. |
| **Block-Diagonal Attention Mask** | Prevents Context Leakage: TEXTURE_2's `<\|seg\|>` hidden state cannot attend to TEXTURE_1's tokens. Proven to reduce inter-texture cosine similarity from 0.74 to 0.16. |
| **Shifted-Zero LM Cliff** | Per-token weight `1 − e^(−2d)` with the zero slot placed one position *past* each assistant `<\|seg\|>`, not at `<\|seg\|>` itself. Weight at `<\|seg\|>` = 1.0 (restores the emission-learning gradient); weight at `<\|seg\|>+1` = 0.0 (geometric freedom); d ≥ 3 ≈ 1.0 (full linguistic anchoring). Cures the Emission Failure that the naïve `λ(SEG)=0` cliff induces. |
| **Masked-Row Trainable SEG Rows** | Unfreezes only the SEG rows of `embed_tokens.weight` and `lm_head.weight` (16,384 effective parameters) via a gradient-mask hook, warm-started from `<|im_end|>`. Calibrates `lm_head[SEG]` so the model actually emits `[SEG]` at inference — LoRA on q/v_proj cannot reach these rows on its own. |
| **Batch Multiplexing** | Each query is fed to SAM independently in its own "slot 1" position. Eliminates SAM 3's pretrained positional bias where only the first query slot received attention (Slot 1: 90.5 %, Slot 2: 0.0 %). |
| **Dustbin Query** | A learned embedding that absorbs non-texture pixels (objects, sky, etc.), preventing false texture assignments. |
| **Permanent SAM Freeze** | SAM stays 100 % frozen. Zero-Shot SAM 3 with an object-aware prompt hit **0.928 mIoU** on RWTD — fine-tuning only destroys this OOD generality. Every prior SAM-LoRA unfreezing attempt in our experimental history produced an OOD regression. |
| **Two-Stage Curriculum** | (1) Bridge warmup (Qwen + SAM frozen); (2) Qwen LoRA thaw at 1e-6 with masked-row SEG rows unfrozen and Bridge LR decayed 10×. No Stage 3. |
| **Micro-Warmup Restart** | LR restart protocol for extension runs: on resume, rebuild the scheduler with each group's peak at `original_base_LR × 0.15`, apply a 2-epoch linear warmup, then cosine-decay over the remaining epochs — gentle LR excursion without shocking weights already in a good basin. |
| **Universal Object-Aware Prompt** | Same prompt for training and inference: "a prominent foreground object and its background, or contrasting materials" — matches the object-aware framing that unlocked the 0.928 ZS SAM result on RWTD. |
| **Two-Pass Inference** | Pass 1: standard causal generation (no repetition). Pass 2: block-diagonal masked forward (decoupled [SEG] extraction). Train–test parity guaranteed. |

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
    |  (LoRA r=8 on q,v +        |     |  Image Encoder             |
    |   masked-row SEG rows)     |     |  1008x1008 -> FPN features |
    |                            |     +-------------+--------------+
    |  Input: image + prompt     |                   |
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
    |  MODULE B: Bridge          |                   |
    |  (4.2M trainable params)   |                   |
    |                            |                   |
    |  Linear(4096 -> 1024) + LN |                   |
    |   + GELU + Dropout(0.4)    |                   |
    +-------------+--------------+                   |
                  |                                  |
                  v                                  |
    +----------------------------+                   |
    |  SAM.resizer (FROZEN)      |                   |
    |  Linear(1024 -> 256)       |                   |
    |  reused from SAM3's own    |                   |
    |  language_backbone         |                   |
    +-------------+--------------+                   |
                  |                                  |
                  +----------------------------------+
                  |                                  |
                  v                                  v
    +-------------------------------------------------------+
    |  MODULE C: SAM3 — Batch Multiplexed (FROZEN)          |
    |  (B, 7, 256) -> flatten -> (B*7, 1, 256)              |
    |  Each query gets its own "slot 1" position            |
    |                                                       |
    |  Fusion Encoder  |  Cross-Attn  |  Pixel Decoder      |
    |  (all frozen — no LoRA)                               |
    +----------------------------+--------------------------+
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
**Status**: Frozen base weights + LoRA (r=8) on `q_proj` and `v_proj` + masked-row trainable SEG rows of `embed_tokens` and `lm_head`.
**Training**: LoRA receives gradients through the spatial path (mask loss → SAM → Bridge → [SEG]) AND linguistic supervision via the Shifted-Zero LM cliff. The SEG rows receive gradient only at their single row each, warm-started from the `<|im_end|>` token.

**Output format** (teacher-forced during training):
```
TEXTURE_1: Texture of rough mossy stone with granular surface in the foreground <|seg|>
TEXTURE_2: Texture of smooth flowing water with reflective sheen in the center <|seg|>
TEXTURE_3: Texture of dry sandy ground with ripple patterns on the right <|seg|>
```

The `<|seg|>` token is a dedicated grounding anchor. Unlike natural language end-of-line tokens (which absorb context from ALL prior tokens via causal attention), `<|seg|>` hidden states are computed under a **block-diagonal attention mask** that isolates each texture block from previous blocks. This produces clean, decoupled visual representations.

**Prompt note**: both SYSTEM and USER prompts include an object-aware example ("a prominent foreground object and its background, or contrasting materials"), and the format template explicitly shows `<|seg|>` per TEXTURE line. Because the prompt contains literal `<|seg|>` text which tokenises as the SEG special token, all seg-position lookups filter by the assistant-region boundary.

---

### Module B: Bridge + Frozen SAM Resizer

```
Bridge (trainable, ~4.2M params):
    Linear(4096 -> 1024) + LayerNorm(1024) + GELU + Dropout(0.4)

Then:
    SAM.backbone.language_backbone.resizer (FROZEN, Linear 1024 -> 256)
```

**Why**: The conventional square-bottleneck projector (`4096 → 512 → 256`) compresses Qwen's rich 4096-D `[SEG]` representation ~16×. A parallel Zero-Shot SAM 3 experiment using a highly descriptive, object-aware prompt (no VLM, no projector, raw text path) achieved **0.928 mIoU on RWTD**. SAM natively understands complex semantic descriptions; the limiter was the projector squashing Qwen's vocabulary before SAM could see it.

Our fix: widen the trainable projection into SAM's **native** text-encoder width (1024-D), then hand the final 1024 → 256 step back to SAM's own pretrained `resizer` layer, kept frozen. The Bridge's job is shape-matching, not semantic translation — SAM's pretrained resizer already knows that map.

**Dropout(0.4)** is load-bearing. At 4.2M params without heavy dropout, the Directional Drift pathology (projector memorising ADE20K-specific manifold directions) would reappear.

---

### Module C: SAM 3 with Batch Multiplexing

Each of the 7 query vectors is fed to SAM **independently** by flattening the query dimension into the batch dimension:

```
(B, 7, 256) -> reshape -> (B*7, 1, 256) -> SAM -> (B*7, 1, H, W) -> reshape -> (B, 7, H, W)
```

This eliminates SAM 3's pretrained positional bias (Slot 1 received 90.5 % of pixels, Slot 2 received 0.0 %). Image features are indexed via `image_ids` (no memory copy).

| Component | Status | Role |
|---|---|---|
| Image Encoder (ViT) | **Frozen** | Multi-scale visual features (FPN) |
| Fusion Encoder | **Frozen** (no LoRA) | Cross-attends image features with queries |
| `cross_attend_prompt` | **Frozen** (no LoRA) | Enriches encoder hidden states |
| Pixel Decoder | **Frozen** | Dense pixel embeddings (B, 256, H, W) |
| `language_backbone.resizer` | **Frozen** (reused by Bridge) | Linear(1024, 256) — semantic channel Qwen→SAM |

**No SAM LoRA.** With the ZS ceiling at 0.928, every training attempt on SAM has been net-negative for OOD generality.

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

## Shifted-Zero LM Loss

Earlier iterations used cosine-decayed LM weighting that kept pressure ~0.5 across most of each block, producing a constant tug-of-war between Dice (wants geometric freedom on `[SEG]`) and LM (wants linguistic fidelity on every text token); Dice plateaued at 0.43.

A sharp exponential cliff with the zero placed **at the `[SEG]` token itself** (the "obvious" design) removes this tug-of-war in the teacher-forced regime but breaks free generation completely: the model never emits `[SEG]`, substituting `<|tool_call|>` or `<|im_end|>` instead, and live RWTD mIoU drops to 0.000.

**Why**: under causal-LM shift semantics, loss at position *i* supervises the prediction of token *i+1*. Setting `λ(SEG) = 0` frees `h_SEG` geometrically, but also erases the gradient that would teach the model to continue correctly after emitting `[SEG]` — and with it, the incentive to emit `[SEG]` in the first place.

**Shifted-Zero fix** — place the zero **one position forward**:

```
Weight(d) = 1 − exp(−α · d),   α = 2.0
     with λ(SEG)   = 1.0       (restore emission-learning gradient)
          λ(SEG+1) = 0.0       (geometric freedom slot)
          λ(other) = 1 − e^(−2d) as before
```

| Position | Weight | Purpose |
|---|---|---|
| `<\|seg\|>` itself | **1.000** | Force emission; supervise post-SEG continuation |
| one token past `<\|seg\|>` | **0.000** | Geometric freedom slot (unused syntactically) |
| two tokens past | 0.865 | Linguistic supervision resumes |
| three tokens past | 0.982 | |
| ≥ four tokens past | ≈ 1.0 | Full linguistic anchoring |
| all other assistant tokens | `1 − e^(−2d)` | Standard cliff |

Prompt-side `<|seg|>` tokens (from the format-template example) are filtered out of the distance calculation — only real assistant-region SEGs count.

---

## Masked-Row Trainable SEG Rows

Shifted-Zero alone recovers teacher-forced Oracle mIoU but still yields live-inference mIoU = 0.000. The root cause is in `lm_head`: the row `lm_head.weight[SEG]` — whose inner product with the final hidden state produces `[SEG]`'s logit — is frozen at random initialisation. LoRA on `q_proj`/`v_proj` cannot reach this row. With an uncalibrated `lm_head[SEG]`, the SEG logit is dominated by well-trained rows like `<|tool_call|>` or `<|im_end|>`, and those tokens are chosen instead.

Fix in two parts:

**1. Warm-start from `<|im_end|>`.** At model initialisation:
```python
embed.weight[SEG_id].copy_(embed.weight[im_end_id])
lm_head.weight[SEG_id].copy_(lm_head.weight[im_end_id])
```
This places `[SEG]` near a token Qwen already associates with "finalise a region of thought" in both the input and output embedding spaces.

**2. Masked-row gradient routing.** Unfreeze the full `embed_tokens.weight` and `lm_head.weight` tensors (so PyTorch computes gradients for them at all) and register a gradient hook that zeros every row except the SEG row:

```python
def _make_mask_hook(row_id):
    def _hook(grad):
        mask = torch.zeros_like(grad)
        mask[row_id] = 1.0
        return grad * mask
    return _hook

for w in (embed.weight, lm_head.weight):
    w.requires_grad_(True)
    w.register_hook(_make_mask_hook(seg_id))
```

This restricts actual weight updates to the two 8192-dimensional SEG rows — 16,384 effective parameters — while leaving the remaining ~150K rows of each tensor exactly as pretrained. The SEG rows are added to the optimizer in their own parameter group at a dedicated learning rate.

Combined effect: live-inference RWTD mIoU jumps from 0.000 to 0.794 at epoch 16, while Oracle continues from 0.738 to 0.781 at the same checkpoint.

---

## Training Process

### Two-Stage Curriculum

| Stage | Epochs | Trainable | Frozen | Key event |
|---|---|---|---|---|
| **1. Bridge Warmup** | 1–8 | Bridge (1e-4) + mask head + dustbin | Qwen LoRA, SEG rows, SAM (full) | Bridge learns to route Qwen's 4096-D into SAM's 1024-D native space |
| **2. Qwen Sync + SEG Rows + Bridge Decay** | 9–30 | + Qwen LoRA (1e-6), + masked-row SEG rows, Bridge LR decayed 10× → 1e-5 | SAM (full) | Qwen learns to emit `[SEG]` correctly and build a grounding embedding on a slowed-down Bridge |

SAM is permanently frozen across all stages (no Stage 3).

### Micro-Warmup Restart (Extension Runs)

For extensions past the initial horizon (e.g. 20 → 30 epochs), a full LR re-warmup to the original base would shock weights already sitting in a good basin. Instead, resume training with:

```bash
python -m training.train \
    --config configs/detecture.yaml \
    --resume checkpoints/epoch_20.pt \
    --resume-lr-scale 0.15 \
    --resume-warmup-epochs 2
```

This rebuilds the LR scheduler with each parameter group's peak set to `original_base_LR × 0.15` (preserving per-group ratios between Bridge, Qwen LoRA, and SEG rows), applies a short 2-epoch linear warmup, then cosine-decays over the remaining 10 epochs only. AdamW moment estimates are preserved across the restart — only the LR envelope changes.

### Forward Pass Walkthrough

```
Step 1: Qwen Forward (teacher forcing with <|seg|> tokens)
  |  Image + prompt + "TEXTURE_1: desc <|seg|>\nTEXTURE_2: desc <|seg|>"
  |  Block-diagonal attention mask applied
  |  -> hidden_states + qwen_logits (for Shifted-Zero weighted CE)
  |
Step 2: Extract <|seg|> Hidden States
  |  Clean token lookup filtered to assistant region (prompt contains literal <|seg|>)
  |  K vectors of dim 4096, each decoupled from other texture blocks
  |
Step 3: Build 7 Query Slots
  |  [DUSTBIN, SEG_1, ..., SEG_K, PAD, ..., PAD]
  |
Step 4: Bridge + frozen SAM resizer
  |  Bridge(4096 -> 1024) + SAM.resizer(1024 -> 256)
  |  -> (B, 7, 256)
  |
Step 5: SAM3 Batch Multiplexed (frozen)
  |  (B, 7, 256) -> (B*7, 1, 256) -> Fusion Encoder -> Pixel Decoder
  |  -> (B, 7, H, W) mask logits
  |
Step 6: Loss
  |  Mask: CE + heavy Dice (weight 3.0) on pixel predictions
  |  LM: Shifted-Zero weighted CE (w=1 at [SEG], w=0 at [SEG]+1, w≈1 elsewhere)
```

### Loss Functions

```
L_total = mask_weight * (CE + 3.0 * Dice) + lm_weight * LM_shifted_zero
```

| Loss | Weight | Notes |
|---|---|---|
| **Cross-Entropy** | 1.0 | Pixel-wise, PAD channels = -inf |
| **Dice** | 3.0 | Per-channel, PAD excluded |
| **LM (Shifted-Zero cliff)** | 0.1 | `Weight(d) = 1 − e^(−2d)` with zero slot at `[SEG]+1` |
| ~~Orthogonal Reg~~ | 0.0 | Retired (no SAM LoRA) |

### Differential Learning Rates (stage-dependent)

| Component | Stage 1 (1–8) | Stage 2 (9–30) |
|---|---|---|
| Bridge (4.2M) | 1e-4 | **1e-5** (decayed 10×) |
| Mask Head + Dustbin | 1e-4 | 1e-4 |
| Qwen LoRA (3.8M) | frozen | 1e-6 (0.01×) |
| SEG rows (16.4K effective) | frozen at warm-start | 1e-5 (masked-row) |
| SAM (full) | frozen | frozen |

Under a Micro-Warmup Restart, all non-frozen rows above scale uniformly by `resume_lr_scale` (default 0.15).

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
  Only assistant-region <|seg|> tokens count (prompt-side ones filtered).

  If <|seg|> tokens are not yet emitted (early training), a regex
  fallback extracts from TEXTURE_N: line-end positions in Pass 1,
  also filtered to the generated region (>= prompt_len).
```

This two-pass approach guarantees train–test parity while preserving generation quality.

---

## Experimental History

Each architectural component was motivated by a specific pathology discovered through rigorous ablation. Entries below are numbered predecessor experiments archived under `ablation/v1/` through `ablation/v7/`; the final row is the method proposed in the current paper.

| Version | Key change | RWTD mIoU (best honest) | Pathology resolved / failure |
|---|---|---|---|
| V1 | Initial VLM + SAM wiring | 0.678 | Qwen LoRA overfitting |
| V2 | Query-slot design | 0.695 | **Slot-1 Addiction** discovered |
| V3 | LLM co-training | 0.703 | Count Collapse + projector drift + SAM regression |
| V4 | 10.5M projector | 0.692 | **Directional Drift** (memorises ADE20K directions) |
| V4-Slim | Bottleneck 2.1M | 0.732 | **First to beat ZS baseline** |
| V5-Oracle | Block-diagonal mask | 0.810 (Oracle) | Geometry-only ceiling; language collapsed in live |
| V5-Live | | 0.136 | **Language Collapse** — LoRA forgot language |
| V6 | Proximity-decayed LM | 0.694 (live exactly_2) | Language preserved; latent co-adaptation churn in Stage 3 |
| Bridge + naïve cliff (λ_SEG=0), Oracle | | 0.738 (Oracle) | **Emission Failure** — live collapses to 0.000 |
| Bridge + Shifted-Zero cliff, Oracle | | 0.781 (Oracle) | — |
| Bridge + Shifted-Zero + Masked-Row SEG + Warm-Start, live (ep 16) | | **0.794** | Emission Failure cured |
| ZS SAM 3 (object-aware prompt) | No training | **0.928** | Reframed the ceiling: SAM natively understands object-aware descriptions |
| **Our final method** | Bridge + Shifted-Zero + Masked-Row + 30-ep Micro-Warmup Restart | *in progress* | Live matches or exceeds Oracle |

Full ablation data: `ablation/v1/` through `ablation/v7/` and `ablation/V7_Shifted_Zero/`.

---

## Project Structure

```
Qwen2SAM_Detecture/
|
+-- configs/
|   +-- detecture.yaml              # Main training config (30 epochs, 2-stage curriculum)
|
+-- models/
|   +-- qwen2sam_detecture.py       # Main model (SEG token, block mask, two-pass inference, Bridge + frozen resizer, masked-row SEG)
|   +-- bridge.py                   # Bridge (Linear 4096 -> 1024 + LN + GELU + Dropout 0.4)
|   +-- losses.py                   # Losses (mask + Shifted-Zero-weighted LM)
|
+-- data/
|   +-- dataset.py                  # DetectureDataset + collator (Shifted-Zero lm_weights)
|
+-- training/
|   +-- train.py                    # Two-stage training loop with Micro-Warmup Restart support
|   +-- monitor.py                  # Sanity checker, logger, plotter, live test evaluator
|   +-- utils.py                    # Config, seed, scheduler, checkpointing
|
+-- scripts/
|   +-- ablation_exact_k2_rwtd.py          # RWTD honest exactly-2 live inference
|   +-- evaluate_bridge_oracle.py          # Oracle (teacher-forced) evaluator
|   +-- eval_checkpoint_all.py             # Batch Oracle+E2E evaluator over multiple checkpoints
|   +-- regenerate_unified_plots.py        # Stitch metrics across resumed runs
|   +-- analyze_vector_collapse.py         # Pre/post projector cosine analysis
|   +-- analyze_visual_bias.py             # Orthogonal subspace projection
|
+-- ablation/
|   +-- v1/ v2/ v3/ v4/ v5/ v6/ v7/        # Per-version ablation studies + analyses
|   +-- V7_Shifted_Zero/experiment_log.md  # Emission-failure diagnosis + Shifted-Zero + Masked-Row fix log
|   +-- vector_collapse_analysis.json
|   +-- visual_bias_analysis.json
|
+-- checkpoints/
|   +-- logs/  plots/  test_results/       # Populated during training
```

---

## Configuration

```yaml
model:
  qwen_model: "Qwen/Qwen3-VL-8B-Instruct"
  lora_r: 8                       # Qwen LoRA rank
  lora_alpha: 16
  qwen_lr_scale: 0.01             # Qwen LR = base * 0.01 = 1e-6
  projector_hidden_dim: 1024      # Bridge output width (SAM3 native text width)
  projector_dropout: 0.4          # Heavy dropout on Bridge

curriculum:
  projector_warmup_epochs: 8      # Stage 1: Bridge warmup (Qwen + SAM frozen)
  projector_lr_decay_at_stage2: 0.1  # Bridge LR x0.1 at Stage 2 entry

loss:
  mask_weight: 1.0
  ce_weight: 1.0
  dice_weight: 3.0
  lm_weight: 0.1                  # Shifted-Zero cliff, alpha=2.0 (hardcoded in dataset.create_labels)
  orthogonal_weight: 0.0          # Retired (no SAM LoRA)

training:
  batch_size: 2
  gradient_accumulation_steps: 4  # Effective batch = 8
  num_epochs: 30                  # 20-epoch cosine run + 10-epoch Micro-Warmup Restart
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
cd /home/aviad/Detecture/Qwen2SAM_Detecture
python -m training.train --config configs/detecture.yaml
```

Resume from checkpoint (default: fast-forward the existing schedule):
```bash
python -m training.train --config configs/detecture.yaml --resume auto
```

Resume with a Micro-Warmup Restart (recommended for extension runs past a late-stage optimum):
```bash
python -m training.train \
    --config configs/detecture.yaml \
    --resume checkpoints/epoch_20.pt \
    --resume-lr-scale 0.15 \
    --resume-warmup-epochs 2
```

### Evaluation

Run Oracle (teacher-forced) + E2E (live exactly-2) on one or many checkpoints in one go:
```bash
python scripts/eval_checkpoint_all.py --checkpoint \
    checkpoints/epoch_4.pt \
    checkpoints/epoch_8.pt \
    checkpoints/epoch_12.pt \
    checkpoints/epoch_16.pt
```

E2E is auto-skipped for checkpoints before epoch 12 (pre-Shifted-Zero-era checkpoints return 0.0 by construction; only Oracle is meaningful there).

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
|                 TRAINABLE COMPONENTS                      |
+-----------------------------------------------------------+
| Qwen3-VL LoRA (r=8, q_proj + v_proj)    3,833,856         |
| Bridge (Linear 4096->1024 + LN + Drop)   4,197,376        |
| Multi-Texture Mask Head                    197,376        |
| DUSTBIN embedding (4096-dim)                 4,096        |
| Masked-row SEG rows (embed + lm_head)       16,384 (eff.) |
+-----------------------------------------------------------+
| TOTAL TRAINABLE                          8,249,088 (~8.2M)|
+-----------------------------------------------------------+

+-----------------------------------------------------------+
|                   FROZEN COMPONENTS                       |
+-----------------------------------------------------------+
| Qwen3-VL base weights                   ~8B params        |
| Qwen3-VL embed/lm_head (non-SEG rows)   2 x (150K x 8192) |
| SAM3 Image Encoder + Fusion + Decoder   ~300M params      |
| SAM3 language_backbone.resizer (reused)  1024x256 + 256   |
+-----------------------------------------------------------+
```
