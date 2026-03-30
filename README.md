# Qwen2SAM-DeTexture

**End-to-End VLM-Guided Multi-Texture Segmentation**

An E2E architecture that fuses a Vision-Language Model (**Qwen3-VL-8B**) with a Geometric Segmentation Engine (**SAM 3**) to segment images into 1-5 distinct texture regions. The model learns to isolate raw materials and textures (e.g., "rough mossy rock", "smooth flowing water") while explicitly absorbing non-texture pixels (objects, background) into a dedicated **Dustbin** channel.

---

## Table of Contents

- [Key Innovations](#key-innovations)
- [Architecture Overview](#architecture-overview)
- [Detailed Architecture](#detailed-architecture)
  - [Module A: Qwen3-VL (The Prompt Generator)](#module-a-qwen3-vl-the-prompt-generator)
  - [Module B: MLP Projector (The Bridge)](#module-b-mlp-projector-the-bridge)
  - [Module C: SAM 3 (The Geometry Engine)](#module-c-sam-3-the-geometry-engine)
  - [Module D: Multi-Texture Mask Head](#module-d-multi-texture-mask-head)
- [The Dustbin Query](#the-dustbin-query)
- [Orthogonal LoRA](#orthogonal-lora)
- [Training Process](#training-process)
  - [Forward Pass Walkthrough](#forward-pass-walkthrough)
  - [Hungarian Matching](#hungarian-matching)
  - [Loss Functions](#loss-functions)
  - [Gradient Strategy](#gradient-strategy)
- [Inference](#inference)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Getting Started](#getting-started)

---

## Key Innovations

| Innovation | What it solves |
|---|---|
| **Variable-K texture segmentation (1-5)** | Prior work was limited to exactly 2 textures (A/B). This model handles any number dynamically. |
| **Dustbin Query** | A learned embedding that absorbs non-texture pixels (objects, sky, etc.), preventing false texture assignments. |
| **Orthogonal LoRA** | Fine-tunes SAM3's cross-attention without catastrophic forgetting of zero-shot capabilities. The LoRA weight updates are regularized to be orthogonal to SAM3's pretrained weight subspace. |
| **Single-token extraction from causal LM** | Instead of mean-pooling multi-token descriptions (which dilutes information), we extract the single hidden state at each `<TEX_i>` marker token. Causal LMs naturally concentrate all context into the final token. |
| **Batched 6-query SAM3 pass** | All 6 query vectors (DUSTBIN + up to 5 textures) are processed in a single forward pass through SAM3's Fusion Encoder, not in a loop. Full GPU parallelization. |
| **Smooth 3-layer MLP projector** | Prevents information loss during the 8x dimensionality reduction (2048 -> 256) with intermediate LayerNorm and GELU activations. |
| **Dynamic prompt (train-test consistency)** | During training, the model sees `N = "1 to 5"` instead of the GT count, forcing it to learn to count textures autonomously. |

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
    |  (frozen + LoRA on q,v)    |     |  Image Encoder             |
    |                            |     |  1008x1008 -> FPN features |
    |  Input: image + prompt     |     +-------------+--------------+
    |  Output: text descriptions |                   |
    |  with <TEX_i> markers      |                   |
    +-------------+--------------+                   |
                  |                                  |
                  v                                  |
    +----------------------------+                   |
    |  Extract <TEX_i> hidden    |                   |
    |  states (single token each)|                   |
    |                            |                   |
    |  K vectors of dim 2048     |                   |
    +-------------+--------------+                   |
                  |                                  |
                  v                                  |
    +----------------------------+                   |
    |  Build 6 Query Slots       |                   |
    |                            |                   |
    |  [DUSTBIN, TEX_1, ...,     |                   |
    |   TEX_K, PAD, ..., PAD]    |                   |
    |                            |                   |
    |  6 vectors of dim 2048     |                   |
    +-------------+--------------+                   |
                  |                                  |
                  v                                  |
    +----------------------------+                   |
    |  MODULE B: MLP Projector   |                   |
    |  (trainable from scratch)  |                   |
    |                            |                   |
    |  2048 -> 1024 + LN + GELU  |                   |
    |  1024 -> 512 + GELU        |                   |
    |  512  -> 256               |                   |
    |                            |                   |
    |  6 vectors of dim 256      |                   |
    +-------------+--------------+                   |
                  |                                  |
                  +----------------------------------+
                  |                                  |
                  v                                  v
    +-------------------------------------------------------+
    |  MODULE C: SAM3 Semantic Path (SINGLE BATCHED PASS)    |
    |                                                        |
    |  Fusion Encoder (frozen + Orthogonal LoRA)             |
    |    - Cross-attends image features with 6 queries       |
    |    - All 6 queries processed simultaneously            |
    |                        |                               |
    |                        v                               |
    |  SegHead cross_attend_prompt (frozen + Orth. LoRA)     |
    |    - Encoder hidden states attend to 6 queries         |
    |                        |                               |
    |                        v                               |
    |  Pixel Decoder (frozen)                                |
    |    - encoder_hs + backbone FPN -> pixel_embed          |
    |    - Output: (B, 256, H, W)                            |
    +----------------------------+---------------------------+
                                 |
                                 v
    +-------------------------------------------------------+
    |  MODULE D: Multi-Texture Mask Head (trainable)         |
    |                                                        |
    |  query_proj = MLP(queries)        -> (B, 6, 256)       |
    |  pixel_proj = Conv1x1(pixel_embed) -> (B, 256, H, W)   |
    |                                                        |
    |  mask_logits = einsum("bqc, bchw -> bqhw")             |
    |                                                        |
    |  Output: (B, 6, H, W) raw logits                       |
    |    Channel 0: DUSTBIN (background/objects)             |
    |    Channels 1-K: Texture regions                       |
    |    Channels K+1 to 5: PAD (masked to -inf)             |
    +-------------------------------------------------------+
                                 |
                                 v
                    +------------------------+
                    |  Softmax(dim=1) -> WTA  |
                    |  Pixel-wise assignment  |
                    +------------------------+
```

---

## Detailed Architecture

### Module A: Qwen3-VL (The Prompt Generator)

**Model**: `Qwen/Qwen3-VL-8B-Instruct`
**Status**: Frozen base weights + LoRA (rank=16) on `q_proj` and `v_proj` attention layers.

**Role**: Analyzes the input image and generates rich texture descriptions. The model is prompted with:

```
System: "You are an expert at analyzing surface textures in images.
         Always respond in the exact format requested, with no extra text."

User: [image] "This image contains exactly 1 to 5 main visually distinct
       regions separated by boundaries..."
```

**Output format** (teacher-forced during training):
```
TEXTURE_1: Texture of rough mossy stone with granular surface in the foreground <TEX_1>
TEXTURE_2: Texture of smooth flowing water with reflective sheen in the center <TEX_2>
TEXTURE_3: Texture of dry sandy ground with ripple patterns on the right <TEX_3>
```

**Key design**: Each description ends with a special `<TEX_i>` token. We extract the **single hidden state** at that token position (not mean-pooling). In causal language models, the final token of a sequence naturally accumulates all preceding context, making it the richest representation.

**5 special tokens** added to Qwen's vocabulary: `<TEX_1>` through `<TEX_5>`.

---

### Module B: MLP Projector (The Bridge)

**Status**: Fully trainable from scratch.

Reduces the 2048-dimensional Qwen hidden states to SAM3's 256-dimensional query space. A naive single-layer projection would destroy semantic information. Instead, we use a **3-layer smooth bottleneck**:

```
Layer 1: Linear(2048 -> 1024) + LayerNorm(1024) + GELU
Layer 2: Linear(1024 -> 512)  + GELU
Layer 3: Linear(512  -> 256)
```

The `LayerNorm` after the first layer stabilizes the distribution shift between the LLM and segmentation domains.

---

### Module C: SAM 3 (The Geometry Engine)

**Components used** (from the SAM3 architecture):

| Component | Status | Role |
|---|---|---|
| Image Encoder (ViT backbone) | **Frozen** | Extracts multi-scale visual features (FPN) |
| Fusion Encoder | **Frozen + Orthogonal LoRA** | Cross-attends image features with our 6 text queries |
| `seg_head.cross_attend_prompt` | **Frozen + Orthogonal LoRA** | Enriches encoder hidden states with query semantics |
| Pixel Decoder | **Frozen** | Produces dense pixel embeddings (B, 256, H, W) |

**Components NOT used**: DETR Decoder (200 object queries), Detector, Tracker, Masklet Matcher, Semantic Seg Head (`Conv2d(256,1)` -- only 1 channel, insufficient for 6 outputs).

**Why not the Semantic Seg Head?** The frozen `Conv2d(256, 1)` can only output a single channel. With 6 queries fused into the pixel embeddings simultaneously, we need 6 output channels. The Multi-Texture Mask Head (Module D) solves this via dot-product between queries and pixel features.

**Single batched pass**: All 6 queries are concatenated into a single prompt tensor `(B, 6, 256)` and processed through the Fusion Encoder in one forward pass. No Python for-loops over queries.

---

### Module D: Multi-Texture Mask Head

**Status**: Fully trainable from scratch.

This is the dot-product mask prediction head, inspired by SAM3's `MaskPredictor`:

```python
query_proj  = MLP(queries)          # (B, 6, 256) -> (B, 6, 256)
pixel_proj  = Conv1x1(pixel_embed)  # (B, 256, H, W) -> (B, 256, H, W)
mask_logits = einsum("bqc, bchw -> bqhw", query_proj, pixel_proj)  # (B, 6, H, W)
```

Each of the 6 queries "votes" for pixels that belong to its texture. The `einsum` computes this dot-product in parallel across all queries and all pixels.

---

## The Dustbin Query

The **DUSTBIN** is a learned embedding (2048-dim, trainable) that occupies **channel 0** of the output. It absorbs all non-texture pixels: objects, sky, background, or anything that doesn't belong to a distinct texture region.

**Why channel 0?** The ground-truth index mask uses value `0` for background/non-texture pixels. By placing DUSTBIN at channel 0, the `CrossEntropyLoss` target indices align directly with the output channels -- no remapping needed.

**Query slot layout** (always 6 slots):
```
Index:  [  0     ,   1   ,   2   , ...,  K  , K+1 , ...,  5  ]
Role:   [DUSTBIN , TEX_1 , TEX_2 , ..., TEX_K, PAD , ..., PAD ]
```

PAD slots have their logits set to `-inf` before softmax/CE, ensuring they never receive probability mass.

---

## Orthogonal LoRA

Standard LoRA fine-tuning risks overwriting SAM3's powerful pretrained features (catastrophic forgetting). **Orthogonal LoRA** constrains the weight updates to directions that don't interfere with the original capabilities.

**How it works:**

1. For each frozen weight matrix `W_0`, compute its top-k singular vectors via SVD:
   ```
   U_k, S_k, V_k = SVD(W_0)  -->  U_k: (out, k) dominant directions
   ```

2. The LoRA update is `DeltaW = B @ A` (low-rank). We penalize its projection onto `W_0`'s dominant subspace:
   ```
   L_orth = || U_k^T @ DeltaW ||_F^2
   ```

3. This penalty is added to the total loss, pushing `DeltaW` into the **null space** of `W_0`'s dominant singular vectors -- the "free directions" that don't disturb pretrained behavior.

**Applied to**: Cross-attention layers in the Fusion Encoder and `seg_head.cross_attend_prompt`, specifically the Q and V projection weights.

```
Fusion Encoder Layer 0: cross_attn_image.q_proj (OrthLoRA), cross_attn_image.v_proj (OrthLoRA)
Fusion Encoder Layer 1: cross_attn_image.q_proj (OrthLoRA), cross_attn_image.v_proj (OrthLoRA)
...
SegHead cross_attend_prompt: q_proj (OrthLoRA), v_proj (OrthLoRA)
```

---

## Training Process

### Forward Pass Walkthrough

Here is the exact flow for a single training image with **K_gt = 3** ground-truth textures, where Qwen predicts **K_pred = 4** (one hallucination):

```
Step 1: Qwen Forward (teacher forcing)
  |  Image + prompt "This image contains 1 to 5 regions..."
  |  -> hidden_states (B, seq_len, 2048) + lm_loss
  |
Step 2: Extract <TEX_i> Hidden States
  |  Find <TEX_1>, <TEX_2>, <TEX_3>, <TEX_4> in token sequence
  |  Extract 4 hidden state vectors (each 2048-dim)
  |  K_pred = 4
  |
Step 3: Build 6 Query Slots
  |  Slot 0: DUSTBIN (learned embedding)
  |  Slot 1: TEX_1 hidden state
  |  Slot 2: TEX_2 hidden state
  |  Slot 3: TEX_3 hidden state
  |  Slot 4: TEX_4 hidden state  (the hallucination)
  |  Slot 5: PAD (zeros)
  |  -> (B, 6, 2048)
  |
Step 4: MLP Projection
  |  (B, 6, 2048) -> (B, 6, 256)
  |
Step 5: SAM3 Backbone (frozen, computed once)
  |  Image -> FPN features
  |
Step 6: SAM3 Semantic Path (single batched pass)
  |  Fusion Encoder: image features x 6 queries
  |  -> cross_attend_prompt -> Pixel Decoder
  |  -> pixel_embed (B, 256, H, W)
  |
Step 7: Mask Head
  |  einsum(6 queries, pixel_embed) -> (B, 6, H, W) logits
  |
Step 8: Hungarian Matching (no_grad)
  |  4 predictions vs 3 GT textures
  |  Cost matrix: text cosine similarity + mask Dice overlap
  |  Result: 3 matched pairs + 1 hallucination -> DUSTBIN
  |
Step 9: Loss Computation
  |  CE + Dice on masks (PAD channel masked to -inf)
  |  Contrastive on text (matched=attract, hallucinated=repel)
  |  MSE on count (4 vs 3)
  |  LM loss on Qwen text generation
  |  Orthogonal penalty on SAM3 LoRA
```

### Hungarian Matching

For each sample in the batch, we solve a bipartite assignment problem:

1. **Build cost matrix** `C[i,j]` of shape `(K_pred, K_gt)`:
   - Text cost: `(1 - cosine_similarity) / 2` between predicted and GT CLIP embeddings
   - Mask cost: `1 - Dice` between predicted soft masks and GT binary masks
   - `C[i,j] = text_weight * text_cost + mask_weight * mask_cost`

2. **Solve** with `scipy.optimize.linear_sum_assignment(C)` (polynomial-time optimal assignment).

3. **Handle mismatches**:
   - **Matched predictions**: Permuted to align with GT channel indices
   - **Unmatched predictions** (hallucinations): Target becomes DUSTBIN (channel 0)
   - **PAD slots**: Excluded from all loss computations

4. **Remap GT mask**: The ground-truth index mask is permuted so that `GT_pixel == channel_idx` for the Cross-Entropy loss.

### Loss Functions

```
L_total = lambda_mask * (L_CE + L_Dice)
        + lambda_text * L_contrastive
        + lambda_count * L_count
        + lambda_lm * L_lm
        + lambda_orth * L_orthogonal
```

| Loss | Formula | Weight | Notes |
|---|---|---|---|
| **Cross-Entropy** | `CE(raw_logits, GT_index_mask)` | 1.0 | Applied to RAW logits (PyTorch applies LogSoftmax internally). PAD channels = -inf. |
| **Dice** | `1 - (2*inter + s) / (union + s)` per channel | 1.0 | Applied to `softmax(logits, dim=1)` probabilities. PAD excluded. |
| **Text Contrastive** | `CosineEmbeddingLoss` | 0.5 | Matched pairs: target=+1 (attract). Hallucinated: target=-1 (repel). |
| **Count Penalty** | `MSE(K_pred, K_gt)` | 0.1 | Gentle regularization. Too high causes mode collapse (model refuses to predict textures). |
| **LM Loss** | Qwen autoregressive CE | 0.5 | Trains Qwen to generate accurate descriptions. |
| **Orthogonal Reg** | `\|\| U_k^T @ DeltaW \|\|_F^2` | 0.01 | Prevents catastrophic forgetting of SAM3 zero-shot capabilities. |

### Gradient Strategy

**Segmentation gradient warmup**: For the first N epochs (default: 10), the texture embeddings are **detached** before entering the SAM3 path. This allows Qwen's LoRA to stabilize on the LM loss and text alignment before the noisy segmentation gradients flow back.

```
Epoch  1-10:  Qwen learns to describe textures (LM + contrastive only)
Epoch 11+:    Full E2E gradients (LM + contrastive + segmentation -> Qwen LoRA)
```

**Differential learning rates**:
| Component | Learning Rate |
|---|---|
| Qwen LoRA | `base_lr` (1e-4) |
| MLP Projector | `base_lr` (1e-4) |
| Mask Head | `base_lr` (1e-4) |
| DUSTBIN embedding | `base_lr` (1e-4) |
| SAM3 Orthogonal LoRA | `base_lr * 0.1` (1e-5) |

---

## Inference

At inference time, the model runs without teacher forcing:

```python
# 1. Prompt Qwen with "1 to 5" (or specific N if known)
# 2. Qwen generates descriptions freely (no teacher forcing)
# 3. Extract <TEX_i> tokens from generated text
# 4. Build query slots -> MLP -> SAM3 -> mask_logits (B, 6, H, W)
# 5. Set PAD channels to -inf
# 6. argmax(dim=1) -> pixel-wise texture assignment
```

The output is a spatial map where each pixel is assigned to either:
- **0**: Background / non-texture (DUSTBIN)
- **1-K**: One of K detected texture regions

---

## Project Structure

```
Qwen2SAM_DeTexture/
|
+-- configs/
|   +-- detexture.yaml           # All hyperparameters
|
+-- models/
|   +-- qwen2sam_detexture.py    # Main E2E model class
|   +-- projector.py             # 3-layer smooth bottleneck MLP
|   +-- orthogonal_lora.py       # Orthogonal LoRA wrapper
|   +-- hungarian.py             # N-way Hungarian matching
|   +-- losses.py                # All loss functions
|
+-- data/
|   +-- dataset.py               # DeTextureDataset + DeTextureCollator
|
+-- training/
|   +-- train.py                 # Training loop
|   +-- utils.py                 # Config, seed, scheduler, checkpointing
|
+-- scripts/
|   +-- evaluate.py              # Evaluation script
|   +-- prepare_data.py          # Data preparation
```

---

## Configuration

All hyperparameters are in `configs/detexture.yaml`:

```yaml
model:
  qwen_model: "Qwen/Qwen3-VL-8B-Instruct"
  lora_r: 16                    # Qwen LoRA rank
  sam3_lora_r: 8                # SAM3 Orthogonal LoRA rank
  sam3_lora_n_singular: 32      # SVD vectors for orthogonality

training:
  batch_size: 1
  gradient_accumulation_steps: 8  # Effective batch = 8
  num_epochs: 100
  learning_rate: 1.0e-4
  seg_grad_warmup_epochs: 10    # Detach seg grads for first 10 epochs

loss:
  mask_weight: 1.0
  text_contrastive_weight: 0.5
  count_weight: 0.1
  lm_weight: 0.5
  orthogonal_weight: 0.01
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

### Data Format

The dataset expects a JSON metadata file:

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

Each mask is a binary PNG where white pixels belong to that texture. The dustbin (background) is implicitly defined as pixels not covered by any texture mask.

---

## Freezing Strategy Summary

```
+-----------------------------------------------------------+
|                    TRAINABLE COMPONENTS                     |
+-----------------------------------------------------------+
| Qwen3-VL LoRA (q_proj, v_proj)          ~4M params        |
| MLP Projector (2048->1024->512->256)     ~1.6M params      |
| Multi-Texture Mask Head                  ~0.2M params      |
| DUSTBIN embedding                        2048 params       |
| SAM3 Orthogonal LoRA (cross-attn Q,V)   ~0.1M params      |
| Text alignment head                      ~1M params        |
+-----------------------------------------------------------+
| TOTAL TRAINABLE                          ~7M params        |
+-----------------------------------------------------------+

+-----------------------------------------------------------+
|                      FROZEN COMPONENTS                     |
+-----------------------------------------------------------+
| Qwen3-VL base weights                   ~8B params        |
| SAM3 Image Encoder (ViT backbone)       ~300M params      |
| SAM3 Fusion Encoder (base weights)      ~9.5M params      |
| SAM3 Pixel Decoder                      ~2M params        |
| CLIP text encoder (for GT embeddings)   ~63M params       |
+-----------------------------------------------------------+
```

Only **~7M parameters** are trained. The vast majority of the model (~8.4B) remains frozen.
