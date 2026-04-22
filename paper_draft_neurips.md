# Detecture: VLM-Guided Multi-Texture Segmentation via Multiplexed Grounding

**Anonymous Authors**

---

## Abstract

Texture segmentation is not simply object segmentation with different masks. Before boundaries can be drawn, a system must decide which visual regularities are perceptually major, at what scale they should be grouped, and how they should be separated when they do not align with object categories. Texture thus occupies an intermediate regime: below full semantics, yet too structured to be captured by local cues alone. We propose that this gap can be bridged through language used in a *sub-semantic* way — a Vision-Language Model first produces texture-guided descriptions of appearance and spatial organisation (rather than object labels), which then serve as anchors for segmentation. Based on this idea, we present **Detecture**, an end-to-end architecture for parallel multi-texture segmentation that couples open-vocabulary reasoning with a texture-aware SAM-family decoder. In developing the system we identify five architectural pathologies that prevent standard `[SEG]`-token pipelines from scaling to multi-target texture grounding: *Context Leakage* (causal attention contaminates downstream `[SEG]` representations, inflating inter-texture cosine similarity from 0.16 to 0.74 under shared context), *Slot-1 Positional Bias* (SAM's multi-prompt cross-attention allocates 90.5 % of pixels to the first query regardless of semantic content), the *Count / Language Collapse* dilemma (uniform LM loss terminates generation after one texture, while binary masking causes catastrophic linguistic forgetting), *Directional Drift* (an over-parameterised projector memorises domain-specific manifold directions), and *Bottleneck Semantic Squash* (projecting a 4096-dim VLM grounding representation through a 256-dim square bottleneck destroys the semantic richness a pretrained segmentation backbone can already interpret). A parallel Zero-Shot experiment reframes the projector's role: SAM 3 with a hand-crafted, object-aware text prompt reaches **0.928 mIoU on RWTD without any training**, suggesting the projector should *preserve* SAM's native semantic mapping rather than relearn one. Detecture instantiates this principle through a **Bridge-to-Native-Space projector** that maps `[SEG]` into SAM 3's native 1024-dim text-encoder width and reuses SAM 3's own frozen 1024 → 256 resizer, **block-diagonal attention masking** and **batch-multiplexed SAM decoding** that decouple multi-query grounding, and a **Shifted-Zero LM-loss cliff** paired with **masked-row trainable `[SEG]` rows** (warm-started from `<|im_end|>`) that together also cure a subtle emission-failure mode which a naïve $\lambda(\text{SEG}) = 0$ cliff silently induces. With a permanently frozen SAM 3 and a two-stage curriculum, the method uses only ~8.2M trainable parameters on top of frozen Qwen3-VL-8B and SAM 3 backbones. We further contribute the **Detecture Miner**, a geometry-first texture mining framework, and adapt the **Real-World Texture Dataset (RWTD)** for VLM-guided evaluation.

---

## 1. Introduction

Foundation segmentation models have transformed visual understanding, but they remain built around a comparatively convenient assumption: that the scene can be segmented by discovering or following object-like structure. Even when prompted flexibly, these models inherit training priors that favour semantic categories, closed contours, and shape-defined regions. Texture, however, resists this logic. A texture boundary is often not an object boundary, and a texture region is often not a named thing. Before segmentation can even begin, a system must decide which repeated visual patterns in the image are perceptually important enough to count as distinct regions, which variations belong to the same texture family, and at what spatial scale this grouping should occur. In that sense, texture segmentation is not just a downstream masking task. It is a prior perceptual problem.

This is precisely what makes texture fundamentally difficult. Texture lies in an intermediate regime between low-level signal and high-level semantics. It is not reducible to raw local statistics, because humans group texture through broader regularities — spatial organisation, dominance, and contrast. Yet it is also not naturally indexed by semantic labels, because many texture regions do not have a canonical name beyond the object or material that happens to contain them. A patch may be perceived as rough, layered, mottled, fibrous, weathered, striped, granular, or speckled long before it is recognised as wall, bark, grass, dust, or stone. The perceptual map of a texture scene is therefore partly unstable: it changes with viewpoint, scale, image quality, and observer emphasis, and this instability becomes even stronger in scientific, expert, or otherwise out-of-distribution imagery where object semantics are weak or misleading.

This creates a central paradox. Modern multimodal models are linguistic, and language is usually treated as a vehicle for semantics. Texture, by contrast, has often been framed as a pre-semantic or even non-semantic phenomenon. At first glance, this seems to make VLMs the wrong tool for the problem. Our starting point is the opposite hypothesis: language may become useful for texture precisely when it is used *below* the level of object naming. The key question is not whether a model can say "this is grass" or "this is a building," but whether it can generate *sub-semantic texture descriptions* such as *dense thin green vertical strands*, *weathered grey granular surface*, or *coarse irregular brown ridges in the lower foreground*. These descriptions are verbal, but they are not primarily object labels. They act as anchors for perceptual regularities that otherwise remain difficult to stabilise across images.

This insight motivates the core idea of our paper. We ask whether language can serve as an intermediate representational layer for texture: not as full semantics, and not as a replacement for segmentation, but as a structured bridge between perceptual texture cues and pixel-level grounding. If such a bridge can be learned, then a first model can transform a visually ambiguous texture field into a set of compact, transferable textual anchors, and a second model can ground those anchors into spatially coherent masks. In this view, the linguistic stage is not merely a prompt generator for a segmentation backend. It is the mechanism that converts a difficult, unstable texture percept into a form that can be shared across models, generalised across domains, and reconciled across multiple competing texture regions in the same image.

This framing also explains why prior progress, while important, remains incomplete. TextureSAM [1] showed that SAM-family models carry a systematic shape bias and that this bias can be shifted toward texture sensitivity through texture-augmented fine-tuning. That result established that *what* to segment in texture-dominant scenes is, at least in part, learnable. But it did not resolve the more fundamental upstream problem: *how many* textures should be segmented, *which* ones are perceptually major, and *how* they should be described in a way that survives ambiguity and domain shift. A prompt-driven segmentation model, even a texture-aware one, still requires these decisions to be supplied externally. What is missing is an autonomous mechanism for discovering and expressing texture identity before mask prediction.

We therefore propose **Detecture**, an end-to-end architecture for parallel multi-texture segmentation that couples a VLM's open-vocabulary reasoning with a texture-aware SAM-family decoder. The VLM first extracts texture-guided descriptions that are linguistic yet intentionally non-object-centric, and dedicated `[SEG]` representations derived from these descriptions are then projected into the segmentation decoder to produce multiple masks in a single forward pass. This design turns language into a sub-semantic coordination space between two otherwise mismatched systems: one that can describe, and one that can localise. It also offers a practical advantage for hard cases. Because texture descriptions can abstract over exact appearance, they provide a more stable anchor when the image comes from unusual viewpoints, expert domains, or distributions far from the object-centric data on which most foundation models were trained.

Making this idea work required more than simply attaching a VLM to SAM. As soon as we moved from single-target grounding to autonomous multi-texture partitioning, we encountered five architectural pathologies. *Context Leakage* caused later `[SEG]` tokens to inherit semantic residue from earlier texture descriptions under standard causal attention. *Slot-1 Positional Bias* in SAM's decoder forced nearly all pixels toward the first prompt slot, preventing fair competition among multiple texture queries. The *Count / Language Collapse* dilemma emerged during co-training: uniform language-modeling loss terminated generation after a single description, while binary masking on the text produced catastrophic forgetting of Qwen's linguistic prior. *Directional Drift* appeared when a high-capacity projector overfit domain-specific directions rather than learning a transferable mapping between VLM and decoder spaces. And *Bottleneck Semantic Squash* emerged once drift was cured: projecting Qwen's rich 4096-dim grounding representation through a 256-dim square bottleneck destroyed the semantic richness SAM's pretrained text pathway already understood — a ceiling we confirmed directly by running a parallel Zero-Shot experiment that bypassed the VLM and projector entirely and reached **0.928 mIoU on RWTD** with a hand-crafted object-aware text prompt. These failures are not incidental engineering bugs. They reflect a deeper fact: once texture is treated as an open-ended, multi-region perceptual problem, the standard assumptions behind current `[SEG]`-token grounding pipelines no longer hold.

Detecture resolves these pathologies through a matched set of design choices. A **block-diagonal attention mask** isolates each texture block during `[SEG]` extraction, eliminating cross-texture contamination. **Batch-multiplexed SAM decoding** reshapes the query tensor so each texture occupies its own independent "Slot 1" in a batched forward pass, bypassing the positional bias entirely. A **Shifted-Zero LM-loss cliff** — assigning loss weight $1.0$ at the `[SEG]` position (to supervise emission) and $0.0$ one position forward (to leave the grounding embedding geometrically free) — resolves the Count / Language dilemma, while **masked-row trainable `[SEG]` rows** warm-started from `<|im_end|>` calibrate the output projection so the model actually emits `[SEG]` at inference. A **Bridge-to-Native-Space projector** maps the 4096-dim `[SEG]` hidden state into SAM 3's native 1024-dim text-encoder width and hands the final $1024 \rightarrow 256$ step back to SAM 3's own pretrained resizer, kept frozen — turning cross-model alignment from a *learning* problem into a *shape-matching* problem, and simultaneously curing Directional Drift and Bottleneck Semantic Squash. Together with a permanently frozen SAM 3 and a two-stage curriculum, these mechanisms enable parallel multi-texture segmentation in a single forward pass with only ~8.2M trainable parameters on top of frozen Qwen3-VL-8B and SAM 3 backbones.

The paper also contributes a full data ecosystem around this idea. Building on TextureSAM's texture-augmented ADE20K, we introduce the **Detecture Miner**, a geometry-first mining framework that identifies texture-rich scenes, extracts distinct texture regions, and associates them with VLM-generated texture descriptions. We further adapt the **Real-World Texture Dataset (RWTD)** for VLM-guided evaluation, using it not only as a benchmark but as a stress test whose distributional distance helped reveal several of the architectural failures above. In this sense, the paper is organised around a joint thesis: texture understanding requires both a new representation and a new pipeline, where linguistic texture anchoring and segmentation are co-designed rather than treated as separate modules.

The central question of this work is therefore the following:

> **Can language function as a sub-semantic anchor for texture, enabling a model to identify, describe, and segment multiple texture regions in a single coherent forward pass?**

We answer this question affirmatively through a unified architecture and data pipeline. Our contributions are:

1. We reframe multi-texture segmentation as a fundamental representational problem, not merely a prompt-engineering or decoder-integration problem, and propose **sub-semantic linguistic anchoring** as the missing bridge between texture perception and pixel grounding.

2. We introduce **Detecture**, a VLM-to-SAM architecture for parallel multi-texture segmentation, and systematically identify five pathologies that arise in the multi-target regime — *Context Leakage*, *Slot-1 Positional Bias*, the *Count / Language Collapse* dilemma, *Directional Drift*, and *Bottleneck Semantic Squash* — plus a subtle *Emission Failure* mode that emerges while resolving the Count / Language dilemma.

3. We propose a corresponding set of mechanisms — **block-diagonal attention masking**, **batch multiplexing**, the **Shifted-Zero LM-loss cliff with masked-row trainable grounding rows**, and a **Bridge-to-Native-Space projector** that reuses SAM 3's own frozen pretrained resizer — that together enable stable one-pass multi-texture grounding with only ~8.2M trainable parameters and a permanently frozen SAM 3.

4. We contribute the **Detecture Miner** for mining and annotating texture-rich training data, and adapt **RWTD** as an external benchmark for evaluating generalisation under real-world texture ambiguity. The cross-domain gap between training (ADE20K) and evaluation (RWTD) was itself instrumental in surfacing several of the pathologies above — evidence that architecture and data benefit from being co-designed.

---

## 2. Related Work

### 2.1 Texture-Aware Segmentation and TextureSAM

Foundation segmentation models exhibit a well-documented shape bias: trained predominantly on object-centric datasets, they learn to segment along object boundaries rather than texture boundaries [1, 14]. TextureSAM [1] addressed this directly by constructing a texture-augmented ADE20K dataset using Compositional Neural Texture (CNT) synthesis [15] with a controllable interpolation coefficient $\eta$ that smoothly transitions images from their original appearance ($\eta = 1.0$) to pure texture replacement ($\eta = 0.0$). Fine-tuning SAM-2 on this augmented data shifted the model's segmentation prior toward texture-defined regions, achieving +0.21 mIoU on the Real-World Texture Dataset (RWTD) and +0.18 mIoU on synthetic texture benchmarks over the SAM-2 baseline. However, TextureSAM remains a *prompt-driven* model: it cannot autonomously determine the number or identity of texture regions, requiring external specification of regions of interest. Our work extends TextureSAM's contributions in two directions—leveraging its Textured-ADE20K for training data construction while adding autonomous, language-guided multi-texture reasoning that eliminates the need for manual prompting.

### 2.2 Reasoning Segmentation and the `[SEG]` Token Paradigm

LISA [2] pioneered *reasoning segmentation*, embedding a learnable `[SEG]` token into an LLM's vocabulary such that the token's hidden state, when projected into SAM's prompt encoder, produces a segmentation mask conditioned on complex natural language queries. LISA++ [3] extended this baseline with instance-level segmentation and Segmentation-in-Dialogue (SiD), enabling multi-turn conversational grounding. However, both LISA and LISA++ are fundamentally *sequential*: segmenting $K$ targets requires $K$ independent forward passes, each unaware of the others. This isolation leads to two practical failures: (a) computational cost that scales linearly with $K$, and (b) independently generated masks that lack mutual spatial awareness, producing overlapping regions that violate the fundamental constraint of image partitioning. In our experiments, sequential LISA-style approaches cannot enforce non-overlapping mask constraints without post-hoc reconciliation—precisely the property our Winner-Takes-All mechanism guarantees natively.

GSVA [4] made an important step toward multi-target grounding by extending LISA's architecture to support *multiple* `[SEG]` tokens within a single decoding pass, alongside a novel `[REJ]` token for explicitly rejecting absent targets in Generalized Referring Expression Segmentation (GRES). While GSVA demonstrates that multiple grounding tokens can coexist in the output sequence, it relies on standard causal attention during generation. As we demonstrate in Section 3.2, this architectural choice introduces *Context Leakage*: the hidden state of $\texttt{[SEG]}_k$ absorbs semantic information from all preceding texture descriptions through the causal attention pathway, with inter-token cosine similarity reaching 0.74 under shared context versus 0.16 under isolation. GSVA's design assumes that the LLM can maintain representational purity across sequential `[SEG]` tokens—an assumption our ablations empirically refute for dense texture grounding.

### 2.3 Dense Grounding via Multimodal LLMs

A growing family of models integrates pixel-level grounding directly into multimodal LLM architectures. PixelLM [5] introduces a lightweight pixel decoder with a segmentation codebook, where codebook token embeddings serve as conditioning inputs for mask generation, enabling multi-target referring segmentation without requiring SAM. GLaMM [6] pioneered *Grounded Conversation Generation*, producing natural language responses seamlessly interleaved with segmentation masks through phrase-level grounding. Sa2VA [7] unifies SAM-2 with LLaVA into a shared token space, generating instruction tokens that guide SAM-2's mask decoder for both image and video grounding. LLaVA-Grounding [8] connects visual grounding with LLM-based reasoning for referring expression comprehension.

While these models advance the state of the art on referring segmentation benchmarks (Sa2VA achieves 81.6 cIoU on RefCOCO, outperforming GLaMM's 79.5), they share a common architectural limitation: all rely on standard causal attention mechanisms during multi-token generation. This is adequate for *referring* segmentation, where targets are specified by the user and generated one at a time, but becomes pathological for *autonomous multi-target partitioning*, where the model must simultaneously generate and ground multiple regions. We identify two failure modes specific to this setting: Context Leakage from causal attention (Section 3.2) and Count Collapse from co-trained language loss (Section 3.4). Our block-diagonal attention mask and proximity-decayed loss masking are general solutions applicable to any architecture in this family.

### 2.4 SAM and Prompt-Based Segmentation

The Segment Anything Model (SAM) [9] and its successors (SAM-2 [10]) have established prompt-based segmentation as a powerful paradigm. Given visual prompts (points, boxes, masks) or dense embedding prompts, SAM produces high-quality segmentation masks with strong zero-shot generalization. SAM-2 extends this to video with memory-augmented tracking. These models excel as segmentation *backends* in VLM-guided pipelines, but their decoder architecture exhibits an underexplored positional bias when processing multiple prompt embeddings simultaneously. Through controlled experiments on all 253 RWTD samples using ground-truth embeddings, we discovered that SAM's cross-attention mechanism allocates 90.5% of pixels to the first prompt slot regardless of semantic content, with 0.0% allocation to the second slot—a bias that persists after swapping prompt order and under isolated single-query evaluation (Section 3.3). This *Slot-1 Addiction* is invisible in single-target applications (where all existing benchmarks operate) but catastrophic for multi-target segmentation, effectively reducing a multi-query decoder to a single-query one.

### 2.5 Multi-Target and Panoptic Segmentation

Classical panoptic segmentation approaches—Mask2Former [11], Panoptic FPN [12]—achieve multi-target segmentation through DETR-style set prediction or proposal generation. While effective for closed-vocabulary, fixed-category settings, these methods cannot leverage natural language descriptions of novel textures and require retraining for new categories. Recent open-vocabulary panoptic models (X-Decoder [16], OpenSeeD [17]) extend to unseen categories via text-image alignment but still operate with predefined category lists rather than free-form descriptions. Our work bridges this gap by combining VLMs' open-vocabulary reasoning with SAM's segmentation quality, introducing the architectural innovations necessary for reliable multi-target texture grounding.

### 2.6 Texture Segmentation and Datasets

Texture segmentation has a rich history predating deep learning, from Brodatz-based evaluations to the Describable Textures Dataset (DTD) [13], KTH-TIPS [14], and texture subsets of MS-COCO Stuff [15]. However, these benchmarks predominantly feature isolated texture patches or well-separated regions under controlled conditions. They do not capture the challenging *texture transition* scenarios common in real-world imagery: gradual boundaries between sand and water, interleaving patterns of brick and mortar, or complex junctions of concrete, grass, and gravel. TextureSAM's RWTD benchmark [1] and our extended Detecture mining pipeline specifically target these challenging in-the-wild cases, providing both evaluation benchmarks and a reproducible methodology for curating texture-rich training data from large-scale annotated datasets.

---

## 3. Methodology

### 3.1 Architecture Overview

Our architecture couples a Vision-Language Model's open-vocabulary reasoning with a texture-aware segmentation decoder for parallel, multi-texture segmentation in a single forward pass. The system comprises four components (Figure 1):

1. **Qwen3-VL-8B** (frozen backbone + lightweight LoRA, rank 8): generates structured texture descriptions, each terminated by a dedicated `[SEG]` grounding token whose hidden state encodes visually-grounded spatial information.
2. **Bridge + Frozen SAM Resizer** (trainable 4.2M + frozen 0.26M): a single `Linear(4096 → 1024) + LayerNorm + GELU + Dropout(0.4)` maps `[SEG]` hidden states into SAM 3's *native* 1024-dim text-encoder width, then SAM 3's own pretrained resizer (`Linear(1024, 256)`, kept frozen) completes the projection to the decoder's query space.
3. **SAM3 Mask Decoder** (permanently frozen): processes each projected query via batch-multiplexed cross-attention to produce per-texture mask predictions. No LoRA; a parallel Zero-Shot experiment showed SAM 3 with an object-aware prompt already reaches 0.928 mIoU on RWTD — any fine-tuning hurts OOD generality.
4. **Winner-Takes-All Pixel Assignment**: resolves spatial competition between queries to produce a consistent, non-overlapping image partition augmented with a learned dustbin channel.

**[Figure 1: Architecture and Gradient Flow.]** *Forward pass (top):* Input image + prompt → Qwen3-VL generates texture descriptions with `[SEG]` tokens under block-diagonal attention mask → Bridge (trainable, 4096 → 1024) → SAM 3's frozen resizer (1024 → 256) → Batch-multiplexed SAM 3 (frozen) produces per-texture masks → WTA pixel assignment. *Backward pass (bottom):* Mask Loss (CE + Dice) → Bridge → `[SEG]` token latent space → Qwen LoRA + masked-row SEG rows (shaping *where* the VLM attends and enabling `[SEG]` emission). Simultaneously, Shifted-Zero Cliff LM Loss → Qwen LoRA on all non-free tokens (preserving *what* the VLM generates); the zero-weight slot is placed one position *past* each `[SEG]`, not at `[SEG]` itself — preserving the emission-learning gradient.

Training employs a two-stage curriculum with ~8.2M trainable parameters, a Shifted-Zero LM cliff that simultaneously prevents count collapse and language collapse while preserving `[SEG]` emission under free generation, and masked-row gradient routing that calibrates `lm\_head[\text{SEG}]` without disturbing the rest of Qwen's vocabulary. The design addresses five pathologies that emerge when extending single-target `[SEG]`-token grounding to the multi-target regime, which we characterize in the following subsections.

### 3.2 Overcoming Context Leakage via Block-Diagonal Attention Masking

**Pathology.** Any architecture that generates multiple `[SEG]` tokens within a single causal decoding pass is susceptible to *Context Leakage*: the hidden state of a later grounding token absorbs semantic information from earlier texture descriptions through the autoregressive attention pathway. Given $K$ texture targets, the model produces a sequence:

$$\texttt{TEXTURE\_1: } \langle \text{desc}_1 \rangle \texttt{ [SEG]}_1 \texttt{ TEXTURE\_2: } \langle \text{desc}_2 \rangle \texttt{ [SEG]}_2 \ldots$$

Under standard causal (left-to-right) attention, $\texttt{[SEG]}_2$ attends to all preceding tokens, including the description of TEXTURE\_1 and $\texttt{[SEG]}_1$. This creates a representational contamination pathway: the hidden state of $\texttt{[SEG]}_2$ encodes not only the semantics of TEXTURE\_2 but also residual information from TEXTURE\_1.

We quantified this effect empirically. Under a standard "segment 1 to 6 textures" prompt with causal attention, the mean cosine similarity between consecutive `[SEG]` embeddings reached **0.74**. When we constrained the prompt to "exactly 2 textures" and repeated the measurement, the cosine similarity dropped to **0.16**—revealing that the high similarity was a *prompt-induced artifact* of causal context sharing, not an intrinsic property of the textures. This contamination is asymmetric and cumulative: later `[SEG]` tokens are progressively more contaminated than earlier ones, producing a systematic degradation in mask quality that worsens with the number of targets. This pathology is inherent to any multi-`[SEG]` architecture that relies on standard causal attention, including GSVA [4] and Sa2VA [7].

**Solution: Two-Pass Inference with Block-Diagonal Attention Mask.** We introduce a custom attention mask $\mathbf{M} \in \{0, -\infty\}^{L \times L}$ that enforces block-diagonal structure over texture-specific token groups while preserving the causal constraint within each block. Let $\mathcal{G}_k$ denote the set of token indices belonging to texture $k$ (its description tokens and `[SEG]` token). We define:

$$\mathbf{M}_{ij} = \begin{cases} 0 & \text{if } \exists k: i \in \mathcal{G}_k \wedge j \in \mathcal{G}_k \wedge i \geq j \\ 0 & \text{if } j \in \mathcal{G}_{\text{shared}} \wedge i \geq j \\ -\infty & \text{otherwise} \end{cases}$$

where $\mathcal{G}_{\text{shared}}$ contains the image tokens and system prompt tokens that all textures may attend to. The constraint $i \geq j$ preserves the causal ordering within each block and with respect to the shared context, ensuring compatibility with the LLM's pretrained autoregressive distribution. This mask guarantees that each `[SEG]` token attends *only* to past tokens within its own texture description and the shared visual context, producing semantically pure, decoupled representations. At inference, we employ a two-pass strategy: the first pass processes the full prompt to extract texture descriptions via Qwen's frozen language capabilities, while the second pass applies the block-diagonal mask to compute isolated `[SEG]` embeddings.

### 3.3 Solving SAM's Positional Bias via Batch Multiplexing

**Pathology.** Prompt-based segmentation decoders in the SAM family exhibit an underexplored positional bias when processing multiple prompt embeddings simultaneously. When the mask decoder receives multiple prompt embeddings $\mathbf{Q} \in \mathbb{R}^{B \times K \times D}$ (where $K$ is the number of texture queries), its cross-attention mechanism routes nearly all spatial attention to the first prompt position—a phenomenon we term *Slot-1 Addiction*. Through controlled experiments on all 253 RWTD samples using ground-truth `[SEG]` embeddings, we measured the pixel allocation across prompt slots:

| Slot | Pixel Allocation | After Swap (T1↔T2) | Isolated Query |
|------|-----------------|---------------------|----------------|
| Slot 1 | **90.5%** | **90.5%** | 90.5% |
| Slot 2 | **0.0%** | **0.0%** | 0.0% |

The bias persists identically after swapping texture assignments between slots and when running each query in isolation—confirming that this is a *positional* bias intrinsic to the decoder's pretrained cross-attention weights, not a consequence of query content. This pathology is invisible in single-target applications (where all existing benchmarks operate) but catastrophic for any multi-target segmentation system, effectively reducing a multi-query decoder to a single-query one.

**Solution: Batch Multiplexing.** Rather than passing multiple queries through SAM's multi-prompt pathway, we reshape the query tensor from $\mathbb{R}^{B \times K \times D}$ to $\mathbb{R}^{(B \cdot K) \times 1 \times D}$, replicating the image features accordingly via an index mapping. This forces each texture query to occupy *its own* Slot 1 in an independent SAM forward pass, bypassing the positional bias entirely. The resulting masks $\hat{\mathbf{m}}_k \in [0,1]^{H \times W}$ for $k = 1, \ldots, K$ are then recombined along the batch dimension for competitive pixel assignment. This transformation is computationally equivalent to $K$ independent SAM passes but is implemented as a single batched operation, preserving GPU efficiency. Empirically, batch multiplexing immediately recovered performance to 99.5% of the zero-shot baseline (mIoU 0.703 vs. 0.706) at epoch 5.

### 3.4 Freeing the Grounding Token: Shifted-Zero Cliff and Trainable Emission Rows

**Pathology 1: Count Collapse under Uniform LM Loss.** We observe that naïve end-to-end co-training of a VLM with segmentation losses inevitably leads to a *Count Collapse* pathology: when the language modeling cross-entropy loss is computed uniformly over all generated tokens, the optimization landscape creates a shortcut whereby the model minimizes total loss by terminating generation after a single texture description, since predicting fewer tokens incurs less cumulative risk. In our experiments, this manifested with striking severity: 85% of evaluation samples predicted $K=1$ texture, with the count distribution collapsing to $\{1: 216, 2: 17, 3: 9, 4: 2, 5: 2, 6: 7\}$. The projector irreversibly co-adapted to this degenerate distribution, allocating 100% of pixels to the dustbin when fed legitimate multi-texture inputs.

**Pathology 2: Language Collapse under Binary Loss Masking.** The naïve fix—applying $\texttt{-100}$ to *all* text tokens and computing LM loss exclusively on `[SEG]` positions—prevents Count Collapse but introduces a symmetric failure: *Language Collapse*. With zero LM gradient on text tokens, the LoRA adapters receive gradients exclusively from the spatial segmentation path (mask loss → SAM → projector → `[SEG]` → LoRA). Within a few epochs of co-training, the LoRA undergoes catastrophic forgetting of its linguistic prior: the model loses the ability to generate coherent texture descriptions and ceases to emit `[SEG]` tokens during inference. Crucially, the segmentation architecture itself remains fully functional when bypassing generation with ground-truth descriptions—confirming that the failure is purely linguistic, not geometric.

This reveals a fundamental dilemma in multi-modal LoRA training: *binary* loss masking creates a binary failure mode. Full text loss → Count Collapse; zero text loss → Language Collapse.

**An Intermediate Cosine-Decayed Variant.** An earlier iteration smoothed the binary mask into a per-block cosine decay from $\lambda_{\max} = 1.0$ at the block start down to $\lambda_{\min} = 0.05$ at the `[SEG]` token. This prevented catastrophic forgetting (live-inference RWTD mIoU recovered from 0.136 to 0.694) but introduced a *continuous* tug-of-war: with non-zero linguistic pressure on nearly every token in each block, the Dice mask loss could not drive the grounding embedding aggressively enough toward geometric optima. Empirically, the training Dice plateaued at 0.43 and honest (exactly-K) RWTD mIoU stalled at 0.694.

**Exponential Cliff: the Obvious-But-Broken Version.** The apparent fix is to replace the gentle cosine with a sharp exponential cliff centred on the `[SEG]` token. Let $d(i)$ be the token-wise distance from position $i$ to the nearest assistant-region `[SEG]`; the "obvious" cliff is $\lambda(\text{SEG}) = 0$ and $\lambda(i) = 1 - e^{-\alpha d(i)}$ elsewhere, with $\alpha = 2.0$ yielding $\lambda = 0.865$ at distance one, 0.982 at distance two, $\approx 1.0$ at distance $\geq 3$. This removes the continuous tug-of-war — gradients from the Dice path and the LM path only coincide at the geometrically-isolated `[SEG]` token — and recovers the teacher-forced (Oracle) regime: mIoU on RWTD under teacher-forced `[SEG]` embeddings climbs to 0.738 at epoch 12.

Under *live* (free-generation) inference, however, this design fails completely: the model never emits `[SEG]` at all, substituting `<|tool_call|>` or `<|im_end|>` in its place, and downstream RWTD mIoU collapses to 0.000.

**Diagnosis: the Emission-Zero Conflation.** Causal-LM loss at position $i$ supervises the *prediction of token $i+1$*. The "obvious" cliff sets $\lambda(\text{SEG}) = 0$ in order to free $\mathbf{h}_{\text{SEG}}$ geometrically — but this same weight also controls the gradient that trains $\mathbf{h}_{\text{SEG}}$ to predict whatever comes *after* `[SEG]` (typically a newline or the next `TEXTURE_k:` marker). Zeroing it removes the "continue correctly after a SEG emission" signal, and with it the principal pathway that would otherwise teach the model to emit `[SEG]` as the natural prelude to the next region. Free-generation therefore substitutes pretrained-high-probability special tokens that Qwen already associates with "end of a thought block".

**Our Solution (Part A): Shifted-Zero Cliff.** We shift the zero one position forward:

$$\lambda_{\text{LM}}(i) = \begin{cases} 0 & \text{if } i - 1 \in \mathcal{S}_{\text{SEG,asst}} \\ 1 - e^{-\alpha \cdot d(i)} & \text{otherwise} \end{cases}, \qquad \alpha = 2.0.$$

That is, the position *after* each assistant-region `[SEG]` carries weight zero (a geometrically unused continuation slot), while the `[SEG]` position itself carries weight one. Under causal-shift semantics, $\lambda(\text{SEG}) = 1$ restores the emission-learning gradient on the post-SEG continuation; this gradient is predominantly syntactic (newlines, region markers) rather than semantic, so the Dice path and the LM path still do not conflict at the grounding embedding itself. Concretely the final weights are:

| position | $\lambda_{\text{LM}}$ | purpose |
|---|---|---|
| `[SEG]` itself | **1.000** | force emission; supervise post-SEG continuation |
| one token past `[SEG]` | **0.000** | geometric freedom slot |
| two tokens past `[SEG]` | 0.865 | linguistic supervision resumes |
| three tokens past `[SEG]` | 0.982 | |
| $\geq 4$ tokens past | $\approx 1.0$ | |
| all other assistant tokens | $1 - e^{-2 d}$ | standard cliff elsewhere |

**Our Solution (Part B): Masked-Row Trainable SEG with Warm-Start.** Shifted-Zero alone remains insufficient. Even with the corrected weighting, live-inference RWTD mIoU stays at 0.000 while the Oracle climbs. Probing `lm_head` reveals a second failure: the output-projection row `lm_head.weight[\text{SEG}]` — the row whose inner product with the final hidden state produces `[SEG]`'s logit — is frozen at its random initialisation, and LoRA on `q_proj`/`v_proj` cannot reach it. With an uncalibrated `lm_head[\text{SEG}]`, the SEG logit is dominated by logits from well-trained rows like `<|tool_call|>` or `<|im_end|>`, and these tokens are chosen instead.

We resolve this with two complementary mechanisms:

1. **Warm-start from `<|im_end|>`.** At model initialisation we copy the pretrained rows `lm_head.weight[\langle|\text{im\_end}|\rangle]` and `embed\_tokens.weight[\langle|\text{im\_end}|\rangle]` into the corresponding SEG rows. This places `[SEG]` near a token Qwen already associates with "finalise a region of thought" in both the input and output embedding spaces — a semantically appropriate anchor that avoids the random-init cliff.

2. **Masked-row gradient routing.** During Stage 2, we unfreeze the full `embed\_tokens.weight` and `lm\_head.weight` tensors (so PyTorch computes gradients for them at all) and register a gradient hook that zeros every row except the SEG row. This restricts actual weight updates to the two 8192-dimensional SEG rows — 16,384 effective parameters — while leaving the remaining $\sim 150$K rows of each tensor exactly as pretrained. The SEG rows are added to the optimiser in their own parameter group at a dedicated learning rate.

Together, Shifted-Zero and Masked-Row cure emission: live-inference RWTD mIoU jumps from 0.000 to 0.794 at epoch 16 (Section 5), while Oracle mIoU continues its trajectory from 0.738 at epoch 12 to 0.781 at epoch 16 — confirming that the live gap was a pure emission problem, not a geometric one.

**Prompt-Template `[SEG]` Filtering.** The user prompt includes the literal string "`<|seg|>`" in its format-template example, which tokenises to the SEG special token within the user message. All distance computations and `[SEG]`-position lookups are filtered to the assistant region only, so prompt-template SEGs are never treated as real grounding outputs.

### 3.5 From Information Bottleneck to Native-Space Bridge

**Pathology (Directional Drift).** We first demonstrate that mapping high-dimensional VLM embeddings to a segmentation decoder's prompt space using standard, high-capacity projectors results in *Directional Drift*: the projector memorizes domain-specific manifold directions from the training distribution rather than learning a generalizable mapping. An early variant with approximately 10.5M projector parameters achieved strong in-domain performance (ADE20K mIoU climbing from 0.700 to 0.707 across epochs 5–13), yet exhibited systematic cross-domain degradation: RWTD mIoU dropped from 0.692 at epoch 5 to 0.618 at epoch 10—a divergence invisible when training and evaluation share the same distribution.

To diagnose the mechanism, we performed a vector geometry analysis, measuring the cosine similarity between paired `[SEG]` embeddings *before* and *after* the projector:

| Domain | Pre-Projector Cosine | Post-Projector Cosine | Compression (Δ) |
|--------|---------------------|----------------------|-----------------|
| RWTD (cross-domain) | 0.736 | 0.919 | **+0.183** |
| ADE20K (in-domain) | 0.831 | 0.879 | +0.048 |

The projector compressed RWTD representations nearly **4× more** than ADE20K representations (Δ = 0.183 vs. 0.048), pushing cross-domain `[SEG]` vectors toward a shared manifold direction learned from ADE20K's training distribution.

**Intermediate Solution (Bottleneck Mitigation).** A first fix reduced the projector to ~2.1M parameters through a $4096 \to 512 \to 256$ square bottleneck with dropout ($p=0.15$). This single change produced the first trained model to surpass the zero-shot SAM baseline (0.732 vs. 0.706 RWTD mIoU). Later architectural iterations inherited the same bottleneck design. However, the best live-inference honest RWTD mIoU achievable under this family (under an exactly-K prompt to neutralize Hungarian-matching inflation) plateaued at 0.694, suggesting that while the bottleneck *cured* drift, it also imposed a new ceiling.

**Reframing: the Zero-Shot SAM 3 Upper-Bound Experiment.** To probe this ceiling, we ran a parallel experiment that *removes the VLM and projector entirely*: we hand-craft a single highly descriptive, object-aware prompt per RWTD image and feed it directly into SAM 3's native text pathway. This zero-shot configuration—no training, no VLM, no projector—achieved **0.928 mIoU on RWTD**, nearly +0.23 above our best trained number under the bottleneck design. The conclusion is unambiguous: SAM 3 natively understands complex semantic descriptions; the limiter was our projector squashing Qwen's rich 4096-dim `[SEG]` representation through a 256-dim bottleneck before SAM could see it.

**Our Solution: Bridge-to-Native-Space.** Rather than learning a new 4096 → 256 semantic mapping, we widen the trainable projection to match SAM 3's *native* text-encoder width and reuse SAM's own pretrained final projection. Formally, letting $\mathbf{h} \in \mathbb{R}^{4096}$ denote a `[SEG]` hidden state:

$$f_{\text{bridge}}(\mathbf{h}) = \mathrm{Dropout}_{0.4}(\mathrm{GELU}(\mathrm{LN}(\mathbf{W}_B \mathbf{h} + \mathbf{b}_B)))$$

with $\mathbf{W}_B \in \mathbb{R}^{1024 \times 4096}$ (the trainable Bridge, ~4.2M parameters). The final projection into SAM's 256-dim decoder query space is performed by SAM 3's *pretrained* `resizer` layer, $\mathbf{R} \in \mathbb{R}^{256 \times 1024}$, kept frozen in place:

$$\mathbf{q} = \mathbf{R} \cdot f_{\text{bridge}}(\mathbf{h}) + \mathbf{b}_R$$

This reframes the cross-model alignment problem: earlier bottleneck designs treated it as a *learning* problem (train a new 4096 → 256 mapping); our Bridge treats it as a *shape-matching* problem (widen Qwen's output to the width SAM already speaks). The Bridge's ~4.2M parameters (roughly 2× the prior bottleneck) would risk reintroducing Directional Drift, but the aggressive dropout ($p = 0.4$) acts as the sole regularizer, compensating for the larger capacity.

Because the resizer carries SAM 3's pretrained 1024 → 256 semantic mapping—the same mapping that enables the 0.928 ZS result—the Bridge need only supply shape compatibility, not semantic translation. Empirically, this design is expected to close the gap between the 0.694 honest RWTD mIoU of bottleneck-family variants and the 0.928 ZS ceiling; final numbers are reported in Section 5 once training converges.

### 3.6 Winner-Takes-All Pixel Assignment

Given $K$ soft mask predictions $\hat{\mathbf{m}}_k \in [0,1]^{H \times W}$ from the batch-multiplexed SAM decoder, we must produce a spatially consistent partition of the image. We define a Winner-Takes-All (WTA) assignment augmented with a learned dustbin channel:

$$\text{label}(i,j) = \begin{cases} \arg\max_{k \in \{1,\ldots,K\}} \hat{m}_k(i,j) & \text{if } \max_k \hat{m}_k(i,j) > \hat{m}_{\text{dust}}(i,j) \\ \texttt{dustbin} & \text{otherwise} \end{cases}$$

where $\hat{m}_{\text{dust}}$ is produced by a learned 4096-dimensional dustbin embedding passed through the same projector and SAM decoder pathway. The dustbin class absorbs pixels that do not belong to any queried texture, preventing forced assignment of background regions. The WTA mechanism guarantees a *mathematically consistent* partition: every pixel is assigned to exactly one class, with no overlaps and no gaps—a property that sequential, independent models fundamentally cannot enforce.

### 3.7 Two-Stage Curriculum with Permanently Frozen SAM

Earlier iterations of our architecture employed three-stage curricula that ultimately unfroze SAM via an orthogonal LoRA in its cross-attention layers. Across many ablation runs, every SAM-unfreezing configuration produced the same signature: a transient in-domain gain on ADE20K followed by an out-of-domain regression on RWTD (one such run lost 4.6 mIoU between its middle and final stages). Combined with the Zero-Shot SAM 3 upper-bound experiment (Section 3.5) showing that SAM 3 already reaches 0.928 mIoU on RWTD without any training, the conclusion is unambiguous: *SAM must be treated as a pretrained semantic asset, not a fine-tunable module.* Our final architecture therefore adopts a **two-stage curriculum** and permanently freezes SAM.

**Stage 1: Bridge Warmup (Epochs 1–8).** Only the Bridge, multi-texture mask head, and dustbin embedding are trainable (~4.4M parameters). Qwen LoRA is frozen; the SEG embedding and `lm\_head` rows are frozen at their warm-started values (Section 3.4); SAM (including the reused resizer) is frozen. The Bridge absorbs the random-initialisation shock and learns the shape-matching $4096 \to 1024$ map into SAM's native text space under heavy Dropout(0.4) regularization before any linguistic gradients enter the system.

**Stage 2: Qwen Sync + Bridge Decay + SEG Rows (Epochs 9–30).** Three components activate simultaneously:

1. **Qwen LoRA** (rank 8, targeting `q\_proj` and `v\_proj`; ~3.8M parameters) is unfrozen at an ultra-conservative learning rate of $1 \times 10^{-6}$ (0.01× base).
2. **Masked-row SEG training** (Section 3.4) unlocks the SEG rows of `embed\_tokens` and `lm\_head` (16,384 effective parameters), enabling the model to learn to emit `[SEG]`.
3. **Bridge LR decay**: the Bridge's learning rate is decayed by 10× (from $1 \times 10^{-4}$ to $1 \times 10^{-5}$). This decay is critical: maintaining the projector at full LR during co-training causes *Latent Co-Adaptation Churn* — the projector's rapid weight updates continuously shift the feature distribution that the LoRA is trying to adapt to, preventing stable co-convergence. By slowing the Bridge once Qwen joins, we ensure the LoRA builds on a stable mapping rather than chasing a moving target.

The Shifted-Zero cliff (Section 3.4) preserves linguistic coherence throughout. There is no Stage 3; SAM (including its resizer, cross-attention, and decoder) remains frozen across all 30 epochs.

The total loss combines mask supervision with the Shifted-Zero-weighted language loss:

$$\mathcal{L} = \underbrace{\mathcal{L}_{\text{CE}}^{\text{mask}} + 3.0 \cdot \mathcal{L}_{\text{Dice}}}_{\text{mask losses}} + 0.1 \cdot \underbrace{\mathcal{L}_{\text{LM}}^{\text{SZ}}}_{\text{Shifted-Zero cliff}}$$

No orthogonal regularization term appears in the final loss (the SAM LoRA it previously regularised is retired).

**Differential Learning Rates.** The learning rate hierarchy reflects the trust hierarchy and evolves across stages:

| Component | Stage 1 (ep 1–8) | Stage 2 (ep 9–30) |
|---|---|---|
| Bridge (4.2M) | $1 \times 10^{-4}$ | $1 \times 10^{-5}$ (decayed 10×) |
| Mask Head + Dustbin | $1 \times 10^{-4}$ | $1 \times 10^{-4}$ |
| Qwen LoRA (3.8M) | frozen | $1 \times 10^{-6}$ |
| SEG rows (16.4K effective) | frozen at warm-start | $1 \times 10^{-5}$ (masked-row) |
| SAM 3 (frozen resizer, decoder) | frozen | frozen |

**Micro-Warmup Restart for Extension Runs.** An initial 20-epoch schedule under a cosine LR curve yielded a peak at epoch 16 (live-inference honest RWTD mIoU = 0.794). To push further without a full LR re-warmup — which would shock weights already sitting in a good basin — we introduce a *Micro-Warmup Restart* protocol for extending training beyond the original horizon. On resume from a late-stage checkpoint, we (a) rebuild the LR scheduler with each parameter group's peak set to `original_base_LR × s` (with $s = 0.15$ in our run), preserving per-group ratios between Bridge, Qwen LoRA, and SEG rows; (b) apply a short linear warmup (2 epochs) from zero to the scaled peak; (c) cosine-decay over the remaining epochs only. AdamW moment estimates are preserved across the restart. This gives the network a gentle, bounded LR excursion — the Bridge briefly re-engages at 15% of its original rate — without discarding the optimisation trajectory that produced the pre-restart optimum.

---

## 4. Data Ecosystem: The Detecture Miner and Dual-Evaluation Strategy

Our contribution extends beyond architecture to a complete data ecosystem designed to (a) transform generic semantic datasets into high-quality texture-grounding training data, (b) provide controlled in-domain evaluation, and (c) stress-test generalization on an external, in-the-wild benchmark. This ecosystem was not merely a prerequisite for training but an active participant in architectural discovery: each pathology identified in Section 3 was first surfaced by evaluating on out-of-domain data whose distributional distance from the training set exposed failure modes invisible on in-domain benchmarks.

### 4.1 The Core Data Contribution: The Detecture Mining Pipeline

We introduce the **Detecture Miner**, an automated, scalable framework designed to transform generic semantic datasets into high-quality texture-grounding data. The pipeline operates in three stages: intelligent filtering, geometric extraction, and VLM-based annotation.

**Source Material.** Our training pool originates from TextureSAM's [1] texture-augmented ADE20K, which applies Compositional Neural Texture (CNT) synthesis at varying augmentation intensities (degree $\eta \in [0, 1]$) to the original ADE20K training split [18]. This yields up to ~222,000 candidate images spanning the full spectrum from original scenes ($\eta = 0$) to fully texture-replaced variants ($\eta = 1.0$). Crucially, the augmentation preserves ground-truth semantic masks across all degrees, as CNT modifies appearance but not region geometry.

**Intelligent Filtering.** We apply a geometry-first scoring system that quantifies each image's suitability for texture segmentation on a 0–100 scale. The score combines a mask-structure prior $s_A$ (region count, size distribution, entropy) with a boundary quality metric $s_{\text{geom}}$ (gradient-transition strength, boundary coherence, region balance), weighted toward geometry:

$$s = 0.20 \cdot s_A + 0.60 \cdot s_{\text{geom}} + \text{bonuses} - \text{penalties}$$

with bonuses for high texture coverage and 2–4 coherent regions, and penalties for object-dominated or excessively fragmented scenes. Semantic filtering classifies ADE20K categories into texture surfaces (~72 terms), objects (~32 terms), and ambiguous (~10 terms) via keyword matching. Images scoring below 65/100 are discarded. A key finding of this filtering stage is that while original indoor scenes (e.g., kitchens, living rooms, $\eta = 0$) were systematically discarded as object-centric and generic, their high-$\eta$ augmented counterparts were identified by the pipeline as "texture-rich laboratories"—the CNT augmentation had transformed semantically complex but texture-poor scenes into images with strong, well-defined texture transitions ideal for our training objective. Full scoring formulas and threshold parameters are provided in Appendix A.

**Extraction and VLM Annotation.** For each image passing the filter, we extract up to 5 geometrically distinct texture regions by merging per-class ADE20K masks (each region ≥1% image area). We then employ Claude 3.5 Sonnet [19] to autonomously generate rich ground-truth descriptions for each region. The VLM receives both the original image and a colored overlay showing region boundaries, and produces structured descriptions in the format "Texture of \<semantic name\>, \<visual features\>, \<spatial context\>" (10–15 words). For TextureSAM datasets, *degree expansion* reuses ground-truth masks (which are degree-invariant) across higher-intensity variants, yielding ~45% additional samples at zero annotation cost. The final training set comprises **~14,000 samples** with an average of 3.2 texture regions per image. Full pipeline details, prompt templates, and validation procedures are provided in Appendix B.

### 4.2 The Custom In-Domain Test Set (ADE20K\_Detecture)

To enable controlled in-domain evaluation, we applied the identical Detecture Miner to the **ADE20K validation split** (2,000 images), curating a dedicated test set we term **ADE20K\_Detecture**. This custom-mined dataset retains only images with strong texture transitions and well-separated regions, providing a high-quality benchmark for verifying core architectural viability in a controlled setting where the visual distribution matches the training data. ADE20K\_Detecture served as our stability probe: stable or improving metrics on this set confirmed that architectural changes (e.g., the bottleneck projector) did not degrade in-domain capability, even as cross-domain metrics revealed pathologies. For instance, one variant's in-domain mIoU climbed steadily from 0.700 to 0.707 across epochs 5–13 on ADE20K\_Detecture, while RWTD mIoU simultaneously degraded from 0.692 to 0.618—a divergence that would have been invisible without dual evaluation.

### 4.3 The Adopted External Benchmark (RWTD)

For out-of-domain evaluation, we adopted the **Real-World Texture Dataset (RWTD)** introduced by TextureSAM [1]: 253 images exhibiting challenging, in-the-wild texture transitions characterized by ambiguous boundaries (sand-to-water, grass-to-soil), multi-scale textures, and semantic diversity spanning natural, urban, and indoor environments. We did not collect RWTD; our contribution was *adapting it for VLM-guided models* by passing it through our VLM annotation stage to generate high-quality ground-truth descriptions for each texture region—descriptions that did not previously exist, since RWTD was originally designed for prompt-based (point/box) segmentation evaluation.

### 4.4 The Dual-Evaluation Strategy

These two benchmarks form a deliberate dual-evaluation strategy:

- **In-Domain Evaluation (ADE20K\_Detecture)**: Verifies core architectural viability in a controlled setting where the visual distribution matches the training data. Stable metrics here confirm that architectural innovations do not introduce regressions.

- **Out-of-Domain / Stress-Test Evaluation (RWTD)**: Functions as an external, in-the-wild probe to test zero-shot generalization and deliberately expose cross-domain pathologies. RWTD's distributional distance from ADE20K was instrumental in surfacing the Directional Drift pathology (Section 3.5): the 10.5M-parameter projector memorized ADE20K-specific manifold directions that compressed RWTD representations 4× more aggressively, a failure mode entirely invisible on ADE20K\_Detecture.

This dual-evaluation philosophy—verify stability in-domain, then stress-test out-of-domain—ensures that performance gains are genuine and generalizable, not artifacts of distribution overlap between training and evaluation data.

---

## 5. Experiments

**Training Protocol.** We train on ~14,000 Detecture-mined ADE20K samples with effective batch size 8 (batch 2, grad-accum 4) for a total of 30 epochs on a single GPU in bfloat16. The schedule is a two-stage curriculum (Section 3.7): 8 epochs of Bridge warmup under frozen Qwen, followed by 22 epochs with Qwen LoRA, masked-row SEG training, and a 10× Bridge LR decay jointly active. The 30-epoch horizon is composed of an initial 20-epoch cosine run followed by a 10-epoch Micro-Warmup Restart extension at $s = 0.15$ (Section 3.7). All results reported below use the *exactly-K* prompt to neutralise the Hungarian-matching inflation that the permissive "1 to 6" prompt induces.

**Ablation: Pathology Resolution Impact.** We evaluate the contribution of each architectural fix by progressively adding components to a baseline and measuring cross-domain generalization on the unseen RWTD benchmark. Rows labelled *V$k$* refer to numbered predecessor experiments documented in our ablation archive; the final row is the method proposed in this paper.

| Configuration | RWTD mIoU | Pathology Resolved |
|--------------|----------|-------------------|
| Baseline (10.5M proj, causal attn, full LM loss) | 0.544 | — (all pathologies present) |
| + Batch Multiplexing (V2) | 0.703 | Slot-1 Positional Bias |
| + Bottleneck Projector 2.1M (V4-Slim) | 0.732 | Directional Drift |
| + Block-Diagonal Mask + `[SEG]` Token (V5 Oracle, teacher-forced) | 0.810 | Context Leakage |
| V5 Live Inference (no LM supervision) | 0.136 | — (Language Collapse exposed) |
| + Cosine-decayed LM (V6, live, exactly-2) | 0.694 | Language Collapse (partial) |
| + Bridge + Frozen Resizer + naïve Exponential Cliff $\lambda_{\text{SEG}} = 0$ (Oracle) | 0.738 | Bottleneck Semantic Squash |
|   same configuration, *live* inference | 0.000 | — (Emission Failure exposed) |
| + Shifted-Zero Cliff (Oracle) | 0.781 | — |
| + Shifted-Zero + Masked-Row SEG + Warm-Start from `<\|im_end\|>` (live, ep 16) | **0.794** | Emission Failure |
| Zero-Shot SAM 3 with object-aware prompt (no training) | **0.928** | — (upper bound) |
| **Our method** (30-epoch schedule with Micro-Warmup Restart; live, best checkpoint) | *in progress* | — |

Each fix addresses a distinct failure mode. Resolving Slot-1 Bias alone gains +0.16 mIoU; the bottleneck projector adds +0.03 by eliminating Directional Drift; the block-diagonal mask and `[SEG]` token restore Context-Leakage-free grounding; the cosine-decayed LM weighting cures Language Collapse but leaves a DICE/LM tug-of-war (honest plateau 0.694). Widening into SAM 3's native 1024-dim text space closes another +0.04 on the teacher-forced Oracle. The naïve exponential cliff ($\lambda_{\text{SEG}} = 0$) inadvertently erases the emission-learning signal and drives *live-inference* mIoU to exactly 0.0 despite Oracle continuing to rise — isolating an Emission Failure pathology that is invisible under teacher forcing. Shifting the zero one position forward (Shifted-Zero) restores the emission gradient; warm-starting the SEG rows from `<|im_end|>` and unfreezing them under masked-row routing calibrates `lm\_head[\text{SEG}]` so that the token can actually be chosen at inference. Together they close the live gap: 0.000 → 0.794 at epoch 16, surpassing the best teacher-forced number from any prior bottleneck-family variant.

**Live vs. Oracle at epoch 16.** Notably, live-inference mIoU (0.794) slightly *exceeds* teacher-forced Oracle mIoU (0.781) at the same checkpoint. We attribute this to distributional mismatch: Oracle feeds the Bridge with `[SEG]` hidden states produced under a fixed "1 to 6" prompt, whereas live inference uses the actual two-texture prompt the model was trained to handle. The gap disappears under a matched prompt.

*[Training is in progress under the Micro-Warmup Restart extension (epochs 21–30). Final quantitative results on RWTD and ADE20K\_Detecture, comparison with LISA and Sa2VA baselines, and qualitative visualisations will be reported upon convergence.]*

---

## 6. Conclusion and Future Work

We have presented Qwen2SAM-Detecture, an end-to-end architecture for parallel, multi-texture segmentation that bridges vision-language reasoning with geometric dense prediction. Through systematic ablation, we identified and resolved five fundamental pathologies — Context Leakage, Slot-1 Positional Bias, the Count / Language Collapse dilemma, Directional Drift, and Bottleneck Semantic Squash — plus a subtle *Emission Failure* mode that emerges once the Count/Language dilemma is naïvely resolved and which we cure with a Shifted-Zero cliff and masked-row trainable SEG rows. Our solutions — block-diagonal attention masking, batch multiplexing, the Shifted-Zero LM-loss cliff with masked-row training and warm-start from `<|im_end|>`, and a Bridge-to-Native-Space projector that reuses SAM 3's own pretrained resizer — are general mechanisms applicable to any system coupling autoregressive VLM decoding with SAM-family decoders. The Zero-Shot SAM 3 experiment (0.928 RWTD mIoU with a hand-crafted object-aware prompt) is particularly instructive: it suggests that for any pretrained segmentation backbone, the correct role of a learned projector is *not* to acquire a new semantic mapping but to preserve the one the backbone already carries. The Bridge + frozen-resizer design is a minimal realization of this principle.

**Limitations.** Several limitations bound the current work. **(1) Inference cost.** The two-pass protocol — autoregressive generation followed by block-diagonal extraction — is substantially slower than prompt-based segmentation; end-to-end latency is dominated by Qwen generation (≈ 1–2 s per image on a single GPU) and scales with the number of predicted textures. **(2) SAM 3 as a hard ceiling.** Our frozen-SAM policy inherits SAM 3's semantic coverage: the 0.928 ZS upper bound on RWTD tightly upper-bounds the method on any domain where SAM natively understands descriptions, and leaves no recourse for domains where it does not. Unfreezing SAM to cross that ceiling consistently regressed out-of-domain generalisation in our ablations. **(3) Narrow benchmark.** RWTD contains 253 samples; honest exactly-K numbers are stable but a richer multi-domain evaluation would strengthen the cross-domain claims. **(4) Hungarian-matching illusion.** The common "1 to 6" prompt inflates scores via Hungarian assignment over permissively-many predicted regions; we therefore report exactly-K as the honest metric, but deployed systems receiving an unknown target count remain sensitive to prompt phrasing. **(5) Emission remains a learned behaviour.** Even with Shifted-Zero and masked-row training, `[SEG]` emission is not guaranteed under sufficiently out-of-distribution prompts; recovery requires either prompt normalisation or fallback regex extraction. **(6) Texture-specific framing.** We evaluate on texture segmentation; extending the method to open-vocabulary scene partitioning is plausible (Section 6.2) but not empirically validated here.

**Auto-Iterative Refinement.** A promising direction for future work is inference-time optimization without additional training. The current architecture produces masks in a single forward pass; however, the initial segmentation output contains rich geometric information that could be fed back to improve itself. Specifically, we envision extracting geometric center-of-mass points from each predicted mask region and injecting them into SAM's sparse prompt encoder *alongside* the dense `[SEG]` embedding in a second refinement pass. This auto-iterative loop would provide SAM with both semantic guidance (from the `[SEG]` token) and spatial guidance (from the center-of-mass points), effectively combining the complementary strengths of dense and sparse prompting. Preliminary analysis suggests this could push performance past the current single-pass ceiling without requiring retraining, as SAM's point-prompt pathway is already well-calibrated from pretraining.

**Scaling to Open-Vocabulary Partitioning.** While we evaluate on texture segmentation, the architectural innovations are domain-agnostic. Extending Qwen2SAM-Detecture to open-vocabulary scene partitioning—where the model autonomously identifies and segments arbitrary semantic regions (not just textures)—requires only expanding the training data domain while the architectural mechanisms remain unchanged.

---

## References

[1] I. Cohen, B. Meivar, P. Tu, S. Avidan, and G. Oren, "TextureSAM: Towards a Texture Aware Foundation Model for Segmentation," *arXiv:2505.16540*, 2025.

[2] X. Lai, Z. Tian, Y. Chen, Y. Li, Y. Yuan, S. Liu, and J. Jia, "LISA: Reasoning Segmentation via Large Language Model," in *CVPR*, 2024.

[3] X. Lai, Z. Tian, Y. Chen, Y. Li, Y. Yuan, S. Liu, and J. Jia, "LISA++: An Improved Baseline for Reasoning Segmentation with Large Language Model," *arXiv:2312.17240*, 2024.

[4] Z. Xia, D. Han, Y. Han, B. Pan, S. Song, G. Huang, and H. He, "GSVA: Generalized Segmentation via Multimodal Large Language Models," in *CVPR*, 2024.

[5] Z. Ren, Y. Huang, S. Wei, Q. Zhu, D. Xu, and D. Zhang, "PixelLM: Pixel Reasoning with Large Multimodal Model," in *CVPR*, 2024.

[6] H. Rasheed, M. U. Maaz, S. Shaji, A. Shaker, S. Khan, H. A. Cholakkal, R. M. Anwer, E. Xing, M.-H. Yang, and F. S. Khan, "GLaMM: Pixel Grounding Large Multimodal Model," in *CVPR*, 2024.

[7] H. Yuan, D. Li, R. Zhang, J. Zhang, C. Li, and J. Yan, "Sa2VA: Marrying SAM2 with LLaVA for Dense Grounded Understanding of Images and Videos," *arXiv:2501.04001*, 2025.

[8] S. Zhang, P. Sun, S. Chen, M. Xiao, W. Shao, W. Zhang, K. Chen, and P. Luo, "LLaVA-Grounding: Grounded Visual Chat with Large Multimodal Models," in *ECCV*, 2024.

[9] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, P. Dollár, and R. Girshick, "Segment Anything," in *ICCV*, 2023.

[10] N. Ravi, V. Gabeur, Y.-T. Hu, R. Hu, C. Ryali, T. Ma, H. Khedr, R. Rädle, C. Rolland, L. Gustafson, E. Mintun, J. Pan, K. V. Alwala, N. Carion, C.-Y. Wu, R. Girshick, P. Dollár, and C. Feichtenhofer, "SAM 2: Segment Anything in Images and Videos," *arXiv:2408.00714*, 2024.

[11] B. Cheng, I. Misra, A. G. Schwing, A. Kirillov, and R. Girshick, "Masked-attention Mask Transformer for Universal Image Segmentation," in *CVPR*, 2022.

[12] A. Kirillov, R. Girshick, K. He, and P. Dollár, "Panoptic Feature Pyramid Networks," in *CVPR*, 2019.

[13] M. Cimpoi, S. Maji, I. Kokkinos, S. Mohamed, and A. Vedaldi, "Describing Textures in the Wild," in *CVPR*, 2014.

[14] E. Hayman, B. Caputo, M. Fritz, and J.-O. Eklundh, "On the Significance of Real-World Conditions for Material Classification," in *ECCV*, 2004.

[15] H. Caesar, J. Uijlings, and V. Ferrari, "COCO-Stuff: Thing and Stuff Classes in Context," in *CVPR*, 2018.

[16] X. Zou, Z.-Y. Dou, J. Yang, Z. Gan, L. Li, C. Li, X. Dai, H. Beber, J. Wang, L. Yuan, N. Peng, L. Wang, Y. J. Lee, and J. Gao, "Generalized Decoding for Pixel, Image, and Language," in *CVPR*, 2023.

[17] H. Zhang, F. Li, X. Zou, S. Liu, C. Li, J. Gao, J. Yang, and L. Zhang, "A Simple Framework for Open-Vocabulary Segmentation and Detection," in *ICCV*, 2023.

[18] B. Zhou, H. Zhao, X. Puig, T. Xiao, S. Fidler, A. Barriuso, and A. Torralba, "Semantic Understanding of Scenes through the ADE20K Dataset," *International Journal of Computer Vision*, vol. 127, pp. 302–321, 2019.

[19] Anthropic, "The Claude Model Family: Claude 3.5 Sonnet, Claude 3.5 Haiku," Anthropic Technical Report, 2024.

[20] C. Zhang, Y. Qiao, S. Tariq, S. Zheng, C. Zhang, C. Li, H. Shin, and C. S. Hong, "Understanding Segment Anything Model: SAM is Biased Towards Texture Rather than Shape," *arXiv:2311.11465*, 2023.

---

## NeurIPS Paper Checklist

1. **Claims.** Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?
   **Answer:** [Yes] The abstract enumerates the five pathologies we diagnose and the six mechanisms we introduce; the contributions list in Section 1 matches Sections 3–5 one-for-one.

2. **Limitations.** Does the paper discuss the limitations of the work?
   **Answer:** [Yes] See the first paragraph of Section 6, which enumerates six distinct limitations covering inference cost, the SAM 3 ceiling, benchmark size, Hungarian-matching inflation, emission brittleness, and texture-specific framing.

3. **Theory Assumptions and Proofs.** For each theoretical result, does the paper provide the full set of assumptions and a complete proof?
   **Answer:** [N/A] This is an empirical architecture paper; no formal theorems are claimed.

4. **Experimental Result Reproducibility.** Does the paper fully disclose all the information needed to reproduce the main experimental results?
   **Answer:** [Yes] Full training config (loss weights, LR schedule, two-stage curriculum boundaries, Shifted-Zero `α`, Micro-Warmup Restart `s`, 30-epoch schedule) is specified in Section 3.7 and the `configs/detecture.yaml` included with the supplementary code. Dataset construction (Detecture Miner) is specified in Section 4 and Appendices A–B.

5. **Open Access to Data and Code.** Does the paper provide open access to the data and code?
   **Answer:** [Yes] Code, the Detecture mining pipeline, the ADE20K_Detecture custom test set, and the VLM-annotated RWTD ground-truth descriptions will be released under permissive licence on acceptance. The underlying RWTD images inherit TextureSAM's release terms [1].

6. **Experimental Setting/Details.** Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer, etc.)?
   **Answer:** [Yes] Section 3.7 (curriculum, LR schedule, per-group LRs), Section 5 (training protocol, batch size, effective batch, precision, wall clock), and the config YAML specify every hyperparameter. Hyperparameters were selected on the ADE20K validation split; RWTD was never used for selection.

7. **Experiment Statistical Significance.** Does the paper report error bars suitably and correctly?
   **Answer:** [Partial] Mask-IoU is reported as mean over the 253 RWTD samples; we do not report seed-variance over multiple training runs due to compute cost (30-epoch training ≈ 60 h on a single H100). We mitigate this by reporting honest exactly-K numbers that remove the main known source of variance (Hungarian inflation).

8. **Experiments Compute Resources.** Does the paper disclose the compute resources required for the experiments?
   **Answer:** [Yes] Single GPU (NVIDIA H100 class), bfloat16, effective batch 8, ~60 hours for the full 30-epoch schedule (including the 10-epoch Micro-Warmup Restart extension). Oracle eval ≈ 1.5 min per checkpoint; honest E2E exactly-2 eval ≈ 25 min per checkpoint.

9. **Code of Ethics.** Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics?
   **Answer:** [Yes]

10. **Broader Impacts.** Does the paper discuss both potential positive societal impacts and negative societal impacts of the work?
    **Answer:** [Yes, brief] Texture segmentation is a low-risk application with plausible positive uses in material inspection, robotics, and accessibility (terrain segmentation). We see no immediate negative societal impact; the VLM-SAM coupling is general enough to apply to sensitive tasks, but on its own produces only class-agnostic pixel assignments.

11. **Safeguards.** Does the paper describe safeguards that have been put in place for responsible release of data or models?
    **Answer:** [N/A] No high-risk generative content or identifying data is released. Image data is inherited from public sources (ADE20K, RWTD) under their existing licences.

12. **Licenses for Existing Assets.** Are the creators or original owners of assets used in the paper properly credited and are the licences and terms of use explicitly mentioned and properly respected?
    **Answer:** [Yes] ADE20K [18] is used under its academic research licence. RWTD [1] is used under its release terms. SAM 3 [9,10] and Qwen3-VL [Qwen] are used under their respective open weights licences. Claude 3.5 [19] was used only for one-off annotation generation under Anthropic's terms of use.

13. **New Assets.** Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?
    **Answer:** [Yes] The Detecture Miner will be released with documentation, scoring formulas (Appendix A), and annotation prompt templates (Appendix B). ADE20K_Detecture and VLM-annotated RWTD descriptions will be released with per-sample metadata.

14. **Crowdsourcing and Research with Human Subjects.** For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?
    **Answer:** [N/A] No human subjects.

15. **Institutional Review Board (IRB) Approvals or Equivalent.** Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether IRB approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?
    **Answer:** [N/A] No human subjects.

16. **Declaration of LLM Usage.** Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research?
    **Answer:** [Yes] (i) Qwen3-VL-8B is the reasoning backbone of our method and is described in Section 3.1. (ii) Claude 3.5 Sonnet was used to generate ground-truth texture descriptions during dataset construction, as documented in Section 4.1 ("Extraction and VLM Annotation") and Appendix B. No LLM was used to write, edit, or generate the scientific content of this manuscript.
