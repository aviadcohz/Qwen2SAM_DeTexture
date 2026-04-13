"""
V5 Bottleneck Projector: Qwen [SEG] hidden space → SAM query space (256).

Deliberately constrained to ~2M parameters to force generalization and
prevent memorizing domain-specific manifold directions (the Directional
Drift pathology identified in V4).

    Linear(4096 → 512) + LayerNorm(512) + GELU
    Linear(512  → 256)

No frozen SAM layers — the [SEG] token representations are visually-grounded
learned vectors, not linguistic embeddings, so anchoring to SAM's CLIP-based
text head is counterproductive.
"""

import torch.nn as nn


class DescriptionProjector(nn.Module):
    """
    V5 Information Bottleneck: maps [SEG] token hidden states from Qwen's
    4096-D space into SAM's 256-D query space.

    Intentionally small (~2.1M params) to act as a strict filter that
    cannot memorize dataset-specific directions.

    Forward path:
        Linear(4096 → 512) + LayerNorm(512) + GELU
        Linear(512  → 256)

    Applied per-slot to the query sequence:
        [DUSTBIN, SEG_1, ..., SEG_K, PAD...]
    """

    def __init__(self, llm_dim: int = 4096, sam_dim: int = 256,
                 hidden_dim: int = 512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(llm_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.15),
            nn.Linear(hidden_dim, sam_dim),
        )

    def forward(self, x):
        """
        Args:
            x: (B, N, llm_dim) — query sequence in LLM space.
        Returns:
            (B, N, sam_dim) — query sequence in SAM3 space.
        """
        return self.proj(x)
