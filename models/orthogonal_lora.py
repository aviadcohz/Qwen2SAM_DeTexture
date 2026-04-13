"""
Orthogonal LoRA for nn.MultiheadAttention and nn.Linear layers.

Applies low-rank adaptation ΔW = B·A while regularizing ΔW to be
orthogonal to the original frozen weights W₀. This prevents catastrophic
forgetting of SAM3's zero-shot capabilities.

The orthogonality penalty projects ΔW onto the dominant singular vectors
of W₀ and penalizes the projection magnitude:
    L_orth = || U_k^T · ΔW ||_F^2
where U_k are the top-k left singular vectors of W₀.

For nn.MultiheadAttention with packed in_proj_weight, we use
torch.nn.utils.parametrize.register_parametrization to inject LoRA
into the separate q/k/v projection weights while preserving PyTorch's
native attention path (FlashAttention / SDPA).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as parametrize


# ===================================================================== #
#  OrthogonalLoRALinear — for wrapping standalone nn.Linear layers       #
# ===================================================================== #

class OrthogonalLoRALinear(nn.Module):
    """
    LoRA adapter for a single nn.Linear layer with orthogonality constraint.

    Wraps a frozen linear layer and adds a trainable low-rank bypass:
        output = frozen_linear(x) + (x @ A^T) @ B^T   [scaled by alpha/r]
    """

    def __init__(
        self,
        frozen_linear: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        n_singular: int = 32,
    ):
        super().__init__()
        self.frozen_linear = frozen_linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_features = frozen_linear.in_features
        out_features = frozen_linear.out_features

        # LoRA matrices: A (r, in) and B (out, r)
        self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # Precompute top-k left singular vectors of W₀
        with torch.no_grad():
            W0 = frozen_linear.weight.data.float()
            k = min(n_singular, min(W0.shape))
            U, S, Vt = torch.linalg.svd(W0, full_matrices=False)
            self.register_buffer("U_k", U[:, :k].clone())

        # Freeze the original linear
        for p in self.frozen_linear.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.frozen_linear(x)
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return base_out + lora_out

    def orthogonal_penalty(self) -> torch.Tensor:
        """|| U_k^T @ ΔW ||_F^2 where ΔW = B @ A * scaling."""
        delta_W = (self.lora_B @ self.lora_A) * self.scaling
        proj = self.U_k.T @ delta_W
        return (proj ** 2).sum()


# ===================================================================== #
#  OrthogonalLoRAParametrization — for MHA projection weights            #
# ===================================================================== #

class OrthogonalLoRAParametrization(nn.Module):
    """
    Parametrization module for register_parametrization().

    Dynamically computes W_effective = W_frozen + (B @ A) * scaling
    on every forward pass. Autograd flows through the matmul to
    lora_A and lora_B, so they train correctly.
    """

    def __init__(self, in_features: int, out_features: int,
                 r: int = 8, alpha: float = 16.0, n_singular: int = 32,
                 frozen_weight: torch.Tensor = None):
        super().__init__()
        self.scaling = alpha / r

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # Top-k SVD of frozen weight for orthogonal penalty
        if frozen_weight is not None:
            with torch.no_grad():
                W0 = frozen_weight.float()
                k = min(n_singular, min(W0.shape))
                U, S, Vt = torch.linalg.svd(W0, full_matrices=False)
                self.register_buffer("U_k", U[:, :k].clone())
        else:
            self.register_buffer("U_k", torch.empty(0))

    def forward(self, W_frozen: torch.Tensor) -> torch.Tensor:
        """Called by parametrize: returns W_frozen + LoRA delta."""
        delta = (self.lora_B @ self.lora_A) * self.scaling
        return W_frozen + delta.to(W_frozen.dtype)

    def orthogonal_penalty(self) -> torch.Tensor:
        """|| U_k^T @ ΔW ||_F^2"""
        if self.U_k.numel() == 0:
            return torch.tensor(0.0, device=self.lora_A.device)
        delta_W = (self.lora_B @ self.lora_A) * self.scaling
        proj = self.U_k.T @ delta_W
        return (proj ** 2).sum()


# ===================================================================== #
#  apply_orthogonal_lora_to_mha — main entry point                      #
# ===================================================================== #

def apply_orthogonal_lora_to_mha(
    mha: nn.MultiheadAttention,
    r: int = 8,
    alpha: float = 16.0,
    n_singular: int = 32,
    target_projections: tuple = ("q", "v"),
) -> dict:
    """
    Apply Orthogonal LoRA to selected projections of an nn.MultiheadAttention.

    For packed in_proj_weight: splits into separate q/k/v_proj_weight
    Parameters and uses register_parametrization for LoRA targets.
    PyTorch's native MHA forward (FlashAttention/SDPA) is preserved.

    For already-separate projections: wraps with OrthogonalLoRALinear.

    Returns dict of modules that have orthogonal_penalty() for loss collection.
    """
    embed_dim = mha.embed_dim
    lora_modules = {}

    if mha.in_proj_weight is not None:
        # --- Packed in_proj_weight: split into separate q/k/v ---
        W = mha.in_proj_weight.data.clone()
        has_bias = mha.in_proj_bias is not None
        b = mha.in_proj_bias.data.clone() if has_bias else None

        # Split weights
        W_q = W[:embed_dim]
        W_k = W[embed_dim:2 * embed_dim]
        W_v = W[2 * embed_dim:]

        # Switch MHA to separate-projection mode
        mha.in_proj_weight = None
        mha.in_proj_bias = None
        mha._qkv_same_embed_dim = False

        # Register separate projection weight Parameters (frozen base)
        mha.q_proj_weight = nn.Parameter(W_q, requires_grad=False)
        mha.k_proj_weight = nn.Parameter(W_k, requires_grad=False)
        mha.v_proj_weight = nn.Parameter(W_v, requires_grad=False)

        # Register separate bias Parameters
        if has_bias:
            b_q = b[:embed_dim]
            b_k = b[embed_dim:2 * embed_dim]
            b_v = b[2 * embed_dim:]
            # MHA forward looks for in_proj_bias when _qkv_same_embed_dim=False
            # and uses bias_k/bias_v. But with separate weights, it expects
            # individual bias handling. Set in_proj_bias to None and handle via
            # the forward's internal logic which concatenates q/k/v biases.
            # PyTorch MHA with separate weights uses in_proj_bias=None but
            # internally slices if present. We register as a concatenated bias.
            mha.register_parameter(
                "in_proj_bias",
                nn.Parameter(torch.cat([b_q, b_k, b_v]), requires_grad=False),
            )

        # Apply parametrization to target projections
        for name in ("q", "k", "v"):
            weight_attr = f"{name}_proj_weight"
            W_proj = getattr(mha, weight_attr).data

            if name in target_projections:
                param_module = OrthogonalLoRAParametrization(
                    in_features=embed_dim, out_features=embed_dim,
                    r=r, alpha=alpha, n_singular=n_singular,
                    frozen_weight=W_proj,
                )
                parametrize.register_parametrization(mha, weight_attr, param_module)
                lora_modules[name] = param_module

        # DO NOT override mha.forward — native attention (FlashAttention) is used

    else:
        # --- Already using separate q_proj, k_proj, v_proj modules ---
        proj_map = {"q": "q_proj", "k": "k_proj", "v": "v_proj"}
        for name in target_projections:
            attr = proj_map[name]
            original = getattr(mha, attr)
            lora_mod = OrthogonalLoRALinear(
                original, r=r, alpha=alpha, n_singular=n_singular,
            )
            setattr(mha, attr, lora_mod)
            lora_modules[name] = lora_mod

    return lora_modules
