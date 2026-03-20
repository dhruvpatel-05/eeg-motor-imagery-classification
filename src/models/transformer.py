"""
EEG Transformer: EEGNet Front-End + Temporal Attention (128 Hz variant)
======================================================================

Architecture for BCI Competition IV Dataset 2a:
  - 22 channels, 256 timepoints (2s @ 128Hz), 4 classes

Design:
  EEGNet-8,2 exact CNN front-end (128 Hz) → Temporal Transformer → Classifier.

  EEGNet:      CNN Block1 → Block2 → Flatten → Linear
  This model:  CNN Block1 → Block2 → Temporal Attention → Linear

  CNN at 128 Hz produces (B, F2=16, 1, T'=8). The transformer attends
  over 8 time steps, each with 16 features. Short sequence = fast attention,
  low parameter overhead.

CNN front-end (identical to EEGNet-8,2 at 128 Hz):
  - F1=8, D=2 → F2=16
  - Temporal kernel = 64 (= 128 // 2)
  - Separable kernel = 16 (= 128 // 8)
  - Pool ×4 → ×8: 256 → 64 → 8 timesteps
  - Max-norm on depthwise (1.0) and classifier (0.25)

Transformer:
  - d_model = 32, n_heads = 4 (8 dims per head)
  - n_layers = 2 (enough for 8-step sequence)
  - ff_dim = 64 (2× d_model)
  - ~12K total params (EEGNet ~2K CNN + ~10K transformer)

Attention visualization:
  - get_temporal_attention() → (B, heads, 8, 8) — which time steps matter
  - get_depthwise_weights() → (F2, 1, 22, 1) — spatial filter patterns
  - Combine for C3/C4 motor cortex analysis
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Max-norm constraint (identical to EEGNet)
# ---------------------------------------------------------------------------
class MaxNormConstraint:
    def __init__(self, param: nn.Parameter, max_norm: float, dim: int = 0):
        self.param = param
        self.max_norm = max_norm
        self.dim = dim

    def __call__(self):
        with torch.no_grad():
            norms = self.param.norm(2, dim=self.dim, keepdim=True).clamp(min=1e-8)
            scale = (norms / self.max_norm).clamp(min=1.0)
            self.param.div_(scale)


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 64, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1), :])


# ---------------------------------------------------------------------------
# Multi-head self-attention with extractable weights
# ---------------------------------------------------------------------------
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, _ = x.shape
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.W_o(context), attn_weights


# ---------------------------------------------------------------------------
# Transformer encoder block
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attn(x_norm)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x, attn_weights


# ---------------------------------------------------------------------------
# EEG Transformer (128 Hz)
# ---------------------------------------------------------------------------
class EEGTransformer(nn.Module):
    """
    EEGNet-8,2 CNN front-end (128 Hz) → Temporal Transformer → Classifier.

    Args:
        n_classes       : output classes (default: 4)
        n_channels      : EEG channels (default: 22)
        n_timepoints    : time samples (default: 256 = 2s @ 128Hz)
        F1              : temporal filters (default: 8, EEGNet-8,2)
        D               : depth multiplier (default: 2, EEGNet-8,2)
        cnn_dropout     : CNN dropout (default: 0.5, EEGNet within-subject)
        d_model         : transformer dim (default: 32)
        n_heads         : attention heads (default: 4)
        n_layers        : transformer blocks (default: 2)
        ff_dim          : feedforward dim (default: 64)
        trans_dropout   : transformer dropout (default: 0.3)
        fc_dropout      : classifier dropout (default: 0.5)
    """

    _TEMPORAL_KERNEL = 64    # 128 // 2
    _SEP_KERNEL = 16         # 128 // 8
    _POOL1 = 4
    _POOL2 = 8

    def __init__(
        self,
        n_classes: int = 4,
        n_channels: int = 22,
        n_timepoints: int = 256,
        # CNN (identical to EEGNet-8,2 at 128 Hz)
        F1: int = 8,
        D: int = 2,
        cnn_dropout: float = 0.5,
        # Transformer
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_dim: int = 64,
        trans_dropout: float = 0.3,
        # Classifier
        fc_dropout: float = 0.5,
    ):
        super().__init__()

        self.n_classes = n_classes
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints
        self.d_model = d_model

        F2 = F1 * D  # 16

        # ══════════════════════════════════════════════════════════════
        # CNN Block 1 (identical to EEGNet at 128 Hz)
        # ══════════════════════════════════════════════════════════════
        self.temporal_conv = nn.Conv2d(
            1, F1, kernel_size=(1, self._TEMPORAL_KERNEL),
            padding=(0, self._TEMPORAL_KERNEL // 2), bias=False,
        )
        self.bn1 = nn.BatchNorm2d(F1)

        self.depthwise_conv = nn.Conv2d(
            F1, F2, kernel_size=(n_channels, 1),
            groups=F1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(F2)
        self.elu1 = nn.ELU()
        self.pool1 = nn.AvgPool2d(kernel_size=(1, self._POOL1))
        self.drop1 = nn.Dropout(p=cnn_dropout)

        self._dw_constraint = MaxNormConstraint(
            self.depthwise_conv.weight, max_norm=1.0, dim=0
        )

        # ══════════════════════════════════════════════════════════════
        # CNN Block 2 (identical to EEGNet at 128 Hz)
        # ══════════════════════════════════════════════════════════════
        self.sep_depthwise = nn.Conv2d(
            F2, F2, kernel_size=(1, self._SEP_KERNEL),
            padding=(0, self._SEP_KERNEL // 2),
            groups=F2, bias=False,
        )
        self.sep_pointwise = nn.Conv2d(F2, F2, kernel_size=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.elu2 = nn.ELU()
        self.pool2 = nn.AvgPool2d(kernel_size=(1, self._POOL2))
        self.drop2 = nn.Dropout(p=cnn_dropout)

        # ══════════════════════════════════════════════════════════════
        # Temporal Transformer
        # ══════════════════════════════════════════════════════════════
        cnn_F2, cnn_T = self._cnn_output_shape()

        # Project F2=16 features up to d_model=32
        self.input_proj = nn.Linear(cnn_F2, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(
            d_model, max_len=cnn_T + 10, dropout=trans_dropout
        )

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_dim, trans_dropout)
            for _ in range(n_layers)
        ])

        self.output_norm = nn.LayerNorm(d_model)

        # ══════════════════════════════════════════════════════════════
        # Classifier
        # ══════════════════════════════════════════════════════════════
        self.classifier_drop = nn.Dropout(fc_dropout)
        self.classifier = nn.Linear(d_model, n_classes)

        self._cls_constraint = MaxNormConstraint(
            self.classifier.weight, max_norm=0.25, dim=1
        )

        self._init_weights()

    # ------------------------------------------------------------------
    def _cnn_output_shape(self) -> tuple[int, int]:
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.n_channels, self.n_timepoints)
            out = self._forward_cnn(dummy)
            _, F2, _, T = out.shape
            return F2, T

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def apply_max_norm_(self):
        """Called by train.py after every optimizer step."""
        self._dw_constraint()
        self._cls_constraint()

    def get_depthwise_weights(self) -> torch.Tensor:
        """(F2, 1, n_channels, 1) — spatial filter patterns."""
        return self.depthwise_conv.weight.data

    # ------------------------------------------------------------------
    def _forward_cnn(self, x: torch.Tensor) -> torch.Tensor:
        """EEGNet Blocks 1+2, identical forward path."""
        # Block 1
        x = self.bn1(self.temporal_conv(x))
        x = self.drop1(self.pool1(self.elu1(self.bn2(self.depthwise_conv(x)))))
        # Block 2
        x = self.drop2(self.pool2(self.elu2(self.bn3(
            self.sep_pointwise(self.sep_depthwise(x))
        ))))
        return x  # (B, F2, 1, T')

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        if x.dim() == 3:
            x = x.unsqueeze(1)

        B = x.size(0)

        # CNN (identical to EEGNet)
        x = self._forward_cnn(x)           # (B, F2, 1, T')
        x = x.squeeze(2).permute(0, 2, 1)  # (B, T', F2)

        # Project to transformer dim
        x = self.input_proj(x)             # (B, T', d_model)
        x = self.pos_enc(x)

        # Temporal transformer
        all_attn_weights = []
        for block in self.transformer_blocks:
            x, attn_weights = block(x)
            all_attn_weights.append(attn_weights)

        # Pool over time
        x = self.output_norm(x)
        x = x.mean(dim=1)                 # (B, d_model)

        # Classify
        x = self.classifier_drop(x)
        logits = self.classifier(x)        # (B, n_classes)

        if return_attention:
            return logits, {
                "temporal": all_attn_weights[-1],
                "all_layers": all_attn_weights,
            }
        return logits

    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_temporal_attention(self, x: torch.Tensor) -> torch.Tensor:
        """(B, n_heads, T', T') — which time steps attend to which."""
        _, attn_dict = self.forward(x, return_attention=True)
        return attn_dict["temporal"]

    def get_all_attention_layers(self, x: torch.Tensor) -> list:
        """List of (B, n_heads, T', T') per layer."""
        _, attn_dict = self.forward(x, return_attention=True)
        return attn_dict["all_layers"]


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print(" EEG Transformer (128 Hz, EEGNet front-end) — Sanity Check")
    print("=" * 60)

    model = EEGTransformer()
    x = torch.randn(8, 1, 22, 256)

    logits = model(x)
    print(f"Input:  {tuple(x.shape)}")
    print(f"Output: {tuple(logits.shape)}")
    print(f"Params: {model.count_parameters():,}")
    assert logits.shape == (8, 4)

    logits2, attn_dict = model(x, return_attention=True)
    print(f"Temporal attn: {tuple(attn_dict['temporal'].shape)}")
    print(f"Attn layers:   {len(attn_dict['all_layers'])}")

    dw = model.get_depthwise_weights()
    print(f"Depthwise:     {tuple(dw.shape)}")

    assert model(torch.randn(4, 22, 256)).shape == (4, 4)

    print("-" * 60)
    print("✓ All checks passed")
    print(f"\nEEGNet-8,2 (128Hz): ~1,716 params")
    print(f"This transformer:   {model.count_parameters():,} params")