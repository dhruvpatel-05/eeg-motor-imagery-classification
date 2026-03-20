"""
CNN+GRU Hybrid — 128 Hz / ERS pipeline variant (alternative).
==============================================================

This is the explicit 128 Hz-pipeline copy of cnn_gru.py, used by train_128.py
and evaluate_128.py under the model name "cnn_gru_alternative".

Architecture for BCI Competition IV Dataset 2a:
    - 22 channels, 256 timepoints (2s @ 128Hz), 4 classes

Design philosophy:
  CNN front-end (inspired by EEGNet) extracts local spatial-temporal features,
  then a bidirectional GRU captures long-range temporal dependencies across the
  sequence of CNN feature frames.

Pipeline:
  1. Temporal Conv   -> learn frequency-specific filters across time
  2. Depthwise Conv  -> learn spatial (channel) mixtures per temporal filter
  3. Separable Conv  -> refine combined spatial-temporal features
  4. Bi-GRU          -> model temporal dynamics across the feature sequence
  5. Attention pool  -> weighted aggregation of GRU hidden states
  6. Classifier      -> linear projection to class logits

128 Hz design correctness:
  - temporal_kernel = sfreq // 2 = 64 at 128 Hz (≥2 Hz frequency resolution)
  - sep_kernel = sfreq // 8 = 16 at 128 Hz
  - Pool ×4 then ×8 → T=256 reduces to T=8 (meaningful GRU sequence length)
  - BatchNorm2d uses track_running_stats=False (correct for cross-subject eval)
  - dropout forwarded from training CLI (--dropout flag)

Tuned defaults (3-round grid search, optimised for cross-subject generalization):
  Round 1: lr=0.001, wd=5e-4
  Round 2: gru_layers=2, batch_size confirmed
  Round 3: gru_hidden=48, cnn_dropout=0.4, fc_dropout=0.5, batch_size=16
           Winner: 64.34% mean val across A02/A03/A07 (hard/strong/mid subjects)
           A07 jumped to 67.4% (vs 25-35% in earlier configs)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Attention pooling - learns to weight GRU timesteps by relevance
# ---------------------------------------------------------------------------
class TemporalAttentionPool(nn.Module):
    """
    Soft attention over the time dimension of GRU output.

    Given H in (batch, T, D), computes:
        e_t  = tanh(W * h_t + b)          -> (batch, T, attn_dim)
        a_t  = softmax(v * e_t)            -> (batch, T)
        c    = sum a_t * h_t               -> (batch, D)
    """

    def __init__(self, hidden_dim: int, attn_dim: int = 48):
        super().__init__()
        self.project = nn.Linear(hidden_dim, attn_dim, bias=True)
        self.context = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: (batch, seq_len, hidden_dim)
        Returns:
            context: (batch, hidden_dim)  -- attention-weighted summary
            weights: (batch, seq_len)     -- attention distribution (for visualisation)
        """
        e = torch.tanh(self.project(h))           # (B, T, attn_dim)
        scores = self.context(e).squeeze(-1)       # (B, T)
        weights = F.softmax(scores, dim=-1)        # (B, T)
        context = torch.bmm(weights.unsqueeze(1), h).squeeze(1)  # (B, D)
        return context, weights


# ---------------------------------------------------------------------------
# CNN + GRU Hybrid
# ---------------------------------------------------------------------------
class CNNGRU(nn.Module):
    """
    CNN+GRU hybrid for EEG motor imagery classification.

    Args:
        n_classes      : number of output classes (default: 4)
        n_channels     : number of EEG channels (default: 22 for BCI-IV-2a)
        n_timepoints   : number of time samples (default: 256 = 2s @ 128Hz)
        sfreq          : sampling frequency in Hz (default: 128)
        F1             : number of temporal filters (default: 8)
        D              : depth multiplier for depthwise conv (default: 2)
        cnn_dropout    : dropout after CNN blocks (default: 0.4)
        gru_hidden     : GRU hidden size per direction (default: 48)
        gru_layers     : number of stacked GRU layers (default: 2)
        gru_dropout    : dropout between GRU layers (default: 0.3)
        attn_dim       : attention projection size (default: 48)
        fc_dropout     : dropout before final classifier (default: 0.5)
        bidirectional  : use bidirectional GRU (default: True)
    """

    def __init__(
        self,
        n_classes: int = 4,
        n_channels: int = 22,
        n_timepoints: int = 256,
        sfreq: int = 128,
        # CNN hyper-parameters
        F1: int = 8,
        D: int = 2,
        cnn_dropout: float = 0.4,
        # GRU hyper-parameters (tuned R3: hidden=48, layers=2)
        gru_hidden: int = 48,
        gru_layers: int = 2,
        gru_dropout: float = 0.3,
        # Attention & classifier (tuned R3: lighter dropout)
        attn_dim: int = 48,
        fc_dropout: float = 0.5,
        bidirectional: bool = True,
    ):
        super().__init__()

        self.n_classes = n_classes
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints
        self.gru_hidden = gru_hidden
        self.bidirectional = bidirectional

        F2 = F1 * D  # pointwise filter count

        # -- CNN Block 1: temporal + depthwise spatial filtering -------------
        temporal_kernel = sfreq // 2  # captures ~2 Hz bandwidth
        self.cnn_block1 = nn.Sequential(
            # Temporal convolution -- frequency-specific feature extraction
            nn.Conv2d(
                1, F1,
                kernel_size=(1, temporal_kernel),
                padding=(0, temporal_kernel // 2),
                bias=False,
            ),
            nn.BatchNorm2d(F1, track_running_stats = False),

            # Depthwise spatial convolution -- learns channel (electrode) mixing
            nn.Conv2d(
                F1, F2,
                kernel_size=(n_channels, 1),
                groups=F1,
                bias=False,
            ),
            nn.BatchNorm2d(F2, track_running_stats = False),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),   # T -> T/4
            nn.Dropout(cnn_dropout),
        )

        # -- CNN Block 2: separable convolution for refined features ---------
        sep_kernel = sfreq // 8
        self.cnn_block2 = nn.Sequential(
            # Depthwise temporal conv
            nn.Conv2d(
                F2, F2,
                kernel_size=(1, sep_kernel),
                padding=(0, sep_kernel // 2),
                groups=F2,
                bias=False,
            ),
            # Pointwise conv
            nn.Conv2d(F2, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2, track_running_stats = False),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),   # T/4 -> T/32
            nn.Dropout(cnn_dropout),
        )

        # -- Compute CNN output shape dynamically ----------------------------
        cnn_out_features, _ = self._cnn_output_shape()

        # -- Bidirectional GRU -- temporal dynamics modelling ----------------
        self.gru = nn.GRU(
            input_size=cnn_out_features,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=gru_dropout if gru_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        gru_out_dim = gru_hidden * (2 if bidirectional else 1)

        # Layer norm on GRU output for stable training
        self.gru_norm = nn.LayerNorm(gru_out_dim)

        # -- Temporal attention pooling --------------------------------------
        self.attention = TemporalAttentionPool(gru_out_dim, attn_dim)

        # -- Classifier head -------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Dropout(fc_dropout),
            nn.Linear(gru_out_dim, n_classes),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _cnn_output_shape(self) -> tuple[int, int]:
        """Run a dummy forward through CNN blocks to get (features, timesteps)."""
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.n_channels, self.n_timepoints)
            out = self.cnn_block2(self.cnn_block1(dummy))
            # out shape: (1, F2, 1, T')
            _, C, H, T = out.shape
            features = C * H   # F2 * 1 (spatial dim collapsed by depthwise conv)
            return features, T

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, 1, n_channels, n_timepoints) or (batch, n_channels, n_timepoints)
            return_attention: if True, also return attention weights for visualisation

        Returns:
            logits: (batch, n_classes)
            attn_weights (optional): (batch, seq_len)
        """
        # Accept 3-D input for convenience
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, C, T) -> (B, 1, C, T)

        # -- CNN feature extraction ------------------------------------------
        x = self.cnn_block1(x)   # (B, F2, 1, T/4)
        x = self.cnn_block2(x)   # (B, F2, 1, T/32)

        # Reshape to (batch, timesteps, features) for the GRU
        B, C, H, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * H)  # (B, T', F2)

        # -- GRU temporal modelling ------------------------------------------
        x, _ = self.gru(x)       # (B, T', gru_out_dim)
        x = self.gru_norm(x)

        # -- Attention pooling -----------------------------------------------
        context, attn_weights = self.attention(x)  # (B, gru_out_dim), (B, T')

        # -- Classification --------------------------------------------------
        logits = self.classifier(context)  # (B, n_classes)

        if return_attention:
            return logits, attn_weights
        return logits

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_cnn_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract CNN feature maps (useful for analysis / probing).

        Returns:
            features: (batch, timesteps, feature_dim)
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.cnn_block2(self.cnn_block1(x))
        B, C, H, T = x.shape
        return x.permute(0, 3, 1, 2).reshape(B, T, C * H)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 65)
    print(" CNN+GRU Hybrid (final R3 tuned) -- Sanity Check")
    print("=" * 65)

    model = CNNGRU()
    x = torch.randn(8, 1, 22, model.n_timepoints)

    # Forward pass
    logits = model(x)
    print(f"Input shape:        {tuple(x.shape)}")
    print(f"Output shape:       {tuple(logits.shape)}")
    print(f"Trainable params:   {model.count_parameters():,}")
    assert logits.shape == (8, 4), f"Unexpected output shape: {logits.shape}"

    # Forward with attention weights
    logits2, attn_w = model(x, return_attention=True)
    print(f"Attention weights:  {tuple(attn_w.shape)}")
    assert attn_w.shape[0] == 8, "Attention batch mismatch"
    assert torch.allclose(attn_w.sum(dim=-1), torch.ones(8), atol=1e-5), (
        "Attention weights don't sum to 1"
    )

    # 3-D input convenience
    x_3d = torch.randn(4, 22, model.n_timepoints)
    logits3 = model(x_3d)
    assert logits3.shape == (4, 4), "3-D input handling failed"

    # CNN feature extraction probe
    feats = model.get_cnn_features(x)
    print(f"CNN features:       {tuple(feats.shape)}")

    print("-" * 65)
    print("All sanity checks passed")
    print("=" * 65)

    # -- Architecture summary ------------------------------------------------
    print("\n-- Layer breakdown --")
    for name, module in model.named_children():
        n_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"  {name:20s}  {n_params:>8,} params")