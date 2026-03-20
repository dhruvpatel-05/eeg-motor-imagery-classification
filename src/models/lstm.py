"""
Pure bidirectional LSTM for EEG motor imagery classification — 128 Hz / ERS pipeline.

Architecture for BCI Competition IV Dataset 2a:
    - 22 channels, 256 timepoints (2s @ 128Hz), 4 classes

Design philosophy:
  No CNN front-end. The LSTM operates directly on EEG channel features after a
  non-learnable AvgPool1d temporal subsampling step and a linear input projection
  over channels, without a conventional CNN feature extractor.

Pipeline:
  1. Input  (B, 22, 256)   -- channels-first from dataloader
  2. Temporal subsampling  AvgPool1d(kernel=4, stride=4)  → (B, 22, 64)  [zero params]
  3. Permute (B, 22, 64) → (B, 64, 22)   -- time-first for Linear projection
  4. Input projection      Linear(22 → proj_dim=24) + LayerNorm(proj_dim) + ELU
                           → (B, 64, 24)
  5. proj_dropout (0.2)   -- regularizes spatial representation before LSTM
  6. Bidirectional LSTM, 1 layer, hidden_size=48   → (B, 64, 96)
  7. lstm_out_dropout (0.4)  -- regularizes recurrent output
  8. LayerNorm on LSTM output (B, 64, 96)
  9. Temporal attention pooling  (B, 64, 96) → (B, 96)   [attn_dim=24]
  10. Classifier  Dropout(0.5) + Linear(96, n_classes) + MaxNorm(0.25)

Temporal subsampling:
  A non-learnable AvgPool1d (kernel_size=pool_factor, stride=pool_factor) reduces
  the sequence length from T=256 to T'=64 timesteps with zero parameters. This
  preserves all channels without cross-channel mixing.

Input projection:
  A single Linear(n_channels, proj_dim) layer maps the 22-channel feature vector
  at each timestep to a proj_dim=24 dimensional representation, followed by
  LayerNorm and ELU. This is the same for both subject_dependent and loso modes.

Bidirectional LSTM:
  The (B, T', proj_dim) feature tensor is fed directly into a single-layer
  bidirectional LSTM with hidden_size=48 per direction, yielding a 96-dimensional
  output at each of the 64 timesteps. Output dropout (p=0.4) and LayerNorm are
  applied before attention. LayerNorm is used throughout in place of BatchNorm to
  avoid accumulating cross-subject running statistics — correct for the
  ERS-normalised 128 Hz pipeline.

Temporal attention pooling and classification:
  The 64 LSTM timesteps are aggregated by a two-layer soft-attention mechanism
  (attn_dim=24): scores e_t = tanh(Wp*ht + b) are projected to scalars,
  normalised with softmax, and used to form a weighted context vector in R^96.
  This passes through dropout (p=0.5) and a linear classifier (96→4) with a
  MaxNorm weight constraint (max_norm=0.25), consistent with the EEGNet baselines.

Two execution modes are supported via the `mode` constructor argument:

  subject_dependent:
    No adversarial components. Forward returns logits or (logits, attn_weights).
    Recommended: lr=0.001, weight_decay=0.0005.

  loso:
    Optionally adds GradientReversal and subject_classifier head on the
    time-averaged input projection (adversarial=True). Forward returns
    (logits, subject_logits) or (logits, attn_weights, subject_logits).
    Training loss adds 0.1 * subject_adversarial_loss with alpha ramping 0→1.
    Recommended: lr=0.0007, weight_decay=0.001.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# ---------------------------------------------------------------------------
# Gradient Reversal Layer (DANN — Ganin et al. 2016)
# ---------------------------------------------------------------------------
class GradientReversalFn(Function):
    """Passes values forward unchanged; reverses (and scales) gradients."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class GradientReversal(nn.Module):
    """Thin nn.Module wrapper around GradientReversalFn."""

    def forward(self, x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        return GradientReversalFn.apply(x, alpha)


# ---------------------------------------------------------------------------
# Attention pooling — identical to cnn_gru_alternative.py
# ---------------------------------------------------------------------------
class TemporalAttentionPool(nn.Module):
    """
    Soft attention over the time dimension of LSTM output.

    Given H in (batch, T, D), computes:
        e_t  = tanh(W * h_t + b)       -> (batch, T, attn_dim)
        a_t  = softmax(v * e_t)        -> (batch, T)
        c    = sum a_t * h_t           -> (batch, D)
    """

    def __init__(self, hidden_dim: int, attn_dim: int = 24):
        super().__init__()
        self.project = nn.Linear(hidden_dim, attn_dim, bias=True)
        self.context = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: (batch, seq_len, hidden_dim)
        Returns:
            context: (batch, hidden_dim)  -- attention-weighted summary
            weights: (batch, seq_len)     -- attention distribution
        """
        e = torch.tanh(self.project(h))           # (B, T, attn_dim)
        scores = self.context(e).squeeze(-1)       # (B, T)
        weights = F.softmax(scores, dim=-1)        # (B, T)
        context = torch.bmm(weights.unsqueeze(1), h).squeeze(1)  # (B, D)
        return context, weights


# ---------------------------------------------------------------------------
# MaxNorm constraint — consistent with cnn_lstm_alternative.py
# ---------------------------------------------------------------------------
class MaxNormConstraint:
    """Clips the L2 norm of a parameter tensor to `max_norm` in-place."""

    def __init__(self, param: nn.Parameter, max_norm: float, dim: int = 0):
        self.param    = param
        self.max_norm = max_norm
        self.dim      = dim

    def __call__(self):
        with torch.no_grad():
            norms = self.param.norm(2, dim=self.dim, keepdim=True).clamp(min=1e-8)
            scale = (norms / self.max_norm).clamp(min=1.0)
            self.param.div_(scale)


# ---------------------------------------------------------------------------
# Pure LSTM model
# ---------------------------------------------------------------------------
class LSTM(nn.Module):
    """
    Pure bidirectional LSTM for EEG motor imagery — 128 Hz / ERS pipeline.

    Args:
        n_classes        : number of output classes (default: 4)
        n_channels       : number of EEG channels (default: 22)
        n_timepoints     : number of time samples (default: 256 = 2s @ 128Hz)
        pool_factor      : AvgPool1d stride for temporal subsampling (default: 4)
        proj_dim         : input projection dimension (default: 24)
        hidden_size      : LSTM hidden size per direction (default: 48)
        num_layers       : number of stacked LSTM layers (default: 1)
        attn_dim         : attention projection size (default: 24)
        dropout_rate     : dropout probability (inter-layer + classifier, default: 0.5)
        proj_dropout     : dropout after input projection, before LSTM (default: 0.2)
        lstm_out_dropout : dropout after LSTM output, before LayerNorm (default: 0.4)
        bidirectional    : use bidirectional LSTM (default: True)
    """

    def __init__(
        self,
        n_classes: int = 4,
        n_channels: int = 22,
        n_timepoints: int = 256,
        pool_factor: int = 4,
        proj_dim: int = 24,
        hidden_size: int = 48,
        num_layers: int = 1,
        attn_dim: int = 24,
        dropout_rate: float = 0.5,
        proj_dropout: float = 0.2,
        lstm_out_dropout: float = 0.4,
        bidirectional: bool = True,
        n_subjects: int = 9,
        mode: str = "subject_dependent",
        adversarial: bool = False,
    ):
        super().__init__()

        assert mode in ["subject_dependent", "loso"], \
            f"mode must be 'subject_dependent' or 'loso', got '{mode}'"
        self.mode = mode

        self.n_channels    = n_channels
        self.n_timepoints  = n_timepoints
        self.pool_factor   = pool_factor
        self.hidden_size   = hidden_size
        self.bidirectional = bidirectional

        # -- Non-learnable temporal subsampling ------------------------------
        # AvgPool1d reduces sequence length from n_timepoints → n_timepoints/pool_factor
        # (256 → 64 with default pool_factor=4). Zero parameters.
        self.time_pool = nn.AvgPool1d(kernel_size=pool_factor, stride=pool_factor)

        # -- Input projection: expand 22 channels → proj_dim before LSTM ----
        self.input_proj = nn.Sequential(
            nn.Linear(n_channels, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ELU(),
        )
        self.proj_drop     = nn.Dropout(proj_dropout)
        self.lstm_out_drop = nn.Dropout(lstm_out_dropout)

        # -- Bidirectional LSTM ----------------------------------------------
        lstm_out_dim = hidden_size * (2 if bidirectional else 1)

        self.lstm = nn.LSTM(
            input_size=proj_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # -- LayerNorm on LSTM output (no running stats) ---------------------
        self.lstm_norm = nn.LayerNorm(lstm_out_dim)

        # -- Temporal attention pooling (same as cnn_gru_alternative) --------
        self.attention = TemporalAttentionPool(lstm_out_dim, attn_dim)

        # -- Classifier head -------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_out_dim, n_classes),
        )

        self._cls_constraint = MaxNormConstraint(
            self.classifier[1].weight, max_norm=0.25, dim=1
        )

        # -- Subject adversarial head (DANN — LOSO only) ----------------------
        # Placed after input_proj (proj_dim) so GRL gradients reach the
        # spatial mixing weights directly, without attenuation through the LSTM.
        # Requires adversarial=True to be instantiated; off by default.
        self.adversarial = adversarial and (self.mode == "loso")
        if self.adversarial:
            self.grad_reversal      = GradientReversal()
            self.subject_classifier = nn.Linear(proj_dim, n_subjects)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x, return_attention=False, alpha=0.0):
        """
        Args:
            x: (batch, n_channels, n_timepoints)  -- channels-first from dataloader
            alpha: gradient-reversal scale for subject adversarial head
                   (0.0 = no adversarial signal; ramps to 1.0 during training)
            return_attention: if True, also return attention weights

        Returns:
            logits:         (batch, n_classes)
            subject_logits: (batch, n_subjects)  -- only when adversarial=True
            attn_weights (optional): (batch, n_timepoints // pool_factor)
        """
        # Drop channel-singleton dimension if present (4-D input from CNN pipeline)
        if x.dim() == 4:
            x = x.squeeze(1)  # (B, 1, C, T) -> (B, C, T)

        # -- Non-learnable temporal subsampling ------------------------------
        # AvgPool1d operates over the last dimension (time)
        x = self.time_pool(x)                         # (B, 22, 256) → (B, 22, 64)

        # Permute channels-first → time-first for Linear input projection
        x = x.permute(0, 2, 1)                        # (B, 22, 64) → (B, 64, 22)

        # -- Input projection: channels → proj_dim ---------------------------
        x = self.input_proj(x)                        # (B, 64, 22) → (B, 64, proj_dim)

        if self.adversarial:
            spatial_pooled = x.mean(dim=1)            # (B, proj_dim) — time-averaged repr

        x = self.proj_drop(x)                         # dropout before LSTM

        # -- LSTM ------------------------------------------------------------
        x, _ = self.lstm(x)      # (B, 64, lstm_out_dim)
        x = self.lstm_out_drop(x)  # dropout after LSTM, before LayerNorm
        x = self.lstm_norm(x)

        # -- Attention pooling -----------------------------------------------
        context, attn_weights = self.attention(x)  # (B, lstm_out_dim), (B, 64)

        # -- Classification --------------------------------------------------
        logits = self.classifier(context)  # (B, n_classes)

        if self.adversarial:
            subject_logits = self.subject_classifier(
                self.grad_reversal(spatial_pooled, alpha=alpha)
            )
            if return_attention:
                return logits, attn_weights, subject_logits
            return logits, subject_logits
        else:
            if return_attention:
                return logits, attn_weights
            return logits

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def apply_max_norm_(self) -> None:
        """Apply MaxNorm constraint to classifier weight. Called by train_128.py."""
        self._cls_constraint()


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for mode in ["subject_dependent", "loso"]:
        model = LSTM(mode=mode)
        x = torch.randn(8, 22, 256)
        out = model(x)
        assert out.shape == (8, 4)
        print(f"{mode} (adversarial=False): {model.count_parameters():,} params — OK")

    # adversarial=True path (loso only)
    model_adv = LSTM(mode="loso", adversarial=True)
    logits, subject_logits = model_adv(x)
    assert logits.shape == (8, 4)
    assert subject_logits.shape == (8, 9)
    print(f"loso (adversarial=True):  {model_adv.count_parameters():,} params — OK")
