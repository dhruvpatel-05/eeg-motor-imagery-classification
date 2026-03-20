"""
EEGNet implementation for BCI Competition IV Dataset 2a.

Architecture matches Lawhern et al. 2018 (Table 2) exactly:
    https://iopscience.iop.org/article/10.1088/1741-2552/aace8c

Parameters are derived directly from the preprocessing pipeline:
    - FS            = 128 Hz  (resampled from 250 Hz in preprocess.py)
    - N_EEG         = 22      (EEG channels only, EOG dropped in preprocess.py)
    - N_TIMEPOINTS  = 256     (0.5–2.5 s post cue at 128 Hz)
    - n_classes     = 4       (Left Hand, Right Hand, Both Feet, Tongue)

Key architecture decisions per paper:
    - Temporal kernel = FS // 2 = 64  (half sampling rate, captures ≥2 Hz)
    - Separable kernel = 16            (fixed — 500 ms at 32 Hz post-pooling)
    - F2 = F1 * D                      (enforced, not a free parameter)
    - DepthwiseConv max norm = 1       (Table 2, Options column)
    - Classifier max norm   = 0.25     (Table 2, Options column)
    - Dropout p = 0.5 within-subject, 0.25 cross-subject (pass via dropout_rate)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Max-norm weight constraint (applied after each optimiser step)
# ---------------------------------------------------------------------------

class MaxNormConstraint:
    """Clips the L2 norm of a parameter tensor to `max_norm` in-place.

    Usage:
        constraint = MaxNormConstraint(module.weight, max_norm=1.0)
        # call constraint() after every optimizer.step()
    """
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
# EEGNet
# ---------------------------------------------------------------------------

class EEGNet(nn.Module):
    """EEGNet-F1,D for BCI Competition IV Dataset 2a.

    Input shape : (batch, 1, N_EEG, N_TIMEPOINTS)  →  (batch, 1, 22, 256)
    Output shape: (batch, n_classes)                →  (batch, 4)

    Args:
        n_classes    : number of output classes. Default 4 (Dataset 2a).
        n_channels   : EEG channels (C). Default 22.
        n_timepoints : time samples (T). Default 512 (2–6 s MI @ 128 Hz).
        F1           : number of temporal filters. Default 8 (EEGNet-8,2).
        D            : depth multiplier — spatial filters per temporal filter.
                       Default 2 (EEGNet-8,2).
        dropout_rate : 0.5 for within-subject, 0.25 for cross-subject.
    """

    # Fixed constants derived from FS=128 and paper Table 2
    _FS              = 128   # Hz
    _TEMPORAL_KERNEL = 64    # = FS // 2  (captures frequency ≥ 2 Hz)
    _SEP_KERNEL      = 16    # fixed 500 ms at 32 Hz (post Block-1 pooling)
    _POOL1           = 4     # AveragePool Block 1
    _POOL2           = 8     # AveragePool Block 2

    def __init__(
        self,
        n_classes   : int   = 4,
        n_channels  : int   = 22,
        n_timepoints: int   = 512,
        F1          : int   = 8,
        D           : int   = 2,
        dropout_rate: float = 0.5,
    ):
        super().__init__()

        F2 = F1 * D   # enforced — not a free parameter (paper Section 2.2.1)

        self.n_classes    = n_classes
        self.n_channels   = n_channels
        self.n_timepoints = n_timepoints
        self.F2           = F2

        # ── Block 1 ────────────────────────────────────────────────────────
        # Step 1: Temporal Conv2D — (1, 64), mode=same, Linear activation
        # Input : (batch, 1,  C, T)
        # Output: (batch, F1, C, T)
        self.temporal_conv = nn.Conv2d(
            in_channels  = 1,
            out_channels = F1,
            kernel_size  = (1, self._TEMPORAL_KERNEL),
            padding      = (0, self._TEMPORAL_KERNEL // 2),   # exact "same"
            bias         = False,
        )
        self.bn1 = nn.BatchNorm2d(F1, track_running_stats=False)

        # Step 2: DepthwiseConv2D — (C, 1), mode=valid, depth=D, max norm=1
        # groups=F1 implements depthwise (each filter processed independently)
        # Input : (batch, F1, C, T)
        # Output: (batch, F1*D, 1, T)
        self.depthwise_conv = nn.Conv2d(
            in_channels  = F1,
            out_channels = F1 * D,
            kernel_size  = (n_channels, 1),
            groups       = F1,
            padding      = (0, 0),   # mode=valid
            bias         = False,
        )
        self.bn2     = nn.BatchNorm2d(F1 * D, track_running_stats=False)
        self.elu1    = nn.ELU()
        self.pool1   = nn.AvgPool2d(kernel_size=(1, self._POOL1))
        self.drop1   = nn.Dropout(p=dropout_rate)

        # Max-norm constraint on depthwise spatial filters (max norm = 1)
        self._dw_constraint = MaxNormConstraint(
            self.depthwise_conv.weight, max_norm=1.0, dim=0
        )

        # ── Block 2 ────────────────────────────────────────────────────────
        # SeparableConv2D = Depthwise (1,16) + Pointwise (1,1), mode=same
        # Input : (batch, F2, 1, T//4)
        # Output: (batch, F2, 1, T//4)
        self.sep_depthwise = nn.Conv2d(
            in_channels  = F2,
            out_channels = F2,
            kernel_size  = (1, self._SEP_KERNEL),
            padding      = (0, self._SEP_KERNEL // 2),   # mode=same
            groups       = F2,
            bias         = False,
        )
        self.sep_pointwise = nn.Conv2d(
            in_channels  = F2,
            out_channels = F2,
            kernel_size  = (1, 1),
            bias         = False,
        )
        self.bn3   = nn.BatchNorm2d(F2, track_running_stats=False)
        self.elu2  = nn.ELU()
        self.pool2 = nn.AvgPool2d(kernel_size=(1, self._POOL2))
        self.drop2 = nn.Dropout(p=dropout_rate)

        # ── Classifier ─────────────────────────────────────────────────────
        # Dense with Softmax, max norm = 0.25
        flat_size = self._get_flat_size(n_timepoints)
        self.classifier = nn.Linear(flat_size, n_classes, bias=True)

        # Max-norm constraint on classifier weights (max norm = 0.25)
        self._cls_constraint = MaxNormConstraint(
            self.classifier.weight, max_norm=0.25, dim=1
        )

        self._init_weights()

    # ── Helpers ────────────────────────────────────────────────────────────

    def _get_flat_size(self, n_timepoints: int) -> int:
        """Compute flattened feature size after Block 2 without allocating GPU mem."""
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.n_channels, n_timepoints)
            out   = self._forward_blocks(dummy)
        return int(out.numel())

    def _init_weights(self):
        """Xavier uniform init for conv layers; constant init for BN."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias,   0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def apply_constraints(self):
        """Call after every optimizer.step() to enforce max-norm constraints.

        Example training loop usage:
            optimizer.step()
            model.apply_constraints()
        """
        self._dw_constraint()
        self._cls_constraint()

    # ── Forward ────────────────────────────────────────────────────────────

    def _forward_blocks(self, x: torch.Tensor) -> torch.Tensor:
        """Run Block 1 and Block 2, return flat tensor before classifier."""

        # Block 1
        # Temporal Conv → BN (no activation — Linear per paper)
        x = self.temporal_conv(x)
        x = self.bn1(x)

        # Trim to exact "valid" output after depthwise (handles off-by-one
        # from even kernel same-padding in temporal conv)
        x = self.depthwise_conv(x)   # valid, collapses channel dim → (B, F1*D, 1, T')
        x = self.bn2(x)
        x = self.elu1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        # Block 2 — SeparableConv2D
        x = self.sep_depthwise(x)
        x = self.sep_pointwise(x)
        x = self.bn3(x)
        x = self.elu2(x)
        x = self.pool2(x)
        x = self.drop2(x)

        return x.flatten(start_dim=1)

    def apply_max_norm_(self) -> None:
        """Alias for apply_constraints() — called by train.py after every optimizer step."""
        self.apply_constraints()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, n_channels, n_timepoints) or (batch, 1, n_channels, n_timepoints)
               The dataloader returns 3-D tensors; the unsqueeze is handled here.

        Returns:
            logits: (batch, n_classes)
            Softmax is NOT applied here — use nn.CrossEntropyLoss which
            applies log-softmax internally (numerically stable).
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)   # (B, C, T) → (B, 1, C, T)
        x = self._forward_blocks(x)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Convenience factory matching paper configurations
# ---------------------------------------------------------------------------

def eegnet_8_2(dropout_rate: float = 0.5, **kwargs) -> EEGNet:
    """EEGNet-8,2 — default configuration from the paper."""
    return EEGNet(F1=8, D=2, dropout_rate=dropout_rate, **kwargs)


def eegnet_4_2(dropout_rate: float = 0.5, **kwargs) -> EEGNet:
    """EEGNet-4,2 — smaller configuration from the paper."""
    return EEGNet(F1=4, D=2, dropout_rate=dropout_rate, **kwargs)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = eegnet_8_2()
    print(model)

    x = torch.randn(16, 1, 22, 512)   # batch=16, 1 channel, 22 EEG ch, 512 timepoints
    logits = model(x)
    print(f"\nInput  shape : {x.shape}")
    print(f"Output shape : {logits.shape}")   # expect (16, 4)

    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {total:,}")