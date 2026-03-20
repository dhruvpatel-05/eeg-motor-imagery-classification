"""
Temporal Convolutional Network (TCN) for EEG Motor Imagery Classification.
Based on: EEG-TCNet (Ingolfsson et al., 2020) - https://arxiv.org/abs/2006.00622
 
Architecture overview:
  1. EEGNet front-end  — compact spatial-temporal feature extraction
  2. TCN blocks        — dilated causal convolutions for long-range temporal modeling
  3. Classifier        — linear layer to 4 classes
 
The EEGNet front-end reduces the 1000-sample input to a compact feature sequence,
then stacked TCN blocks with exponentially increasing dilation capture temporal
dependencies at multiple timescales without recurrence.
 
Input:  (batch, 22, 1000)  — 22 EEG channels, 1000 timepoints (4s @ 250Hz)
Output: (batch, 4)         — logits for 4 motor imagery classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Temporal block — single dilated causal conv residual block
# ---------------------------------------------------------------------------
 
class TemporalBlock(nn.Module):
    """
    Single TCN residual block with dilated causal convolution.
 
    Uses two conv layers with the same dilation, each followed by
    BatchNorm + ELU + Dropout. A residual connection with optional
    1x1 projection handles channel dimension changes.
 
    Args:
        in_channels  : input channel count
        out_channels : output channel count
        kernel_size  : temporal kernel size (default: 4, per EEG-TCNet)
        dilation     : dilation factor (1, 2, 4, 8, ...)
        dropout      : dropout probability
    """
 
    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int = 4,
        dilation:     int = 1,
        dropout:      float = 0.3,
        track_running_stats: bool = True
    ):
        super().__init__()
 
        # Causal padding: pad only the left side so output doesn't depend on future
        padding = (kernel_size - 1) * dilation
 
        self.conv1 = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.bn1 = nn.BatchNorm1d(out_channels, track_running_stats=track_running_stats)
        self.elu1    = nn.ELU()
        self.drop1   = nn.Dropout(dropout)
 
        self.conv2 = nn.Conv1d(
            out_channels, out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.bn2 = nn.BatchNorm1d(out_channels, track_running_stats=track_running_stats)
        self.elu2    = nn.ELU()
        self.drop2   = nn.Dropout(dropout)
 
        # 1x1 projection for residual if channel dims differ
        self.residual_proj = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else nn.Identity()
        )
 
        self._padding = padding
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_proj(x)
 
        # Conv 1 — trim causal padding from right
        out = self.conv1(x)
        out = out[:, :, :-self._padding] if self._padding > 0 else out
        out = self.elu1(self.bn1(out))
        out = self.drop1(out)
 
        # Conv 2 — trim causal padding from right
        out = self.conv2(out)
        out = out[:, :, :-self._padding] if self._padding > 0 else out
        out = self.elu2(self.bn2(out))
        out = self.drop2(out)
 
        return out + residual
 
# ---------------------------------------------------------------------------
# TCN
# ---------------------------------------------------------------------------
 
class TCN(nn.Module):
    """
    EEG-TCNet: EEGNet front-end + stacked dilated TCN blocks.
 
    Args:
        n_classes      : number of output classes (default: 4)
        n_channels     : number of EEG channels (default: 22)
        n_timepoints   : number of time samples (default: 1000)
        sfreq          : sampling frequency in Hz (default: 250)
        F1             : temporal filters in EEGNet front-end (default: 8)
        D              : depth multiplier in EEGNet front-end (default: 2)
        eegnet_dropout : dropout in EEGNet front-end (default: 0.3)
        tcn_filters    : number of filters in each TCN block (default: 12)
        tcn_kernel     : kernel size in TCN blocks (default: 4)
        tcn_layers     : number of TCN blocks (default: 2)
        tcn_dropout    : dropout in TCN blocks (default: 0.3)
        fc_dropout     : dropout before classifier (default: 0.5)
    """
 
    def __init__(
        self,
        n_classes:      int   = 4,
        n_channels:     int   = 22,
        n_timepoints:   int   = 1000,
        sfreq:          int   = 250,
        # EEGNet front-end
        F1:             int   = 8,
        D:              int   = 2,
        eegnet_dropout: float = 0.2,
        # TCN blocks
        tcn_filters:    int   = 12,
        tcn_kernel:     int   = 4,
        tcn_layers:     int   = 2,
        tcn_dropout:    float = 0.4,
        # Classifier
        fc_dropout:     float = 0.5,
        track_running_stats: bool = True
    ):
        super().__init__()
 
        self.n_channels   = n_channels
        self.n_timepoints = n_timepoints
        F2 = F1 * D
 
        # ── EEGNet front-end ──────────────────────────────────────────────
        # Temporal conv: (1, sfreq//2) — captures ≥2 Hz
        temporal_kernel = sfreq // 2   # 125 for 250 Hz
 
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(
                1, F1,
                kernel_size=(1, temporal_kernel),
                padding=(0, temporal_kernel // 2),
                bias=False,
            ),
            nn.BatchNorm2d(F1, track_running_stats=track_running_stats),
        )
 
        # Depthwise spatial conv: collapses channel dimension
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                F1, F2,
                kernel_size=(n_channels, 1),
                groups=F1,
                bias=False,
            ),
            nn.BatchNorm2d(F2, track_running_stats=track_running_stats),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),   # 250Hz: 1000→125, 128Hz: 256→32
            nn.Dropout(eegnet_dropout),
        )
 
        # ── TCN blocks ────────────────────────────────────────────────────
        # Exponentially increasing dilation: 1, 2, 4, ...
        tcn_blocks = []
        for i in range(tcn_layers):
            in_ch  = F2 if i == 0 else tcn_filters
            dilation = 2 ** i
            tcn_blocks.append(
                TemporalBlock(
                    in_channels=in_ch,
                    out_channels=tcn_filters,
                    kernel_size=tcn_kernel,
                    dilation=dilation,
                    dropout=tcn_dropout,
                    track_running_stats=track_running_stats,
                )
            )
        self.tcn = nn.Sequential(*tcn_blocks)
 
        # ── Classifier ────────────────────────────────────────────────────
        flat_size = self._get_flat_size()
        self.classifier = nn.Sequential(
            nn.Dropout(fc_dropout),
            nn.Linear(flat_size, n_classes),
        )

    def _get_flat_size(self) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.n_channels, self.n_timepoints)
            out = self._forward_features(dummy)
        return out.shape[-1]
 
 
    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """EEGNet front-end + TCN, returns flat feature vector."""
        x = self.temporal_conv(x)    # (B, F1, C, T)
        x = self.depthwise_conv(x)   # (B, F2, 1, T//8)
        x = x.squeeze(2)             # (B, F2, T//8)
        x = self.tcn(x)              # (B, tcn_filters, T//8)
        x = x.mean(dim=-1)            # take last timestep — causal output
        return x
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_channels, n_timepoints) or (batch, 1, n_channels, n_timepoints)
        Returns:
            logits: (batch, n_classes)
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self._forward_features(x)
        return self.classifier(x)
 
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
 
 
# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
 
if __name__ == "__main__":
    print("=" * 55)
    print(" EEG-TCNet — Sanity Check")
    print("=" * 55)
 
    model = TCN()
    x = torch.randn(8, 22, 1000)
 
    out = model(x)
    print(f"Input shape:      {tuple(x.shape)}")
    print(f"Output shape:     {tuple(out.shape)}")
    print(f"Trainable params: {model.count_parameters():,}")
    assert out.shape == (8, 4), f"Unexpected output shape: {out.shape}"
 
    # 4D input
    x4 = torch.randn(8, 1, 22, 1000)
    out4 = model(x4)
    assert out4.shape == (8, 4)
 
    print("-" * 55)
    print("All sanity checks passed")
 
    print("\n── Layer breakdown ──")
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"  {name:25s}  {n:>8,} params")