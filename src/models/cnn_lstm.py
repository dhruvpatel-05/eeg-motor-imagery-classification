"""
CNN+LSTM hybrid architecture for EEG motor imagery classification — 128 Hz / ERS pipeline.
Based on: Bashivan et al. (2016) - https://arxiv.org/abs/1511.06448

This is the 128 Hz-adapted variant of cnn_lstm.py. Key differences:
  - sfreq parameter: temporal kernel derived as sfreq // 2 (= 64 at 128 Hz)
    matching EEGNet convention of ≥2 Hz frequency resolution, rather than
    the hardcoded kernel=25 (≈195 ms at 128 Hz) of the 250 Hz original.
  - BatchNorm2d uses track_running_stats=False: at eval time the model uses
    per-batch statistics instead of accumulated running stats, which is
    correct for cross-subject generalisation where the test distribution
    shifts. Consistent with cnn_gru.py and alternative_eegnet.py.
  - dropout_rate is forwarded from the training CLI (--dropout flag).

Architecture overview:
  1. Temporal conv  — learns frequency-specific features along time axis
  2. Depthwise spatial conv — learns which electrode combinations matter
  3. LSTM           — models long-range temporal dependencies
  4. Classifier     — linear layer to 4 classes

Input:  (batch, 22, 256)  — 22 EEG channels, 256 timepoints (2s @ 128Hz)
Output: (batch, 4)        — logits for 4 motor imagery classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class CNNLSTM(nn.Module):
    """
    CNN+LSTM hybrid for EEG motor imagery — 128 Hz / ERS pipeline variant.

    Args:
        n_classes     : number of output classes (default: 4)
        n_channels    : number of EEG channels (default: 22)
        n_timepoints  : number of time samples (default: 256 = 2s @ 128Hz)
        sfreq         : sampling frequency in Hz (default: 128)
                        temporal kernel is derived as sfreq // 2
        F1            : number of temporal filters (default: 32)
        F2            : number of pointwise filters = F1 * D (default: 64)
        lstm_hidden   : LSTM hidden state size (default: 128)
        lstm_layers   : number of LSTM layers (default: 2)
        dropout_rate  : dropout probability (default: 0.5)
    """

    def __init__(
        self,
        n_classes: int = 4,
        n_channels: int = 22,
        n_timepoints: int = 256,
        sfreq: int = 128,
        F1: int = 32,
        F2: int = 64,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout_rate: float = 0.5,
    ):
        super().__init__()

        self.n_channels = n_channels
        self.n_timepoints = n_timepoints
        self.lstm_hidden = lstm_hidden

        temporal_kernel = sfreq // 2   # 64 at 128 Hz — matches ≥2 Hz resolution
        padding = temporal_kernel // 2

        # -------------------------------------------------------------------
        # Stage 1: Temporal convolution
        # Kernel slides along time axis only.
        # temporal_kernel = sfreq // 2 (64 at 128 Hz) guarantees that the
        # filter covers at least one full cycle of the lowest frequency (2 Hz),
        # matching the EEGNet / CNN-GRU design convention.
        # track_running_stats=False: use per-batch stats at eval time —
        # correct for cross-subject evaluation where distribution shifts.
        # -------------------------------------------------------------------
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, temporal_kernel), padding=(0, padding), bias=False),
            nn.BatchNorm2d(F1, track_running_stats=False),
            nn.ELU()
        )

        # -------------------------------------------------------------------
        # Stage 2: Depthwise spatial convolution
        # Kernel (n_channels, 1) collapses the electrode dimension.
        # groups=F1 means each temporal filter gets its own spatial filter
        # (depthwise — no cross-filter mixing yet).
        # Followed by pooling to reduce time from T -> T/4.
        # -------------------------------------------------------------------
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(F1, F2, kernel_size=(n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F2, track_running_stats=False),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout_rate),
        )

        # -------------------------------------------------------------------
        # Stage 3: LSTM
        # Input: (batch, seq_len=T/4, input_size=F2).
        # At 128 Hz: T=256 -> seq_len=64 after pooling.
        # batch_first=True so input/output are (batch, seq, features)
        # dropout between LSTM layers when lstm_layers > 1
        # -------------------------------------------------------------------
        self.lstm = nn.LSTM(
            input_size=F2,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout_rate if lstm_layers > 1 else 0.0
        )

        # -------------------------------------------------------------------
        # Classifier
        # -------------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_hidden, n_classes),
        )

        self._dw_constraint = MaxNormConstraint(
            self.spatial_conv[0].weight, max_norm=1.0, dim=0
        )
        self._cls_constraint = MaxNormConstraint(
            self.classifier[1].weight, max_norm=0.25, dim=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_channels, n_timepoints) or (batch, 1, n_channels, n_timepoints)
        Returns:
            logits: (batch, n_classes)
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # CNN stages
        x = self.temporal_conv(x)   # (batch, F1, n_channels, T)
        x = self.spatial_conv(x)    # (batch, F2, 1, T/4)

        # Reshape for LSTM: (batch, F2, 1, T/4) → (batch, T/4, F2)
        x = x.squeeze(2)            # (batch, F2, T/4)
        x = x.permute(0, 2, 1)     # (batch, T/4, F2)

        # LSTM: mean-pool over all timesteps
        x, _ = self.lstm(x)
        x = x.mean(dim=1)

        return self.classifier(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def apply_constraints(self):
        self._dw_constraint()
        self._cls_constraint()

    def apply_max_norm_(self) -> None:
        self.apply_constraints()


# sanity check
if __name__ == "__main__":
    model = CNNLSTM(sfreq=128, n_timepoints=256)
    x = torch.randn(8, 22, 256)    # batch of 8, 128 Hz input
    out = model(x)
    print(f"Input:           {tuple(x.shape)}")
    print(f"Output:          {tuple(out.shape)}")
    print(f"Parameters:      {model.count_parameters():,}")
    print(f"Temporal kernel: {model.temporal_conv[0].kernel_size}")  # expect (1, 64)
    assert out.shape == (8, 4), "Unexpected output shape"
    assert model.temporal_conv[0].kernel_size == (1, 64), "Temporal kernel should be 64 at 128 Hz"
    print("CNN+LSTM (128 Hz alternative) sanity check passed")
