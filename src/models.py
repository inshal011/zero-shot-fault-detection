"""Model components for zero-shot fault detection.

Architecture: shared convolutional encoder trained with
(1) masked spectrogram reconstruction (self-supervised, no labels needed)
(2) domain-adversarial training via gradient reversal (DANN) so the
    embedding space cannot discriminate the source machine type,
    encouraging machine-agnostic representations of "normal" dynamics.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReversal(torch.autograd.Function):
    """Identity forward; reversed (scaled) gradient backward."""

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd=1.0):
    return GradientReversal.apply(x, lambd)


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1),
            nn.BatchNorm2d(c_out),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class Encoder(nn.Module):
    """CNN encoder for (1, n_mels, n_frames) log-mel patches -> embedding."""

    def __init__(self, n_mels=128, n_frames=64, emb_dim=128):
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBlock(1, 32),    # 64x32
            ConvBlock(32, 64),   # 32x16
            ConvBlock(64, 128),  # 16x8
            ConvBlock(128, 256), # 8x4
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(256, emb_dim)
        self.emb_dim = emb_dim

    def forward(self, x):
        h = self.backbone(x)
        z = self.pool(h).flatten(1)
        return self.proj(z)


class Decoder(nn.Module):
    """Reconstructs the full log-mel patch from the embedding."""

    def __init__(self, n_mels=128, n_frames=64, emb_dim=128):
        super().__init__()
        self.n_mels, self.n_frames = n_mels, n_frames
        self.fc = nn.Linear(emb_dim, 256 * (n_mels // 16) * (n_frames // 16))
        self.h0, self.w0 = n_mels // 16, n_frames // 16
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),
        )

    def forward(self, z):
        h = self.fc(z).view(-1, 256, self.h0, self.w0)
        return self.net(h)


class DomainClassifier(nn.Module):
    """Predicts source machine type from embedding (trained adversarially)."""

    def __init__(self, emb_dim=128, n_domains=4, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden), nn.GELU(),
            nn.Linear(hidden, n_domains),
        )

    def forward(self, z, lambd=1.0):
        return self.net(grad_reverse(z, lambd))


class ZeroShotModel(nn.Module):
    def __init__(self, n_mels=128, n_frames=64, emb_dim=128, n_domains=4,
                 use_adversarial=True):
        super().__init__()
        self.encoder = Encoder(n_mels, n_frames, emb_dim)
        self.decoder = Decoder(n_mels, n_frames, emb_dim)
        self.use_adversarial = use_adversarial
        if use_adversarial:
            self.domain_head = DomainClassifier(emb_dim, n_domains)

    def forward(self, x, mask_ratio=0.3, grl_lambda=1.0):
        """x: (B, 1, n_mels, n_frames). Masks random time columns, encodes
        the masked input, reconstructs the *unmasked* target."""
        if self.training and mask_ratio > 0:
            B, _, M, T = x.shape
            mask = (torch.rand(B, 1, 1, T, device=x.device) > mask_ratio).float()
            x_in = x * mask
        else:
            x_in = x
        z = self.encoder(x_in)
        recon = self.decoder(z)
        out = {"z": z, "recon": recon}
        if self.use_adversarial:
            out["domain_logits"] = self.domain_head(z, grl_lambda)
        return out


def reconstruction_error(model, x):
    """Per-sample MSE between unmasked input and its reconstruction (eval)."""
    model.eval()
    with torch.no_grad():
        z = model.encoder(x)
        recon = model.decoder(z)
        err = F.mse_loss(recon, x, reduction="none").mean(dim=(1, 2, 3))
    return err, z
