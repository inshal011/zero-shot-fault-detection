"""Anomaly scoring without target-domain priors.

All statistics are estimated ONLY from source-domain normal data:
  - Mahalanobis distance in embedding space against the pooled
    source-normal Gaussian (Ledoit-Wolf shrinkage covariance).
  - Reconstruction error of the masked autoencoder head.
  - Combined score: z-normalize each (using source statistics) and sum.

No target-domain samples, labels, or statistics are used at any point.
"""
import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from torch.utils.data import DataLoader

from .models import reconstruction_error


@torch.no_grad()
def embed_dataset(model, dataset, device, batch_size=64):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    zs, errs, labels, fids = [], [], [], []
    for x, _, y, fid in loader:
        x = x.to(device)
        err, z = reconstruction_error(model, x)
        zs.append(z.cpu().numpy())
        errs.append(err.cpu().numpy())
        labels.append(y.numpy())
        fids.append(fid.numpy())
    return (np.concatenate(zs), np.concatenate(errs),
            np.concatenate(labels), np.concatenate(fids))


class SourceNormalScorer:
    def __init__(self):
        self.mu = None
        self.prec = None
        self.recon_mu = None
        self.recon_sd = None
        self.maha_mu = None
        self.maha_sd = None

    def fit(self, z_src, recon_err_src):
        lw = LedoitWolf().fit(z_src)
        self.mu = lw.location_
        self.prec = lw.precision_
        maha = self._maha(z_src)
        self.maha_mu, self.maha_sd = maha.mean(), maha.std() + 1e-9
        self.recon_mu = recon_err_src.mean()
        self.recon_sd = recon_err_src.std() + 1e-9
        return self

    def _maha(self, z):
        d = z - self.mu
        return np.sqrt(np.einsum("bi,ij,bj->b", d, self.prec, d))

    def score(self, z, recon_err, mode="combined"):
        maha = self._maha(z)
        if mode == "maha":
            return maha
        if mode == "recon":
            return recon_err
        z_m = (maha - self.maha_mu) / self.maha_sd
        z_r = (recon_err - self.recon_mu) / self.recon_sd
        return z_m + z_r


def file_level_scores(patch_scores, patch_labels, patch_fids):
    """Aggregate patch scores to file level (mean); returns (scores, labels)."""
    out_s, out_y = [], []
    for fid in np.unique(patch_fids):
        m = patch_fids == fid
        out_s.append(patch_scores[m].mean())
        out_y.append(patch_labels[m].max())
    return np.array(out_s), np.array(out_y)
