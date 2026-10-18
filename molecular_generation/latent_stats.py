"""Per-dimension calibration statistics of flow latents z against the
prior's own (mu, sigma). TensorFlow-free so it can be unit tested locally.

For each latent dimension d, with u = (z_d - mu_d) / sigma_d over all atoms
of one molecule size N:

  zscore_rms   sqrt(E[u^2]). The likelihood-optimal sigma_d is the RMS of
               (z_d - mu_d), so 1.0 means the prior's spread matches the
               data's in that dimension; 1.5 means the data is 1.5x wider
               than the prior assumes.
  kurtosis     E[u^4] / E[u^2]^2. 3.0 for a Gaussian; larger = heavy tails.
  abs_ratio    E|u| / sqrt(E[u^2]). 0.798 for a Gaussian; smaller = heavier
               tails / more peaked.
  bias         E[u], the mean offset in sigma units.

It also reproduces inspect_embedding_space.py's pooled statistic (one std
over every atom and dimension, compared with the arithmetic mean of the
per-dimension sigmas) so the two can be compared directly: that statistic
is inflated whenever sigma differs across dimensions, even for a perfectly
calibrated prior.
"""

import numpy as np

GAUSSIAN_ABS_RATIO = float(np.sqrt(2.0 / np.pi))


class LatentCalibrationStats(object):

    def __init__(self, dim):
        self.dim = dim
        self.count = 0
        self.s1 = np.zeros(dim)
        self.s2 = np.zeros(dim)
        self.sa = np.zeros(dim)
        self.s4 = np.zeros(dim)
        self.pooled_sum = 0.0
        self.pooled_sum_sq = 0.0
        self.pooled_count = 0
        self.mu = None
        self.sigma = None

    def update(self, z, mu, sigma):
        """z: [num_atoms, dim]; mu, sigma: [dim] (constant for one N)."""
        z = np.asarray(z, dtype=np.float64)
        self.mu = np.asarray(mu, dtype=np.float64)
        self.sigma = np.asarray(sigma, dtype=np.float64)
        u = (z - self.mu) / self.sigma
        self.count += u.shape[0]
        self.s1 += u.sum(axis=0)
        self.s2 += (u ** 2).sum(axis=0)
        self.sa += np.abs(u).sum(axis=0)
        self.s4 += (u ** 4).sum(axis=0)
        self.pooled_sum += z.sum()
        self.pooled_sum_sq += (z ** 2).sum()
        self.pooled_count += z.size

    def summary(self, near_floor=0.15, wide=1.25, narrow=0.8):
        n = float(self.count)
        q = self.s2 / n
        rms_u = np.sqrt(q)
        kurt = (self.s4 / n) / q ** 2
        abs_ratio = (self.sa / n) / rms_u
        bias = self.s1 / n
        pooled_mean = self.pooled_sum / self.pooled_count
        pooled_std = float(np.sqrt(self.pooled_sum_sq / self.pooled_count -
                                   pooled_mean ** 2))
        sig = self.sigma
        sigma_mean = float(sig.mean())
        sigma_rms = float(np.sqrt((sig ** 2).mean()))
        return {
            'n_atoms': int(self.count),
            'sigma_mean': sigma_mean,
            'sigma_rms': sigma_rms,
            'sigma_geomean': float(np.exp(np.log(sig).mean())),
            'sigma_min': float(sig.min()),
            'sigma_max': float(sig.max()),
            'dims_sigma_near_floor': int((sig < near_floor).sum()),
            'zscore_rms': float(np.sqrt(q.mean())),
            'zscore_rms_dim_median': float(np.median(rms_u)),
            'zscore_rms_dim_p10': float(np.percentile(rms_u, 10)),
            'zscore_rms_dim_p90': float(np.percentile(rms_u, 90)),
            'frac_dims_data_wider': float((rms_u > wide).mean()),
            'frac_dims_data_narrower': float((rms_u < narrow).mean()),
            'abs_ratio': float(abs_ratio.mean()),
            'kurtosis_mean': float(kurt.mean()),
            'kurtosis_median': float(np.median(kurt)),
            'bias_abs_mean': float(np.abs(bias).mean()),
            'pooled_std': pooled_std,
            'pooled_over_sigma_mean': pooled_std / sigma_mean,
            'pooled_over_sigma_rms': pooled_std / sigma_rms,
        }
