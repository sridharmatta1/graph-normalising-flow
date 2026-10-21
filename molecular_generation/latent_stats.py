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


def _rbf_mmd2_unbiased(x, y, bandwidth):
    def sqdist(a, b):
        return np.maximum(
            (a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None, :] - 2.0 * a @ b.T,
            0.0)
    gamma = 1.0 / (2.0 * bandwidth ** 2)
    kxx = np.exp(-gamma * sqdist(x, x))
    kyy = np.exp(-gamma * sqdist(y, y))
    kxy = np.exp(-gamma * sqdist(x, y))
    n, m = len(x), len(y)
    return float((kxx.sum() - np.trace(kxx)) / (n * (n - 1)) +
                 (kyy.sum() - np.trace(kyy)) / (m * (m - 1)) -
                 2.0 * kxy.mean())


def compare_embeddings(real_mols, gen_mols, rng, max_mmd_atoms=1500):
    """Compares real encoder embeddings with embeddings sampled from a flow.

    real_mols, gen_mols: [num_molecules, N, D] (one molecule size N).
    Reports, between the two sets of per-atom D-vectors:
      std_ratio_*       per-dimension std(gen)/std(real); 1.0 = same spread.
      mean_shift_*      |mean(gen)-mean(real)| / std(real) per dimension.
      corr_*            64x64 correlation matrices: mean |off-diagonal|
                        for each, and ||C_gen - C_real||_F relative to
                        ||C_real - I||_F (how much of the real correlation
                        structure is mismatched).
      molmean_std_ratio per-dimension std of the per-molecule mean
                        embedding, gen/real (between-molecule variation).
      mmd2 / mmd2_null  unbiased RBF-MMD^2 real-vs-gen, against the same
                        statistic between two halves of the real data
                        (the noise floor: mmd2 near mmd2_null = no
                        detectable difference).
    """
    d = real_mols.shape[-1]
    real = real_mols.reshape(-1, d).astype(np.float64)
    gen = gen_mols.reshape(-1, d).astype(np.float64)

    std_r, std_g = real.std(axis=0), gen.std(axis=0)
    ratio = std_g / std_r
    shift = np.abs(gen.mean(axis=0) - real.mean(axis=0)) / std_r

    def corr(a):
        return np.nan_to_num(np.corrcoef(a.T))
    c_r, c_g = corr(real), corr(gen)
    off = ~np.eye(d, dtype=bool)
    denom = np.linalg.norm(c_r - np.eye(d))

    mm_r = real_mols.astype(np.float64).mean(axis=1).std(axis=0)
    mm_g = gen_mols.astype(np.float64).mean(axis=1).std(axis=0)

    def subsample(a, k):
        return a[rng.choice(len(a), size=min(k, len(a)), replace=False)]
    xr = subsample(real, max_mmd_atoms)
    xg = subsample(gen, max_mmd_atoms)
    half = len(xr) // 2
    pair = np.sqrt(np.maximum(
        (xr ** 2).sum(1)[:, None] + (xr ** 2).sum(1)[None, :] -
        2.0 * xr @ xr.T, 0.0))
    bandwidth = float(np.median(pair[pair > 0]))

    return {
        'std_ratio_mean': float(ratio.mean()),
        'std_ratio_p10': float(np.percentile(ratio, 10)),
        'std_ratio_p90': float(np.percentile(ratio, 90)),
        'std_ratio_min': float(ratio.min()),
        'std_ratio_max': float(ratio.max()),
        'mean_shift_mean': float(shift.mean()),
        'mean_shift_max': float(shift.max()),
        'corr_offdiag_real': float(np.abs(c_r[off]).mean()),
        'corr_offdiag_gen': float(np.abs(c_g[off]).mean()),
        'corr_mismatch_rel': float(np.linalg.norm(c_g - c_r) /
                                   max(denom, 1e-12)),
        'molmean_std_ratio': float((mm_g / mm_r).mean()),
        'mmd2': _rbf_mmd2_unbiased(xr, xg, bandwidth),
        'mmd2_null': _rbf_mmd2_unbiased(xr[:half], xr[half:2 * half],
                                        bandwidth),
        'n_real_atoms': int(len(real)),
        'n_gen_atoms': int(len(gen)),
    }
