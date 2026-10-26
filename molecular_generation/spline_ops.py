"""Monotonic rational-quadratic spline transform (Durkan et al., 2019,
"Neural Spline Flows") with identity linear tails, written once against a
tiny ops interface so the same code runs on NumPy (where it is unit tested
locally) and on TensorFlow (where the flow uses it).

Per scalar x, K bins on [-bound, bound]; outside that interval the map is
the identity (log-derivative 0). Parameters per element (last axis):
  raw_w [K]    bin widths (softmax),
  raw_h [K]    bin heights (softmax),
  raw_d [K-1]  interior knot derivatives (softplus); the two boundary
               derivatives are fixed to 1 so the spline joins the identity
               tails smoothly.
All-zero raw parameters give exactly the identity map (uniform bins, unit
slopes), which is what lets a zero-initialised conditioner start the flow
as the identity, like the affine coupling layers do.
"""

import numpy as np


class NumpyOps(object):

    @staticmethod
    def softmax(a):
        e = np.exp(a - a.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    @staticmethod
    def softplus(a):
        return np.logaddexp(a, 0.0)

    exp = staticmethod(np.exp)
    log = staticmethod(np.log)
    sqrt = staticmethod(np.sqrt)
    maximum = staticmethod(np.maximum)
    zeros_like = staticmethod(np.zeros_like)
    ones_like = staticmethod(np.ones_like)

    @staticmethod
    def cumsum(a):
        return np.cumsum(a, axis=-1)

    @staticmethod
    def concat(parts):
        return np.concatenate(parts, axis=-1)

    @staticmethod
    def clip(a, lo, hi):
        return np.clip(a, lo, hi)

    @staticmethod
    def where(cond, a, b):
        return np.where(cond, a, b)

    @staticmethod
    def logical_and(a, b):
        return np.logical_and(a, b)

    @staticmethod
    def bin_index(x, inner_knots):
        return (x[..., None] >= inner_knots).sum(axis=-1)

    @staticmethod
    def select(params, idx):
        return np.take_along_axis(params, idx[..., None], axis=-1)[..., 0]


class TFOps(object):
    """The same interface on TensorFlow. Imported lazily so this module
    (and the NumPy tests) work without TensorFlow installed."""

    def __init__(self):
        import tensorflow as tf
        self.tf = tf

    def softmax(self, a):
        return self.tf.nn.softmax(a, axis=-1)

    def softplus(self, a):
        return self.tf.nn.softplus(a)

    def exp(self, a):
        return self.tf.exp(a)

    def log(self, a):
        return self.tf.log(a)

    def sqrt(self, a):
        return self.tf.sqrt(a)

    def maximum(self, a, b):
        return self.tf.maximum(a, b)

    def zeros_like(self, a):
        return self.tf.zeros_like(a)

    def ones_like(self, a):
        return self.tf.ones_like(a)

    def cumsum(self, a):
        return self.tf.cumsum(a, axis=-1)

    def concat(self, parts):
        return self.tf.concat(parts, axis=-1)

    def clip(self, a, lo, hi):
        return self.tf.clip_by_value(a, lo, hi)

    def where(self, cond, a, b):
        return self.tf.where(cond, a, b)

    def logical_and(self, a, b):
        return self.tf.logical_and(a, b)

    def bin_index(self, x, inner_knots):
        tf = self.tf
        return tf.reduce_sum(
            tf.cast(tf.expand_dims(x, -1) >= inner_knots, tf.int32), axis=-1)

    def select(self, params, idx):
        tf = self.tf
        depth = params.get_shape().as_list()[-1]
        onehot = tf.one_hot(idx, depth, dtype=params.dtype)
        return tf.reduce_sum(onehot * params, axis=-1)


def _knots(raw_w, raw_h, raw_d, ops, num_bins, bound, min_bin, min_deriv):
    widths = min_bin + (1.0 - min_bin * num_bins) * ops.softmax(raw_w)
    heights = min_bin + (1.0 - min_bin * num_bins) * ops.softmax(raw_h)
    zeros = ops.zeros_like(widths[..., :1])
    knots_x = -bound + 2.0 * bound * ops.concat([zeros, ops.cumsum(widths)])
    knots_y = -bound + 2.0 * bound * ops.concat([zeros, ops.cumsum(heights)])
    # raw_d == 0 maps to derivative exactly 1 (identity slope).
    offset = float(np.log(np.exp(1.0 - min_deriv) - 1.0))
    inner = min_deriv + ops.softplus(raw_d + offset)
    ones = ops.ones_like(inner[..., :1])
    derivs = ops.concat([ones, inner, ones])
    return knots_x, knots_y, derivs


def _bin_values(idx, knots_x, knots_y, derivs, ops):
    sel = ops.select
    x_k, x_k1 = sel(knots_x[..., :-1], idx), sel(knots_x[..., 1:], idx)
    y_k, y_k1 = sel(knots_y[..., :-1], idx), sel(knots_y[..., 1:], idx)
    d_k, d_k1 = sel(derivs[..., :-1], idx), sel(derivs[..., 1:], idx)
    return x_k, x_k1 - x_k, y_k, y_k1 - y_k, d_k, d_k1


def _forward_in_bin(xi, y_k, h, s, d_k, d_k1):
    xi1 = xi * (1.0 - xi)
    denom = s + (d_k1 + d_k - 2.0 * s) * xi1
    y = y_k + h * (s * xi * xi + d_k * xi1) / denom
    deriv = (s * s) * (d_k1 * xi * xi + 2.0 * s * xi1 +
                       d_k * (1.0 - xi) * (1.0 - xi)) / (denom * denom)
    return y, deriv


def rqs_forward(x, raw_w, raw_h, raw_d, ops, num_bins=8, bound=5.0,
                min_bin=1e-3, min_deriv=1e-3):
    """Returns (y, log|dy/dx|), elementwise."""
    knots_x, knots_y, derivs = _knots(raw_w, raw_h, raw_d, ops, num_bins,
                                      bound, min_bin, min_deriv)
    inside = ops.logical_and(x > -bound, x < bound)
    xc = ops.clip(x, -bound, bound)
    idx = ops.bin_index(xc, knots_x[..., 1:-1])
    x_k, w, y_k, h, d_k, d_k1 = _bin_values(idx, knots_x, knots_y, derivs, ops)
    xi = ops.clip((xc - x_k) / w, 0.0, 1.0)
    y, deriv = _forward_in_bin(xi, y_k, h, h / w, d_k, d_k1)
    return (ops.where(inside, y, x),
            ops.where(inside, ops.log(deriv), ops.zeros_like(x)))


def rqs_inverse(y, raw_w, raw_h, raw_d, ops, num_bins=8, bound=5.0,
                min_bin=1e-3, min_deriv=1e-3):
    """Returns (x, log|dy/dx| evaluated at that x) -- the log-derivative of
    the FORWARD map, so the log-det of the inverse is its negative."""
    knots_x, knots_y, derivs = _knots(raw_w, raw_h, raw_d, ops, num_bins,
                                      bound, min_bin, min_deriv)
    inside = ops.logical_and(y > -bound, y < bound)
    yc = ops.clip(y, -bound, bound)
    idx = ops.bin_index(yc, knots_y[..., 1:-1])
    x_k, w, y_k, h, d_k, d_k1 = _bin_values(idx, knots_x, knots_y, derivs, ops)
    s = h / w
    dy = yc - y_k
    a = dy * (d_k1 + d_k - 2.0 * s) + h * (s - d_k)
    b = h * d_k - dy * (d_k1 + d_k - 2.0 * s)
    c = -s * dy
    disc = ops.maximum(b * b - 4.0 * a * c, 0.0 * b)
    xi = ops.clip(2.0 * c / (-b - ops.sqrt(disc)), 0.0, 1.0)
    x = xi * w + x_k
    _, deriv = _forward_in_bin(xi, y_k, h, s, d_k, d_k1)
    return (ops.where(inside, x, y),
            ops.where(inside, ops.log(deriv), ops.zeros_like(y)))
