from gnn import *
from n_conditioning import NEmbedding, make_film_conditioned_gnn_fn


class NConditionedGNFBlock(snt.AbstractModule):
    """Same coupling-layer structure as GNFBlock, but s and t are computed
    by N-conditioned networks (see n_conditioning.py): N is embedded once
    per forward pass and fed into every s/t network's own FiLM generator.

    Also mirrors GNFBlock's per-step batch-norm renormalization, applied to
    x0/x1 immediately before they're fed into that step's s/t network.
    Without it, the unclamped affine scale exp(s) compounds across stacked
    coupling layers -- each layer's output is the next layer's s/t input
    with no renormalization in between -- and reliably overflows to inf/nan
    within a handful of layers. The batch-norm term contributes its own
    additional piece to log_det_jacobian, exactly as in GNFBlock.

    Unlike GNFBlock, each batch-norm step uses a *pair* of bijectors
    (make_batch_norm_pair(), not make_batch_norm()) sharing one underlying
    layer: bn_train (training=True, current-batch statistics) for f()'s
    .inverse() calls, and bn_generate (training=False, the layer's
    accumulated running average) for g()'s .forward() calls. A single
    training=True instance used for both, as GNFBlock does, normalizes
    generation-time samples using statistics computed fresh from that
    generated batch -- not the real-data-derived distribution the layer's
    gamma/beta were actually calibrated against during training. That
    mismatch was confirmed empirically: a trained checkpoint's generated
    embeddings had ~5-11x the L2 norm of real encoder embeddings, enough
    to make the (unchanged) decoder predict "no edge" almost everywhere.

    N never touches the half of the nodes that passes through unchanged
    (x0 in the first sub-step, x1 in the second) and never appears in the
    log-det term directly — it can only ever influence log_det_jacobian
    through s, same as before the batch-norm addition.
    """

    def __init__(self,
                 num_timesteps,
                 node_embedding_dim,
                 hidden_dim,
                 n_embed_dim=32,
                 weight_sharing=False,
                 use_batch_norm=False,
                 name="NConditionedGNFBlock"):
        super(NConditionedGNFBlock, self).__init__(name=name)
        self.num_timesteps = num_timesteps
        self.weight_sharing = weight_sharing
        self.use_batch_norm = use_batch_norm
        with self._enter_variable_scope():
            self.n_embedding = NEmbedding(n_embed_dim)
            make_s_fn = make_film_conditioned_gnn_fn(hidden_dim,
                                                      node_embedding_dim)
            make_t_fn = make_film_conditioned_gnn_fn(hidden_dim,
                                                      node_embedding_dim)
            if weight_sharing:
                self.s = [make_s_fn(), make_s_fn()]
                self.t = [make_t_fn(), make_t_fn()]
            else:
                self.s = [
                    get_gnns(num_timesteps, make_s_fn),
                    get_gnns(num_timesteps, make_s_fn)
                ]
                self.t = [
                    get_gnns(num_timesteps, make_t_fn),
                    get_gnns(num_timesteps, make_t_fn)
                ]
            self.bns = [[make_batch_norm_pair() for _ in range(num_timesteps)],
                        [make_batch_norm_pair() for _ in range(num_timesteps)]]

    def f(self, x):
        log_det_jacobian = 0
        n_embedding = self.n_embedding(x)
        x0, x1 = tf.split(x.nodes, num_or_size_splits=2, axis=1)
        x0 = x.replace(nodes=x0)
        x1 = x.replace(nodes=x1)
        for i in range(self.num_timesteps):
            if self.use_batch_norm:
                bn_train, _ = self.bns[0][i]
                log_det_jacobian += bn_train.inverse_log_det_jacobian(
                    x0.nodes, 2)
                x0 = x0.replace(nodes=bn_train.inverse(x0.nodes))
            if self.weight_sharing:
                s = self.s[0](x0, n_embedding).nodes
                t = self.t[0](x0, n_embedding).nodes
            else:
                s = self.s[0][i](x0, n_embedding).nodes
                t = self.t[0][i](x0, n_embedding).nodes
            log_det_jacobian += tf.reduce_sum(s)
            x1 = x1.replace(nodes=x1.nodes * tf.exp(s) + t)

            if self.use_batch_norm:
                bn_train, _ = self.bns[1][i]
                log_det_jacobian += bn_train.inverse_log_det_jacobian(
                    x1.nodes, 2)
                x1 = x1.replace(nodes=bn_train.inverse(x1.nodes))
            if self.weight_sharing:
                s = self.s[1](x1, n_embedding).nodes
                t = self.t[1](x1, n_embedding).nodes
            else:
                s = self.s[1][i](x1, n_embedding).nodes
                t = self.t[1][i](x1, n_embedding).nodes
            log_det_jacobian += tf.reduce_sum(s)
            x0 = x0.replace(nodes=x0.nodes * tf.exp(s) + t)

        x = x.replace(nodes=tf.concat([x0.nodes, x1.nodes], axis=1))
        return x, log_det_jacobian

    def g(self, z):
        n_embedding = self.n_embedding(z)
        z0, z1 = tf.split(z.nodes, num_or_size_splits=2, axis=1)
        z0 = z.replace(nodes=z0)
        z1 = z.replace(nodes=z1)
        for i in reversed(range(self.num_timesteps)):
            if self.weight_sharing:
                s = self.s[1](z1, n_embedding).nodes
                t = self.t[1](z1, n_embedding).nodes
            else:
                s = self.s[1][i](z1, n_embedding).nodes
                t = self.t[1][i](z1, n_embedding).nodes
            if self.use_batch_norm:
                _, bn_generate = self.bns[1][i]
                z1 = z1.replace(nodes=bn_generate.forward(z1.nodes))
            z0 = z0.replace(nodes=(z0.nodes - t) * tf.exp(-s))

            if self.weight_sharing:
                s = self.s[0](z0, n_embedding).nodes
                t = self.t[0](z0, n_embedding).nodes
            else:
                s = self.s[0][i](z0, n_embedding).nodes
                t = self.t[0][i](z0, n_embedding).nodes
            if self.use_batch_norm:
                _, bn_generate = self.bns[0][i]
                z0 = z0.replace(nodes=bn_generate.forward(z0.nodes))
            z1 = z1.replace(nodes=(z1.nodes - t) * tf.exp(-s))
        return z.replace(nodes=tf.concat([z0.nodes, z1.nodes], axis=1))

    def _build(self, input, inverse=True):
        func = self.f if inverse else self.g
        return func(input)


class GNFBlock(snt.AbstractModule):
    def __init__(self,
                 make_gnn_fn,
                 num_timesteps,
                 node_embedding_dim,
                 use_batch_norm=False,
                 weight_sharing=False,
                 use_efficient_backprop=True,
                 name="GNFBlock"):
        super(GNFBlock, self).__init__(name=name)
        self.num_timesteps = num_timesteps
        self.weight_sharing = weight_sharing
        self.use_batch_norm = use_batch_norm
        self.use_efficient_backprop = use_efficient_backprop
        if weight_sharing:
            self.s = [make_gnn_fn(), make_gnn_fn()]
            self.t = [make_gnn_fn(), make_gnn_fn()]
        else:
            self.s = [
                get_gnns(num_timesteps, make_gnn_fn),
                get_gnns(num_timesteps, make_gnn_fn)
            ]
            self.t = [
                get_gnns(num_timesteps, make_gnn_fn),
                get_gnns(num_timesteps, make_gnn_fn)
            ]
        self.bns = [[make_batch_norm() for _ in range(num_timesteps)],
                    [make_batch_norm() for _ in range(num_timesteps)]]

    def f(self, x):
        log_det_jacobian = 0
        x0, x1 = tf.split(x.nodes, num_or_size_splits=2, axis=1)
        x0 = x.replace(nodes=x0)
        x1 = x.replace(nodes=x1)
        for i in range(self.num_timesteps):
            if self.use_batch_norm:
                bn = self.bns[0][i]
                log_det_jacobian += bn.inverse_log_det_jacobian(x0.nodes, 2)
                x0 = x0.replace(nodes=bn.inverse(x0.nodes))
            if self.weight_sharing:
                s = self.s[0](x0).nodes
                t = self.t[0](x0).nodes
            else:
                s = self.s[0][i](x0).nodes
                t = self.t[0][i](x0).nodes
            log_det_jacobian += tf.reduce_sum(s)
            x1 = x1.replace(nodes=x1.nodes * tf.exp(s) + t)

            if self.use_batch_norm:
                bn = self.bns[1][i]
                log_det_jacobian += bn.inverse_log_det_jacobian(x1.nodes, 2)
                x1 = x1.replace(nodes=bn.inverse(x1.nodes))
            if self.weight_sharing:
                s = self.s[1](x1).nodes
                t = self.t[1](x1).nodes
            else:
                s = self.s[1][i](x1).nodes
                t = self.t[1][i](x1).nodes
            log_det_jacobian += tf.reduce_sum(s)
            x0 = x0.replace(nodes=x0.nodes * tf.exp(s) + t)

        x = x.replace(nodes=tf.concat([x0.nodes, x1.nodes], axis=1))
        return x, log_det_jacobian

    def g(self, z):
        z0, z1 = tf.split(z.nodes, num_or_size_splits=2, axis=1)
        z0 = z.replace(nodes=z0)
        z1 = z.replace(nodes=z1)
        for i in reversed(range(self.num_timesteps)):
            if self.weight_sharing:
                s = self.s[1](z1).nodes
                t = self.t[1](z1).nodes
            else:
                s = self.s[1][i](z1).nodes
                t = self.t[1][i](z1).nodes
            if self.use_batch_norm:
                z1 = z1.replace(nodes=self.bns[1][i].forward(z1.nodes))
            z0 = z0.replace(nodes=(z0.nodes - t) * tf.exp(-s))

            if self.weight_sharing:
                s = self.s[0](z0).nodes
                t = self.t[0](z0).nodes
            else:
                s = self.s[0][i](z0).nodes
                t = self.t[0][i](z0).nodes
            if self.use_batch_norm:
                z0 = z0.replace(nodes=self.bns[0][i].forward(z0.nodes))
            z1 = z1.replace(nodes=(z1.nodes - t) * tf.exp(-s))
        return z.replace(nodes=tf.concat([z0.nodes, z1.nodes], axis=1))

    def _build(self, input, inverse=True):
        func = self.f if inverse else self.g
        return func(input)
