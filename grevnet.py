from gnn import *


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
