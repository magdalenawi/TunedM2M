from utils import *


class TunedM2M():
    def __init__(self, hid_units, Aprop_num, n_heads, n_nodes, f_dimension, nb_classes,
                 lr, reg, prob, prob_att, beta, adj_idx, idx_size):
        self.X = tf.sparse_placeholder(tf.float32, name='X')
        self.y = tf.placeholder('float32', name='y')
        self.mask = tf.placeholder('float32', name='Mask')
        self.adj = tf.sparse_placeholder(tf.float32, name='ADJ')
        self.dropout = tf.placeholder('float32', name='dropout')
        self.dropout_att = tf.placeholder('float32', name='dropout_att')
        self.nodes = n_nodes
        self.prob = prob
        self.prob_att = prob_att
        self.reg = reg

        attns = []
        z_save = []
        for _ in range(n_heads):
            att_z, z = NI_ATT(
                self.X,
                in_sz=f_dimension,
                adj_mat=self.adj,
                out_sz=hid_units,
                activation=tf.nn.elu,
                nb_nodes=n_nodes,
                att_drop=self.dropout_att
            )
            attns.append(att_z)
            z_save.append(z)
        z_att = tf.concat(attns, axis=-1)
        z = tf.concat(z_save, axis=-1)
        self.recon_loss = self.recon_reg(adj_idx, idx_size, z, n_nodes, beta=beta)

        self.concat_z = [z_att]
        for i in range(Aprop_num - 1):
            self.concat_z.append(dot(self.adj, self.concat_z[i], True))
        h = tf.concat(self.concat_z, 1)
        h = tf.nn.dropout(h, self.dropout)
        logits = tf.layers.dense(
            h, nb_classes,
            activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False)
        )
        self.loss(logits, lr)

    def accuracy(self, X, Y, mask, adj):
        """Get accuracy"""
        return self.sess.run(self.acc,
                             feed_dict={self.adj: adj,
                                        self.X: X,
                                        self.y: Y,
                                        self.mask: mask,
                                        self.dropout: 1.0,
                                        self.dropout_att: 1.0})

    def train(self, batch_xs, batch_ys, mask, adj):
        _ = self.sess.run(self.trains, feed_dict={self.y: batch_ys,
                                                  self.adj: adj,
                                                  self.X: batch_xs,
                                                  self.mask: mask,
                                                  self.dropout: self.prob,
                                                  self.dropout_att: self.prob_att})

    def loss(self, logits, lr):
        var = tf.trainable_variables()
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in var if 'bias' not in v.name]) * self.reg

        self.cost = masked_softmax_cross_entropy(logits, self.y, self.mask) + lossL2 + self.recon_loss

        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9)
        self.trains = self.optimizer.minimize(self.cost)
        self.acc = masked_accuracy(logits, self.y, self.mask)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def recon_reg(self, adj_idx, idx_size, z, n_nodes, beta):
        idx_pos = np.random.choice(np.arange(len(adj_idx)), size=idx_size, replace=False)
        node_self = tf.gather(z, adj_idx[idx_pos][:, 0])
        neighborhood = tf.gather(z, adj_idx[idx_pos][:, 1])
        dot_product = node_self * neighborhood
        adj_pos = tf.nn.sigmoid(tf.reduce_sum(dot_product, axis=-1))
        adj_one = tf.ones(idx_size)
        positive = beta * (adj_one - adj_pos)

        idx_neg = np.random.choice(np.arange(n_nodes), size=idx_size, replace=False)
        negative_node = tf.gather(z, idx_neg)
        dot_product = node_self * negative_node
        adj_neg = tf.nn.sigmoid(tf.reduce_sum(dot_product, axis=-1))
        negative = (1 - beta) * (adj_one - adj_neg)
        return tf.reduce_sum(tf.maximum(0.0, positive - negative))/tf.constant(idx_size, dtype=tf.float32)

    def get_loss(self, batch_xs, batch_ys, mask, adj ):
        c = self.sess.run(self.cost, feed_dict={self.X: batch_xs,
                                                self.y: batch_ys,
                                                self.mask: mask,
                                                self.dropout: 1.0,
                                                self.dropout_att: 1.0,
                                                self.adj: adj})
        return c
