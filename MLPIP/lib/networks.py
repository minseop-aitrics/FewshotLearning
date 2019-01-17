import tensorflow as tf 
import os
import pdb

relu = tf.nn.relu
elu = tf.nn.elu
normal = tf.distributions.Normal
kldv = tf.distributions.kl_divergence

class Network(object):
    def __init__(self, name):
        self.name = name
        self.eps = 1e-3
        self.cdim = 64

    def dense(self, x, units, name='dense', reuse=None):
        with tf.variable_scope(name, reuse=reuse):
            kernel = tf.get_variable('kernel', [x.shape[1].value, units])
            bias = tf.get_variable('bias', [units],
                    initializer=tf.zeros_initializer())
            x = tf.matmul(x, kernel) + bias
            return x

    def conv(self, x, filters, kernel_size=3, strides=1, padding='SAME',
            name='conv', reuse=None):
        with tf.variable_scope(name, reuse=reuse):
            kernel = tf.get_variable('kernel',
                    [kernel_size, kernel_size, x.shape[-1].value, filters])
            x = tf.nn.conv2d(x, kernel, [1, 1, strides, strides],
                    padding=padding)
            return x

    def batch_norm(self, x, training, decay=0.9, name='batch_norm', reuse=None):
        with tf.variable_scope(name, reuse=reuse):
            dim = x.shape[-1].value
            moving_mean = tf.get_variable('moving_mean', [dim],
                    initializer=tf.zeros_initializer(), trainable=False)
            moving_var = tf.get_variable('moving_var', [dim],
                    initializer=tf.ones_initializer(), trainable=False)
            beta = tf.get_variable('beta', [dim],
                    initializer=tf.zeros_initializer())
            gamma = tf.get_variable('gamma', [dim],
                    initializer=tf.ones_initializer())

            if training:
                x, batch_mean, batch_var = tf.nn.fused_batch_norm(x, gamma, beta)
                update_mean = moving_mean.assign_sub((1-decay)*(moving_mean - batch_mean))
                update_var = moving_var.assign_sub((1-decay)*(moving_var - batch_var))
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mean)
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_var)
            else:
                x, batch_mean, batch_var = tf.nn.fused_batch_norm(x, gamma, beta,
                        mean=moving_mean, variance=moving_var, is_training=False)
            return x

    def global_avg_pool(self, x):
        return tf.reduce_mean(x, [2, 3])

    def simple_conv(self, in_x, reuse=False, isTr=True):
        def conv_block(x, name, reuse, isTr):
            x = self.conv(x, self.cdim, name=name+'/conv', reuse=reuse)
            x = self.batch_norm(x, isTr, name=name+'/bn', reuse=reuse)
            x = relu(x)
#            if isTr:
#                x = tf.nn.dropout(x, 0.5)
            x = tf.nn.max_pool(x, [1,2,2,1], [1,2,2,1], 'VALID')
            return x
        x = in_x
        for i in range(5):
            x = conv_block(x, 'b{}'.format(i+1), reuse=reuse, isTr=isTr)
        x = tf.layers.flatten(x)
        return x

    def qpsi_H(self, sh, reuse=False):
        h = self.dense(sh, self.hdim, name='q1', reuse=reuse)
        proto_h = tf.reshape(h, [self.nway, -1, self.hdim])
        proto_h = tf.reduce_mean(proto_h, axis=1)
        mu_w = tf.nn.elu(self.dense(proto_h, self.hdim, name='qmu', reuse=reuse))
        sig_w = tf.nn.elu(self.dense(proto_h, self.hdim, name='qsig', reuse=reuse))
        #sig_w = tf.clip_by_value(tf.exp(sig_w * .5), 1e-4, 10.)
        return mu_w, sig_w

class MLPIP(Network):
    def __init__(self, name, nway, kshot, qsize, isTr, reuse=False):
        self.name = name
        self.nway = nway
        self.kshot = kshot
        self.qsize = qsize
        self.cdim = 64
        self.hdim = 256

        self.inputs = {\
                'sx': tf.placeholder(tf.float32, [None,84,84,3]),
                'qx': tf.placeholder(tf.float32, [None,84,84,3]),
                'qy': tf.placeholder(tf.float32, [None,None]),
                'lr': tf.placeholder(tf.float32),
                'tr': tf.placeholder(tf.bool)}
        self.outputs = {}

        with tf.variable_scope(name):
            self._build_network(isTr, reuse=reuse)

    def _build_network(self, isTr, reuse):
        ip = self.inputs
        sq_inputs = tf.concat([ip['sx'], ip['qx']], axis=0)
        sq_outputs = self.simple_conv(sq_inputs, reuse, isTr)
        support_h = sq_outputs[:self.nway*self.kshot]
        query_h = sq_outputs[self.nway*self.kshot:]

        mu_ws, sig_ws = self.qpsi_H(support_h, reuse=reuse)
        samples = normal(mu_ws, sig_ws).sample()
        # samples.shape : (n, hdim)
        pred = tf.matmul(query_h, samples, transpose_b=True)
        pred = tf.nn.softmax(pred)
        self.outputs['pred'] = pred
        self.outputs['loss'] = cross_entropy(pred, ip['qy'])
        self.outputs['acc'] = tf_acc(pred, ip['qy'])
        self.probe = [mu_ws, sig_ws]





def cross_entropy(pred, label): 
    return -tf.reduce_mean(tf.reduce_sum(label*tf.log(pred+1e-10), axis=1))

def cross_entropy_with_metabatch(pred, label):
    # shape of pred, label: (metabatch, batch, nway)
    return -tf.reduce_mean(tf.reduce_sum(label*tf.log(pred+1e-10), axis=2), axis=1)

def tf_acc(p, y): 
    acc = tf.equal(tf.argmax(y,1), tf.argmax(p,1))
    acc = tf.reduce_mean(tf.cast(acc, 'float'))
    return acc

def ckpt_restore_with_prefix(sess, ckpt_dir, prefix):
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=prefix)
    var_list_name = [i.name.split(':')[0] for i in var_list]

    for var_name, _ in tf.contrib.framework.list_variables(ckpt_dir):
        var = tf.contrib.framework.load_variable(ckpt_dir, var_name)
        new_name = prefix + '/' + var_name
        if new_name in var_list_name:
            with tf.variable_scope(prefix, reuse=True):
                tfvar = tf.get_variable(var_name)
                sess.run(tfvar.assign(var))
