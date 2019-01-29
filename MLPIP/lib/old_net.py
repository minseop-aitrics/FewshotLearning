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
        # sh.shape : (nk, hdim)
        h = self.dense(sh, self.hdim, name='q1', reuse=reuse)
        proto_h = tf.reshape(h, [self.nway, -1, self.hdim])
        proto_h = tf.reduce_mean(proto_h, axis=1)
        mu_w = tf.nn.elu(self.dense(proto_h, self.hdim, name='qmu', reuse=reuse))
        logsig_w2 = tf.nn.elu(self.dense(proto_h, self.hdim, name='qsig', reuse=reuse))
        #sig_w = tf.exp(logsig_w2*.5)
        sig_w = tf.clip_by_value(tf.exp(logsig_w2 * .5), 1e-8, 10.)
        return mu_w, sig_w


class MLPIP(Network):
    # meta-batch version
    def __init__(self, name, nway, kshot, qsize, mbsize, isTr=True, reuse=False):
        self.name = name
        self.nway = nway
        self.kshot = kshot
        self.qsize = qsize
        self.mbsize = mbsize
        self.n_samples = 10
        self.cdim = 64
        self.hdim = 256

        self.inputs = {\
                'sx': tf.placeholder(tf.float32, [mbsize,None,84,84,3]),
                'sy': tf.placeholder(tf.float32, [mbsize,None,nway]),
                'qx': tf.placeholder(tf.float32, [mbsize,None,84,84,3]),
                'qy': tf.placeholder(tf.float32, [mbsize,None,nway]),
                'lr': tf.placeholder(tf.float32)}
        self.outputs = {}

        with tf.variable_scope(name):
            self._build_network(isTr, reuse=reuse)

    def _build_network(self, isTr, reuse):
        ip = self.inputs
        # let just feed all of them for batch stats.
        sx_all = tf.reshape(ip['sx'], [-1,84,84,3]) # (mbsize*nway*kshot, -)
        qx_all = tf.reshape(ip['qx'], [-1,84,84,3]) # (mbsize*nway*qsize, -)
        x_all = tf.concat([sx_all, qx_all], axis=0) # (m*n*k+m*n*q, -)
            
        # if training mode, moving averages will be saved and batch stats will be used
        # by all of support and query sets, and  in test mode, just 
        # m.a will be used. 
        h_all = self.simple_conv(x_all, reuse, isTr)
        support_h = h_all[:self.mbsize*self.nway*self.kshot]
        support_h = tf.reshape(support_h, [self.mbsize, self.nway*self.kshot,-1])

        query_h = h_all[self.mbsize*self.nway*self.kshot:]
        query_h = tf.reshape(query_h, [self.mbsize, self.nway*self.qsize,-1])
        
        # for reuse term 
        _, _ = self.qpsi_H(support_h[0], reuse=reuse)

        def single_batch_process(inputs):
            sh, qh, qy = inputs # support_h, query_h, qeury_y
            mu_ws, sig_ws = self.qpsi_H(sh, reuse=True)
            samples = normal(mu_ws, sig_ws).sample(self.n_samples)
            # samples.shape : (n_samples, n, hdim)
            psis = tf.transpose(samples, [0,2,1])
            xs = tf.tile(tf.expand_dims(qh, 0), [self.n_samples,1,1])
#            pred = tf.nn.softmax(tf.matmul(xs, psis))
#            pred = tf.reduce_mean(pred, axis=0)
            pred = tf.reduce_mean(tf.matmul(xs, psis), axis=0)
            pred = tf.nn.softmax(pred)
            loss = cross_entropy(pred, qy)
            acc = tf_acc(pred, qy)
            return loss, acc

        def simple_baseline(inputs): 
            # this is for meta-batch prototypical network 
            sh, qh, qy = inputs
            proto_vec = tf.reshape(sh, [self.nway, self.kshot, self.hdim])
            proto_vec = tf.reduce_mean(proto_vec, axis=1)
            _p = tf.expand_dims(proto_vec, axis=0)
            _q = tf.expand_dims(qh, axis=1)
            emb = (_p-_q)**2
            dist = tf.reduce_mean(emb, axis=2)
            pred = tf.nn.softmax(-dist)
            loss = cross_entropy(pred, qy)
            acc = tf_acc(pred, qy)
            return loss, acc

        elems = (support_h, query_h, ip['qy'])
        out_dtype = (tf.float32, tf.float32)

        loss, acc = tf.map_fn(simple_baseline, elems, dtype=out_dtype,
                parallel_iterations=self.mbsize)

        self.outputs['loss'] = tf.reduce_mean(loss)
        self.outputs['acc'] = acc

        if isTr:
            opt = tf.train.AdamOptimizer(ip['lr'])
            update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_op):
                self.train_op = opt.minimize(loss)
            

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
