import tensorflow as tf 
import os
import pdb

class Network(object):
    def __init__(self, name):
        self.name = name

    def batch_norm(self, x, training, decay=0.9, name='batch_norm', reuse=None):
        with tf.variable_scope(name, reuse=reuse):
            dim = x.shape[1].value
            moving_mean = tf.get_variable('moving_mean', [dim],
                    initializer=tf.zeros_initializer(), trainable=False)
            moving_var = tf.get_variable('moving_var', [dim],
                    initializer=tf.ones_initializer(), trainable=False)
            beta = tf.get_variable('beta', [dim],
                    initializer=tf.zeros_initializer())
            gamma = tf.get_variable('gamma', [dim],
                    initializer=tf.ones_initializer())

            if training:
                x, batch_mean, batch_var = tf.nn.fused_batch_norm(x, gamma, beta, data_format='NCHW')
                update_mean = moving_mean.assign_sub((1-decay)*(moving_mean - batch_mean))
                update_var = moving_var.assign_sub((1-decay)*(moving_var - batch_var))
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mean)
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_var)
            else:
                x, batch_mean, batch_var = tf.nn.fused_batch_norm(x, gamma, beta,
                        mean=moving_mean, variance=moving_var, is_training=False,
                        data_format='NCHW')
            return x

class MAMLNet(Network):
    def __init__(self, name, nway, kshot, qsize, mbsize, norm=False, reuse=False, 
            inner_loop_iter=5, stop_grad=True, isTr=True):
        self.name = name
        self.nway = nway
        self.kshot = kshot
        self.qsize = qsize
        self.mbsize = mbsize
        self.norm = norm 
        self.inner_loop_iter = inner_loop_iter
        self.stop_grad = stop_grad
        
        self.inputs = {\
                'sx': tf.placeholder(tf.float32, [mbsize,nway*kshot,84,84,3], 
                    name='ph_sx'),
                'sy': tf.placeholder(tf.float32, [mbsize,nway*kshot,nway],
                    name='ph_sy'),
                'qx': tf.placeholder(tf.float32, [mbsize,nway*qsize,84,84,3],
                    name='ph_qx'),
                'qy': tf.placeholder(tf.float32, [mbsize,nway*qsize,nway],
                    name='ph_qy'),
                'tr': tf.placeholder(tf.bool, name='ph_isTr'),
                'lr_alpha': tf.placeholder(tf.float32, [], name='ph_alpha'),
                'lr_beta': tf.placeholder(tf.float32, [], name='ph_beta')}

        self.outputs = {\
                'preda': None,
                'predb': None,
                'lossa': None,
                'lossb': None,
                'accuracy': None}

            
        with tf.variable_scope(self.name, reuse=reuse):
            self._build_network(isTr, reuse=reuse)

    def _build_network(self, isTr, reuse=False):
        ip = self.inputs
        weights = self.construct_weights()

        meta_lossbs = []; meta_predbs = []; # (# metabatch, # inner updates)
        _ = self.forward(ip['sx'][0], weights, reuse=False)
        
        def singlebatch_graph(inputs):
            sx, sy, qx, qy = inputs
            # when sinlge batch is given, get the lossbs, predbs
            # sy.shape : (nk, n)  / qy.shape : (nq, n)
            lossbs, predbs = [], []
            preda = self.forward(sx, weights, reuse=True)
            lossa = tf.reduce_mean(cross_entropy(preda, sy))
            grads = tf.gradients(lossa, list(weights.values()))
            if self.stop_grad:
                grads = [tf.stop_gradient(grad) for grad in grads]
            gradients = dict(zip(weights.keys(), grads))
            adapted_weights = dict(zip(weights.keys(), 
                [weights[key] - ip['lr_alpha'] * gradients[key] \
                        for key in weights.keys()]))
            predb = self.forward(qx, adapted_weights, reuse=True)
            lossb = cross_entropy(predb, qy)

            predbs.append(predb)
            lossbs.append(lossb)

            for _ in range(self.inner_loop_iter - 1):
                preda = self.forward(sx, adapted_weights, reuse=True)
                lossa = tf.reduce_mean(cross_entropy(preda, sy))
                grads = tf.gradients(lossa, list(adapted_weights.values()))
                if self.stop_grad:
                    grads = [tf.stop_gradient(grad) for grad in grads]
                gradients = dict(zip(adapted_weights.keys(), grads))
                adapted_weights = dict(zip(adapted_weights.keys(),
                    [adapted_weights[key] - ip['lr_alpha'] * gradients[key] \
                            for key in adapted_weights.keys()]))
                predb = self.forward(qx, adapted_weights, reuse=True)
                lossb = cross_entropy(predb, qy)

                predbs.append(predb)
                lossbs.append(lossb)
            # predbs.shape = (# of inner iter, nq, n)
            # lossbs.shape = (# of inner iter, nq)
            return predbs, lossbs

        elems = (ip['sx'], ip['sy'], ip['qx'], ip['qy'])
        out_dtype = ([tf.float32]*self.inner_loop_iter,
                [tf.float32]*self.inner_loop_iter)
        meta_predbs, meta_lossbs = \
                tf.map_fn(singlebatch_graph, elems, dtype=out_dtype,
                parallel_iterations=self.mbsize)

        # loop version doesn't work at all. Don't know why
#        lvmeta_predbs, lvmeta_lossbs = [], []
#        for mbi in range(self.mbsize):
#            predbs, lossbs = singlebatch_graph((ip['sx'][mbi],
#                    ip['sy'][mbi], ip['qx'][mbi], ip['qy'][mbi]))
#            lvmeta_lossbs.append(lossbs)
#            lvmeta_predbs.append(predbs)
#        meta_lossbs = tf.transpose(lvmeta_lossbs, [1,0,2])
#        meta_predbs = tf.transpose(lvmeta_predbs, [1,0,2,3])

        # meta_predbs.shape = (#inner_loop, meta_batch, nq, n)
        # meta_lossbs.shape = (#inner_loop, meta_batch, nq)
        
        self.outputs['predb'] = meta_predbs
        self.outputs['lossb'] = meta_lossbs 
        #self.outputs['lossa'] = meta_lossas
        # lossb (metabatch, innerup, nq)

        if isTr:
            opt_loss = tf.reduce_mean(meta_lossbs, (1,2))[-1] 
            opt = tf.train.AdamOptimizer(ip['lr_beta'])
            gvs = opt.compute_gradients(opt_loss)
            gvs = [(tf.clip_by_value(grad, -10, 10), var) for grad, var in gvs]
            self.train_op = opt.apply_gradients(gvs)
            self.gvs = gvs
    

        self.outputs['accuracy'] = \
                [tf_acc(meta_predbs[-1][mi], ip['qy'][mi]) \
                for mi in range(self.mbsize)]
#        self.outputs['accuracy'] = [tf_acc(mp, ip['qy'][mi]) \
#                for mi, mp in enumerate(meta_predbs[-1])]

    def construct_weights(self):
        weights = {}
        f32 = tf.float32
        conv_init = tf.contrib.layers.xavier_initializer_conv2d(dtype=f32)
        fc_init = tf.contrib.layers.xavier_initializer(dtype=f32)
        
        # first conv 3->32
        weights['conv{}'.format(1)] = tf.get_variable('conv{}'.format(1),
                [3,3,3,32], initializer=conv_init, dtype=f32)
        weights['bias{}'.format(1)] = tf.get_variable('bias{}'.format(1),
                initializer=tf.zeros([32]))
        for i in range(2, 5):
            weights['conv{}'.format(i)] = tf.get_variable('conv{}'.format(i),
                    [3,3,32,32], initializer=conv_init, dtype=f32)
            weights['bias{}'.format(i)] = tf.get_variable('bias{}'.format(i),
                    initializer=tf.zeros([32]))
#            weights['beta{}'.format(i)] = tf.get_variable('beta{}'.format(i),
#                    initializer=tf.zeros([32]))
#            weights['gamma{}'.format(i)] = tf.get_variable('gamma{}'.format(i),
#                    initializer=tf.ones([32]))

        weights['w5'] = tf.get_variable('w5', 
                [32*5*5, self.nway], initializer=fc_init)
        weights['b5'] = tf.get_variable('b5',
                initializer=tf.zeros([self.nway]))
        return weights

    def forward(self, x, weights, reuse=False, scope=''):

        def conv(x, w, b, reuse, scope):
            h = tf.nn.conv2d(x, w, [1,1,1,1], 'SAME') + b
            h = tf.contrib.layers.batch_norm(h, activation_fn=tf.nn.relu,
                    reuse=reuse, scope=scope)
            #h = batch_norm(h, True, name=scope+'bn', reuse=reuse)
            #h, _, _ = tf.nn.fused_batch_norm(h, gamma, beta)
            return tf.nn.max_pool(h, [1,2,2,1], [1,2,2,1], 'VALID')
    
        for i in range(1, 5):
            x = conv(x, weights['conv{}'.format(i)], weights['bias{}'.format(i)],
                    reuse, scope+'{}'.format(i))
                    #weights['beta{}'.format(i)], weights['gamma{}'.format(i)], 
        dim = 1
        for s in x.get_shape().as_list()[1:]:
            dim *= s
        x = tf.reshape(x, [-1, dim])
        out = tf.matmul(x, weights['w5']) + weights['b5']
        return out


#def batch_norm(x, training, decay=0.9, name='batch_norm', reuse=None):
#    with tf.variable_scope(name, reuse=reuse):
#        dim = x.shape[-1].value
#        moving_mean = tf.get_variable('moving_mean', [dim],
#                initializer=tf.zeros_initializer(), trainable=False)
#        moving_var = tf.get_variable('moving_var', [dim],
#                initializer=tf.ones_initializer(), trainable=False)
#        beta = tf.get_variable('beta', [dim],
#                initializer=tf.zeros_initializer())
#        gamma = tf.get_variable('gamma', [dim],
#                initializer=tf.ones_initializer())
#        if training:
#            x, batch_mean, batch_var = tf.nn.fused_batch_norm(x, gamma, beta, data_format='NHWC')
#            update_mean = moving_mean.assign_sub((1-decay)*(moving_mean - batch_mean))
#            update_var = moving_var.assign_sub((1-decay)*(moving_var - batch_var))
#            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mean)
#            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_var)
#        else:
#            x, batch_mean, batch_var = tf.nn.fused_batch_norm(x, gamma, beta,
#                    mean=moving_mean, variance=moving_var, is_training=False,
#                    data_format='NHWC')
#        return x

def cross_entropy(pred, label):
    ce = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label)
    return ce
    #return -tf.reduce_mean(tf.reduce_sum(label*tf.log(pred+1e-10), axis=1))

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
